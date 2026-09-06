"""Face -> Web Search -> Candidate Verification -> Fingerprint -> Blockchain.

NON-COMMERCIAL research / demo. Intended ONLY for a consenting participant or a
controlled test image. Do not use for surveillance, mass identification, or
building dossiers on people.

Phase 1 + 2 + 3:
  [1] load image
  [2] detect single face + embedding
  [3] reverse image search via SerpApi Google Lens
  [4] download candidate images + verify candidate faces

Phase 4:
  [5] create SHA-256 fingerprint of the Phase 3 verification data
  [6] record fingerprint on Ethereum Sepolia
  [7] retrieve fingerprint and verify it

Usage:
    python app.py --image data/input/test.jpg
    python app.py --image data/input/test.jpg --debug-search
    python app.py --image data/input/test.jpg --max-candidates 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import config
from utils import logger
from pipeline import face_detector as fd
from pipeline import reverse_search as rs
from pipeline import candidate_matcher as cm
from pipeline.fingerprint import (
    sha256_fingerprint,
    sha256_file,
    build_canonical_evidence,
)
from pipeline.blockchain import (
    BlockchainError,
    record_fingerprint,
    get_web3,
    get_onchain_fingerprint,
    fetch_tx_state,
    SEPOLIA_CHAIN_ID,
)
from web3.exceptions import TimeExhausted


log = logger.get_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Detect a single face, reverse-search the image with "
            "SerpApi Google Lens, verify returned candidate images, "
            "fingerprint the results, and record the fingerprint on "
            "Ethereum Sepolia."
        ),
    )

    parser.add_argument(
        "--image",
        required=False,
        default=None,
        help="Path to the input image (a consenting subject or test image).",
    )

    parser.add_argument(
        "--verify-proof",
        default=None,
        metavar="PATH",
        help=(
            "Re-verify a saved proof file (e.g. "
            "data/results/blockchain_proof.json) against the blockchain, "
            "without sending a new transaction. Rebuilds the canonical "
            "evidence, recomputes the SHA-256, and compares it to the "
            "on-chain fingerprint."
        ),
    )

    parser.add_argument(
        "--recover-tx",
        default=None,
        metavar="TX_HASH",
        help=(
            "READ-ONLY recovery for an already-submitted transaction (e.g. "
            "after a receipt-wait timeout). Fetches the receipt, decodes the "
            "on-chain fingerprint, independently recomputes the canonical "
            "fingerprint from the current verification.json, compares them, "
            "and writes blockchain_proof.json. Never sends a transaction."
        ),
    )

    parser.add_argument(
        "--verification",
        default=None,
        metavar="PATH",
        help=(
            "Path to the Phase 3 verification.json used to rebuild canonical "
            "evidence during --recover-tx "
            "(default: data/results/verification.json)."
        ),
    )

    parser.add_argument(
        "--crop-out",
        default=None,
        help=(
            "Where to save the face crop "
            "(default: data/results/<image>_face.jpg)."
        ),
    )

    parser.add_argument(
        "--model",
        default=config.insightface_model,
        help=(
            f"InsightFace model pack "
            f"(default: {config.insightface_model})."
        ),
    )

    parser.add_argument(
        "--det-size",
        type=int,
        default=config.det_size,
        help=f"Detector input size (default: {config.det_size}).",
    )

    parser.add_argument(
        "--det-threshold",
        type=float,
        default=config.det_threshold,
        help=(
            "Minimum detector confidence "
            f"(default: {config.det_threshold})."
        ),
    )

    parser.add_argument(
        "--debug-search",
        action="store_true",
        help=(
            "Write the raw SerpApi JSON (credentials scrubbed) to "
            "data/results/lens_response.json."
        ),
    )

    parser.add_argument(
        "--max-show",
        type=int,
        default=10,
        help="Max search candidates to print (default: 10).",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help=(
            "Max Google Lens candidates to download and verify "
            "(default: 10)."
        ),
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=cm.MATCH_THRESHOLD,
        help=(
            "Similarity >= this value is a match "
            f"(default: {cm.MATCH_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--uncertain-threshold",
        type=float,
        default=cm.UNCERTAIN_THRESHOLD,
        help=(
            "Similarity >= this value but below match threshold is uncertain "
            f"(default: {cm.UNCERTAIN_THRESHOLD})."
        ),
    )

    return parser


def run(args: argparse.Namespace) -> int:
    config.ensure_dirs()

    # Standalone re-verification mode: no image / pipeline needed.
    if args.verify_proof:
        return _verify_proof_file(Path(args.verify_proof))

    # READ-ONLY recovery of an already-submitted transaction. Never sends.
    if args.recover_tx:
        verification_path = (
            Path(args.verification)
            if args.verification
            else config.results_dir / "verification.json"
        )
        return _recover_tx(args.recover_tx, verification_path)

    if not args.image:
        logger.failure("--image is required (or use --verify-proof PATH).")
        return 1

    if args.max_show < 0:
        logger.failure("--max-show must be >= 0.")
        return 13

    if args.max_candidates < 0:
        logger.failure("--max-candidates must be >= 0.")
        return 14

    try:
        cm.classify_similarity(
            0.0,
            match_threshold=args.match_threshold,
            uncertain_threshold=args.uncertain_threshold,
        )
    except ValueError as e:
        logger.failure(f"Invalid matching thresholds: {e}")
        return 15

    image_path = Path(args.image)

    # Default crop output path derived from the input filename.
    if args.crop_out:
        crop_out = Path(args.crop_out)
    else:
        crop_out = config.results_dir / f"{image_path.stem}_face.jpg"

    # -------------------------------------------------------------------------
    # Stage 1: load image
    # -------------------------------------------------------------------------
    logger.stage(1, 5, f"Loading image  ({image_path})")

    try:
        image = fd.load_image(image_path)

    except fd.ImageLoadError as e:
        logger.failure(str(e))
        return 2

    h, w = image.shape[:2]

    logger.info(f"Image loaded: {w}x{h} px")

    # Stable SHA-256 of the original input bytes — part of the Phase 4 evidence
    # contract. Same image → same hash, every run.
    input_image_hash = sha256_file(image_path)

    # -------------------------------------------------------------------------
    # Stage 2: detect face + embedding
    # -------------------------------------------------------------------------
    logger.stage(2, 5, "Detecting face")

    try:
        result = fd.analyze_image(
            image_path,
            crop_path=crop_out,
            model_name=args.model,
            det_size=args.det_size,
            det_threshold=args.det_threshold,
            use_cpu=config.use_cpu,
        )

    except fd.NoFaceError:
        logger.failure(
            "No face detected. "
            "Provide an image with exactly one clear face."
        )
        return 3

    except fd.MultipleFacesError as e:
        logger.failure(
            f"{e}  (the MVP accepts exactly one face)"
        )
        return 4

    except fd.FaceDetectionError as e:
        logger.failure(
            f"Face stage failed: {e}"
        )
        return 5

    logger.success("Face detected")

    logger.info(
        f"Bounding box (x1,y1,x2,y2): {result.bbox}"
    )

    logger.info(
        f"Detector confidence: {result.det_score:.4f}"
    )

    logger.success("Embedding generated")

    logger.info(
        f"Embedding: {result.embedding_preview()}"
    )

    if result.crop_path:
        logger.info(
            f"Face crop saved: {result.crop_path}"
        )

    # -------------------------------------------------------------------------
    # Stage 3: reverse image search
    # -------------------------------------------------------------------------
    logger.stage(3, 5, "Searching web with Google Lens")

    try:
        search = rs.search_image(
            image_path=image_path,
            api_key=config.serpapi_key,
            engine=config.serpapi_engine,
        )

    except rs.MissingAPIKeyError as e:
        logger.failure(str(e))
        return 6

    except rs.InvalidAPIKeyError as e:
        logger.failure(
            f"SerpApi rejected the API key: {e}"
        )
        return 7

    except rs.RateLimitError as e:
        logger.failure(
            f"SerpApi rate limit / quota exceeded: {e}"
        )
        return 8

    except rs.SearchNetworkError as e:
        logger.failure(
            f"Network/timeout during search: {e}"
        )
        return 9

    except rs.MalformedResponseError as e:
        logger.failure(
            f"Could not parse SerpApi response: {e}"
        )
        return 10

    except rs.ImageInputError as e:
        logger.failure(
            f"Image problem for search: {e}"
        )
        return 11

    except rs.ReverseSearchError as e:
        logger.failure(
            f"Reverse search failed: {e}"
        )
        return 12

    if args.debug_search:
        _write_debug_search(search)

    _print_search_results(
        search,
        max_show=args.max_show,
    )

    # -------------------------------------------------------------------------
    # No candidates
    # -------------------------------------------------------------------------
    if search.count == 0:
        print()

        logger.info(
            "No web candidates returned. "
            "Candidate verification skipped."
        )

        verification_path = _write_verification_results(
            image_path=image_path,
            search=search,
            matches=[],
            max_candidates=args.max_candidates,
            match_threshold=args.match_threshold,
            uncertain_threshold=args.uncertain_threshold,
            input_image_hash=input_image_hash,
        )

        logger.info(
            f"Verification results written to: {verification_path}"
        )

        # Even with zero candidates, we fingerprint the resulting
        # canonical evidence so the pipeline remains deterministic.
        return _run_phase4(verification_path)

    # -------------------------------------------------------------------------
    # Stage 4: candidate verification
    # -------------------------------------------------------------------------
    logger.stage(
        4,
        5,
        f"Verifying candidate faces (max {args.max_candidates})",
    )

    try:
        matches = cm.verify_candidates(
            result.embedding,
            search.candidates,
            limit=args.max_candidates,
            output_dir=config.candidates_dir,
            model_name=args.model,
            det_size=args.det_size,
            det_threshold=args.det_threshold,
            use_cpu=config.use_cpu,
            match_threshold=args.match_threshold,
            uncertain_threshold=args.uncertain_threshold,
        )

    except Exception as e:
        logger.failure(
            f"Candidate verification failed unexpectedly: {e}"
        )
        return 16

    _print_verification_results(matches)

    verification_path = _write_verification_results(
        image_path=image_path,
        search=search,
        matches=matches,
        max_candidates=args.max_candidates,
        match_threshold=args.match_threshold,
        uncertain_threshold=args.uncertain_threshold,
        input_image_hash=input_image_hash,
    )

    logger.info(
        f"Verification results written to: {verification_path}"
    )

    # -------------------------------------------------------------------------
    # Stage 5: Phase 4
    # -------------------------------------------------------------------------
    return _run_phase4(verification_path)


def _load_verification(verification_path: Path) -> dict:
    """Load the Phase 3 verification.json, raising ValueError if malformed."""
    with verification_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("verification.json is not a JSON object")
    return data


def _evidence_and_fingerprint(verification_data: dict) -> tuple[dict, str]:
    """Build canonical evidence and its SHA-256 fingerprint (canonical, not raw)."""
    canonical = build_canonical_evidence(verification_data)
    fingerprint = sha256_fingerprint(canonical)
    return canonical, fingerprint

def _print_final_summary(
    *,
    verification_path: Path,
    proof_path: Path,
    fingerprint: str,
    tx_hash: str,
    blockchain_result: str,
) -> None:
    """Print a concise final summary for the CLI/demo."""

    try:
        verification_data = _load_verification(verification_path)
        results = verification_data.get(
            "verification",
            {},
        ).get(
            "results",
            [],
        )

        scored = [
            result
            for result in results
            if result.get("best_similarity") is not None
        ]

        best = (
            max(
                scored,
                key=lambda result: result["best_similarity"],
            )
            if scored
            else None
        )

    except Exception:
        best = None

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    print()
    print("Pipeline")
    print("  ✓ Face detected")
    print("  ✓ Google Lens search completed")
    print("  ✓ Candidate verification completed")
    print("  ✓ Evidence fingerprint created")
    print(f"  ✓ Blockchain proof: {blockchain_result}")

    if best is not None:
        print()
        print("Best candidate")
        print(
            f"  Title:      "
            f"{best.get('title') or '(no title)'}"
        )
        print(
            f"  Similarity: "
            f"{best['best_similarity']:.4f}"
        )
        print(
            f"  Status:     "
            f"{best.get('status', '(unknown)')}"
        )
        print(
            f"  Source:     "
            f"{best.get('source') or '(unknown)'}"
        )

    print()
    print("Proof")
    print(f"  SHA-256:    {fingerprint}")
    print(f"  TX hash:    {tx_hash}")
    print(f"  Verification: {proof_path}")
    print(f"  Candidates:   {verification_path}")

    print()
    print("=" * 60)

def _run_phase4(verification_path: Path) -> int:
    """Canonical evidence → SHA-256 → Sepolia write → read → recompute → verify."""

    print()
    print("=" * 50)
    print("PHASE 4 — FINGERPRINT + BLOCKCHAIN")
    print("=" * 50)
    print()

    # ---------------------------------------------------------------------
    # Step 1: canonical evidence + fingerprint
    # ---------------------------------------------------------------------
    logger.info("  Step 1/5 — Building canonical evidence + SHA-256 fingerprint")

    try:
        verification_data = _load_verification(verification_path)
        canonical, fingerprint = _evidence_and_fingerprint(verification_data)
    except Exception as e:
        logger.failure(f"Fingerprint creation failed: {e}")
        return 17

    logger.success("Canonical evidence fingerprinted")
    logger.info(f"SHA-256: {fingerprint}")

    # ---------------------------------------------------------------------
    # Step 2: record fingerprint on Sepolia
    # ---------------------------------------------------------------------
    logger.info("  Step 2/5 — Recording fingerprint on Ethereum Sepolia")

    try:
        tx_hash = record_fingerprint(fingerprint)
    except BlockchainError as e:
        logger.failure(f"Blockchain recording failed: {e}")
        return 18
    except Exception as e:
        logger.failure(f"Blockchain recording failed: {type(e).__name__}: {e}")
        return 18

    logger.success("Fingerprint transaction sent")
    logger.info(f"Transaction hash: {tx_hash}")

    # ---------------------------------------------------------------------
    # Step 3: wait for confirmation
    # ---------------------------------------------------------------------
    logger.info("Waiting for blockchain confirmation...")

    try:
        w3 = get_web3()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    except TimeExhausted:
        # The transaction WAS submitted; we simply have not observed a receipt
        # yet. Do NOT resubmit — preserve the tx hash and let the user recover.
        logger.failure(
            "Receipt not observed before timeout — the transaction was "
            "already submitted and was NOT resubmitted."
        )
        logger.info(f"Transaction hash (preserved): {tx_hash}")
        logger.info(
            "Recover it later once mined with:\n"
            f"    python app.py --recover-tx {tx_hash}"
        )
        return 26
    except Exception as e:
        logger.failure(f"Transaction confirmation failed: {type(e).__name__}: {e}")
        return 19

    block_number = receipt["blockNumber"]
    logger.success("Transaction confirmed")
    logger.info(f"Block number: {block_number}")

    # ---------------------------------------------------------------------
    # Step 4: TRUE re-verification — rebuild + recompute independently.
    # We recompute the hash from evidence again (not the in-memory value) and
    # compare it to the fingerprint actually stored on-chain.
    # ---------------------------------------------------------------------
    logger.info("Retrieving fingerprint from blockchain and re-verifying...")

    try:
        _, recomputed = _evidence_and_fingerprint(_load_verification(verification_path))
        onchain = get_onchain_fingerprint(tx_hash, w3)
        blockchain_result = "VERIFIED" if recomputed == onchain else "TAMPERED"
    except Exception as e:
        logger.failure(f"Blockchain verification failed: {type(e).__name__}: {e}")
        return 20

    # ---------------------------------------------------------------------
    # Step 5: persist a re-verifiable proof (no secrets).
    # ---------------------------------------------------------------------
    proof_path = _write_proof_file(
        canonical=canonical,
        fingerprint=fingerprint,
        tx_hash=tx_hash,
        block_number=block_number,
        verification_status=blockchain_result,
    )
    logger.info(f"Proof written to: {proof_path}")

    if blockchain_result == "VERIFIED":
        logger.success("BLOCKCHAIN VERIFICATION: VERIFIED")
    else:
        logger.failure("BLOCKCHAIN VERIFICATION: TAMPERED")
        return 21

    _print_final_summary(
        verification_path=verification_path,
        proof_path=proof_path,
        fingerprint=fingerprint,
        tx_hash=tx_hash,
        blockchain_result=blockchain_result,
    )

    return 0


def _write_proof_file(
    *,
    canonical: dict,
    fingerprint: str,
    tx_hash: str,
    block_number: int | None,
    verification_status: str,
) -> Path:
    """Persist non-secret proof metadata for later re-verification."""
    out = config.results_dir / "blockchain_proof.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    proof = {
        "network": "sepolia",
        "chain_id": SEPOLIA_CHAIN_ID,
        "tx_hash": tx_hash,
        "block_number": block_number,
        "fingerprint": fingerprint,
        "canonical_evidence": canonical,
        "verification_status": verification_status,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2, ensure_ascii=False)

    return out


def _verify_proof_file(proof_path: Path) -> int:
    """Re-verify a saved proof against the chain (no new transaction).

    Independently recomputes SHA-256 from the proof's stored canonical evidence
    and compares it to the fingerprint recorded on-chain in the referenced tx.
    """
    print()
    print("=" * 50)
    print("PHASE 4 — RE-VERIFY SAVED PROOF")
    print("=" * 50)
    print()

    try:
        with proof_path.open("r", encoding="utf-8") as f:
            proof = json.load(f)
        if not isinstance(proof, dict):
            raise ValueError("proof file is not a JSON object")
        canonical = proof["canonical_evidence"]
        tx_hash = proof["tx_hash"]
        if not isinstance(canonical, dict) or not tx_hash:
            raise ValueError("proof file missing canonical_evidence/tx_hash")
    except FileNotFoundError:
        logger.failure(f"Proof file not found: {proof_path}")
        return 22
    except Exception as e:
        logger.failure(f"Malformed proof file: {e}")
        return 22

    # Recompute the fingerprint from the (stored) canonical evidence — do NOT
    # trust the fingerprint field written alongside it.
    recomputed = sha256_fingerprint(canonical)
    logger.info(f"Recomputed SHA-256: {recomputed}")
    logger.info(f"Transaction:        {tx_hash}")

    try:
        onchain = get_onchain_fingerprint(tx_hash)
    except BlockchainError as e:
        logger.failure(f"Blockchain read failed: {e}")
        return 20
    except Exception as e:
        logger.failure(f"Blockchain read failed: {type(e).__name__}: {e}")
        return 20

    logger.info(f"On-chain fingerprint: {onchain}")

    if recomputed == onchain:
        logger.success("BLOCKCHAIN VERIFICATION: VERIFIED")
        return 0

    logger.failure("BLOCKCHAIN VERIFICATION: TAMPERED")
    return 21


def _recover_tx(tx_hash: str, verification_path: Path) -> int:
    """READ-ONLY recovery of an already-submitted transaction.

    Never sends a transaction. Fetches the receipt for ``tx_hash``, decodes the
    on-chain fingerprint, independently rebuilds the canonical evidence from the
    CURRENT ``verification.json`` (not from any stored proof) and recomputes the
    SHA-256, compares the two, and writes ``blockchain_proof.json``.

    Exit codes: 0 VERIFIED, 21 TAMPERED, 20 read/decode error, 22 evidence
    problem, 23 missing tx, 24 pending, 25 mined-but-failed receipt.
    """
    print()
    print("=" * 50)
    print("PHASE 4 — RECOVER SUBMITTED TRANSACTION (read-only)")
    print("=" * 50)
    print()

    logger.info(f"Transaction: {tx_hash}")

    # --- Step 1: fetch state (no submission). -----------------------------
    try:
        w3 = get_web3()
        state = fetch_tx_state(tx_hash, w3)
    except BlockchainError as e:
        logger.failure(f"Blockchain read failed: {e}")
        return 20
    except Exception as e:
        logger.failure(f"Blockchain read failed: {type(e).__name__}: {e}")
        return 20

    if not state["exists"]:
        logger.failure("Transaction does not exist on-chain. Nothing recovered.")
        return 23

    if state["state"] == "PENDING":
        logger.failure(
            "Transaction is PENDING (not mined yet). Re-run recovery later; "
            "no transaction was sent."
        )
        return 24

    if state["receipt_status"] == 0:
        logger.failure(
            f"Transaction FAILED on-chain (receipt status 0) in block "
            f"{state['block_number']}. Nothing to verify."
        )
        return 25

    if state["receipt_status"] != 1:
        logger.failure("Receipt not available yet; try again shortly.")
        return 24

    block_number = state["block_number"]
    logger.success(f"Transaction MINED and SUCCESSFUL (block {block_number})")

    # --- Step 2: decode the on-chain fingerprint. -------------------------
    try:
        onchain = get_onchain_fingerprint(tx_hash, w3)
    except BlockchainError as e:
        logger.failure(f"Could not decode on-chain fingerprint: {e}")
        return 20

    # --- Step 3: independently rebuild + recompute from verification.json. -
    try:
        verification_data = _load_verification(verification_path)
        canonical, recomputed = _evidence_and_fingerprint(verification_data)
    except FileNotFoundError:
        logger.failure(f"Verification file not found: {verification_path}")
        return 22
    except Exception as e:
        logger.failure(f"Could not rebuild canonical evidence: {e}")
        return 22

    logger.info(f"Local recomputed fingerprint: {recomputed}")
    logger.info(f"On-chain fingerprint:         {onchain}")

    result = "VERIFIED" if recomputed == onchain else "TAMPERED"

    # --- Step 4: persist the recovered proof (no secrets). ----------------
    proof_path = _write_proof_file(
        canonical=canonical,
        fingerprint=recomputed,
        tx_hash=tx_hash,
        block_number=block_number,
        verification_status=result,
    )
    logger.info(f"Proof written to: {proof_path}")

    if result == "VERIFIED":
        logger.success("BLOCKCHAIN VERIFICATION: VERIFIED")
        return 0

    logger.failure("BLOCKCHAIN VERIFICATION: TAMPERED")
    return 21


def _write_debug_search(
    search: "rs.ReverseSearchResult",
) -> None:
    """Persist raw SerpApi JSON with credentials scrubbed."""

    out = config.results_dir / "lens_response.json"

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rs.sanitize_raw(search.raw_response),
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        f"Raw SerpApi JSON written to: {out}"
    )


def _print_search_results(
    search: "rs.ReverseSearchResult",
    max_show: int = 10,
) -> None:
    """Print a compact, human-readable summary of Lens candidates."""

    print()
    print("=" * 50)
    print("GOOGLE LENS SEARCH")
    print("=" * 50)
    print()

    logger.success(
        "Search completed"
    )

    if search.image_id:
        logger.info(
            f"Uploaded image id: {search.image_id}"
        )

    logger.info(
        f"Candidates found: {search.count}"
    )

    if search.count == 0:
        print()
        logger.info(
            "No candidates returned for this image."
        )
        return

    for candidate in search.candidates[:max_show]:

        idx = (
            candidate.position
            if candidate.position is not None
            else "?"
        )

        print()

        print(
            f"[{idx}] "
            f"{candidate.title or '(no title)'}"
        )

        print(
            f"    Source: "
            f"{candidate.source or candidate.domain or '(unknown)'}"
        )

        print(
            f"    URL:    "
            f"{candidate.url or '(none)'}"
        )

    remaining = search.count - min(
        search.count,
        max_show,
    )

    if remaining > 0:
        print()

        logger.info(
            f"... and {remaining} more "
            f"(use --max-show to see more)."
        )


def _print_verification_results(
    matches: list[cm.CandidateMatch],
) -> None:
    """Print candidate face-verification results."""

    print()
    print("=" * 50)
    print("CANDIDATE FACE VERIFICATION")
    print("=" * 50)

    if not matches:
        print()
        logger.info(
            "No candidates were verified."
        )
        return

    for match in matches:

        candidate = match.candidate

        position = (
            candidate.position
            if candidate.position is not None
            else "?"
        )

        print()

        print(
            f"[{position}] "
            f"{candidate.title or '(no title)'}"
        )

        print(
            f"    Status:       {match.status}"
        )

        if match.best_similarity is not None:
            print(
                f"    Similarity:   "
                f"{match.best_similarity:.4f}"
            )

        if match.face_count:
            print(
                f"    Faces found:  "
                f"{match.face_count}"
            )

        if match.best_face_index is not None:
            print(
                f"    Best face:    "
                f"{match.best_face_index}"
            )

        if match.image_path:
            print(
                f"    Image:        "
                f"{match.image_path}"
            )

        if match.error:
            print(
                f"    Error:        "
                f"{match.error}"
            )

        print(
            f"    Source:       "
            f"{candidate.source or candidate.domain or '(unknown)'}"
        )

        print(
            f"    URL:          "
            f"{candidate.url or '(none)'}"
        )



def _write_verification_results(
    *,
    image_path: Path,
    search: "rs.ReverseSearchResult",
    matches: list[cm.CandidateMatch],
    max_candidates: int,
    match_threshold: float,
    uncertain_threshold: float,
    input_image_hash: str | None = None,
) -> Path:
    """Write structured Phase 3 output.

    ``input_image_hash`` (SHA-256 of the original input bytes) is recorded so
    Phase 4 can rebuild its canonical evidence from this file alone, without
    re-reading the original image.
    """

    out = config.results_dir / "verification.json"

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "phase": 3,
        "status": "complete",
        "input_image": str(image_path),
        "input_image_hash": input_image_hash,
        "search": {
            "engine": search.engine,
            "image_id": search.image_id,
            "image_url": search.image_url,
            "search_id": search.search_id,
            "candidate_count": search.count,
        },
        "verification": {
            "max_candidates": max_candidates,
            "match_threshold": match_threshold,
            "uncertain_threshold": uncertain_threshold,
            "results": [
                match.to_dict()
                for match in matches
            ],
        },
        "next_phase": {
            "fingerprint": True,
            "blockchain": True,
        },
    }

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        return run(args)

    except KeyboardInterrupt:
        logger.failure(
            "Interrupted."
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())