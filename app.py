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
from pipeline.fingerprint import sha256_fingerprint
from pipeline.blockchain import (
    record_fingerprint,
    retrieve_and_verify,
    get_web3,
)


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
        required=True,
        help="Path to the input image (a consenting subject or test image).",
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
        )

        logger.info(
            f"Verification results written to: {verification_path}"
        )

        # Even with zero candidates, we fingerprint the resulting
        # Phase 3 output so that the pipeline remains deterministic.
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
    )

    logger.info(
        f"Verification results written to: {verification_path}"
    )

    # -------------------------------------------------------------------------
    # Stage 5: Phase 4
    # -------------------------------------------------------------------------
    return _run_phase4(verification_path)


def _run_phase4(verification_path: Path) -> int:
    """Create fingerprint, record it on Sepolia, then verify it."""

    print()
    print("=" * 50)
    print("PHASE 4 — FINGERPRINT + BLOCKCHAIN")
    print("=" * 50)
    print()

    # ---------------------------------------------------------------------
    # Step 1: fingerprint
    # ---------------------------------------------------------------------
    logger.stage(
        5,
        5,
        "Creating SHA-256 fingerprint",
    )

    try:
        with verification_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            verification_data = json.load(f)

        fingerprint = sha256_fingerprint(
            verification_data
        )

    except Exception as e:
        logger.failure(
            f"Fingerprint creation failed: {e}"
        )
        return 17

    logger.success(
        "Fingerprint created"
    )

    logger.info(
        f"SHA-256: {fingerprint}"
    )

    # ---------------------------------------------------------------------
    # Step 2: record fingerprint on Sepolia
    # ---------------------------------------------------------------------
    logger.stage(
        5,
        5,
        "Recording fingerprint on Ethereum Sepolia",
    )

    try:
        tx_hash = record_fingerprint(
            fingerprint
        )

    except Exception as e:
        logger.failure(
            f"Blockchain recording failed: {e}"
        )
        return 18

    logger.success(
        "Fingerprint transaction sent"
    )

    logger.info(
        f"Transaction hash: {tx_hash}"
    )

    # ---------------------------------------------------------------------
    # Step 3: wait for confirmation
    # ---------------------------------------------------------------------
    logger.info(
        "Waiting for blockchain confirmation..."
    )

    try:
        w3 = get_web3()

        receipt = w3.eth.wait_for_transaction_receipt(
            tx_hash
        )

    except Exception as e:
        logger.failure(
            f"Transaction confirmation failed: {e}"
        )
        return 19

    logger.success(
        "Transaction confirmed"
    )

    logger.info(
        f"Block number: {receipt['blockNumber']}"
    )

    # ---------------------------------------------------------------------
    # Step 4: retrieve + verify
    # ---------------------------------------------------------------------
    logger.info(
        "Retrieving fingerprint from blockchain..."
    )

    try:
        blockchain_result = retrieve_and_verify(
            tx_hash,
            fingerprint,
        )

    except Exception as e:
        logger.failure(
            f"Blockchain verification failed: {e}"
        )
        return 20

    if blockchain_result == "VERIFIED":
        logger.success(
            "BLOCKCHAIN VERIFICATION: VERIFIED"
        )
    else:
        logger.failure(
            "BLOCKCHAIN VERIFICATION: TAMPERED"
        )
        return 21

    print()
    logger.success(
        "Phase 4 complete."
    )

    return 0


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
) -> Path:
    """Write structured Phase 3 output."""

    out = config.results_dir / "verification.json"

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "phase": 3,
        "status": "complete",
        "input_image": str(image_path),
        "search": {
            "engine": search.engine,
            "image_id": search.image_id,
            "image_url": search.image_url,
            "search_id": search.search_id,
            "candidate_count": search.count,
        },
        "verification": {
            "max_candidates": max_candidates,
            "match_threshold": cm.MATCH_THRESHOLD,
            "uncertain_threshold": cm.UNCERTAIN_THRESHOLD,
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