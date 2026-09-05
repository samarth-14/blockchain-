"""Phase 4 — canonical evidence + SHA-256 fingerprinting.

This module turns the *raw* Phase 3 verification output into a small, stable
**canonical evidence record** and fingerprints that record with SHA-256.

Design rules (see the Phase 4 audit):
  * We fingerprint the canonical evidence record, NOT the raw verification.json.
  * The canonical record contains only stable evidence fields. It deliberately
    excludes mutable/raw API data (SerpApi image_id / search_id), absolute local
    filesystem paths, raw responses, timestamps, and debug fields.
  * Serialization is deterministic: the same canonical evidence always produces
    the exact same bytes and therefore the exact same SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

# Bump this if the canonical evidence *shape* changes in a breaking way, so old
# proofs remain interpretable and re-verification stays meaningful.
CANONICAL_VERSION = 1


# ---------------------------------------------------------------------------
# Low-level hashing helpers
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's bytes (streamed).

    The same file always produces the same hash. Used to derive a stable
    ``input_image_hash`` from the original input image.
    """
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Canonicalization + fingerprint
# ---------------------------------------------------------------------------
def canonicalize(record: dict[str, Any]) -> str:
    """Serialize a record into a deterministic JSON string.

    Deterministic because: keys are sorted, separators are compact, and the
    output is stable UTF-8. The same record always yields the same string.
    """
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_fingerprint(record: dict[str, Any]) -> str:
    """Return the SHA-256 fingerprint of a (canonical) record.

    Callers should pass a canonical evidence record from
    :func:`build_canonical_evidence` — not the raw Phase 3 verification.json.
    """
    canonical = canonicalize(record)
    return sha256_bytes(canonical.encode("utf-8"))


# ---------------------------------------------------------------------------
# Canonical evidence record
# ---------------------------------------------------------------------------
def _select_best_result(results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the single most-representative candidate deterministically.

    Highest similarity wins; ``None`` similarity ranks below any real score.
    Ties break on ascending ``position`` (then title) so the choice is stable.
    """
    if not results:
        return None

    def sort_key(r: dict[str, Any]) -> tuple:
        sim = r.get("best_similarity")
        has_sim = sim is not None
        # position/title may be None; coerce to sortable, stable values.
        pos = r.get("position")
        pos = pos if isinstance(pos, int) else 1_000_000
        title = r.get("title") or ""
        # Sort so that (has_sim desc, sim desc, pos asc, title asc) → first is best.
        return (not has_sim, -(sim if has_sim else 0.0), pos, title)

    return sorted(results, key=sort_key)[0]


def build_canonical_evidence(
    verification_data: dict[str, Any],
    input_image_hash: Optional[str] = None,
) -> dict[str, Any]:
    """Build the stable canonical evidence record from Phase 3 output.

    Parameters
    ----------
    verification_data:
        The parsed ``data/results/verification.json`` produced by Phase 3.
    input_image_hash:
        SHA-256 of the original input image. If ``None``, falls back to
        ``verification_data["input_image_hash"]`` when present.

    Returns
    -------
    A dict containing ONLY stable evidence fields. Excluded on purpose:
    SerpApi image_id/search_id, image_id-bearing search block, absolute local
    paths (input_image, image_path), raw responses, and debug fields.

    Raises
    ------
    ValueError: if ``verification_data`` is not a dict / is malformed.
    """
    if not isinstance(verification_data, dict):
        raise ValueError("verification_data must be a dict")

    verification = verification_data.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("verification_data missing 'verification' block")

    results = verification.get("results")
    if results is None:
        results = []
    if not isinstance(results, list):
        raise ValueError("verification.results must be a list")

    if input_image_hash is None:
        maybe = verification_data.get("input_image_hash")
        if isinstance(maybe, str) and maybe:
            input_image_hash = maybe

    best = _select_best_result(results)

    if best is None:
        best_match: Optional[dict[str, Any]] = None
        verification_status = "no_candidates"
    else:
        sim = best.get("best_similarity")
        best_match = {
            "title": best.get("title"),
            "source": best.get("source"),
            # Phase 3 stores the page URL under "page_url".
            "source_url": best.get("page_url"),
            "candidate_image_url": best.get("image_url"),
            # Round to keep floats byte-stable across runs/serializers.
            "similarity": round(sim, 6) if isinstance(sim, (int, float)) else None,
            "status": best.get("status"),
        }
        verification_status = best.get("status") or "unknown"

    return {
        "canonical_version": CANONICAL_VERSION,
        "input_image_hash": input_image_hash,
        "verification_status": verification_status,
        "best_match": best_match,
    }
