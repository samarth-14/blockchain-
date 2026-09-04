"""Canonicalize metadata and create a SHA-256 fingerprint."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize(metadata: dict[str, Any]) -> str:
    """Convert metadata into a deterministic JSON string."""
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_fingerprint(metadata: dict[str, Any]) -> str:
    """Return the SHA-256 fingerprint of the canonical metadata."""
    canonical = canonicalize(metadata)

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()