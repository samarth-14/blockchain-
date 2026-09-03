"""Canonicalize discovered metadata and produce a SHA-256 fingerprint.

PHASE 4 — NOT IMPLEMENTED YET.

Placeholder. When implemented it will canonicalize the matched post's metadata
(stable key order, normalized encoding) and return its SHA-256 hex digest, so
the same input always yields the same fingerprint for on-chain comparison.
"""

from __future__ import annotations


def canonicalize(*args, **kwargs):
    raise NotImplementedError("fingerprint is Phase 4 and not implemented yet.")


def sha256_fingerprint(*args, **kwargs):
    raise NotImplementedError("fingerprint is Phase 4 and not implemented yet.")
