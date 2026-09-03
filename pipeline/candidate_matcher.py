"""Compare candidate faces against the input embedding.

PHASE 3 — NOT IMPLEMENTED YET.

Placeholder. When implemented it will download publicly accessible candidate
images (discovered in Phase 2), compute embeddings with
`pipeline.face_detector.get_embedding`, and rank them by cosine similarity to
the input face to find the strongest match.
"""

from __future__ import annotations


def match_candidates(*args, **kwargs):
    raise NotImplementedError(
        "candidate_matcher is Phase 3 and not implemented yet."
    )
