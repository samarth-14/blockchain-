"""Phase 4 tests: canonical evidence, determinism, and fingerprint stability."""

from __future__ import annotations

import hashlib

import pytest

from pipeline.fingerprint import (
    build_canonical_evidence,
    canonicalize,
    sha256_bytes,
    sha256_fingerprint,
    sha256_file,
)


def _verification(**overrides):
    """A representative Phase 3 verification.json payload."""
    data = {
        "phase": 3,
        "status": "complete",
        "input_image": "/Users/someone/face-web-blockchain/data/input/test.jpg",
        "input_image_hash": "a" * 64,
        "search": {
            "engine": "google_lens",
            "image_id": "img_ABC123",       # mutable / per-run
            "image_url": "https://serpapi.example/uploads/img_ABC123.jpg",
            "search_id": "search_XYZ789",   # mutable / per-run
            "candidate_count": 2,
        },
        "verification": {
            "max_candidates": 10,
            "match_threshold": 0.40,
            "uncertain_threshold": 0.30,
            "results": [
                {
                    "position": 1,
                    "title": "Example Person - Wikipedia",
                    "source": "Wikipedia",
                    "domain": "en.wikipedia.org",
                    "page_url": "https://en.wikipedia.org/wiki/Example",
                    "image_url": "https://upload.example/example.jpg",
                    "image_path": "/tmp/candidates/cand_1.jpg",  # local path
                    "status": "match",
                    "face_count": 1,
                    "best_similarity": 0.7123456789,
                    "best_face_index": 0,
                    "error": None,
                },
                {
                    "position": 2,
                    "title": "Someone Else",
                    "source": "Example News",
                    "domain": "news.example",
                    "page_url": "https://news.example/story",
                    "image_url": "https://news.example/img.jpg",
                    "image_path": "/tmp/candidates/cand_2.jpg",
                    "status": "no_match",
                    "face_count": 1,
                    "best_similarity": 0.12,
                    "best_face_index": 0,
                    "error": None,
                },
            ],
        },
        "next_phase": {"fingerprint": True, "blockchain": True},
    }
    data.update(overrides)
    return data


# --- A. deterministic canonicalization ------------------------------------
def test_canonicalization_is_deterministic():
    ev = build_canonical_evidence(_verification())
    assert canonicalize(ev) == canonicalize(build_canonical_evidence(_verification()))


def test_canonicalization_is_key_order_independent():
    ev = build_canonical_evidence(_verification())
    reordered = dict(reversed(list(ev.items())))
    assert canonicalize(ev) == canonicalize(reordered)


# --- B. deterministic SHA-256 --------------------------------------------
def test_fingerprint_is_deterministic():
    f1 = sha256_fingerprint(build_canonical_evidence(_verification()))
    f2 = sha256_fingerprint(build_canonical_evidence(_verification()))
    assert f1 == f2
    assert len(f1) == 64


# --- C. mutable/raw Phase 3 fields are excluded ---------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["search"].__setitem__("image_id", "img_DIFFERENT"),
        lambda d: d["search"].__setitem__("search_id", "search_DIFFERENT"),
        lambda d: d["search"].__setitem__("image_url", "https://other/upload.jpg"),
        lambda d: d.__setitem__("input_image", "/completely/different/path.jpg"),
        lambda d: d["verification"]["results"][0].__setitem__(
            "image_path", "/tmp/other/location.jpg"
        ),
        lambda d: d["verification"]["results"][0].__setitem__("best_face_index", 5),
        lambda d: d["verification"]["results"][0].__setitem__("face_count", 9),
        lambda d: d.__setitem__("next_phase", {"debug": "changed"}),
    ],
)
def test_mutable_fields_do_not_change_fingerprint(mutate):
    base = sha256_fingerprint(build_canonical_evidence(_verification()))
    d = _verification()
    mutate(d)
    assert sha256_fingerprint(build_canonical_evidence(d)) == base


# --- D. meaningful evidence changes DO change fingerprint ------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["verification"]["results"][0].__setitem__("title", "New Title"),
        lambda d: d["verification"]["results"][0].__setitem__(
            "page_url", "https://en.wikipedia.org/wiki/Different"
        ),
        lambda d: d["verification"]["results"][0].__setitem__(
            "image_url", "https://upload.example/different.jpg"
        ),
        lambda d: d["verification"]["results"][0].__setitem__("source", "Britannica"),
        lambda d: d["verification"]["results"][0].__setitem__("status", "uncertain"),
        lambda d: d["verification"]["results"][0].__setitem__("best_similarity", 0.999),
        lambda d: d.__setitem__("input_image_hash", "b" * 64),
    ],
)
def test_meaningful_changes_change_fingerprint(mutate):
    base = sha256_fingerprint(build_canonical_evidence(_verification()))
    d = _verification()
    mutate(d)
    assert sha256_fingerprint(build_canonical_evidence(d)) != base


# --- Canonical record content / exclusions --------------------------------
def test_canonical_record_only_contains_stable_fields():
    ev = build_canonical_evidence(_verification())
    assert set(ev.keys()) == {
        "canonical_version",
        "input_image_hash",
        "verification_status",
        "best_match",
    }
    bm = ev["best_match"]
    assert set(bm.keys()) == {
        "title",
        "source",
        "source_url",
        "candidate_image_url",
        "similarity",
        "status",
    }
    # No raw/mutable leakage anywhere in the serialized record.
    blob = canonicalize(ev)
    for forbidden in ("img_ABC123", "search_XYZ789", "/tmp/", "image_path", "best_face_index"):
        assert forbidden not in blob


def test_best_match_selection_prefers_highest_similarity():
    ev = build_canonical_evidence(_verification())
    assert ev["best_match"]["title"] == "Example Person - Wikipedia"
    assert ev["verification_status"] == "match"
    assert ev["best_match"]["similarity"] == pytest.approx(0.712346, abs=1e-6)


def test_no_candidates_yields_null_best_match():
    d = _verification()
    d["verification"]["results"] = []
    ev = build_canonical_evidence(d)
    assert ev["best_match"] is None
    assert ev["verification_status"] == "no_candidates"


def test_input_image_hash_falls_back_to_verification_field():
    d = _verification()
    ev = build_canonical_evidence(d)  # no explicit hash arg
    assert ev["input_image_hash"] == "a" * 64
    # explicit arg overrides
    ev2 = build_canonical_evidence(d, input_image_hash="c" * 64)
    assert ev2["input_image_hash"] == "c" * 64


def test_malformed_verification_raises():
    with pytest.raises(ValueError):
        build_canonical_evidence("not a dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_canonical_evidence({"no": "verification"})


# --- E. file / byte hashing helpers ---------------------------------------
def test_sha256_bytes_matches_hashlib():
    assert sha256_bytes(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_sha256_file_is_stable(tmp_path):
    p = tmp_path / "img.bin"
    p.write_bytes(b"\x00\x01\x02fake-image-bytes")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2 == hashlib.sha256(b"\x00\x01\x02fake-image-bytes").hexdigest()
