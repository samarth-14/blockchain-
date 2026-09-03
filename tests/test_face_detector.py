"""Tests for the face stage.

Two tiers:
  * Fast, always-run unit tests that need NO model download — they exercise the
    pure-Python logic (image loading errors, single-face rule, cropping,
    embedding normalization) using a tiny fake `Face` object.
  * An optional integration test that runs the real InsightFace model on a real
    image. It is skipped automatically unless a test image exists AND insightface
    is importable, so `pytest` stays green on a fresh checkout without models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline import face_detector as fd


# --- Fakes ---------------------------------------------------------------------

class FakeFace:
    """Minimal stand-in for an InsightFace Face object."""
    def __init__(self, bbox, det_score=0.9, embedding=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = det_score
        if embedding is not None:
            self.embedding = np.array(embedding, dtype=np.float32)


def _blank_image(w=100, h=80):
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- load_image ----------------------------------------------------------------

def test_load_image_missing_raises():
    with pytest.raises(fd.ImageLoadError):
        fd.load_image("does/not/exist.jpg")


def test_load_image_reads_real_file(tmp_path):
    import cv2
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), _blank_image())
    img = fd.load_image(p)
    assert img.shape == (80, 100, 3)


# --- get_single_face rule (monkeypatched detector) -----------------------------

def test_get_single_face_rejects_zero(monkeypatch):
    monkeypatch.setattr(fd, "detect_faces", lambda *a, **k: [])
    with pytest.raises(fd.NoFaceError):
        fd.get_single_face(_blank_image())


def test_get_single_face_rejects_multiple(monkeypatch):
    faces = [FakeFace([0, 0, 10, 10]), FakeFace([20, 20, 30, 30])]
    monkeypatch.setattr(fd, "detect_faces", lambda *a, **k: faces)
    with pytest.raises(fd.MultipleFacesError):
        fd.get_single_face(_blank_image())


def test_get_single_face_accepts_one(monkeypatch):
    face = FakeFace([1, 2, 3, 4])
    monkeypatch.setattr(fd, "detect_faces", lambda *a, **k: [face])
    assert fd.get_single_face(_blank_image()) is face


# --- embedding normalization ---------------------------------------------------

def test_get_embedding_normalizes_raw():
    face = FakeFace([0, 0, 1, 1], embedding=[3.0, 4.0])  # norm 5
    emb = fd.get_embedding(face)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-6)
    np.testing.assert_allclose(emb, [0.6, 0.8], atol=1e-6)


def test_get_embedding_prefers_normed():
    face = FakeFace([0, 0, 1, 1])
    face.normed_embedding = np.array([0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(fd.get_embedding(face), [0.0, 1.0])


# --- crop ----------------------------------------------------------------------

def test_crop_face_clamps_and_nonempty():
    img = _blank_image(w=100, h=80)
    face = FakeFace([40, 30, 60, 50])
    crop = fd.crop_face(img, face, margin=0.2)
    assert crop.size > 0
    assert crop.shape[2] == 3


def test_crop_face_bbox_at_edge():
    img = _blank_image(w=50, h=50)
    face = FakeFace([-5, -5, 20, 20])  # negative coords must clamp to 0
    crop = fd.crop_face(img, face)
    assert crop.size > 0


def test_save_crop_rejects_empty(tmp_path):
    with pytest.raises(fd.FaceDetectionError):
        fd.save_crop(np.zeros((0, 0, 3), dtype=np.uint8), tmp_path / "x.jpg")


# --- Optional integration test (real model, real image) ------------------------

_TEST_IMAGE = Path(__file__).resolve().parent.parent / "data" / "input" / "test.jpg"


@pytest.mark.skipif(
    not _TEST_IMAGE.exists(), reason="no data/input/test.jpg present"
)
def test_integration_real_image(tmp_path):
    pytest.importorskip("insightface")
    result = fd.analyze_image(_TEST_IMAGE, crop_path=tmp_path / "face.jpg")
    assert len(result.bbox) == 4
    assert result.embedding.shape[0] in (512,)  # buffalo_* produces 512-d
    np.testing.assert_allclose(np.linalg.norm(result.embedding), 1.0, atol=1e-3)
    assert result.crop_path and result.crop_path.exists()
