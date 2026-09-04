"""Tests for Phase 3 candidate image downloading."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline import candidate_matcher as cm
from pipeline.reverse_search import Candidate


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        status_code: int = 200,
        headers: dict | None = None,
    ):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "image/jpeg"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=65536):
        yield self.content


class FakeSession:
    def __init__(self, response=None, exception=None):
        self.response = response
        self.exception = exception
        self.last_get = None

    def get(self, url, timeout=None, stream=False):
        self.last_get = {
            "url": url,
            "timeout": timeout,
            "stream": stream,
        }

        if self.exception:
            raise self.exception

        return self.response


def make_jpeg_bytes() -> bytes:
    """Create a tiny valid JPEG entirely in memory."""
    image = np.full((60, 60, 3), 128, dtype=np.uint8)

    ok, encoded = cv2.imencode(".jpg", image)

    assert ok
    return encoded.tobytes()


def make_candidate(
    image_url: str | None = "https://example.com/face.jpg",
    position: int = 1,
) -> Candidate:
    return Candidate(
        position=position,
        title="Test candidate",
        url="https://example.com/page",
        source="example.com",
        image_url=image_url,
    )


def test_download_candidate_success(tmp_path):
    response = FakeResponse(make_jpeg_bytes())
    session = FakeSession(response=response)

    candidate = make_candidate()

    result = cm.download_candidate(
        candidate,
        index=1,
        output_dir=tmp_path,
        session=session,
    )

    assert result.status == "downloaded"
    assert result.success is True
    assert result.path is not None
    assert result.path.exists()

    assert session.last_get["url"] == "https://example.com/face.jpg"
    assert session.last_get["stream"] is True


def test_download_candidate_without_image_url(tmp_path):
    candidate = make_candidate(image_url=None)

    result = cm.download_candidate(
        candidate,
        index=1,
        output_dir=tmp_path,
    )

    assert result.status == "no_image"
    assert result.success is False
    assert result.path is None


def test_download_candidate_invalid_image(tmp_path):
    response = FakeResponse(b"this is not an image")
    session = FakeSession(response=response)

    candidate = make_candidate()

    result = cm.download_candidates(
        [candidate],
        limit=1,
        output_dir=tmp_path,
        session=session,
    )[0]

    assert result.status == "invalid_image"
    assert result.path is None
    assert result.error is not None


def test_download_candidates_respects_limit(tmp_path):
    response = FakeResponse(make_jpeg_bytes())
    session = FakeSession(response=response)

    candidates = [
        make_candidate(position=1),
        make_candidate(position=2),
        make_candidate(position=3),
    ]

    results = cm.download_candidates(
        candidates,
        limit=2,
        output_dir=tmp_path,
        session=session,
    )

    assert len(results) == 2


def test_download_candidates_continues_after_failure(tmp_path):
    first = make_candidate(
        image_url="https://example.com/bad.jpg",
        position=1,
    )
    second = make_candidate(
        image_url="https://example.com/good.jpg",
        position=2,
    )

    class SequenceSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout=None, stream=False):
            self.calls += 1

            if "bad" in url:
                return FakeResponse(b"not an image")

            return FakeResponse(make_jpeg_bytes())

    session = SequenceSession()

    results = cm.download_candidates(
        [first, second],
        limit=2,
        output_dir=tmp_path,
        session=session,
    )

    assert len(results) == 2
    assert results[0].status == "invalid_image"
    assert results[1].status == "downloaded"
    assert results[1].path is not None
    assert results[1].path.exists()


def test_download_candidates_handles_request_failure(tmp_path):
    import requests

    session = FakeSession(
        exception=requests.ConnectionError("connection failed")
    )

    candidate = make_candidate()

    results = cm.download_candidates(
        [candidate],
        limit=1,
        output_dir=tmp_path,
        session=session,
    )

    assert len(results) == 1
    assert results[0].status == "download_failed"
    assert results[0].path is None
    assert "connection failed" in results[0].error


def test_image_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MAX_IMAGE_BYTES", 10)

    response = FakeResponse(b"0123456789ABCDEF")
    session = FakeSession(response=response)

    candidate = make_candidate()

    results = cm.download_candidates(
        [candidate],
        limit=1,
        output_dir=tmp_path,
        session=session,
    )

    assert results[0].status == "download_failed"
    assert results[0].path is None

def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert cm.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert cm.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    assert cm.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_zero_vector():
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        cm.cosine_similarity(a, b)


def test_cosine_similarity_rejects_dimension_mismatch():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        cm.cosine_similarity(a, b)


def test_classify_similarity():
    assert cm.classify_similarity(0.50) == "match"
    assert cm.classify_similarity(0.40) == "match"
    assert cm.classify_similarity(0.35) == "uncertain"
    assert cm.classify_similarity(0.30) == "uncertain"
    assert cm.classify_similarity(0.20) == "no_match"


def test_classify_similarity_custom_thresholds():
    assert (
        cm.classify_similarity(
            0.70,
            match_threshold=0.75,
            uncertain_threshold=0.50,
        )
        == "uncertain"
    )

    assert (
        cm.classify_similarity(
            0.45,
            match_threshold=0.75,
            uncertain_threshold=0.50,
        )
        == "no_match"
    )


def test_classify_similarity_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        cm.classify_similarity(
            0.50,
            match_threshold=0.30,
            uncertain_threshold=0.40,
        )


def test_match_candidates_preserves_failed_downloads(tmp_path):
    candidate = make_candidate()

    downloaded = cm.DownloadedCandidate(
        candidate=candidate,
        path=None,
        status="download_failed",
        error="test failure",
    )

    input_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    results = cm.match_candidates(
        input_embedding,
        [downloaded],
    )

    assert len(results) == 1
    assert results[0].status == "download_failed"
    assert results[0].error == "test failure"


def test_candidate_match_to_dict(tmp_path):
    candidate = make_candidate()

    result = cm.CandidateMatch(
        candidate=candidate,
        image_path=tmp_path / "candidate.jpg",
        status="match",
        face_count=2,
        best_similarity=0.82,
        best_face_index=1,
    )

    data = result.to_dict()

    assert data["status"] == "match"
    assert data["face_count"] == 2
    assert data["best_similarity"] == pytest.approx(0.82)
    assert data["best_face_index"] == 1
    assert data["page_url"] == candidate.url
    assert data["image_url"] == candidate.image_url

def test_match_downloaded_candidate_match(monkeypatch, tmp_path):
    candidate = make_candidate()
    image_path = tmp_path / "candidate.jpg"
    image_path.write_bytes(b"fake-image")

    input_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    class FakeFace:
        embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        cm.cv2,
        "imdecode",
        lambda *args, **kwargs: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    monkeypatch.setattr(
        cm,
        "_validate_image_bytes",
        lambda data: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    from pipeline import face_detector as fd

    monkeypatch.setattr(
        fd,
        "load_image",
        lambda path: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    monkeypatch.setattr(
        fd,
        "detect_faces",
        lambda *args, **kwargs: [FakeFace()],
    )

    monkeypatch.setattr(
        fd,
        "get_embedding",
        lambda face: face.embedding,
    )

    result = cm.match_downloaded_candidate(
        candidate,
        image_path,
        input_embedding,
    )

    assert result.status == "match"
    assert result.face_count == 1
    assert result.best_face_index == 0
    assert result.best_similarity == pytest.approx(1.0)


def test_match_downloaded_candidate_uncertain(monkeypatch, tmp_path):
    candidate = make_candidate()
    image_path = tmp_path / "candidate.jpg"

    input_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    class FakeFace:
        embedding = np.array(
            [0.35, np.sqrt(1 - 0.35**2), 0.0],
            dtype=np.float32,
        )

    from pipeline import face_detector as fd

    monkeypatch.setattr(
        fd,
        "load_image",
        lambda path: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    monkeypatch.setattr(
        fd,
        "detect_faces",
        lambda *args, **kwargs: [FakeFace()],
    )

    monkeypatch.setattr(
        fd,
        "get_embedding",
        lambda face: face.embedding,
    )

    result = cm.match_downloaded_candidate(
        candidate,
        image_path,
        input_embedding,
    )

    assert result.status == "uncertain"
    assert result.best_similarity == pytest.approx(0.35)


def test_match_downloaded_candidate_no_match(monkeypatch, tmp_path):
    candidate = make_candidate()
    image_path = tmp_path / "candidate.jpg"

    input_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    class FakeFace:
        embedding = np.array(
            [0.1, np.sqrt(1 - 0.1**2), 0.0],
            dtype=np.float32,
        )

    from pipeline import face_detector as fd

    monkeypatch.setattr(
        fd,
        "load_image",
        lambda path: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    monkeypatch.setattr(
        fd,
        "detect_faces",
        lambda *args, **kwargs: [FakeFace()],
    )

    monkeypatch.setattr(
        fd,
        "get_embedding",
        lambda face: face.embedding,
    )

    result = cm.match_downloaded_candidate(
        candidate,
        image_path,
        input_embedding,
    )

    assert result.status == "no_match"
    assert result.best_similarity == pytest.approx(0.1)


def test_match_downloaded_candidate_no_face(monkeypatch, tmp_path):
    candidate = make_candidate()
    image_path = tmp_path / "candidate.jpg"

    from pipeline import face_detector as fd

    monkeypatch.setattr(
        fd,
        "load_image",
        lambda path: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    monkeypatch.setattr(
        fd,
        "detect_faces",
        lambda *args, **kwargs: [],
    )

    result = cm.match_downloaded_candidate(
        candidate,
        image_path,
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    assert result.status == "no_face"
    assert result.face_count == 0
    assert result.best_similarity is None


def test_match_downloaded_candidate_selects_best_face(
    monkeypatch,
    tmp_path,
):
    candidate = make_candidate()
    image_path = tmp_path / "candidate.jpg"

    input_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    class FaceOne:
        embedding = np.array(
            [0.20, np.sqrt(1 - 0.20**2), 0.0],
            dtype=np.float32,
        )

    class FaceTwo:
        embedding = np.array(
            [0.85, np.sqrt(1 - 0.85**2), 0.0],
            dtype=np.float32,
        )

    class FaceThree:
        embedding = np.array(
            [0.50, np.sqrt(1 - 0.50**2), 0.0],
            dtype=np.float32,
        )

    from pipeline import face_detector as fd

    monkeypatch.setattr(
        fd,
        "load_image",
        lambda path: np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        ),
    )

    monkeypatch.setattr(
        fd,
        "detect_faces",
        lambda *args, **kwargs: [
            FaceOne(),
            FaceTwo(),
            FaceThree(),
        ],
    )

    monkeypatch.setattr(
        fd,
        "get_embedding",
        lambda face: face.embedding,
    )

    result = cm.match_downloaded_candidate(
        candidate,
        image_path,
        input_embedding,
    )

    assert result.status == "match"
    assert result.face_count == 3
    assert result.best_face_index == 1
    assert result.best_similarity == pytest.approx(0.85)


def test_match_downloaded_candidate_processing_failure(
    monkeypatch,
    tmp_path,
):
    candidate = make_candidate()
    image_path = tmp_path / "candidate.jpg"

    from pipeline import face_detector as fd

    monkeypatch.setattr(
        fd,
        "load_image",
        lambda path: (_ for _ in ()).throw(
            fd.ImageLoadError("test image failure")
        ),
    )

    result = cm.match_downloaded_candidate(
        candidate,
        image_path,
        np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    assert result.status == "processing_failed"
    assert "test image failure" in result.error


def test_match_candidates_ranks_by_similarity(
    monkeypatch,
    tmp_path,
):
    from pipeline import face_detector as fd

    input_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    candidates = [
        make_candidate(position=1),
        make_candidate(position=2),
        make_candidate(position=3),
    ]

    downloaded = [
        cm.DownloadedCandidate(
            candidate=candidates[0],
            path=tmp_path / "one.jpg",
            status="downloaded",
        ),
        cm.DownloadedCandidate(
            candidate=candidates[1],
            path=tmp_path / "two.jpg",
            status="downloaded",
        ),
        cm.DownloadedCandidate(
            candidate=candidates[2],
            path=tmp_path / "three.jpg",
            status="downloaded",
        ),
    ]

    similarities = {
        "one.jpg": 0.45,
        "two.jpg": 0.85,
        "three.jpg": 0.65,
    }

    def fake_match(
        candidate,
        image_path,
        input_embedding,
        **kwargs,
    ):
        similarity = similarities[image_path.name]

        return cm.CandidateMatch(
            candidate=candidate,
            image_path=image_path,
            status=cm.classify_similarity(similarity),
            face_count=1,
            best_similarity=similarity,
            best_face_index=0,
        )

    monkeypatch.setattr(
        cm,
        "match_downloaded_candidate",
        fake_match,
    )

    results = cm.match_candidates(
        input_embedding,
        downloaded,
    )

    assert [r.best_similarity for r in results] == [
        pytest.approx(0.85),
        pytest.approx(0.65),
        pytest.approx(0.45),
    ]

    assert [r.candidate.position for r in results] == [
        2,
        3,
        1,
    ]