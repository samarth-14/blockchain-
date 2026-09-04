"""Phase 3: download and verify Google Lens candidate images.

This module consumes Candidate objects produced by Phase 2 and performs:

    Candidate
        -> download public image
        -> validate image
        -> detect faces
        -> generate face embeddings
        -> compare against input embedding
        -> rank candidates

Phase 3 does NOT perform fingerprinting or blockchain operations.

Safety scope:
    * Intended for consenting participants and controlled/test images.
    * Candidate images must be publicly accessible.
    * No login-protected/private content is accessed.
    * Candidate images may contain multiple faces.
    * The input image is expected to contain exactly one face, enforced by
      Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from config import config
from pipeline.reverse_search import Candidate
from utils.http import build_session


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Maximum size of a downloaded candidate image.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# HTTP timeout for candidate image downloads.
DEFAULT_TIMEOUT = 20  # seconds

# Initial demo thresholds.
#
# These are NOT scientifically validated biometric identity thresholds.
# They are configurable values for the hackathon demo.
MATCH_THRESHOLD = 0.40
UNCERTAIN_THRESHOLD = 0.30


# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------

class CandidateMatchError(Exception):
    """Base class for Phase 3 candidate-processing errors."""


class CandidateDownloadError(CandidateMatchError):
    """Candidate image could not be downloaded."""


class InvalidCandidateImageError(CandidateMatchError):
    """Downloaded bytes were not a valid decodable image."""


# ------------------------------------------------------------------------------
# Download result
# ------------------------------------------------------------------------------

@dataclass
class DownloadedCandidate:
    """Result of downloading one candidate image."""

    candidate: Candidate
    path: Optional[Path]
    status: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True when the candidate image was successfully downloaded."""
        return self.status == "downloaded"


# ------------------------------------------------------------------------------
# Matching result
# ------------------------------------------------------------------------------

@dataclass
class FaceMatch:
    """Similarity result for one detected face."""

    face_index: int
    similarity: float


@dataclass
class CandidateMatch:
    """Verification result for one candidate image."""

    candidate: Candidate
    image_path: Optional[Path]
    status: str
    face_count: int = 0
    best_similarity: Optional[float] = None
    best_face_index: Optional[int] = None
    error: Optional[str] = None

    @property
    def matched(self) -> bool:
        """Return True when this candidate meets the match threshold."""
        return self.status == "match"

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "position": self.candidate.position,
            "title": self.candidate.title,
            "source": self.candidate.source,
            "domain": self.candidate.domain,
            "page_url": self.candidate.url,
            "image_url": self.candidate.image_url,
            "image_path": (
                str(self.image_path)
                if self.image_path is not None
                else None
            ),
            "status": self.status,
            "face_count": self.face_count,
            "best_similarity": self.best_similarity,
            "best_face_index": self.best_face_index,
            "error": self.error,
        }


# ------------------------------------------------------------------------------
# Download helpers
# ------------------------------------------------------------------------------

def _safe_filename(candidate: Candidate, index: int) -> str:
    """Create a deterministic local filename.

    Remote URLs are never used directly as filenames.
    """
    position = (
        candidate.position
        if candidate.position is not None
        else index
    )

    return f"candidate_{position}_{index}.jpg"


def _validate_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes with OpenCV.

    Raises:
        InvalidCandidateImageError: if the bytes are empty or invalid.
    """
    if not data:
        raise InvalidCandidateImageError(
            "Downloaded image is empty."
        )

    array = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise InvalidCandidateImageError(
            "Downloaded content could not be decoded as an image."
        )

    if image.size == 0:
        raise InvalidCandidateImageError(
            "Decoded image is empty."
        )

    return image


def download_candidate(
    candidate: Candidate,
    index: int,
    *,
    output_dir: Optional[str | Path] = None,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> DownloadedCandidate:
    """Download and validate one candidate image.

    Only the public `image_url` returned by Phase 2 is used.

    The candidate webpage itself is NOT scraped here.

    Args:
        candidate: Candidate returned by reverse image search.
        index: Local processing index.
        output_dir: Directory for validated candidate images.
        session: Optional requests session.
        timeout: HTTP timeout in seconds.

    Returns:
        DownloadedCandidate describing the outcome.

    Raises:
        CandidateDownloadError:
            Network/download/size/write failure.
        InvalidCandidateImageError:
            Response was not a valid image.
    """
    if not candidate.image_url:
        return DownloadedCandidate(
            candidate=candidate,
            path=None,
            status="no_image",
            error="Candidate has no image_url.",
        )

    output = (
        Path(output_dir)
        if output_dir is not None
        else config.candidates_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output / _safe_filename(
        candidate,
        index,
    )

    sess = session or build_session()

    try:
        response = sess.get(
            candidate.image_url,
            timeout=timeout,
            stream=True,
        )

        response.raise_for_status()

        # Check Content-Length when the server provides it.
        content_length = response.headers.get(
            "Content-Length"
        )

        if content_length:
            try:
                if int(content_length) > MAX_IMAGE_BYTES:
                    raise CandidateDownloadError(
                        f"Image exceeds {MAX_IMAGE_BYTES} byte limit."
                    )
            except ValueError:
                # Invalid Content-Length is not fatal.
                # The streaming size check below still applies.
                pass

        chunks: list[bytes] = []
        total = 0

        for chunk in response.iter_content(
            chunk_size=64 * 1024
        ):
            if not chunk:
                continue

            total += len(chunk)

            if total > MAX_IMAGE_BYTES:
                raise CandidateDownloadError(
                    f"Image exceeds {MAX_IMAGE_BYTES} byte limit."
                )

            chunks.append(chunk)

        data = b"".join(chunks)

    except CandidateDownloadError:
        raise

    except requests.Timeout as exc:
        raise CandidateDownloadError(
            f"Timed out downloading candidate image: {exc}"
        ) from exc

    except requests.RequestException as exc:
        raise CandidateDownloadError(
            f"Could not download candidate image: {exc}"
        ) from exc

    # Validate the bytes before writing anything to disk.
    image = _validate_image_bytes(data)

    # Normalize everything to JPEG.
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
    )

    if not ok:
        raise InvalidCandidateImageError(
            "Could not normalize candidate image to JPEG."
        )

    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise CandidateDownloadError(
            f"Could not save candidate image to {path}: {exc}"
        ) from exc

    return DownloadedCandidate(
        candidate=candidate,
        path=path,
        status="downloaded",
    )


def download_candidates(
    candidates: list[Candidate],
    *,
    limit: int = 10,
    output_dir: Optional[str | Path] = None,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[DownloadedCandidate]:
    """Download up to `limit` candidate images.

    A failure for one candidate does not stop processing the others.

    Args:
        candidates: Phase 2 Google Lens candidates.
        limit: Maximum number of candidates to process.
        output_dir: Local candidate-image directory.
        session: Optional shared HTTP session.
        timeout: HTTP timeout.

    Returns:
        One DownloadedCandidate result per processed candidate.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")

    results: list[DownloadedCandidate] = []

    for index, candidate in enumerate(
        candidates[:limit],
        start=1,
    ):
        try:
            result = download_candidate(
                candidate,
                index,
                output_dir=output_dir,
                session=session,
                timeout=timeout,
            )

        except InvalidCandidateImageError as exc:
            result = DownloadedCandidate(
                candidate=candidate,
                path=None,
                status="invalid_image",
                error=str(exc),
            )

        except CandidateDownloadError as exc:
            result = DownloadedCandidate(
                candidate=candidate,
                path=None,
                status="download_failed",
                error=str(exc),
            )

        results.append(result)

    return results


# ------------------------------------------------------------------------------
# Similarity
# ------------------------------------------------------------------------------

def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Calculate cosine similarity between two embeddings.

    InsightFace normally returns normalized embeddings, but normalization is
    performed defensively here.

    Returns:
        Float in the normal cosine-similarity range [-1, 1].

    Raises:
        ValueError:
            If vectors have incompatible dimensions or zero magnitude.
    """
    a = np.asarray(
        embedding_a,
        dtype=np.float32,
    )

    b = np.asarray(
        embedding_b,
        dtype=np.float32,
    )

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            "Embeddings must be one-dimensional vectors."
        )

    if a.shape != b.shape:
        raise ValueError(
            "Embedding dimensions do not match: "
            f"{a.shape} vs {b.shape}"
        )

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "Cannot calculate similarity for a zero vector."
        )

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


def classify_similarity(
    similarity: float,
    *,
    match_threshold: float = MATCH_THRESHOLD,
    uncertain_threshold: float = UNCERTAIN_THRESHOLD,
) -> str:
    """Classify a similarity score for the demo.

    Returns:
        "match"
        "uncertain"
        "no_match"
    """
    if uncertain_threshold > match_threshold:
        raise ValueError(
            "uncertain_threshold cannot exceed match_threshold."
        )

    if similarity >= match_threshold:
        return "match"

    if similarity >= uncertain_threshold:
        return "uncertain"

    return "no_match"


# ------------------------------------------------------------------------------
# Single candidate matching
# ------------------------------------------------------------------------------

def match_downloaded_candidate(
    candidate: Candidate,
    image_path: Path,
    input_embedding: np.ndarray,
    *,
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    use_cpu: bool = True,
    match_threshold: float = MATCH_THRESHOLD,
    uncertain_threshold: float = UNCERTAIN_THRESHOLD,
) -> CandidateMatch:
    """Detect and compare every face in one candidate image.

    Unlike the input image, candidate images are allowed to contain multiple
    faces.

    Example:

        Candidate image
            face 0 -> similarity 0.31
            face 1 -> similarity 0.74
            face 2 -> similarity 0.42

        Candidate score = 0.74

    The strongest face similarity becomes the candidate's score.
    """
    from pipeline import face_detector as fd

    try:
        image = fd.load_image(
            image_path
        )

        faces = fd.detect_faces(
            image,
            model_name=model_name,
            det_size=det_size,
            det_threshold=det_threshold,
            use_cpu=use_cpu,
        )

        if not faces:
            return CandidateMatch(
                candidate=candidate,
                image_path=image_path,
                status="no_face",
                face_count=0,
            )

        best_similarity: Optional[float] = None
        best_face_index: Optional[int] = None

        for face_index, face in enumerate(faces):
            candidate_embedding = fd.get_embedding(
                face
            )

            similarity = cosine_similarity(
                input_embedding,
                candidate_embedding,
            )

            if (
                best_similarity is None
                or similarity > best_similarity
            ):
                best_similarity = similarity
                best_face_index = face_index

        # The loop above always finds a score because `faces` is non-empty.
        assert best_similarity is not None
        assert best_face_index is not None

        status = classify_similarity(
            best_similarity,
            match_threshold=match_threshold,
            uncertain_threshold=uncertain_threshold,
        )

        return CandidateMatch(
            candidate=candidate,
            image_path=image_path,
            status=status,
            face_count=len(faces),
            best_similarity=best_similarity,
            best_face_index=best_face_index,
        )

    except fd.FaceDetectionError as exc:
        return CandidateMatch(
            candidate=candidate,
            image_path=image_path,
            status="processing_failed",
            error=str(exc),
        )

    except (ValueError, TypeError) as exc:
        return CandidateMatch(
            candidate=candidate,
            image_path=image_path,
            status="processing_failed",
            error=str(exc),
        )


# ------------------------------------------------------------------------------
# Full candidate matching
# ------------------------------------------------------------------------------

def match_candidates(
    input_embedding: np.ndarray,
    downloaded_candidates: list[DownloadedCandidate],
    *,
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    use_cpu: bool = True,
    match_threshold: float = MATCH_THRESHOLD,
    uncertain_threshold: float = UNCERTAIN_THRESHOLD,
) -> list[CandidateMatch]:
    """Compare the input embedding against all downloaded candidates.

    Candidates that failed during downloading are retained in the results so
    the final verification report accurately explains what happened.

    Successfully downloaded candidates are passed through InsightFace.

    Results are sorted from highest similarity to lowest similarity. Candidates
    without a similarity score are placed at the bottom.
    """
    results: list[CandidateMatch] = []

    for downloaded in downloaded_candidates:
        if (
            not downloaded.success
            or downloaded.path is None
        ):
            results.append(
                CandidateMatch(
                    candidate=downloaded.candidate,
                    image_path=downloaded.path,
                    status=downloaded.status,
                    error=downloaded.error,
                )
            )
            continue

        result = match_downloaded_candidate(
            downloaded.candidate,
            downloaded.path,
            input_embedding,
            model_name=model_name,
            det_size=det_size,
            det_threshold=det_threshold,
            use_cpu=use_cpu,
            match_threshold=match_threshold,
            uncertain_threshold=uncertain_threshold,
        )

        results.append(result)

    # Highest similarity first.
    # Candidates with no score are always placed last.
    results.sort(
        key=lambda result: (
            result.best_similarity is not None,
            (
                result.best_similarity
                if result.best_similarity is not None
                else -1.0
            ),
        ),
        reverse=True,
    )

    return results


# ------------------------------------------------------------------------------
# Convenience helper
# ------------------------------------------------------------------------------

def verify_candidates(
    input_embedding: np.ndarray,
    candidates: list[Candidate],
    *,
    limit: int = 10,
    output_dir: Optional[str | Path] = None,
    session: Optional[requests.Session] = None,
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    use_cpu: bool = True,
    match_threshold: float = MATCH_THRESHOLD,
    uncertain_threshold: float = UNCERTAIN_THRESHOLD,
) -> list[CandidateMatch]:
    """Download and verify Google Lens candidates in one call.

    This is the main Phase 3 API that app.py can eventually use:

        results = verify_candidates(
            input_embedding,
            search.candidates,
            limit=10,
        )
    """
    downloaded = download_candidates(
        candidates,
        limit=limit,
        output_dir=output_dir,
        session=session,
    )

    return match_candidates(
        input_embedding,
        downloaded,
        model_name=model_name,
        det_size=det_size,
        det_threshold=det_threshold,
        use_cpu=use_cpu,
        match_threshold=match_threshold,
        uncertain_threshold=uncertain_threshold,
    )