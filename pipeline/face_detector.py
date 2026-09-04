"""Face detection + embedding using InsightFace.

This is the entry stage of the pipeline. It is deliberately modular so Phase 2
(reverse image search) can reuse the exact same embedding routine on downloaded
candidate images to compare against the input face.

Public API
----------
- load_image(path)            -> np.ndarray (BGR, as OpenCV/InsightFace expect)
- detect_faces(image)         -> list[insightface Face]
- get_single_face(image)      -> one Face, enforcing the MVP single-face rule
- get_embedding(face)         -> np.ndarray (normalized 512-d vector)
- crop_face(image, face)      -> np.ndarray (the cropped face region)
- save_crop(crop, path)       -> Path
- analyze_image(path, ...)    -> FaceResult  (convenience: does the whole stage)

MVP safety rules (per project scope):
  * exactly ONE face must be present.
  * zero faces  -> NoFaceError
  * multiple    -> MultipleFacesError
This keeps the demo focused on a single consenting subject and avoids anything
resembling multi-person identification.

The InsightFace model is loaded lazily and cached, so importing this module is
cheap and tests can import it without triggering a model download.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@contextlib.contextmanager
def _suppress_stdout():
    """Temporarily suppress Python-level stdout.

    InsightFace can print model-download/provider messages during its first
    initialization. Using a Python-level redirect is safer on Windows than
    swapping the underlying stdout file descriptor with os.dup2(), which can
    leave PowerShell/Python's stdout stream in an invalid state.
    """
    saved_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        yield
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = saved_stdout
# --- Errors --------------------------------------------------------------------

class FaceDetectionError(Exception):
    """Base class for face-stage errors."""


class ImageLoadError(FaceDetectionError):
    """The image path was missing or could not be decoded."""


class NoFaceError(FaceDetectionError):
    """No face was detected in the image."""


class MultipleFacesError(FaceDetectionError):
    """More than one face was detected (rejected for the MVP)."""


# --- Result container ----------------------------------------------------------

@dataclass
class FaceResult:
    """Everything the face stage produces for a single input image."""
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2) integer pixel coords
    det_score: float                     # detector confidence
    embedding: np.ndarray                # L2-normalized 512-d float32 vector
    crop_path: Optional[Path] = None     # where the face crop was saved (if any)

    def embedding_preview(self, n: int = 5) -> str:
        vals = ", ".join(f"{v:+.4f}" for v in self.embedding[:n])
        return f"[{vals}, ... dim={self.embedding.shape[0]}]"


# --- Lazy model singleton ------------------------------------------------------

_APP = None  # cached insightface.app.FaceAnalysis instance


def _get_app(model_name: str = "buffalo_l", det_size: int = 640, use_cpu: bool = True):
    """Load (once) and return the InsightFace FaceAnalysis app.

    The heavy import and model download happen here, on first real use, so that
    simply importing this module (e.g. in unit tests) stays fast and offline.
    """
    global _APP
    if _APP is not None:
        return _APP

    # Imported lazily so the module imports even before insightface is installed.
    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"] if use_cpu else [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    # InsightFace prints model paths / provider info to stdout on load (and a
    # download progress bar on first run). Silence it so the CLI stages stay
    # clean; the one-time model download still happens, just quietly.
    with _suppress_stdout():
        app = FaceAnalysis(name=model_name, providers=providers)
        # ctx_id=-1 forces CPU inside InsightFace regardless of provider list.
        app.prepare(ctx_id=-1 if use_cpu else 0, det_size=(det_size, det_size))
    _APP = app
    return _APP


# --- Public API ----------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR numpy array (OpenCV convention).

    Raises ImageLoadError if the file is missing or not a decodable image.
    """
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"Image not found: {p}")
    # cv2.imread silently returns None on failure, so check explicitly.
    image = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Could not decode image (unsupported/corrupt?): {p}")
    return image


def detect_faces(
    image: np.ndarray,
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    use_cpu: bool = True,
) -> List:
    """Detect all faces in a BGR image, filtered by detector confidence.

    Returns a list of InsightFace `Face` objects (may be empty).
    """
    app = _get_app(model_name=model_name, det_size=det_size, use_cpu=use_cpu)
    faces = app.get(image)
    return [f for f in faces if float(getattr(f, "det_score", 0.0)) >= det_threshold]


def get_single_face(
    image: np.ndarray,
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    use_cpu: bool = True,
):
    """Detect exactly one face, enforcing the MVP single-subject rule.

    Raises NoFaceError for zero faces and MultipleFacesError for >1.
    """
    faces = detect_faces(
        image,
        model_name=model_name,
        det_size=det_size,
        det_threshold=det_threshold,
        use_cpu=use_cpu,
    )
    if len(faces) == 0:
        raise NoFaceError("No face detected in the image.")
    if len(faces) > 1:
        raise MultipleFacesError(
            f"Expected exactly one face for the MVP but detected {len(faces)}."
        )
    return faces[0]


def get_embedding(face) -> np.ndarray:
    """Return an L2-normalized 512-d embedding for a detected face.

    InsightFace exposes `normed_embedding` (already unit-length); we fall back to
    normalizing the raw `embedding` if needed. Normalizing guarantees cosine
    similarity in Phase 2 is a plain dot product.
    """
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
        if emb is None:
            raise FaceDetectionError("Face object has no embedding.")
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
    return np.asarray(emb, dtype=np.float32)


def _bbox_ints(face, image_shape) -> Tuple[int, int, int, int]:
    """Clamp a face bbox to integer pixel coordinates within the image."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def crop_face(image: np.ndarray, face, margin: float = 0.2) -> np.ndarray:
    """Return the cropped face region, with an optional margin around the bbox.

    `margin` is a fraction of the box size added on each side (0.2 = 20%),
    clamped to image bounds. A margin gives Phase 2 comparisons more context.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _bbox_ints(face, image.shape)
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    return image[cy1:cy2, cx1:cx2].copy()


def save_crop(crop: np.ndarray, path: str | Path) -> Path:
    """Write a face crop to disk, creating parent dirs. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if crop.size == 0:
        raise FaceDetectionError("Refusing to save an empty face crop.")
    ok = cv2.imwrite(str(p), crop)
    if not ok:
        raise FaceDetectionError(f"Failed to write face crop to {p}")
    return p


def analyze_image(
    path: str | Path,
    crop_path: Optional[str | Path] = None,
    model_name: str = "buffalo_l",
    det_size: int = 640,
    det_threshold: float = 0.5,
    use_cpu: bool = True,
) -> FaceResult:
    """Run the full face stage on one image and return a FaceResult.

    This is the single call app.py needs. It loads the image, enforces the
    single-face rule, computes the embedding, and (optionally) saves a crop.
    """
    image = load_image(path)
    face = get_single_face(
        image,
        model_name=model_name,
        det_size=det_size,
        det_threshold=det_threshold,
        use_cpu=use_cpu,
    )
    bbox = _bbox_ints(face, image.shape)
    embedding = get_embedding(face)

    saved: Optional[Path] = None
    if crop_path is not None:
        crop = crop_face(image, face)
        saved = save_crop(crop, crop_path)

    return FaceResult(
        bbox=bbox,
        det_score=float(getattr(face, "det_score", 0.0)),
        embedding=embedding,
        crop_path=saved,
    )
