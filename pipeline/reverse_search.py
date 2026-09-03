"""Genuine reverse image search via the official SerpApi Google Lens API.

Phase 2 of the pipeline. Given a local image, this module performs a REAL
external search and returns normalized candidate objects for Phase 3 to consume.
There are NO mocks, NO hardcoded URLs, and NO manual Google scraping here.

How a local image reaches Google Lens
-------------------------------------
SerpApi's Google Lens engine does not accept a raw local file directly; it needs
either a publicly reachable image URL or an `image_id` obtained from SerpApi's
own Image Upload API. We use SerpApi's official upload endpoint so the image is
handled entirely within SerpApi (no arbitrary third-party host):

    1. POST the image bytes to  https://serpapi.com/image   -> { image_id }
    2. GET  https://serpapi.com/search?engine=google_lens&image_id=... -> results

Alternatively, if the caller already has a public URL for the image, we pass it
straight through via the `url` parameter and skip the upload entirely. The image
submission strategy is deliberately factored out (`ImageSubmission`) so this
choice is explicit and swappable.

Docs:
  * Google Lens API:      https://serpapi.com/google-lens-api
  * Upload an image:      https://serpapi.com/google-lens-upload-an-image

Credentials come only from config/.env and are never logged or written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

SERPAPI_SEARCH_URL = "https://serpapi.com/search"
SERPAPI_UPLOAD_URL = "https://serpapi.com/image"

# SerpApi's upload endpoint caps images at 500 KB. Stay safely under it.
_MAX_UPLOAD_BYTES = 490_000
_DEFAULT_TIMEOUT = 30  # seconds


# --- Errors --------------------------------------------------------------------

class ReverseSearchError(Exception):
    """Base class for reverse-search errors."""


class MissingAPIKeyError(ReverseSearchError):
    """No SerpApi key was configured (SERPAPI_API_KEY)."""


class InvalidAPIKeyError(ReverseSearchError):
    """SerpApi rejected the key (HTTP 401)."""


class RateLimitError(ReverseSearchError):
    """SerpApi rate limit / quota exhausted (HTTP 429)."""


class SearchNetworkError(ReverseSearchError):
    """Network failure or timeout talking to SerpApi."""


class MalformedResponseError(ReverseSearchError):
    """SerpApi returned something we could not parse or an API error body."""


class ImageInputError(ReverseSearchError):
    """The local image is missing, unreadable, or unsupported."""


# --- Data objects --------------------------------------------------------------

@dataclass
class Candidate:
    """A single normalized reverse-image-search result.

    Not every SerpApi result carries every field, so all are optional except the
    fields we can always synthesize. Phase 3 will consume `url`/`image_url`.
    """
    position: Optional[int] = None
    title: Optional[str] = None
    url: Optional[str] = None            # page URL where the image was found
    source: Optional[str] = None         # human source label from SerpApi
    domain: Optional[str] = None         # derived from url
    thumbnail_url: Optional[str] = None  # small preview hosted by Google/SerpApi
    image_url: Optional[str] = None      # full candidate image, if provided
    result_type: str = "visual_match"
    raw: Dict[str, Any] = field(default_factory=dict)  # reference to raw result

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "position": self.position,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "domain": self.domain,
            "thumbnail_url": self.thumbnail_url,
            "image_url": self.image_url,
            "result_type": self.result_type,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ReverseSearchResult:
    """The full outcome of one reverse image search."""
    candidates: List[Candidate]
    engine: str
    image_id: Optional[str] = None       # set when we uploaded a local image
    image_url: Optional[str] = None      # set when a public URL was searched
    search_id: Optional[str] = None      # SerpApi search_metadata.id
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.candidates)


# --- HTTP session --------------------------------------------------------------

def _build_session() -> requests.Session:
    """A session with NO automatic retries.

    We intentionally avoid retry/backoff here: SerpApi calls consume a metered
    quota, and silently retrying a 429/5xx could burn the free plan. The caller
    sees the error and decides.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "face-web-blockchain-research/0.1 (non-commercial demo)"})
    return session


# --- Image submission strategy -------------------------------------------------

def _domain_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc or None
    except Exception:
        return None


def _read_upload_bytes(image_path: Path) -> tuple[bytes, str]:
    """Return (bytes, filename) for upload, guaranteed <= the SerpApi size cap.

    If the original file is already small enough it is sent as-is; otherwise it
    is re-encoded as JPEG and progressively downscaled until it fits. This keeps
    a genuine picture of the subject (never a fabricated stand-in).
    """
    if not image_path.exists():
        raise ImageInputError(f"Image not found: {image_path}")
    try:
        data = image_path.read_bytes()
    except OSError as e:
        raise ImageInputError(f"Could not read image {image_path}: {e}") from e
    if not data:
        raise ImageInputError(f"Image is empty: {image_path}")

    if len(data) <= _MAX_UPLOAD_BYTES:
        return data, image_path.name

    # Too large: re-encode/downscale with OpenCV.
    import cv2  # local import; keeps module import cheap and Phase-1-independent

    img = cv2.imread(str(image_path))
    if img is None:
        raise ImageInputError(f"Unsupported or corrupt image: {image_path}")

    scale = 1.0
    for _ in range(8):
        resized = img
        if scale < 1.0:
            h, w = img.shape[:2]
            resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok and buf.nbytes <= _MAX_UPLOAD_BYTES:
            return buf.tobytes(), image_path.stem + ".jpg"
        scale *= 0.8
    raise ImageInputError(
        f"Could not compress {image_path} under {_MAX_UPLOAD_BYTES} bytes for upload."
    )


def upload_image(image_path: Path, api_key: str, session: requests.Session,
                 timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Upload a local image to SerpApi's Image API and return its image_id.

    This is the explicit, official image-hosting step. The image is sent to
    SerpApi only (not to an arbitrary third party).
    """
    payload, filename = _read_upload_bytes(image_path)
    try:
        resp = session.post(
            SERPAPI_UPLOAD_URL,
            data={"api_key": api_key},
            files={"image": (filename, payload, "application/octet-stream")},
            timeout=timeout,
        )
    except requests.Timeout as e:
        raise SearchNetworkError(f"Timed out uploading image to SerpApi: {e}") from e
    except requests.RequestException as e:
        raise SearchNetworkError(f"Network error uploading image to SerpApi: {e}") from e

    _raise_for_http_status(resp)
    try:
        body = resp.json()
    except ValueError as e:
        raise MalformedResponseError(f"Upload response was not JSON: {e}") from e

    image_id = body.get("image_id")
    if not image_id:
        raise MalformedResponseError(f"Upload response missing image_id: {body!r}")
    return image_id


# --- Response parsing ----------------------------------------------------------

def _candidate_from_visual_match(item: Dict[str, Any]) -> Candidate:
    url = item.get("link")
    thumb = item.get("thumbnail")
    # SerpApi nests the full image under "image" (dict) or as "image"/"original".
    image_url = None
    img_field = item.get("image")
    if isinstance(img_field, dict):
        image_url = img_field.get("link") or img_field.get("original")
    elif isinstance(img_field, str):
        image_url = img_field
    image_url = image_url or item.get("original") or item.get("original_image")

    return Candidate(
        position=item.get("position"),
        title=item.get("title"),
        url=url,
        source=item.get("source"),
        domain=_domain_of(url),
        thumbnail_url=thumb if isinstance(thumb, str) else (thumb or {}).get("link") if isinstance(thumb, dict) else None,
        image_url=image_url,
        result_type="visual_match",
        raw=item,
    )


def parse_response(data: Dict[str, Any]) -> List[Candidate]:
    """Extract candidate objects from a SerpApi Google Lens JSON response.

    Primary source is `visual_matches`; `exact_matches` (when present) are
    appended and tagged. Missing fields are tolerated. Raises
    MalformedResponseError if the response carries a SerpApi `error`.
    """
    if not isinstance(data, dict):
        raise MalformedResponseError("Response is not a JSON object.")
    if data.get("error"):
        # SerpApi reports problems (bad key, no results, etc.) in this field.
        raise MalformedResponseError(str(data["error"]))

    candidates: List[Candidate] = []

    visual = data.get("visual_matches")
    if isinstance(visual, list):
        for item in visual:
            if isinstance(item, dict):
                candidates.append(_candidate_from_visual_match(item))

    exact = data.get("exact_matches")
    if isinstance(exact, list):
        for item in exact:
            if isinstance(item, dict):
                c = _candidate_from_visual_match(item)
                c.result_type = "exact_match"
                candidates.append(c)

    return candidates


# --- HTTP status handling ------------------------------------------------------

def _raise_for_http_status(resp: requests.Response) -> None:
    """Map SerpApi HTTP status codes to our typed errors."""
    if resp.status_code == 200:
        return
    # SerpApi returns a JSON body with an "error" message on failures.
    detail = ""
    try:
        detail = str(resp.json().get("error", "")).strip()
    except Exception:
        detail = (resp.text or "").strip()[:200]

    if resp.status_code == 401:
        raise InvalidAPIKeyError(detail or "SerpApi rejected the API key (401).")
    if resp.status_code == 429:
        raise RateLimitError(detail or "SerpApi rate limit / quota exceeded (429).")
    raise MalformedResponseError(
        f"SerpApi returned HTTP {resp.status_code}: {detail or 'unknown error'}"
    )


# --- Public API ----------------------------------------------------------------

def search_image(
    image_path: Optional[str | Path] = None,
    *,
    image_url: Optional[str] = None,
    api_key: str = "",
    engine: str = "google_lens",
    result_type: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> ReverseSearchResult:
    """Perform a genuine reverse image search and return normalized candidates.

    Provide exactly one image source:
      * `image_path` — a local file; uploaded to SerpApi to obtain an image_id.
      * `image_url`  — a public image URL; passed straight to Google Lens.

    Raises the typed ReverseSearchError subclasses for the documented failure
    modes (missing/invalid key, rate limit, network/timeout, malformed response,
    bad image). Zero results is NOT an error — it returns an empty candidate list.
    """
    if not api_key:
        raise MissingAPIKeyError(
            "SERPAPI_API_KEY is not set. Add it to your .env (see .env.example)."
        )
    if bool(image_path) == bool(image_url):
        raise ImageInputError("Provide exactly one of image_path or image_url.")

    sess = session or _build_session()

    params: Dict[str, Any] = {"engine": engine, "api_key": api_key}
    if result_type:
        params["type"] = result_type

    image_id: Optional[str] = None
    if image_path is not None:
        image_id = upload_image(Path(image_path), api_key, sess, timeout=timeout)
        params["image_id"] = image_id
    else:
        params["url"] = image_url

    try:
        resp = sess.get(SERPAPI_SEARCH_URL, params=params, timeout=timeout)
    except requests.Timeout as e:
        raise SearchNetworkError(f"Timed out contacting SerpApi Google Lens: {e}") from e
    except requests.RequestException as e:
        raise SearchNetworkError(f"Network error contacting SerpApi: {e}") from e

    _raise_for_http_status(resp)
    try:
        data = resp.json()
    except ValueError as e:
        raise MalformedResponseError(f"SerpApi response was not valid JSON: {e}") from e

    candidates = parse_response(data)
    search_id = None
    meta = data.get("search_metadata")
    if isinstance(meta, dict):
        search_id = meta.get("id")

    return ReverseSearchResult(
        candidates=candidates,
        engine=engine,
        image_id=image_id,
        image_url=image_url,
        search_id=search_id,
        raw_response=data,
    )


# --- Debug helper --------------------------------------------------------------

def sanitize_raw(data: Any) -> Any:
    """Return a deep copy of `data` with any `api_key` field scrubbed.

    SerpApi does not echo the key in responses, but we defensively strip any key
    named `api_key` before writing debug output to disk, so credentials can
    never leak into data/results/lens_response.json.
    """
    if isinstance(data, dict):
        return {k: ("<redacted>" if k == "api_key" else sanitize_raw(v)) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_raw(v) for v in data]
    return data
