"""Shared HTTP helpers.

Phase 1 does not perform any network requests, but the reverse-image search
(Phase 2) and candidate-image download (Phase 3) both will. Centralising the
`requests.Session`, timeouts, retries and a polite User-Agent here now means
those phases plug in without each reinventing HTTP hygiene.

Nothing in this module is imported by the face pipeline; it exists so the
structure is ready for Phase 2.
"""

from __future__ import annotations

from typing import Optional

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; import location has moved across versions.
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - defensive
    Retry = None  # type: ignore

DEFAULT_TIMEOUT = 20  # seconds
DEFAULT_USER_AGENT = (
    "face-web-blockchain-research/0.1 (non-commercial demo; contact via repo)"
)


def build_session(user_agent: str = DEFAULT_USER_AGENT, retries: int = 3) -> requests.Session:
    """Create a `requests.Session` with sensible retry/backoff defaults."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    if Retry is not None:
        retry = Retry(
            total=retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    return session


def get(url: str, session: Optional[requests.Session] = None, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> requests.Response:
    """GET a URL with a default timeout. Raises for HTTP errors."""
    sess = session or build_session()
    resp = sess.get(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp
