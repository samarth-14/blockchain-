"""Tests for the Phase 2 reverse image search.

These tests MOCK the HTTP layer only — they never touch the live SerpApi API and
never consume quota. A representative (but trimmed) SerpApi Google Lens response
shape is used to exercise parsing, missing fields, zero results, malformed
bodies, and error mapping.

For a genuine end-to-end check against the real API, see
`tests/live_search.py` (run manually with a real key).
"""

from __future__ import annotations

import json

import pytest
import requests

from pipeline import reverse_search as rs


# --- Fake HTTP plumbing --------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Stand-in for requests.Session with scripted get/post responses."""
    def __init__(self, get_response=None, post_response=None, get_exc=None):
        self._get_response = get_response
        self._post_response = post_response
        self._get_exc = get_exc
        self.headers = {}
        self.last_get = None
        self.last_post = None

    def get(self, url, params=None, timeout=None):
        self.last_get = {"url": url, "params": params, "timeout": timeout}
        if self._get_exc is not None:
            raise self._get_exc
        return self._get_response

    def post(self, url, data=None, files=None, timeout=None):
        self.last_post = {"url": url, "data": data, "files": files, "timeout": timeout}
        return self._post_response


# --- Representative SerpApi response --------------------------------------------

SAMPLE_RESPONSE = {
    "search_metadata": {"id": "abc123", "status": "Success"},
    "search_parameters": {"engine": "google_lens"},
    "visual_matches": [
        {
            "position": 1,
            "title": "Astronaut portrait - Example News",
            "link": "https://www.example-news.com/astronaut",
            "source": "example-news.com",
            "thumbnail": "https://serpapi.example/thumb1.jpg",
            "image": {"link": "https://cdn.example-news.com/astro_full.jpg"},
        },
        {
            # Missing several optional fields on purpose.
            "position": 2,
            "title": "Space Agency Gallery",
            "link": "https://gallery.example.org/photo/42",
        },
    ],
    "exact_matches": [
        {
            "position": 1,
            "title": "Original upload",
            "link": "https://archive.example.net/item/7",
            "source": "archive.example.net",
        }
    ],
}


# --- Parsing -------------------------------------------------------------------

def test_parse_representative_response():
    candidates = rs.parse_response(SAMPLE_RESPONSE)
    # 2 visual + 1 exact
    assert len(candidates) == 3

    c1 = candidates[0]
    assert c1.position == 1
    assert c1.title == "Astronaut portrait - Example News"
    assert c1.url == "https://www.example-news.com/astronaut"
    assert c1.domain == "example-news.com"  # www. stripped
    assert c1.thumbnail_url == "https://serpapi.example/thumb1.jpg"
    assert c1.image_url == "https://cdn.example-news.com/astro_full.jpg"
    assert c1.result_type == "visual_match"
    assert c1.raw is SAMPLE_RESPONSE["visual_matches"][0]


def test_parse_missing_optional_fields():
    candidates = rs.parse_response(SAMPLE_RESPONSE)
    c2 = candidates[1]
    assert c2.title == "Space Agency Gallery"
    assert c2.url == "https://gallery.example.org/photo/42"
    assert c2.domain == "gallery.example.org"
    # Optional fields absent -> None, not a crash.
    assert c2.thumbnail_url is None
    assert c2.image_url is None
    assert c2.source is None
    # to_dict drops the None fields.
    assert "thumbnail_url" not in c2.to_dict()


def test_parse_exact_match_tagged():
    candidates = rs.parse_response(SAMPLE_RESPONSE)
    exact = [c for c in candidates if c.result_type == "exact_match"]
    assert len(exact) == 1
    assert exact[0].url == "https://archive.example.net/item/7"


def test_parse_zero_results():
    assert rs.parse_response({"visual_matches": []}) == []
    assert rs.parse_response({"search_metadata": {"id": "x"}}) == []


def test_parse_error_body_raises():
    with pytest.raises(rs.MalformedResponseError):
        rs.parse_response({"error": "Google hasn't returned any results."})


def test_parse_non_dict_raises():
    with pytest.raises(rs.MalformedResponseError):
        rs.parse_response(["not", "a", "dict"])


# --- search_image (fully mocked) -----------------------------------------------

def test_search_missing_api_key():
    with pytest.raises(rs.MissingAPIKeyError):
        rs.search_image(image_url="https://example.com/x.jpg", api_key="")


def test_search_requires_exactly_one_source():
    with pytest.raises(rs.ImageInputError):
        rs.search_image(api_key="k")  # neither path nor url
    with pytest.raises(rs.ImageInputError):
        rs.search_image(image_path="a.jpg", image_url="b", api_key="k")


def test_search_with_url_success():
    session = FakeSession(get_response=FakeResponse(200, SAMPLE_RESPONSE))
    result = rs.search_image(
        image_url="https://example.com/astro.jpg", api_key="k", session=session
    )
    assert result.count == 3
    assert result.search_id == "abc123"
    assert result.image_url == "https://example.com/astro.jpg"
    # The key was sent to SerpApi but is not stored on the result object.
    assert session.last_get["params"]["url"] == "https://example.com/astro.jpg"
    assert session.last_get["params"]["engine"] == "google_lens"


def test_search_zero_results_is_not_error():
    session = FakeSession(get_response=FakeResponse(200, {"visual_matches": []}))
    result = rs.search_image(image_url="https://example.com/x.jpg", api_key="k", session=session)
    assert result.count == 0


def test_search_invalid_key_maps_401():
    session = FakeSession(get_response=FakeResponse(401, {"error": "Invalid API key"}))
    with pytest.raises(rs.InvalidAPIKeyError):
        rs.search_image(image_url="https://example.com/x.jpg", api_key="bad", session=session)


def test_search_rate_limit_maps_429():
    session = FakeSession(get_response=FakeResponse(429, {"error": "run out of searches"}))
    with pytest.raises(rs.RateLimitError):
        rs.search_image(image_url="https://example.com/x.jpg", api_key="k", session=session)


def test_search_network_error():
    session = FakeSession(get_exc=requests.ConnectionError("boom"))
    with pytest.raises(rs.SearchNetworkError):
        rs.search_image(image_url="https://example.com/x.jpg", api_key="k", session=session)


def test_search_timeout():
    session = FakeSession(get_exc=requests.Timeout("slow"))
    with pytest.raises(rs.SearchNetworkError):
        rs.search_image(image_url="https://example.com/x.jpg", api_key="k", session=session)


def test_search_malformed_json():
    session = FakeSession(get_response=FakeResponse(200, json_body=None, text="<html>"))
    with pytest.raises(rs.MalformedResponseError):
        rs.search_image(image_url="https://example.com/x.jpg", api_key="k", session=session)


# --- upload path (mocked) ------------------------------------------------------

def test_upload_and_search_local_image(tmp_path, monkeypatch):
    # Create a tiny real image so the (default small-file) path reads bytes.
    import cv2, numpy as np
    img_path = tmp_path / "face.jpg"
    cv2.imwrite(str(img_path), np.full((60, 60, 3), 128, np.uint8))

    session = FakeSession(
        get_response=FakeResponse(200, SAMPLE_RESPONSE),
        post_response=FakeResponse(200, {"image_id": "IMG_ID_123"}),
    )
    result = rs.search_image(image_path=img_path, api_key="k", session=session)
    assert result.image_id == "IMG_ID_123"
    # image_id (not url) must have been sent to the search endpoint.
    assert session.last_get["params"]["image_id"] == "IMG_ID_123"
    assert "url" not in session.last_get["params"]
    # The upload POST carried the api_key as a form field and the image file.
    assert session.last_post["data"]["api_key"] == "k"
    assert "image" in session.last_post["files"]


def test_upload_missing_image_id():
    session = FakeSession(post_response=FakeResponse(200, {"message": "ok"}))
    with pytest.raises(rs.MalformedResponseError):
        rs.upload_image(_p := __import__("pathlib").Path(__file__), "k", session)


def test_upload_image_not_found(tmp_path):
    session = FakeSession()
    with pytest.raises(rs.ImageInputError):
        rs.upload_image(tmp_path / "nope.jpg", "k", session)


# --- sanitize_raw --------------------------------------------------------------

def test_sanitize_raw_scrubs_api_key():
    data = {"a": 1, "api_key": "SECRET", "nested": {"api_key": "SECRET2", "ok": 3}, "list": [{"api_key": "S3"}]}
    clean = rs.sanitize_raw(data)
    assert clean["api_key"] == "<redacted>"
    assert clean["nested"]["api_key"] == "<redacted>"
    assert clean["nested"]["ok"] == 3
    assert clean["list"][0]["api_key"] == "<redacted>"
    # original untouched
    assert data["api_key"] == "SECRET"
    # round-trips as JSON
    json.dumps(clean)
