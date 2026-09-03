"""Manual, opt-in LIVE integration check for Phase 2.

This is deliberately NOT a pytest test — the normal `pytest` suite must never
consume SerpApi quota. Run it explicitly, with a real key configured in .env:

    ./.venv/bin/python tests/live_search.py
    ./.venv/bin/python tests/live_search.py --image data/input/test.jpg

It performs ONE genuine SerpApi Google Lens search and prints the number of
candidates plus the first few discovered URLs, so you can confirm results are
returned live by SerpApi and are not hardcoded. It makes a single request (no
automatic retries) to respect the free-plan quota.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python tests/live_search.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config  # noqa: E402
from pipeline import reverse_search as rs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Live SerpApi Google Lens check.")
    parser.add_argument("--image", default=str(config.input_dir / "test.jpg"))
    args = parser.parse_args()

    if not config.serpapi_key:
        print("SERPAPI_API_KEY not set in .env — cannot run live check.", file=sys.stderr)
        return 6

    print(f"Running ONE live Google Lens search for: {args.image}")
    try:
        result = rs.search_image(
            image_path=args.image,
            api_key=config.serpapi_key,
            engine=config.serpapi_engine,
        )
    except rs.ReverseSearchError as e:
        print(f"Live search failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"image_id: {result.image_id}")
    print(f"search_id: {result.search_id}")
    print(f"candidates: {result.count}")
    for c in result.candidates[:5]:
        print(f"  [{c.position}] {c.domain or c.source}  ->  {c.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
