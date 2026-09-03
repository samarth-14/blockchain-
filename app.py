"""Face -> Web Search -> Blockchain Verification  |  CLI entry point.

NON-COMMERCIAL research / demo. Intended ONLY for a consenting participant or a
controlled test image. Do not use for surveillance, mass identification, or
building dossiers on people.

Current scope (Phase 1 + 2):
  [1/3] load image  ->  [2/3] detect single face + embedding
  ->  [3/3] genuine reverse image search via SerpApi Google Lens.

  Usage:  python app.py --image data/input/test.jpg
          python app.py --image data/input/test.jpg --debug-search

Later phases (candidate matching, fingerprint, blockchain) will extend this CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import config
from utils import logger
from pipeline import face_detector as fd
from pipeline import reverse_search as rs

log = logger.get_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Detect a single face, embed it, then reverse-search the "
                    "image with SerpApi Google Lens.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (a consenting subject or test image).",
    )
    parser.add_argument(
        "--crop-out",
        default=None,
        help="Where to save the face crop (default: data/results/<image>_face.jpg).",
    )
    parser.add_argument(
        "--model",
        default=config.insightface_model,
        help=f"InsightFace model pack (default: {config.insightface_model}).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=config.det_size,
        help=f"Detector input size (default: {config.det_size}).",
    )
    parser.add_argument(
        "--det-threshold",
        type=float,
        default=config.det_threshold,
        help=f"Minimum detector confidence (default: {config.det_threshold}).",
    )
    parser.add_argument(
        "--debug-search",
        action="store_true",
        help="Write the raw SerpApi JSON (credentials scrubbed) to "
             "data/results/lens_response.json.",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=10,
        help="Max candidates to print (default: 10).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config.ensure_dirs()

    image_path = Path(args.image)

    # Default crop output path derived from the input filename.
    if args.crop_out:
        crop_out = Path(args.crop_out)
    else:
        crop_out = config.results_dir / f"{image_path.stem}_face.jpg"

    # ---- Stage 1: load image -------------------------------------------------
    logger.stage(1, 3, f"Loading image  ({image_path})")
    try:
        image = fd.load_image(image_path)
    except fd.ImageLoadError as e:
        logger.failure(str(e))
        return 2
    h, w = image.shape[:2]
    logger.info(f"Image loaded: {w}x{h} px")

    # ---- Stage 2: detect face + embedding -----------------------------------
    logger.stage(2, 3, "Detecting face")
    try:
        result = fd.analyze_image(
            image_path,
            crop_path=crop_out,
            model_name=args.model,
            det_size=args.det_size,
            det_threshold=args.det_threshold,
            use_cpu=config.use_cpu,
        )
    except fd.NoFaceError:
        logger.failure("No face detected. Provide an image with exactly one clear face.")
        return 3
    except fd.MultipleFacesError as e:
        logger.failure(f"{e}  (the MVP accepts exactly one face)")
        return 4
    except fd.FaceDetectionError as e:
        logger.failure(f"Face stage failed: {e}")
        return 5

    logger.success("Face detected")
    logger.info(f"Bounding box (x1,y1,x2,y2): {result.bbox}")
    logger.info(f"Detector confidence: {result.det_score:.4f}")
    logger.success("Embedding generated")
    logger.info(f"Embedding: {result.embedding_preview()}")
    if result.crop_path:
        logger.info(f"Face crop saved: {result.crop_path}")

    # ---- Stage 3: reverse image search (Google Lens via SerpApi) ------------
    logger.stage(3, 3, "Searching web with Google Lens")
    try:
        # We search with the FULL input image (more context than a tight crop,
        # so Google Lens returns better candidates). The face stage above has
        # already enforced the single-consenting-face rule.
        search = rs.search_image(
            image_path=image_path,
            api_key=config.serpapi_key,
            engine=config.serpapi_engine,
        )
    except rs.MissingAPIKeyError as e:
        logger.failure(str(e))
        return 6
    except rs.InvalidAPIKeyError as e:
        logger.failure(f"SerpApi rejected the API key: {e}")
        return 7
    except rs.RateLimitError as e:
        logger.failure(f"SerpApi rate limit / quota exceeded: {e}")
        return 8
    except rs.SearchNetworkError as e:
        logger.failure(f"Network/timeout during search: {e}")
        return 9
    except rs.MalformedResponseError as e:
        logger.failure(f"Could not parse SerpApi response: {e}")
        return 10
    except rs.ImageInputError as e:
        logger.failure(f"Image problem for search: {e}")
        return 11
    except rs.ReverseSearchError as e:
        logger.failure(f"Reverse search failed: {e}")
        return 12

    if args.debug_search:
        _write_debug_search(search)

    _print_search_results(search, max_show=args.max_show)

    print()
    logger.success("Phase 2 complete.")
    return 0


def _write_debug_search(search: "rs.ReverseSearchResult") -> None:
    """Persist the raw SerpApi JSON (credentials scrubbed) for inspection."""
    out = config.results_dir / "lens_response.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(rs.sanitize_raw(search.raw_response), f, indent=2, ensure_ascii=False)
    logger.info(f"Raw SerpApi JSON written to: {out}")


def _print_search_results(search: "rs.ReverseSearchResult", max_show: int = 10) -> None:
    """Print a compact, human-readable summary of the candidates."""
    print()
    print("=" * 50)
    print("GOOGLE LENS SEARCH")
    print("=" * 50)
    print()
    logger.success("Search completed")
    if search.image_id:
        logger.info(f"Uploaded image id: {search.image_id}")
    logger.info(f"Candidates found: {search.count}")

    if search.count == 0:
        print()
        logger.info("No candidates returned for this image.")
        return

    for c in search.candidates[:max_show]:
        idx = c.position if c.position is not None else "?"
        print()
        print(f"[{idx}] {c.title or '(no title)'}")
        print(f"    Source: {c.source or c.domain or '(unknown)'}")
        print(f"    URL:    {c.url or '(none)'}")

    remaining = search.count - min(search.count, max_show)
    if remaining > 0:
        print()
        logger.info(f"... and {remaining} more (use --max-show to see more).")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.failure("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
