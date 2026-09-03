"""Small logging helpers for consistent, readable CLI output.

The pipeline is a multi-stage CLI, so we want two things:
  1. A conventional `logging.Logger` for diagnostics / debug output.
  2. A few tiny helpers that print the clean, human-facing stage markers
     the spec asks for ([1/2] ..., u2713 ...).

Keeping these together means every phase (face -> search -> chain) shares
one visual language.
"""

from __future__ import annotations

import logging
import sys

# ANSI colors. Kept intentionally minimal and disabled automatically when the
# output is not a TTY (e.g. piped to a file or CI logs).
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def get_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# --- Human-facing stage markers -------------------------------------------------

def stage(current: int, total: int, message: str) -> None:
    """Print a numbered stage marker, e.g. `[1/2] Loading image`."""
    print(f"{_c('36;1', f'[{current}/{total}]')} {message}")


def success(message: str) -> None:
    """Print a success line, e.g. `u2713 Face detected`."""
    print(f"{_c('32;1', '✓')} {message}")


def info(message: str) -> None:
    print(f"  {message}")


def failure(message: str) -> None:
    """Print a failure line to stderr, e.g. `u2717 No face detected`."""
    print(f"{_c('31;1', '✗')} {message}", file=sys.stderr)
