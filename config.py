"""Central configuration, loaded from environment / .env.

Secrets (SerpApi key, Ethereum RPC URL, private key) are NEVER hardcoded and
NEVER committed. They live in a local `.env` file (git-ignored) and are read
here via python-dotenv. Phase 1 only needs the face-detection settings; the
Phase 2/4 fields are declared now so later phases have a single source of
truth, but they are optional and unused until then.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present. Never raises if it's missing.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    input_dir: Path = PROJECT_ROOT / "data" / "input"
    candidates_dir: Path = PROJECT_ROOT / "data" / "candidates"
    results_dir: Path = PROJECT_ROOT / "data" / "results"

    # --- Face detection (Phase 1) ---
    # InsightFace model pack. "buffalo_l" is the standard full pack; "buffalo_s"
    # is smaller/faster. Overridable via env for constrained machines.
    insightface_model: str = field(default_factory=lambda: os.getenv("INSIGHTFACE_MODEL", "buffalo_l"))
    # Detection input size (square). Larger = more accurate, slower.
    det_size: int = field(default_factory=lambda: int(os.getenv("DET_SIZE", "640")))
    # Minimum detector confidence to accept a face.
    det_threshold: float = field(default_factory=lambda: float(os.getenv("DET_THRESHOLD", "0.5")))
    # Force CPU execution provider. GPU can be enabled later by setting this off.
    use_cpu: bool = field(default_factory=lambda: _get_bool("USE_CPU", True))

    # --- Reverse image search (Phase 2) ---
    # SERPAPI_API_KEY is the canonical name; SERPAPI_KEY kept as a fallback so
    # older .env files still work. Never logged or written to disk.
    serpapi_key: str = field(
        default_factory=lambda: os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY", "")
    )
    serpapi_engine: str = field(default_factory=lambda: os.getenv("SERPAPI_ENGINE", "google_lens"))

    # --- Blockchain (Phase 4) ---
    # Backward-compatible names: prefer ETH_* if present, else fall back to the
    # existing SEPOLIA_RPC_URL / PRIVATE_KEY names. Never logged or written out.
    # WALLET_ADDRESS is intentionally NOT read here — the sender address is
    # derived from the private key in pipeline/blockchain.py.
    eth_rpc_url: str = field(
        default_factory=lambda: os.getenv("ETH_RPC_URL") or os.getenv("SEPOLIA_RPC_URL", "")
    )
    eth_private_key: str = field(
        default_factory=lambda: os.getenv("ETH_PRIVATE_KEY") or os.getenv("PRIVATE_KEY", "")
    )
    eth_chain_id: int = field(default_factory=lambda: int(os.getenv("ETH_CHAIN_ID", "11155111")))  # Sepolia

    def ensure_dirs(self) -> None:
        for d in (self.input_dir, self.candidates_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


# A shared, ready-to-use instance.
config = Config()
