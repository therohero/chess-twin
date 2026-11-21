"""Runtime configuration (environment variables with sensible defaults)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CHESSTWIN_DATA", ROOT / "data")).resolve()
BASE_MODEL = Path(os.environ.get("CHESSTWIN_BASE_MODEL", ROOT / "models" / "base.pt")).resolve()
LICHESS_TOKEN = os.environ.get("LICHESS_TOKEN") or None   # optional: raises Lichess rate limits
DEVICE = os.environ.get("CHESSTWIN_DEVICE") or None        # "cpu" / "cuda"; default: auto-detect with CPU fallback
STOCKFISH = os.environ.get("CHESSTWIN_STOCKFISH") or None   # path to a Stockfish binary (default: `stockfish` on PATH)
