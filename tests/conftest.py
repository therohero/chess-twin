import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schess.rules import Board  # noqa: E402
from chesstwin.san import to_san  # noqa: E402


def make_pgn_games(user, n, seed=0, plies=60):
    """Random legal games in Lichess PGN shape, with `user` playing alternate colours. -> list of (headers, movetext)"""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        b, sans = Board(), []
        for _ in range(plies):
            legal = b.legal_moves()
            if not legal or b.outcome(legal):
                break
            m = rnd.choice(legal)
            sans.append(to_san(b, m, legal))
            b.make(m)
        text = " ".join(f"{k // 2 + 1}. {s}" if k % 2 == 0 else s for k, s in enumerate(sans)) + " *"
        w, bl = (user, "opponent") if i % 2 == 0 else ("opponent", user)
        out.append(({"Event": "Rated Blitz game", "White": w, "Black": bl, "Variant": "Standard"}, text))
    return out
