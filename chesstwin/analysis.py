"""Stockfish game analysis: per-move quality, accuracy, and training weights derived from it.

Stockfish is driven over the UCI protocol with a tiny client (no extra dependencies). One search is run per
position of the game (N+1 searches for N moves): the search at position i gives the best move and the value of
the position *before* move i, and the search at position i+1 gives the value *after* it, so the cost of a move is
the drop in the mover's winning chances.
"""
import math
import os
import shutil
import subprocess
import threading

import numpy as np

from schess.encoding import encode_move, encode_position, history_moves
from schess.rules import Board, WHITE, move_uci

from .san import to_san

MATE_CP = 10_000
DEFAULT_DEPTH = 12
LABELS = ("best", "excellent", "good", "inaccuracy", "mistake", "blunder")
GOOD_LABELS = ("best", "excellent", "good")
# win-probability points lost -> label (a move that is not the engine's top choice but loses <= 0.5 points is "best")
_THRESHOLDS = ((0.5, "best"), (2.0, "excellent"), (5.0, "good"), (10.0, "inaccuracy"), (20.0, "mistake"))
BAD_AT = 20.0           # a loss of this many win% points counts as "fully bad" for weighting


class StockfishError(Exception):
    """An error with a message that is safe to show the user."""


def find_stockfish(path=None):
    """Explicit path > $CHESSTWIN_STOCKFISH > `stockfish` on PATH. Returns an executable path or None."""
    for cand in (path, os.environ.get("CHESSTWIN_STOCKFISH")):
        if cand:
            found = shutil.which(cand)
            if found:
                return found
    # Debian/Ubuntu install it to /usr/games, which is usually not on PATH
    return shutil.which("stockfish") or shutil.which("stockfish", path="/usr/games:/usr/local/games")


# ---------------------------------------------------------------------------------------------- UCI client
class UciEngine:
    def __init__(self, path=None, threads=1, hash_mb=64):
        exe = find_stockfish(path)
        if not exe:
            raise StockfishError("Stockfish was not found. Install it (e.g. `apt install stockfish`, `pacman -S stockfish`, "
                                 "`brew install stockfish`) or set CHESSTWIN_STOCKFISH to its path.")
        try:
            self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1)
        except OSError as e:
            raise StockfishError(f"Could not start Stockfish ({e}).") from e
        self.path = exe
        self._lock = threading.Lock()
        self._send("uci")
        self._wait("uciok")
        self._send(f"setoption name Threads value {int(threads)}")
        self._send(f"setoption name Hash value {int(hash_mb)}")
        self._send("isready")
        self._wait("readyok")

    def _send(self, line):
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise StockfishError("Stockfish stopped responding.") from e

    def _wait(self, token):
        for line in self.proc.stdout:
            if line.strip().startswith(token):
                return line.strip()
        raise StockfishError("Stockfish exited unexpectedly.")

    def set_option(self, name, value):
        with self._lock:
            self._send(f"setoption name {name} value {value}")
            self._send("isready")
            self._wait("readyok")

    def analyse(self, fen, depth=DEFAULT_DEPTH, movetime=None):
        """Search `fen`. -> (score_cp from the side to move's view, best move in UCI or None). Mate = +-MATE_CP."""
        with self._lock:
            self._send(f"position fen {fen}")
            self._send(f"go movetime {int(movetime)}" if movetime else f"go depth {int(depth)}")
            score = 0
            for line in self.proc.stdout:
                tok = line.split()
                if not tok:
                    continue
                if tok[0] == "info" and "score" in tok and "lowerbound" not in tok and "upperbound" not in tok:
                    i = tok.index("score")
                    kind, val = tok[i + 1], int(tok[i + 2])
                    if kind == "cp":
                        score = val
                    elif kind == "mate":
                        score = (MATE_CP - abs(val)) * (1 if val > 0 else -1)
                elif tok[0] == "bestmove":
                    best = tok[1] if len(tok) > 1 and tok[1] not in ("(none)", "0000") else None
                    return score, best
            raise StockfishError("Stockfish exited unexpectedly.")

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except (StockfishError, subprocess.TimeoutExpired):
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------------------------- maths
def win_pct(cp):
    """Centipawns -> winning chances in percent (the Lichess model)."""
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * max(-MATE_CP, min(MATE_CP, cp)))) - 1.0)


def move_accuracy(loss):
    """Win-percent points lost by one move -> 0..100 (the Lichess accuracy curve)."""
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss) - 3.1669))


def classify(loss, is_best):
    if is_best:
        return "best"
    for limit, label in _THRESHOLDS:
        if loss <= limit:
            return label
    return "blunder"


def parse_uci(board, uci):
    for m in board.legal_moves():
        if move_uci(m) == uci:
            return m
    return None


# ---------------------------------------------------------------------------------------------- analysis
def analyse_game(ucis, engine, depth=DEFAULT_DEPTH, movetime=None, progress=None, cancel=None):
    """Analyse a game given as UCI moves from the start position.

    -> {"moves": [...], "evals": [cp from White's view after 0..N plies], "white": summary, "black": summary}
    """
    board = Board()
    positions = []   # (fen, legal moves, outcome) before each move, plus the final position
    sans = []
    for u in ucis:
        legal = board.legal_moves()
        m = parse_uci(board, u)
        if m is None:
            raise ValueError(f"illegal move {u} in {board.fen()}")
        positions.append((board.fen(), board.turn, board.outcome(legal)))
        sans.append(to_san(board, m, legal))
        board.make(m)
    positions.append((board.fen(), board.turn, board.outcome()))

    values = []      # per position: (cp for side to move, best uci or None)
    for i, (fen, turn, outcome) in enumerate(positions):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        if outcome is not None:   # game over: checkmate is lost for the side to move, anything else is a draw
            values.append((-MATE_CP if outcome[1] == "checkmate" else 0, None))
        else:
            values.append(engine.analyse(fen, depth, movetime))
        if progress:
            progress((i + 1) / len(positions), f"Analysing move {min(i + 1, len(ucis))}/{len(ucis)}")

    moves = []
    for i, u in enumerate(ucis):
        color = positions[i][1]
        before, best = values[i]
        after = -values[i + 1][0]                       # value of the played move for the same side
        after = min(after, before) if best == u else after   # the engine's own move can't be worse than its estimate
        loss = max(0.0, win_pct(before) - win_pct(after))
        bmove = parse_uci(Board(positions[i][0]), best) if best else None
        label = classify(loss, best == u)
        moves.append({
            "ply": i + 1, "color": "white" if color == WHITE else "black", "uci": u, "san": sans[i],
            "label": label, "cp_before": before, "cp_after": after, "loss": round(loss, 2),
            "accuracy": round(move_accuracy(loss), 1),
            "best_uci": best, "best_san": to_san(Board(positions[i][0]), bmove) if bmove else None,
        })
    evals = [(v if positions[i][1] == WHITE else -v) for i, (v, _) in enumerate(values)]
    out = {"moves": moves, "evals": evals, "depth": depth, "movetime": movetime}
    for color in ("white", "black"):
        out[color] = summarise([m for m in moves if m["color"] == color])
    return out


def summarise(moves):
    counts = {k: 0 for k in LABELS}
    for m in moves:
        counts[m["label"]] += 1
    n = len(moves)
    return {"moves": n, "counts": counts,
            "accuracy": round(sum(m["accuracy"] for m in moves) / n, 1) if n else None,
            "avg_loss": round(sum(m["loss"] for m in moves) / n, 2) if n else None}


# ---------------------------------------------------------------------------------------------- learning weights
TIE = 3.0        # accuracy difference (points) below which neither side is considered better
FULL_GAP = 20.0  # accuracy difference at which the re-weighting reaches full strength


def weighting(analysis, human_color):
    """How should the human's moves be weighted when teaching the twin from this game?

    The twin imitates the player. If the player played *worse* than the bot, the bot is currently too strong
    compared with the person it is meant to mirror, so the player's mistakes are the informative moves (weight up)
    and their good moves are ones the bot already makes (weight down). If the player played *better* than the bot,
    it is the other way round: the bot should pick up the player's good moves and not their slips.
    Returns {"mode": "player_worse"|"player_better"|"even", "strength": 0..1, "gap": accuracy difference}.
    """
    me, bot = analysis[human_color], analysis["black" if human_color == "white" else "white"]
    if me["accuracy"] is None or bot["accuracy"] is None:
        return {"mode": "even", "strength": 0.0, "gap": 0.0}
    gap = bot["accuracy"] - me["accuracy"]        # > 0: the bot played better
    if abs(gap) < TIE:
        return {"mode": "even", "strength": 0.0, "gap": round(gap, 1)}
    strength = min(1.0, abs(gap) / FULL_GAP)
    return {"mode": "player_worse" if gap > 0 else "player_better", "strength": round(strength, 2), "gap": round(gap, 1)}


def move_weights(analysis, human_color, w=None):
    """Per human move weight (mean 1). Bad = 1 at a >= BAD_AT win% loss, good = 0 at no loss."""
    w = w or weighting(analysis, human_color)
    mine = [m for m in analysis["moves"] if m["color"] == human_color]
    bad = np.array([min(1.0, m["loss"] / BAD_AT) for m in mine], dtype=np.float64)
    if w["mode"] == "player_worse":
        raw = 1.0 + w["strength"] * (1.5 * bad - 0.5 * (1 - bad))    # blunder -> 2.5x, perfect move -> 0.5x at full strength
    elif w["mode"] == "player_better":
        raw = 1.0 + w["strength"] * (0.5 * (1 - bad) - 0.8 * bad)    # good -> 1.5x, blunder -> 0.2x
    else:
        raw = np.ones(len(mine))
    raw = np.maximum(raw, 0.05)
    return raw / raw.mean() if len(raw) else raw


def training_samples(analysis, human_color):
    """-> dict(squares, meta, move, hist, weight) for the human's moves in an analysed game, or None if empty."""
    board = Board()
    turn_color = WHITE if human_color == "white" else BLACK
    sq, meta, mv, hist, idx = [], [], [], [], []
    for m in analysis["moves"]:
        move = parse_uci(board, m["uci"])
        if board.turn == turn_color:
            s, mt = encode_position(board)
            sq.append(s); meta.append(mt); mv.append(encode_move(move, board.turn)); idx.append(m["ply"] - 1)
            hist.append(history_moves(board))
        board.make(move)
    if not sq:
        return None
    wts = move_weights(analysis, human_color)
    return {"squares": np.stack(sq), "meta": np.array(meta, np.uint8), "move": np.array(mv, np.int64), "hist": np.stack(hist),
            "weight": wts.astype(np.float32)}
