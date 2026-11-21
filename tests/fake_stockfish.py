"""A tiny UCI engine for tests: 1-ply material search using the repo's own rules. Not strong, but deterministic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schess.rules import Board, WHITE, move_uci  # noqa: E402

VAL = [0, 100, 300, 300, 500, 900, 0]


def material(b):   # from the side to move's view
    s = 0
    for p in b.sq:
        if p:
            v = VAL[(p - 1) % 6 + 1]
            s += v if (p <= 6) == (b.turn == WHITE) else -v
    return s


def search(b):
    best, bs = None, -10 ** 9
    for m in b.legal_moves():
        b.make(m)
        if b.legal_moves():
            sc = -material(b)
        else:
            sc = 9000 if b.in_check() else 0
        b.unmake()
        if sc > bs:
            best, bs = m, sc
    return best, bs


def main():
    b = Board()
    for line in sys.stdin:
        t = line.split()
        if not t:
            continue
        if t[0] == "uci":
            print("id name FakeFish\nuciok", flush=True)
        elif t[0] == "isready":
            print("readyok", flush=True)
        elif t[0] == "position":
            b = Board(" ".join(t[2:]))
        elif t[0] == "go":
            m, sc = search(b)
            if m is None:
                print("info depth 0 score cp 0\nbestmove (none)", flush=True)
            elif sc >= 9000:
                print("info depth 1 score mate 1\nbestmove " + move_uci(m), flush=True)
            else:
                print(f"info depth 1 score cp {sc}\nbestmove {move_uci(m)}", flush=True)
        elif t[0] == "quit":
            return


main()
