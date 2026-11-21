"""Check how good a policy model (default: models/base.pt) is.

  1. Prediction accuracy: top-1 / top-5 on human moves from your preprocessed data (needs --data, no Stockfish).
  3. Match mode (--levels): the model plays Stockfish at each "Skill Level" (0-20) and the score per level shows
     where it stops winning.
  2. Playing strength: the model plays games against itself at several temperatures and Stockfish rates every
     move it chose (accuracy, win-% lost per move, share of best moves / mistakes / blunders).

    python analyze_model.py                                   # both checks, defaults
    python analyze_model.py --model checkpoints/model.pt --games 20 --depth 12
    python analyze_model.py --skip-play                       # accuracy only
    python analyze_model.py --data none                       # strength only
    python analyze_model.py --levels 0-20 --games 8           # match against Stockfish skill levels 0..20 (only that)
    python analyze_model.py --levels 0-5 --search 2           # same, with the app's blunder check

Temperature 0 = always the model's favourite move, 1 = sample like the training players, >1 = looser.
Note: the held-out shard may overlap with the data the model was trained on if you trained on all shards.
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

from chesstwin import config
from chesstwin.analysis import DEFAULT_DEPTH, LABELS, StockfishError, UciEngine, analyse_game
from chesstwin.engine import Engine
from schess.encoding import HISTORY_MAX, sequence_history
from schess.model import load_model
from schess.rules import Board, move_uci


def prediction_accuracy(net, data_dir, n, device):
    files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not files:
        print(f"(no shard_*.npz in {data_dir} - skipping the prediction check)\n")
        return
    d = np.load(files[-1])
    n = min(n, len(d["move"]))
    h = torch.from_numpy(sequence_history(d["move"][-n:], d["game"][-n:], HISTORY_MAX)).to(device)
    s, m, y = (torch.from_numpy(d["squares"][-n:]).to(device), torch.from_numpy(d["meta"][-n:]).to(device),
               torch.from_numpy(d["move"][-n:].astype(np.int64)).to(device))
    top1 = top5 = 0
    loss = 0.0
    with torch.no_grad():
        for i in range(0, n, 2048):
            out = net(s[i:i + 2048], m[i:i + 2048], h[i:i + 2048])
            t5 = out.topk(5, 1).indices
            top1 += (t5[:, 0] == y[i:i + 2048]).sum().item()
            top5 += (t5 == y[i:i + 2048, None]).any(1).sum().item()
            loss += torch.nn.functional.cross_entropy(out, y[i:i + 2048], reduction="sum").item()
    print(f"Prediction of human moves ({n:,} positions from {os.path.basename(files[-1])})")
    print(f"  top-1 {top1 / n:.1%}   top-5 {top5 / n:.1%}   cross-entropy {loss / n:.3f}\n")


SEARCH = 0   # blunder-check plies for the model's moves (set by --search)


class _Store:
    def __init__(self, path):
        self.path = __import__("pathlib").Path(path)

    def model_path(self, pid):
        return self.path


def play_game(engine, temperature, random_plies, max_plies, rnd):
    """Model plays both sides. The first `random_plies` moves are sampled at T=1 for variety. -> list of UCI"""
    board, ucis = Board(), []
    while len(ucis) < max_plies:
        legal = board.legal_moves()
        if not legal or board.outcome(legal):
            break
        t = 1.0 if len(ucis) < random_plies else temperature
        m = engine.choose("m", board, t, legal, search=SEARCH)
        ucis.append(move_uci(m))
        board.make(m)
    return ucis


def strength(net_path, device, temps, games, depth, random_plies, max_plies, sf_path, seed):
    engine = Engine(_Store(net_path), device)
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    with UciEngine(sf_path) as sf:
        for t in temps:
            moves = []
            for g in range(games):
                ucis = play_game(engine, t, random_plies, max_plies, rnd)
                print(f"\r  T={t}: analysing game {g + 1}/{games} ({len(ucis)} plies)  ", end="", file=sys.stderr, flush=True)
                a = analyse_game(ucis, sf, depth)
                moves += [m for m in a["moves"] if m["ply"] > random_plies]
            print("\r" + " " * 70 + "\r", end="", file=sys.stderr)
            if not moves:
                continue
            n = len(moves)
            acc = sum(m["accuracy"] for m in moves) / n
            loss = sum(m["loss"] for m in moves) / n
            cnt = {k: sum(m["label"] == k for m in moves) / n for k in LABELS}
            print(f"  T={t:<4} accuracy {acc:5.1f}   avg loss {loss:5.2f} win-%/move   " +
                  "  ".join(f"{k} {cnt[k]:.0%}" for k in ("best", "inaccuracy", "mistake", "blunder")) + f"   ({n} moves)")


def match(net_path, device, level, games, movetime, model_temp, random_plies, max_plies, sf_path, seed):
    """Model vs Stockfish at `Skill Level` level; colours alternate. -> (wins, draws, losses) from the model's view."""
    engine = Engine(_Store(net_path), device)
    w = d = l = 0
    with UciEngine(sf_path) as sf:
        sf.set_option("Skill Level", level)
        for g in range(games):
            torch.manual_seed(seed + g)
            model_white = g % 2 == 0
            board, plies = Board(), 0
            while plies < max_plies:
                legal = board.legal_moves()
                if not legal or board.outcome(legal):
                    break
                if (board.turn == 0) == model_white:
                    m = engine.choose("m", board, 1.0 if plies < random_plies else model_temp, legal,
                                      search=0 if plies < random_plies else SEARCH)
                else:
                    _, best = sf.analyse(board.fen(), movetime=movetime)
                    m = next(x for x in legal if move_uci(x) == best)
                board.make(m)
                plies += 1
            out = board.outcome()
            if out is None or out[0] == "1/2-1/2":
                d += 1
            elif (out[0] == "1-0") == model_white:
                w += 1
            else:
                l += 1
    return w, d, l


def run_levels(args, device):
    lo, hi = (int(x) for x in args.levels.split("-"))
    print(f"Model vs Stockfish skill levels {lo}-{hi}: {args.games} games per level (colours alternate), "
          f"Stockfish {args.movetime} ms/move, model T={args.model_temp}, "
          f"blunder check {args.search or 'off'}\n")
    print(f"{'level':>5} {'W':>3} {'D':>3} {'L':>3}  score")
    for lvl in range(lo, hi + 1):
        w, d, l = match(args.model, device, lvl, args.games, args.movetime, args.model_temp, args.random_plies,
                        args.max_plies, args.stockfish, args.seed)
        print(f"{lvl:>5} {w:>3} {d:>3} {l:>3}  {(w + d / 2) / args.games:5.0%}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(config.BASE_MODEL), help="checkpoint to test (default: models/base.pt)")
    ap.add_argument("--data", default="data_base", help="preprocessed shard directory, or 'none'")
    ap.add_argument("--positions", type=int, default=100_000, help="positions for the prediction check")
    ap.add_argument("--skip-play", action="store_true", help="skip the Stockfish strength check")
    ap.add_argument("--games", type=int, default=10, help="self-play games per temperature")
    ap.add_argument("--temps", default="0,1", help="comma-separated temperatures (default 0,1)")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Stockfish depth per position")
    ap.add_argument("--random-plies", type=int, default=6, help="opening plies sampled at T=1 for variety (not rated in strength mode)")
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--stockfish", help="path to the Stockfish binary")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", help="match mode: play Stockfish skill levels, e.g. 0-20 (replaces the other checks)")
    ap.add_argument("--movetime", type=int, default=100, help="match mode: Stockfish milliseconds per move")
    ap.add_argument("--search", type=int, default=0, choices=range(4),
                    help="blunder-check plies for the model's moves (0 = pure policy, as in the app's default)")
    ap.add_argument("--model-temp", type=float, default=0.0, help="match mode: model temperature")
    args = ap.parse_args()
    global SEARCH
    SEARCH = args.search
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(args.model, device)
    print(f"Model: {args.model}  {net.config}  ({sum(p.numel() for p in net.parameters()):,} parameters, {device})\n")
    if args.levels:
        try:
            run_levels(args, device)
        except StockfishError as e:
            sys.exit(f"\n{e}")
        return
    if args.data.lower() != "none":
        prediction_accuracy(net, args.data, args.positions, device)
    if not args.skip_play:
        print(f"Strength vs Stockfish (depth {args.depth}), model plays both sides; moves after the first "
              f"{args.random_plies} plies are rated:")
        try:
            strength(args.model, device, [float(t) for t in args.temps.split(",")], args.games, args.depth,
                     args.random_plies, args.max_plies, args.stockfish, args.seed)
        except StockfishError as e:
            sys.exit(f"\n{e}\n(use --skip-play to only run the prediction check)")
        print("\nFor reference: accuracy ~95+ is strong-engine play, ~80-90 a decent club player, <70 beginner-level.")


if __name__ == "__main__":
    main()
