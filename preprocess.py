"""Convert a Lichess PGN database (.pgn / .pgn.zst / .pgn.bz2) into training shards.

  python preprocess.py lichess_db_standard_rated_2024-01.pgn.zst --out data --min-elo 1800

Each shard is data/shard_00000.npz with arrays: squares int8[N,64], meta uint8[N], move int16[N], game int32[N] (game index).
Every ply of every accepted game becomes one (position -> move played) sample.
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

from schess.encoding import encode_move, encode_position
from schess.pgn import iter_games, movetext_to_sans, open_text, parse_san
from schess.rules import Board


def process_batch(games):
    """games: list of (movetext, skip_plies). Returns arrays for all positions."""
    sqs, metas, moves, gids = [], [], [], []
    bad = 0
    for text, skip, gid in games:
        board = Board()
        try:
            for ply, san in enumerate(movetext_to_sans(text)):
                m = parse_san(board, san)
                if ply >= skip:
                    sq, meta = encode_position(board)
                    sqs.append(sq)
                    metas.append(meta)
                    moves.append(encode_move(m, board.turn))
                    gids.append(gid)
                board.make(m)
        except ValueError:
            bad += 1
    if not sqs:
        return np.zeros((0, 64), np.int8), np.zeros(0, np.uint8), np.zeros(0, np.int16), np.zeros(0, np.int32), bad
    return np.stack(sqs), np.array(metas, np.uint8), np.array(moves, np.int16), np.array(gids, np.int32), bad


def accepted(h, args):
    if h.get("Variant", "Standard") not in ("Standard", "From Position") or "FEN" in h or "SetUp" in h:
        return False
    if h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    try:
        w, b = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        return False
    if min(w, b) < args.min_elo or max(w, b) > args.max_elo:
        return False
    if args.no_forfeit and h.get("Termination") in ("Time forfeit", "Abandoned", "Rules infraction"):
        return False
    return True


def batches(args):
    f = open_text(args.pgn)
    seen = taken = 0
    chunk = []
    for h, text in iter_games(f):
        seen += 1
        if accepted(h, args):
            taken += 1
            chunk.append((text, args.skip_plies, taken - 1))
            if len(chunk) == 256:
                yield chunk
                chunk = []
            if args.max_games and taken >= args.max_games:
                break
        if seen % 200000 == 0:
            print(f"  scanned {seen:,} games, kept {taken:,}", file=sys.stderr, flush=True)
    if chunk:
        yield chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-elo", type=int, default=1800, help="both players must be at least this rated")
    ap.add_argument("--max-elo", type=int, default=4000)
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-plies", type=int, default=0, help="don't train on the first N plies")
    ap.add_argument("--no-forfeit", action="store_true", help="drop games ended by time forfeit/abandonment")
    ap.add_argument("--shard-size", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    buf = {"s": [], "m": [], "v": [], "g": []}
    n_buf = shard = total = bad = 0

    def flush():
        nonlocal buf, n_buf, shard
        if not n_buf:
            return
        np.savez(os.path.join(args.out, f"shard_{shard:05d}.npz"),
                 squares=np.concatenate(buf["s"]), meta=np.concatenate(buf["m"]), move=np.concatenate(buf["v"]),
                 game=np.concatenate(buf["g"]))
        print(f"wrote shard {shard} ({n_buf:,} positions)", flush=True)
        buf, n_buf, shard = {"s": [], "m": [], "v": [], "g": []}, 0, shard + 1

    with mp.Pool(args.workers) as pool:
        for s, m, v, g, b in pool.imap(process_batch, batches(args), chunksize=1):
            bad += b
            buf["s"].append(s); buf["m"].append(m); buf["v"].append(v); buf["g"].append(g)
            n_buf += len(v)
            total += len(v)
            if n_buf >= args.shard_size:
                flush()
    flush()
    print(f"done: {total:,} positions, {bad} games skipped due to parse errors")


if __name__ == "__main__":
    main()
