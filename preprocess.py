"""Convert a Lichess PGN database (.pgn / .pgn.zst / .pgn.bz2) into training shards.

  python preprocess.py lichess_db_standard_rated_2024-01.pgn.zst --out data --elo-range 1800-2200
  python preprocess.py games.pgn.zst --out data --elo-range 1200-2400 --elo-bin 200   # sorted into data/elo_1200-1399/ ...

Elo filter: --elo-range MIN-MAX (or --min-elo / --max-elo), applied according to --elo-mode:
  both    both players must be inside the range (default)
  avg     the average of the two ratings must be inside the range
  either  at least one player must be inside the range
With --elo-bin N, accepted games are sorted into N-wide rating bands (by the player ratings per --elo-mode:
lower rating for "both", average for "avg", the in-range player's rating for "either"), one subdirectory per band.

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


def parse_elo_range(text):
    try:
        lo, hi = text.replace(":", "-").split("-")
        return int(lo or 0), int(hi or 4000)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad --elo-range {text!r}, expected MIN-MAX like 1800-2200")


def elo_key(w, b, args):
    """Return the rating used for range checks/binning, or None if the game is outside the range."""
    lo, hi = args.min_elo, args.max_elo
    if args.elo_mode == "both":
        return min(w, b) if lo <= min(w, b) and max(w, b) <= hi else None
    if args.elo_mode == "avg":
        a = (w + b) // 2
        return a if lo <= a <= hi else None
    inside = [r for r in (w, b) if lo <= r <= hi]
    return min(inside) if inside else None


def bucket_of(elo, args):
    if not args.elo_bin:
        return None
    return args.min_elo + (elo - args.min_elo) // args.elo_bin * args.elo_bin


def accepted(h, args):
    """Returns (ok, bucket)."""
    if h.get("Variant", "Standard") not in ("Standard", "From Position") or "FEN" in h or "SetUp" in h:
        return False, None
    if h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False, None
    try:
        w, b = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        return False, None
    elo = elo_key(w, b, args)
    if elo is None:
        return False, None
    if args.no_forfeit and h.get("Termination") in ("Time forfeit", "Abandoned", "Rules infraction"):
        return False, None
    return True, bucket_of(elo, args)


def batches(args):
    """Yields (bucket, chunk); chunks never mix elo buckets (bucket is None without --elo-bin)."""
    f = open_text(args.pgn)
    seen = taken = 0
    chunks = {}
    for h, text in iter_games(f):
        seen += 1
        ok, bucket = accepted(h, args)
        if ok:
            taken += 1
            chunk = chunks.setdefault(bucket, [])
            chunk.append((text, args.skip_plies, taken - 1))
            if len(chunk) == 256:
                yield bucket, chunk
                chunks[bucket] = []
            if args.max_games and taken >= args.max_games:
                break
        if seen % 200000 == 0:
            print(f"  scanned {seen:,} games, kept {taken:,}", file=sys.stderr, flush=True)
    for bucket, chunk in chunks.items():
        if chunk:
            yield bucket, chunk


def work(item):
    bucket, games = item
    return (bucket,) + process_batch(games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--out", default="data")
    ap.add_argument("--elo-range", type=parse_elo_range, metavar="MIN-MAX", help="rating range, e.g. 1800-2200 (overrides --min-elo/--max-elo)")
    ap.add_argument("--min-elo", type=int, default=1800)
    ap.add_argument("--max-elo", type=int, default=4000)
    ap.add_argument("--elo-mode", choices=["both", "avg", "either"], default="both",
                    help="both: both players in range; avg: average rating in range; either: at least one player in range")
    ap.add_argument("--elo-bin", type=int, default=0, metavar="N",
                    help="sort games into N-wide rating bands, one output subdirectory per band (0 = single output)")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-plies", type=int, default=0, help="don't train on the first N plies")
    ap.add_argument("--no-forfeit", action="store_true", help="drop games ended by time forfeit/abandonment")
    ap.add_argument("--shard-size", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.elo_range:
        args.min_elo, args.max_elo = args.elo_range
    if args.min_elo > args.max_elo:
        ap.error("min elo is greater than max elo")
    print(f"elo filter: {args.min_elo}-{args.max_elo} (mode={args.elo_mode}, bin={args.elo_bin or 'off'})", file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)

    bufs = {}      # bucket -> {"s","m","v","g","n"}
    shards = {}    # bucket -> next shard index
    totals = {}    # bucket -> positions written/buffered
    bad = 0

    def out_dir(bucket):
        if bucket is None:
            return args.out
        return os.path.join(args.out, f"elo_{bucket}-{bucket + args.elo_bin - 1}")

    def flush(bucket):
        buf = bufs.get(bucket)
        if not buf or not buf["n"]:
            return
        d = out_dir(bucket)
        os.makedirs(d, exist_ok=True)
        i = shards.get(bucket, 0)
        np.savez(os.path.join(d, f"shard_{i:05d}.npz"),
                 squares=np.concatenate(buf["s"]), meta=np.concatenate(buf["m"]), move=np.concatenate(buf["v"]),
                 game=np.concatenate(buf["g"]))
        print(f"wrote {d}/shard_{i:05d}.npz ({buf['n']:,} positions)", flush=True)
        shards[bucket] = i + 1
        bufs[bucket] = {"s": [], "m": [], "v": [], "g": [], "n": 0}

    with mp.Pool(args.workers) as pool:
        for bucket, s, m, v, g, b in pool.imap(work, batches(args), chunksize=1):
            bad += b
            buf = bufs.setdefault(bucket, {"s": [], "m": [], "v": [], "g": [], "n": 0})
            buf["s"].append(s); buf["m"].append(m); buf["v"].append(v); buf["g"].append(g)
            buf["n"] += len(v)
            totals[bucket] = totals.get(bucket, 0) + len(v)
            if buf["n"] >= args.shard_size:
                flush(bucket)
    for bucket in list(bufs):
        flush(bucket)
    total = sum(totals.values())
    if args.elo_bin:
        for bucket in sorted(totals):
            print(f"  {out_dir(bucket)}: {totals[bucket]:,} positions")
    print(f"done: {total:,} positions, {bad} games skipped due to parse errors")


if __name__ == "__main__":
    main()
