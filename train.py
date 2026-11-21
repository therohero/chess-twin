"""Supervised training: predict the move a human played from the position.

  python train.py --data data --epochs 3
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from schess.encoding import sequence_history
from schess.model import ChessNet

HISTORY = 0   # set from --history in main(); number of previous moves fed to the network


def load_data(path, max_positions, max_games):
    sq, meta, mv, hist = [], [], [], []
    n = 0
    for f in sorted(glob.glob(os.path.join(path, "shard_*.npz"))):
        d = np.load(f)
        s, me, m, g = d["squares"], d["meta"], d["move"], d["game"]
        if max_games:
            keep = g < max_games
            s, me, m, g = s[keep], me[keep], m[keep], g[keep]
        sq.append(s); meta.append(me); mv.append(m); hist.append(sequence_history(m, g, HISTORY))
        n += len(mv[-1])
        print(f"loaded {f} ({n:,} positions)", flush=True)
        if max_positions and n >= max_positions:
            break
        if max_games and not keep.all():
            break
    if not sq:
        raise SystemExit(f"no shard_*.npz found in {path} - run preprocess.py first")
    sq, meta, mv, hist = np.concatenate(sq), np.concatenate(meta), np.concatenate(mv), np.concatenate(hist)
    if max_positions:
        sq, meta, mv, hist = sq[:max_positions], meta[:max_positions], mv[:max_positions], hist[:max_positions]
    return torch.from_numpy(sq), torch.from_numpy(meta), torch.from_numpy(hist), torch.from_numpy(mv.astype(np.int64))


def shard_files(path):
    files = sorted(glob.glob(os.path.join(path, "shard_*.npz")))
    if not files:
        raise SystemExit(f"no shard_*.npz found in {path} - run preprocess.py first")
    return files


def load_chunks(chunks):
    """chunks: list of (file, start, stop). Returns tensors for the concatenated slices."""
    sq, meta, mv, hist = [], [], [], []
    for f, a, b in chunks:
        d = np.load(f)
        sq.append(d["squares"][a:b]); meta.append(d["meta"][a:b]); mv.append(d["move"][a:b])
        hist.append(sequence_history(d["move"][a:b], d["game"][a:b], HISTORY))
    return (torch.from_numpy(np.concatenate(sq)), torch.from_numpy(np.concatenate(meta)),
            torch.from_numpy(np.concatenate(hist)), torch.from_numpy(np.concatenate(mv).astype(np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--val-size", type=int, default=50000, help="held-out positions (taken from the end)")
    ap.add_argument("--max-games", type=int, default=0, help="train on only the first N games (quick test run)")
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--stream", type=int, default=0, metavar="K",
                    help="don't load everything into RAM: train on K shards at a time, shuffled across them "
                         "(for datasets larger than memory; ignores --max-games/--max-positions)")
    ap.add_argument("--history", type=int, default=0, metavar="K", help="feed the last K moves to the network (0 = off)")
    ap.add_argument("--shards", type=int, default=0, metavar="N", help="use only the first N shards (quick experiments)")
    args = ap.parse_args()
    global HISTORY
    HISTORY = args.history

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    if args.stream:
        files = shard_files(args.data)
        if args.shards:
            files = files[:args.shards]
        sizes = [len(np.load(f)["move"]) for f in files]
        if sizes[-1] <= args.val_size:
            raise SystemExit("last shard is smaller than --val-size; lower --val-size")
        # validation = tail of the last shard (mostly unseen games), as in the in-memory mode
        cut = sizes[-1] - args.val_size
        train_chunks = [(f, 0, k) for f, k in zip(files[:-1], sizes[:-1])] + [(files[-1], 0, cut)]
        vsq, vmeta, vhist, vmv = load_chunks([(files[-1], cut, sizes[-1])])
        n = sum(b - a for _, a, b in train_chunks)
        print(f"streaming {len(files)} shards, {n:,} training positions, {args.stream} shards in memory", flush=True)
    else:
        sq, meta, hist, mv = load_data(args.data, args.max_positions, args.max_games)
        n = len(mv) - args.val_size
        if n <= 0:
            raise SystemExit("not enough data for the validation split; lower --val-size")
        # positions within a game are consecutive in the files, so the tail split is mostly unseen games
        vsq, vmeta, vhist, vmv = sq[n:], meta[n:], hist[n:], mv[n:]
        sq, meta, hist, mv = sq[:n], meta[:n], hist[:n], mv[:n]

    net = ChessNet(args.channels, args.blocks, args.history).to(dev)
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state_dict"])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = n // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps_per_epoch * args.epochs, pct_start=0.05)
    start_epoch, best = 0, 0.0
    if ck is not None and "optimizer" in ck:   # full-state checkpoint (last.pt): continue exactly where it stopped
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"], ck.get("best", 0.0)
        print(f"resumed full state at epoch {start_epoch}")
    elif ck is not None:
        print("resumed weights only: optimizer/LR schedule start fresh")
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)   # fp16 needs loss scaling; also skips overflowing steps
    skipped = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def evaluate():
        net.eval()
        loss = top1 = top3 = 0.0
        with torch.no_grad():
            for i in range(0, len(vmv), 4096):
                s, m, y = vsq[i:i + 4096].to(dev), vmeta[i:i + 4096].to(dev), vmv[i:i + 4096].to(dev)
                h = vhist[i:i + 4096].to(dev)
                with torch.autocast(dev, enabled=use_amp):
                    lg = net(s, m, h)
                loss += F.cross_entropy(lg.float(), y, reduction="sum").item()
                t3 = lg.topk(3, 1).indices
                top1 += (t3[:, 0] == y).sum().item()
                top3 += (t3 == y[:, None]).any(1).sum().item()
        net.train()
        k = len(vmv)
        return loss / k, top1 / k, top3 / k

    def batches_in_memory():
        perm = torch.randperm(n)
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            yield sq[idx], meta[idx], hist[idx], mv[idx]

    def batches_streamed():
        order = train_chunks[:]
        random.shuffle(order)
        rest = None   # positions left over from the previous group, carried into the next one
        for g in range(0, len(order), args.stream):
            group = load_chunks(order[g:g + args.stream])
            if rest is not None:
                group = tuple(torch.cat([r, x]) for r, x in zip(rest, group))
            perm = torch.randperm(len(group[-1]))
            full = len(perm) // args.batch_size * args.batch_size
            for i in range(0, full, args.batch_size):
                idx = perm[i:i + args.batch_size]
                yield tuple(x[idx] for x in group)
            idx = perm[full:]
            rest = tuple(x[idx] for x in group)
            del group

    for epoch in range(start_epoch, args.epochs):
        batches = batches_streamed() if args.stream else batches_in_memory()
        t0, run_loss, run_acc = time.time(), 0.0, 0.0
        for step, (s, m, h, y) in zip(range(steps_per_epoch), batches):
            s, m, h, y = s.to(dev, non_blocking=True), m.to(dev), h.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=use_amp):
                lg = net(s, m, h)
            loss = F.cross_entropy(lg.float(), y)
            if not torch.isfinite(loss):          # bad batch: skip it instead of poisoning the weights
                skipped += 1
                if skipped > 100:
                    raise SystemExit("too many non-finite losses; aborting (last.pt from the previous epoch is still good)")
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() if step % 20 == 0 else 0
            run_acc += (lg.argmax(1) == y).float().mean().item() if step % 20 == 0 else 0
            if (step + 1) % 1000 == 0:
                k = 1000 / 20
                print(f"epoch {epoch+1} step {step+1}/{steps_per_epoch} loss {run_loss/k:.3f} "
                      f"acc {run_acc/k:.3f} ({(step+1)*args.batch_size/(time.time()-t0):.0f} pos/s)", flush=True)
                run_loss = run_acc = 0.0
        vl, v1, v3 = evaluate()
        print(f"== epoch {epoch+1}: val loss {vl:.3f} top1 {v1:.3f} top3 {v3:.3f}", flush=True)
        if not all(torch.isfinite(p).all() for p in net.state_dict().values() if p.is_floating_point()):
            raise SystemExit("weights contain NaN/inf; not saving a checkpoint")
        if v1 >= best:
            best = v1
            torch.save({"config": net.config, "state_dict": net.state_dict(), "val_top1": v1}, args.out)
            print("saved", args.out)
        torch.save({"config": net.config, "state_dict": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch + 1, "best": best},
                   os.path.join(os.path.dirname(args.out) or ".", "last.pt"))


if __name__ == "__main__":
    main()
