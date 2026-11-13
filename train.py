"""Supervised training: predict the move a human played from the position.

  python train.py --data data --epochs 3
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from schess.model import ChessNet


def load_data(path, max_positions, max_games):
    sq, meta, mv = [], [], []
    n = 0
    for f in sorted(glob.glob(os.path.join(path, "shard_*.npz"))):
        d = np.load(f)
        s, me, m = d["squares"], d["meta"], d["move"]
        if max_games:
            keep = d["game"] < max_games
            s, me, m = s[keep], me[keep], m[keep]
        sq.append(s); meta.append(me); mv.append(m)
        n += len(mv[-1])
        print(f"loaded {f} ({n:,} positions)", flush=True)
        if max_positions and n >= max_positions:
            break
        if max_games and not keep.all():
            break
    if not sq:
        raise SystemExit(f"no shard_*.npz found in {path} - run preprocess.py first")
    sq, meta, mv = np.concatenate(sq), np.concatenate(meta), np.concatenate(mv)
    if max_positions:
        sq, meta, mv = sq[:max_positions], meta[:max_positions], mv[:max_positions]
    return torch.from_numpy(sq), torch.from_numpy(meta), torch.from_numpy(mv.astype(np.int64))


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
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    sq, meta, mv = load_data(args.data, args.max_positions, args.max_games)
    n = len(mv) - args.val_size
    if n <= 0:
        raise SystemExit("not enough data for the validation split; lower --val-size")
    # positions within a game are consecutive in the files, so the tail split is mostly unseen games
    vsq, vmeta, vmv = sq[n:], meta[n:], mv[n:]
    sq, meta, mv = sq[:n], meta[:n], mv[:n]

    net = ChessNet(args.channels, args.blocks).to(dev)
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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def evaluate():
        net.eval()
        loss = top1 = top3 = 0.0
        with torch.no_grad():
            for i in range(0, len(vmv), 4096):
                s, m, y = vsq[i:i + 4096].to(dev), vmeta[i:i + 4096].to(dev), vmv[i:i + 4096].to(dev)
                with torch.autocast(dev, enabled=use_amp):
                    lg = net(s, m)
                loss += F.cross_entropy(lg.float(), y, reduction="sum").item()
                t3 = lg.topk(3, 1).indices
                top1 += (t3[:, 0] == y).sum().item()
                top3 += (t3 == y[:, None]).any(1).sum().item()
        net.train()
        k = len(vmv)
        return loss / k, top1 / k, top3 / k

    for epoch in range(start_epoch, args.epochs):
        perm = torch.randperm(n)
        t0, run_loss, run_acc = time.time(), 0.0, 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            s, m, y = sq[idx].to(dev, non_blocking=True), meta[idx].to(dev), mv[idx].to(dev)
            with torch.autocast(dev, enabled=use_amp):
                lg = net(s, m)
            loss = F.cross_entropy(lg.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
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
        if v1 >= best:
            best = v1
            torch.save({"config": net.config, "state_dict": net.state_dict(), "val_top1": v1}, args.out)
            print("saved", args.out)
        torch.save({"config": net.config, "state_dict": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch + 1, "best": best},
                   os.path.join(os.path.dirname(args.out) or ".", "last.pt"))


if __name__ == "__main__":
    main()
