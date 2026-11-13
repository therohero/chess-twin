"""Fine-tune (or train from scratch) a policy network on one player's moves."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from schess.model import ChessNet, load_model

MIN_POSITIONS = 300


def _split(data, val_frac=0.1):
    """Hold out the last games (by game index) for validation so no game is in both sets."""
    n_games = int(data["game"].max()) + 1
    n_val_games = max(1, int(n_games * val_frac)) if n_games >= 10 else 0
    cut = n_games - n_val_games
    is_val = data["game"] >= cut
    return ~is_val, is_val


def _tensors(data, mask, device):
    return (torch.from_numpy(data["squares"][mask]).to(device), torch.from_numpy(data["meta"][mask]).to(device),
            torch.from_numpy(data["move"][mask].astype(np.int64)).to(device))


def _top1(net, vs, vm, vy):
    if len(vy) == 0:
        return None
    net.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, len(vy), 2048):
            hit += (net(vs[i:i + 2048], vm[i:i + 2048]).argmax(1) == vy[i:i + 2048]).sum().item()
    net.train()
    return hit / len(vy)


def train_profile(data, out_path, base_path, mode="finetune", device="cpu", progress=None, cancel=None):
    """Train on `data` (dict from lichess.collect) and write the best checkpoint to `out_path`. Returns a stats dict."""
    n_total = len(data["move"])
    if n_total < MIN_POSITIONS:
        raise ValueError(f"Only {n_total} of your moves were found - at least {MIN_POSITIONS} are needed. "
                         "Download more games or relax the filters.")
    tr, va = _split(data)
    ts, tm, ty = _tensors(data, tr, device)
    vs, vm, vy = _tensors(data, va, device)
    n = len(ty)

    base = load_model(base_path, device) if base_path else None
    base_top1 = _top1(base, vs, vm, vy) if base is not None else None
    if mode == "fresh":
        net = ChessNet(**(base.config if base is not None else {})).to(device)
        lr, epochs = 2e-3, int(np.clip(round(400_000 / n), 6, 30))
    else:
        net = base
        lr, epochs = 4e-4, int(np.clip(round(60_000 / n), 2, 6))
    net.train()
    bs = 256 if n >= 2000 else 64
    steps = max(1, n // bs)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps * epochs, pct_start=0.15)

    best = base_top1 if (mode != "fresh" and base_top1 is not None) else -1.0
    best_state, best_epoch = None, 0
    if mode != "fresh":   # epoch 0 = the untouched base model; keep it if fine-tuning does not beat it
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        run = 0.0
        for st in range(steps):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            idx = perm[st * bs:(st + 1) * bs]
            loss = F.cross_entropy(net(ts[idx], tm[idx]), ty[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run = 0.9 * run + 0.1 * loss.item() if st else loss.item()
            if progress and st % 10 == 0:
                done = ep * steps + st
                progress(done / (steps * epochs), f"Epoch {ep + 1}/{epochs} - loss {run:.2f}")
        acc = _top1(net, vs, vm, vy)
        score = acc if acc is not None else float("inf")   # no validation games: just keep the latest weights
        if score >= best:
            best, best_epoch = score, ep + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if progress:
            progress((ep + 1) / epochs, f"Epoch {ep + 1}/{epochs} done" + (f" - predicts {acc:.0%} of your held-out moves" if acc is not None else ""))
    net.load_state_dict(best_state)
    torch.save({"config": net.config, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "val_top1": best if 0 <= best <= 1 else None}, out_path)
    return {"positions": n_total, "train_positions": n, "val_positions": len(vy), "epochs": epochs, "best_epoch": best_epoch,
            "val_top1": best if (0 <= best <= 1 and len(vy)) else None, "base_val_top1": base_top1,
            "seconds": round(time.time() - t0, 1)}
