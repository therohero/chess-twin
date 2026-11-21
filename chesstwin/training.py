"""Fine-tune (or train from scratch) a policy network on one player's moves."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from schess.model import ChessNet, load_model

MIN_POSITIONS = 300


def _split(data, every=10):
    """Hold out every `every`-th game for validation (whole games, so no game is in both sets).

    Interleaved rather than a block: games arrive newest first, so a block at either end would judge the twin
    only on the player's oldest or only on their newest games."""
    n_games = int(data["game"].max()) + 1
    is_val = (data["game"] % every == every // 2) if n_games >= 10 else np.zeros(len(data["game"]), bool)
    return ~is_val, is_val


def _tensors(data, mask, device):
    return (torch.from_numpy(data["squares"][mask]).to(device), torch.from_numpy(data["meta"][mask]).to(device),
            torch.from_numpy(data["hist"][mask]).to(device),
            torch.from_numpy(data["move"][mask].astype(np.int64)).to(device))


def _evaluate(net, vs, vm, vh, vy):
    """-> (top-1 accuracy, mean log-loss) on the validation moves, or (None, None) without validation data."""
    if len(vy) == 0:
        return None, None
    was_training = net.training
    net.eval()
    hit, nll = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(vy), 2048):
            lg, y = net(vs[i:i + 2048], vm[i:i + 2048], vh[i:i + 2048]), vy[i:i + 2048]
            hit += (lg.argmax(1) == y).sum().item()
            nll += F.cross_entropy(lg, y, reduction="sum").item()
    net.train(was_training)
    return hit / len(vy), nll / len(vy)


def train_profile(data, out_path, base_path, mode="finetune", device="cpu", progress=None, cancel=None):
    """Train on `data` (dict from lichess.collect) and write the best checkpoint to `out_path`. Returns a stats dict."""
    n_total = len(data["move"])
    if n_total < MIN_POSITIONS:
        raise ValueError(f"Only {n_total} of your moves were found - at least {MIN_POSITIONS} are needed. "
                         "Download more games or relax the filters.")
    tr, va = _split(data)
    ts, tm, th, ty = _tensors(data, tr, device)
    vs, vm, vh, vy = _tensors(data, va, device)
    n = len(ty)

    base = load_model(base_path, device) if base_path else None
    base_top1, base_nll = _evaluate(base, vs, vm, vh, vy) if base is not None else (None, None)
    if mode == "fresh":
        net = ChessNet(**(base.config if base is not None else {})).to(device)
        lr, epochs = 2e-3, int(np.clip(round(400_000 / n), 6, 30))
    else:
        # ~6 epochs measured best on a 2,900-game player (vs 2 before: +0.8 pts top-1, better log-loss)
        net = base
        lr, epochs = 4e-4, int(np.clip(round(480_000 / n), 4, 8))
    net.train()
    bs = 256 if n >= 2000 else 64
    steps = max(1, n // bs)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps * epochs, pct_start=0.15)

    # Keep the epoch with the lowest validation log-loss, not the best top-1: "Like you" samples from the
    # probabilities, and late epochs keep top-1 flat while becoming overconfident.
    best_nll = base_nll if (mode != "fresh" and base_nll is not None) else float("inf")
    best_top1 = base_top1 if mode != "fresh" else None
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
            loss = F.cross_entropy(net(ts[idx], tm[idx], th[idx]), ty[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run = 0.9 * run + 0.1 * loss.item() if st else loss.item()
            if progress and st % 10 == 0:
                done = ep * steps + st
                progress(done / (steps * epochs), f"Epoch {ep + 1}/{epochs} - loss {run:.2f}")
        acc, nll = _evaluate(net, vs, vm, vh, vy)
        score = nll if nll is not None else -float("inf")  # no validation games: just keep the latest weights
        if score <= best_nll:
            best_nll, best_top1, best_epoch = score, acc, ep + 1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if progress:
            progress((ep + 1) / epochs, f"Epoch {ep + 1}/{epochs} done" + (f" - predicts {acc:.0%} of your held-out moves" if acc is not None else ""))
    net.load_state_dict(best_state)
    torch.save({"config": net.config, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "val_top1": best_top1}, out_path)
    return {"positions": n_total, "train_positions": n, "val_positions": len(vy), "epochs": epochs, "best_epoch": best_epoch,
            "val_top1": best_top1, "val_nll": round(best_nll, 4) if len(vy) else None, "base_val_top1": base_top1,
            "seconds": round(time.time() - t0, 1)}


LEARN_LR = 1e-4       # small: one game should nudge the twin, not rewrite it
LEARN_STEPS = 12


def learn_from_game(samples, model_path, out_path, device="cpu"):
    """Nudge an existing twin towards the human's moves of one analysed game, weighted per move.

    samples: dict from chesstwin.analysis.training_samples (squares, meta, move, hist, weight).
    Writes the updated checkpoint to out_path and returns {"positions", "loss_before", "loss_after"}.
    """
    net = load_model(model_path, device)       # eval mode: batch-norm statistics stay frozen (the batch is tiny)
    s = torch.from_numpy(samples["squares"]).to(device)
    h = torch.from_numpy(samples["hist"]).to(device)
    m = torch.from_numpy(samples["meta"]).to(device)
    y = torch.from_numpy(samples["move"]).to(device)
    w = torch.from_numpy(samples["weight"]).to(device)

    def loss_fn():
        return (F.cross_entropy(net(s, m, h), y, reduction="none") * w).mean()
    with torch.no_grad():
        before = loss_fn().item()
    opt = torch.optim.AdamW(net.parameters(), lr=LEARN_LR, weight_decay=0.0)
    for _ in range(LEARN_STEPS):
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        opt.step()
    with torch.no_grad():
        after = loss_fn().item()
    torch.save({"config": net.config, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "val_top1": None}, out_path)
    return {"positions": len(y), "loss_before": round(before, 4), "loss_after": round(after, 4)}
