import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chesstwin import analysis as an
from chesstwin.config import ROOT
from chesstwin.server import create_app

BASE = str(ROOT / "models" / "base.pt")
FAKE = Path(__file__).parent / "fake_stockfish.py"
# Fool's mate: 1.f3 e5 2.g4 Qh4#  (white blunders away the game)
FOOLS = ["f2f3", "e7e5", "g2g4", "d8h4"]


@pytest.fixture()
def stockfish(tmp_path, monkeypatch):
    exe = tmp_path / "stockfish"
    exe.write_text(f"#!{sys.executable}\nimport runpy; runpy.run_path({str(FAKE)!r}, run_name='__main__')\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr("chesstwin.config.STOCKFISH", str(exe))
    return str(exe)


def test_missing_stockfish(monkeypatch):
    monkeypatch.delenv("CHESSTWIN_STOCKFISH", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(an.StockfishError, match="not found"):
        an.UciEngine()


def test_analyse_fools_mate(stockfish):
    with an.UciEngine(stockfish) as eng:
        a = an.analyse_game(FOOLS, eng, depth=1)
    assert [m["san"] for m in a["moves"]] == ["f3", "e5", "g4", "Qh4#"]
    assert a["moves"][3]["label"] == "best"                  # mate in one found and played
    assert a["moves"][2]["label"] == "blunder"               # g4 allows mate
    assert a["moves"][2]["best_san"] != "g4"
    assert a["evals"][-1] < -9000 and len(a["evals"]) == 5   # black has mated white
    assert a["black"]["accuracy"] > a["white"]["accuracy"]
    assert a["white"]["counts"]["blunder"] >= 1


def test_labels_and_accuracy():
    assert an.classify(0.0, True) == "best" and an.classify(30, False) == "blunder"
    assert an.classify(3, False) == "good" and an.classify(7, False) == "inaccuracy" and an.classify(15, False) == "mistake"
    assert an.move_accuracy(0) > 99 and an.move_accuracy(50) < 15
    assert an.win_pct(0) == 50 and an.win_pct(an.MATE_CP) > 99.9


def _fake_analysis(white_losses, black_losses):
    mv = []
    for i in range(max(len(white_losses), len(black_losses)) * 2):
        losses = white_losses if i % 2 == 0 else black_losses
        if i // 2 >= len(losses):
            continue
        loss = losses[i // 2]
        mv.append({"ply": i + 1, "color": "white" if i % 2 == 0 else "black", "uci": ["e2e4", "e7e5", "g1f3", "b8c6"][i % 4] if i < 4 else "a2a3",
                   "loss": loss, "accuracy": an.move_accuracy(loss), "label": an.classify(loss, loss == 0)})
    return {"moves": mv, "white": an.summarise([m for m in mv if m["color"] == "white"]),
            "black": an.summarise([m for m in mv if m["color"] == "black"])}


def test_weights_player_worse_than_bot():
    a = _fake_analysis([0, 25], [0, 0])               # human (white) blundered, bot perfect
    w = an.weighting(a, "white")
    assert w["mode"] == "player_worse" and w["strength"] == 1.0
    wt = an.move_weights(a, "white")
    assert wt[1] > wt[0] * 3 and abs(wt.mean() - 1) < 1e-6    # bad move valued more, good move less


def test_weights_player_better_than_bot():
    a = _fake_analysis([0, 0], [0, 25])
    assert an.weighting(a, "white")["mode"] == "player_better"
    wt = an.move_weights(a, "white")
    assert np.allclose(wt, 1.0)                                # all human moves are good -> equal weights
    a = _fake_analysis([0, 8, 0], [30, 30, 30])
    wt = an.move_weights(a, "white")
    assert wt[0] > wt[1]                                       # good moves valued more than the slip


def test_weights_even_game_is_neutral():
    a = _fake_analysis([0, 4], [0, 4])
    assert an.weighting(a, "white")["mode"] == "even"
    assert np.allclose(an.move_weights(a, "white"), 1.0)


def test_training_samples_only_human_moves(stockfish):
    with an.UciEngine(stockfish) as eng:
        a = an.analyse_game(FOOLS, eng, depth=1)
    s = an.training_samples(a, "white")
    assert len(s["move"]) == 2 and s["squares"].shape == (2, 64) and abs(s["weight"].mean() - 1) < 1e-5


def wait_analysis(c, gid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/games/{gid}/analysis").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise TimeoutError


def play(c, gid, ucis_white):
    g = None
    for u in ucis_white:
        g = c.post(f"/api/games/{gid}/move", json={"uci": u}).json()
        assert "moves" in g, g
        g = c.post(f"/api/games/{gid}/bot-move", json={"temperature": 0}).json()
    return g


def test_api_analysis_and_learning(tmp_path, stockfish):
    c = TestClient(create_app(data_dir=tmp_path, base_model=BASE, device="cpu"))
    assert c.get("/api/status").json()["stockfish"] is True
    # make a "twin" profile by copying the base model
    from chesstwin.profiles import ProfileStore
    import shutil
    shutil.copy(BASE, tmp_path / "bob.pt")          # save() moves the file, so never hand it the real base model
    ProfileStore(tmp_path, BASE).save("bob", tmp_path / "bob.pt", {"games": 1, "positions": 1})
    g = c.post("/api/games", json={"profile": "bob", "color": "white"}).json()
    assert c.post(f"/api/games/{g['id']}/analysis").status_code == 422      # nothing played yet
    play(c, g["id"], ["e2e4", "g1f3", "f1c4"])
    j = c.post(f"/api/games/{g['id']}/analysis", json={"depth": 6}).json()
    assert j["status"] in ("running", "done")
    a = wait_analysis(c, g["id"])
    assert a["status"] == "done" and a["human"] == "white" and len(a["moves"]) == 6 and a["can_learn"]
    assert c.get(f"/api/games/{g['id']}").json()["analysed"] is True

    before = (tmp_path / "profiles" / "bob" / "model.pt").read_bytes()
    r = c.post(f"/api/games/{g['id']}/learn")
    assert r.status_code == 200, r.text
    assert r.json()["positions"] == 3 and r.json()["loss_after"] <= r.json()["loss_before"]
    assert (tmp_path / "profiles" / "bob" / "model.pt").read_bytes() != before
    assert [p for p in c.get("/api/profiles").json() if p["id"] == "bob"][0]["games_learned"] == 1
    assert c.post(f"/api/games/{g['id']}/learn").status_code == 409          # only once per game
    assert not list(tmp_path.glob("tmp*"))                                    # no stray temp dirs

    # undo invalidates the analysis
    c.post(f"/api/games/{g['id']}/undo")
    assert c.get(f"/api/games/{g['id']}").json()["analysed"] is False


def test_generic_opponent_cannot_learn(tmp_path, stockfish):
    c = TestClient(create_app(data_dir=tmp_path, base_model=BASE, device="cpu"))
    g = c.post("/api/games", json={"profile": "base", "color": "white"}).json()
    play(c, g["id"], ["e2e4", "g1f3"])
    c.post(f"/api/games/{g['id']}/analysis", json={"depth": 6})
    a = wait_analysis(c, g["id"])
    assert a["status"] == "done" and a["can_learn"] is False
    assert c.post(f"/api/games/{g['id']}/learn").status_code == 409


def test_analysis_without_stockfish(tmp_path, monkeypatch):
    monkeypatch.setattr("chesstwin.config.STOCKFISH", None)
    monkeypatch.delenv("CHESSTWIN_STOCKFISH", raising=False)
    monkeypatch.setenv("PATH", "")
    c = TestClient(create_app(data_dir=tmp_path, base_model=BASE, device="cpu"))
    assert c.get("/api/status").json()["stockfish"] is False
    g = c.post("/api/games", json={"profile": "base", "color": "white"}).json()
    play(c, g["id"], ["e2e4", "g1f3"])
    r = c.post(f"/api/games/{g['id']}/analysis")
    assert r.status_code == 503 and "Stockfish" in r.json()["detail"]
