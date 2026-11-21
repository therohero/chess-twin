import time

import pytest
from fastapi.testclient import TestClient

from chesstwin.server import create_app
from conftest import make_pgn_games

from chesstwin.config import ROOT
BASE = str(ROOT / "models" / "base.pt")


@pytest.fixture()
def client(tmp_path):
    games = make_pgn_games("alice", 40, plies=70)

    def source(username, max_games, tcs, color, rated_only, cancel):
        assert username == "alice"
        yield from games
    return TestClient(create_app(data_dir=tmp_path, base_model=BASE, device="cpu", games_source=source))


def wait_done(c, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get("/api/train").json()
        if s["status"] != "running":
            return s
        time.sleep(0.2)
    raise TimeoutError


def test_train_then_play(client):
    assert client.get("/").status_code == 200
    assert [p["id"] for p in client.get("/api/profiles").json()] == ["base"]
    assert client.post("/api/train", json={"username": "not valid!"}).status_code == 422
    r = client.post("/api/train", json={"username": "alice", "max_games": 50})
    assert r.status_code == 200
    s = wait_done(client)
    assert s["status"] == "done", s
    assert s["result"]["games"] == 40 and s["result"]["positions"] > 300
    profs = client.get("/api/profiles").json()
    assert [p["id"] for p in profs] == ["base", "alice"]

    g = client.post("/api/games", json={"profile": "alice", "color": "white"}).json()
    assert g["human"] == "white" and "e2" in g["legal"] and not g["bot_to_move"]
    assert client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e5"}).status_code == 422
    g = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"}).json()
    assert g["bot_to_move"] and g["moves"] == ["e4"]
    g = client.post(f"/api/games/{g['id']}/bot-move", json={"temperature": 0.5}).json()
    assert len(g["moves"]) == 2 and not g["bot_to_move"] and g["last"]
    g = client.post(f"/api/games/{g['id']}/undo").json()
    assert g["moves"] == [] and g["turn"] == "white"
    g = client.post(f"/api/games/{g['id']}/resign").json()
    assert g["result"] == {"score": "0-1", "reason": "resignation"}

    # bot opens when the human plays black; temperature 0 is deterministic
    g = client.post("/api/games", json={"profile": "base", "color": "black"}).json()
    assert g["bot_to_move"]
    a = client.post(f"/api/games/{g['id']}/bot-move", json={"temperature": 0}).json()["moves"]
    g2 = client.post("/api/games", json={"profile": "base", "color": "black"}).json()
    b = client.post(f"/api/games/{g2['id']}/bot-move", json={"temperature": 0}).json()["moves"]
    assert a == b and len(a) == 1
    # blunder check: accepted 0..3, rejected above
    g3 = client.post("/api/games", json={"profile": "base", "color": "black"}).json()
    assert len(client.post(f"/api/games/{g3['id']}/bot-move", json={"temperature": 0, "search": 2}).json()["moves"]) == 1
    assert client.post(f"/api/games/{g3['id']}/bot-move", json={"search": 9}).status_code == 422

    assert client.delete("/api/profiles/alice").status_code == 200
    assert client.delete("/api/profiles/base").status_code == 404
    assert client.post("/api/games", json={"profile": "alice"}).status_code == 404


def test_promotion_and_game_over(client):
    from chesstwin.games import Game
    from schess.rules import Board, WHITE
    g = Game("base", WHITE)
    g.board = Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    st = g.state()
    assert "a7a8" in st["promotions"]
    g.apply(g.parse("a7a8q"))
    assert g.sans == ["a8=Q"]


def test_lichess_errors_reach_the_user(tmp_path):
    from chesstwin.lichess import LichessError

    def source(*a):
        raise LichessError('Lichess user "alice" was not found.')
        yield
    c = TestClient(create_app(data_dir=tmp_path, base_model=BASE, device="cpu", games_source=source))
    c.post("/api/train", json={"username": "alice"})
    s = wait_done(c)
    assert s["status"] == "error" and "not found" in s["error"]
