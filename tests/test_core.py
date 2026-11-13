import random
import threading

import pytest

from chesstwin import lichess
from chesstwin.san import to_san
from conftest import make_pgn_games
from schess.pgn import parse_san
from schess.rules import Board


def test_san_roundtrip_on_random_games():
    rnd = random.Random(1)
    for _ in range(30):
        b = Board()
        for _ in range(80):
            legal = b.legal_moves()
            if not legal or b.outcome(legal):
                break
            m = rnd.choice(legal)
            san = to_san(b, m, legal)
            assert parse_san(b, san) == m, (b.fen(), san)   # our SAN must be parseable back to the same move
            b.make(m)


def test_san_specials():
    b = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert to_san(b, (4, 6, 0)) == "O-O" and to_san(b, (4, 2, 0)) == "O-O-O"
    b = Board("4k3/P7/8/8/8/8/8/R3K2R w - - 0 1")
    assert to_san(b, (48, 56, 5)) == "a8=Q+"
    b = Board("6k1/8/8/8/8/8/8/R3R1K1 w - - 0 1")   # two rooks can reach e... disambiguation by file
    assert to_san(b, (0, 2, 0)) == "Rac1" or to_san(b, (0, 2, 0)) == "Rc1"


def test_game_samples_only_users_moves():
    (h, text), = make_pgn_games("alice", 1, plies=20)    # alice is white
    s = lichess.game_samples(h, text, "Alice")
    assert len(s[2]) == 10 and len(s[0]) == 10
    assert lichess.game_samples(h, text, "nobody") is None
    assert lichess.game_samples({**h, "Variant": "Chess960"}, text, "alice") is None


def test_collect_progress_and_empty():
    games = make_pgn_games("alice", 6)
    seen = []
    data = lichess.collect(iter(games), "alice", 6, lambda *a: seen.append(a))
    assert data["games"] == 6 and len(data["squares"]) == len(data["move"]) == len(data["game"]) and seen[-1][0] == 6
    with pytest.raises(lichess.LichessError):
        lichess.collect(iter(games), "bob", 6)
    ev = threading.Event(); ev.set()
    with pytest.raises(lichess.Cancelled):
        lichess.collect(iter(games), "alice", 6, cancel=ev)


def test_username_validation():
    assert lichess.valid_username("DrNykterstein") and lichess.valid_username("a_b-c9")
    assert not lichess.valid_username("") and not lichess.valid_username("a") and not lichess.valid_username("x y") \
        and not lichess.valid_username("../etc")


def test_stream_games_over_real_http(monkeypatch):
    """Exercise the real requests streaming path (content-type without charset -> bytes pitfall)."""
    import http.server

    games = make_pgn_games("alice", 4, plies=24)
    body = "".join("".join(f'[{k} "{v}"]\n' for k, v in h.items()) + "\n" + t + "\n\n" for h, t in games).encode()
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"], seen["accept"] = self.path, self.headers.get("Accept")
            code = 404 if "ghost" in self.path else 200
            self.send_response(code)
            self.send_header("Content-Type", "application/x-chess-pgn")
            self.end_headers()
            if code == 200:
                self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(lichess, "API", f"http://127.0.0.1:{srv.server_port}/api/games/user/{{}}")
    try:
        got = list(lichess.stream_games("alice", 10, ["blitz", "rapid"], "white"))
        assert len(got) == 4 and got[0][0]["White"] == "alice"
        assert "perfType=blitz%2Crapid" in seen["path"] and "color=white" in seen["path"] and seen["accept"] == "application/x-chess-pgn"
        with pytest.raises(lichess.LichessError, match="not found"):
            list(lichess.stream_games("ghost"))
    finally:
        srv.shutdown()
