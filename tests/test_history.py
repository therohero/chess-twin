"""Optional move-history input (schess.encoding history helpers, ChessNet(history=K))."""
import random

import numpy as np
import torch

from chesstwin.config import ROOT
from schess.encoding import encode_move, history_moves, sequence_history
from schess.model import ChessNet, load_model
from schess.rules import Board, N



def test_shard_history_matches_live_board_history():
    rnd = random.Random(0)
    mv, gid, hs = [], [], []
    for g in range(20):
        b = Board()
        for _ in range(60):
            legal = b.legal_moves()
            if not legal:
                break
            m = rnd.choice(legal)
            hs.append(history_moves(b, 4)); mv.append(encode_move(m, b.turn)); gid.append(g)
            b.make(m)
    assert (sequence_history(np.array(mv), np.array(gid), 4) == np.stack(hs)).all()

def test_underpromotion_history_is_decoded():
    b = Board("8/1P2k3/8/8/8/8/6K1/8 w - - 0 1")
    m = (49, 57, N)                                         # b7-b8=N
    enc = encode_move(m, b.turn)
    b.make(m)
    live = history_moves(b, 1)
    shard = sequence_history(np.array([enc, 0]), np.array([0, 0]), 1)[1]
    assert live[0] == shard[0] == (49 ^ 56) * 64 + (57 ^ 56)     # seen from Black's side

def test_old_checkpoints_load_and_ignore_history():
    net = load_model(ROOT / "models" / "base.pt")
    assert net.history == 0
    sq = torch.zeros(1, 64, dtype=torch.int8)
    m = torch.zeros(1, dtype=torch.uint8)
    assert torch.equal(net(sq, m), net(sq, m, torch.zeros(1, 4, dtype=torch.long)))

def test_history_model_uses_the_moves():
    torch.manual_seed(0)
    net = ChessNet(8, 1, history=2).eval()
    assert net.config["history"] == 2
    b = Board()
    sq = torch.from_numpy(np.array(b.sq, np.int8))[None]
    m = torch.zeros(1, dtype=torch.uint8)
    none = net(sq, m)                                        # missing history = "no moves yet"
    assert torch.equal(none, net(sq, m, torch.full((1, 2), -1)))
    assert not torch.equal(none, net(sq, m, torch.tensor([[12 * 64 + 28, -1]])))


def test_twin_training_and_play_with_a_history_model(tmp_path):
    from chesstwin import lichess
    from chesstwin.engine import Engine
    from chesstwin.training import learn_from_game, train_profile
    from conftest import make_pgn_games

    base = tmp_path / "base.pt"
    torch.save({"config": ChessNet(8, 1, history=4).config, "state_dict": ChessNet(8, 1, history=4).state_dict()}, base)
    data = lichess.collect(make_pgn_games("alice", 40, plies=40), "alice", 40)
    assert data["hist"].shape == (len(data["move"]), 8)
    out = tmp_path / "twin.pt"
    stats = train_profile(data, out, base, device="cpu")
    assert stats["val_top1"] is not None and load_model(out).history == 4

    class Store:
        def model_path(self, pid):
            return out
    b = Board()
    for _ in range(6):
        b.make(Engine(Store(), "cpu").choose("t", b, 1.0))
    samples = {"squares": data["squares"][:5], "meta": data["meta"][:5], "move": data["move"][:5].astype(np.int64),
               "hist": data["hist"][:5], "weight": np.ones(5, np.float32)}
    assert learn_from_game(samples, out, tmp_path / "t2.pt")["positions"] == 5
