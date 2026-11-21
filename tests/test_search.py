"""Blunder-check search (schess.search) and its use in the engine."""
import random

import torch

from chesstwin.config import ROOT
from chesstwin.engine import Engine
from schess.rules import Board, name_sq
from schess.search import MATE, material, root_scores

BASE = ROOT / "models" / "base.pt"


def mv(uci):
    return name_sq(uci[:2]), name_sq(uci[2:4]), 0


class _Store:
    def model_path(self, pid):
        return BASE


def test_material_is_from_the_movers_view():
    b = Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert material(b) == 900
    b = Board("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1")
    assert material(b) == -900


def test_finds_hanging_piece_and_mate():
    b = Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")           # Rxd5 wins the queen
    s = root_scores(b, b.legal_moves(), depth=2)
    assert max(s, key=s.get) == mv("d1d5")
    b = Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")            # Ra8 is mate
    s = root_scores(b, b.legal_moves(), depth=1)
    assert s[mv("a1a8")] >= MATE - 10


def test_sees_that_a_move_hangs_the_queen():
    b = Board("4k3/8/8/4p3/8/8/8/3QK3 w - - 0 1")
    s = root_scores(b, b.legal_moves(), depth=2)
    assert s[mv("d1d2")] == 800                               # scores are absolute material: Q vs P
    assert s[mv("d1d4")] == -100                              # exd4 wins the queen


def test_sees_a_knight_fork():
    # after Qc3?? Ne2+ forks king and queen
    b = Board("4k3/8/8/Q7/3n4/8/8/6K1 w - - 0 1")
    s = root_scores(b, b.legal_moves(), depth=3)
    assert max(s.values()) == 900 and s[mv("a5c3")] == -300   # Qxd4 wins the knight instead


def test_board_is_restored_even_when_the_node_budget_runs_out():
    rnd = random.Random(3)
    b = Board()
    for _ in range(20):
        b.make(rnd.choice(b.legal_moves()))
    fen, n = b.fen(), len(b.stack)
    s = root_scores(b, b.legal_moves(), depth=3, max_nodes=500)
    assert b.fen() == fen and len(b.stack) == n
    assert len(s) < len(b.legal_moves())


def test_engine_blunder_check_never_hangs_the_queen():
    torch.manual_seed(0)
    eng = Engine(_Store(), "cpu")
    b = Board("4k3/8/8/4p3/8/8/8/3QK3 w - - 0 1")
    legal = b.legal_moves()
    bad = {mv("d1d4"), mv("d1d7"), mv("d1d8")}               # squares the pawn or king can take on
    for _ in range(30):
        assert eng.choose("x", b, temperature=3.0, legal=legal, search=2) not in bad
