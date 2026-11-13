"""Sanity tests for the rules engine (perft against known node counts) and SAN/encoding."""
from schess.rules import Board, perft, START_FEN
from schess.pgn import parse_san
from schess.encoding import encode_position, encode_move, NUM_MOVES

CASES = [
    (START_FEN, [20, 400, 8902]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
]

for fen, counts in CASES:
    b = Board(fen)
    for d, expected in enumerate(counts, 1):
        got = perft(b, d)
        assert got == expected, (fen, d, got, expected)
        assert b.fen() == Board(fen).fen()
    print("ok", fen)

b = Board()
for san in "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O".split():
    b.make(parse_san(b, san))
print(b.fen())
assert b.fen().startswith("r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1")
b = Board()
for san in "f3 e5 g4 Qh4#".split():
    b.make(parse_san(b, san))
assert b.outcome() == ("0-1", "checkmate")
# every legal move must have a distinct in-range index
for fen, _ in CASES:
    b = Board(fen)
    idx = [encode_move(m, b.turn) for m in b.legal_moves()]
    assert len(set(idx)) == len(idx) and all(0 <= i < NUM_MOVES for i in idx)
    encode_position(b)
print("all tests passed")
