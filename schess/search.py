"""Small tactical search used as a blunder check on top of the policy network.

The network has no value head, so the search scores positions by material only: a negamax alpha-beta of
`depth` plies followed by a capture-only quiescence search (check evasions are searched fully, so mates are
seen). It is not meant to find the best move; `root_scores` tells the engine how much material each
candidate move keeps against the opponent's best replies, so moves that hang a piece can be filtered out
while the network still picks among the safe ones.

Scores are in centipawns from the side to move's point of view; mate is +-MATE.
"""
from .rules import B, K, N, P, Q, R

MATE = 100_000
VALUE = {P: 100, N: 300, B: 320, R: 500, Q: 900, K: 0}
_PIECE_VALUE = [0] + [VALUE[t] for t in (P, N, B, R, Q, K)] * 2      # indexed by piece code 0..12
MAX_QDEPTH = 8
DELTA = 200


class NodeLimit(Exception):
    pass


def material(board):
    """Material balance in centipawns from the side to move's view."""
    s = 0
    for p in board.sq:
        if p:
            s += _PIECE_VALUE[p] if p <= 6 else -_PIECE_VALUE[p]
    return s if board.turn == 0 else -s


def _captures(board):
    """[(material gain, -attacker value, move)] for pseudo-legal captures and promotions, best first."""
    sq, ep = board.sq, board.ep
    caps = []
    for m in board.pseudo_moves():
        f, t, pr = m
        victim = sq[t]
        if victim or pr or (t == ep and (sq[f] - 1) % 6 + 1 == P):
            gain = (_PIECE_VALUE[victim] if victim else VALUE[P] if not pr else 0) + (VALUE[pr] - VALUE[P] if pr else 0)
            caps.append((gain, -_PIECE_VALUE[sq[f]], m))
    caps.sort(key=lambda x: -x[0])
    return caps                                          # pseudo-legal; legality is checked when played


def _make_legal(board, m):
    """Play a pseudo-legal move; if it leaves the own king in check, take it back and return False."""
    us = board.turn
    board.make(m)
    if board.is_attacked(board.king_square(us), 1 - us):
        board.unmake()
        return False
    return True


def _ordered(board, moves):
    sq = board.sq
    return sorted(moves, key=lambda m: -(_PIECE_VALUE[sq[m[1]]] * 10 - _PIECE_VALUE[sq[m[0]]] + (900 if m[2] else 0)))


class Searcher:
    def __init__(self, max_nodes=300_000):
        self.max_nodes, self.nodes = max_nodes, 0

    def _tick(self):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise NodeLimit

    def qsearch(self, board, alpha, beta, ply, qdepth=0):
        self._tick()
        if board.in_check():
            moves = board.legal_moves()
            if not moves:
                return -MATE + ply
            if qdepth >= MAX_QDEPTH:
                return material(board)
            moves = _ordered(board, moves)
        else:
            stand = material(board)
            if stand >= beta or qdepth >= MAX_QDEPTH:
                return stand
            alpha = max(alpha, stand)
            # delta pruning: skip captures that cannot lift the score to alpha even if the piece is free
            moves = [m for gain, _, m in _captures(board) if stand + gain + DELTA > alpha]
        for m in moves:
            if not _make_legal(board, m):
                continue
            v = -self.qsearch(board, -beta, -alpha, ply + 1, qdepth + 1)
            board.unmake()
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply)
        self._tick()
        if board.halfmove >= 100 or board.keys.count(board.keys[-1]) >= 2:
            return 0                                     # repetition inside the tree counts as a draw
        best, any_legal = -MATE, False
        for m in _ordered(board, board.pseudo_moves()):
            if not _make_legal(board, m):
                continue
            any_legal = True
            v = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.unmake()
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        if not any_legal:
            return -MATE + ply if board.in_check() else 0
        return best


def root_scores(board, moves, depth=2, margin=None, max_nodes=150_000):
    """-> {move: score} after playing each move and searching `depth - 1` further plies plus quiescence.

    With `margin`, only moves within `margin` of the best score need exact values: later moves are searched
    with a narrowed window, so a score below `best - margin` is only an upper bound (much faster).
    Moves are searched in the given order (put the network's favourites first). If the node budget runs out,
    the moves not searched yet are left out of the result."""
    s, out, n0, best = Searcher(max_nodes), {}, len(board.stack), -MATE
    try:
        for m in moves:
            alpha = -MATE if margin is None or best == -MATE else best - margin - 1
            board.make(m)
            v = out[m] = -s.negamax(board, depth - 1, -MATE, -alpha, 1)
            board.unmake()
            best = max(best, v)
    except NodeLimit:
        while len(board.stack) > n0:                     # the search was cut off somewhere deep in the tree
            board.unmake()
    return out
