"""Standard algebraic notation for moves (the repo only had a SAN *parser*)."""
from schess.rules import P, K, sq_name

_LETTER = {2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}


def to_san(board, m, legal=None):
    """SAN of legal move `m` in the current position of `board` (board is left unchanged)."""
    legal = board.legal_moves() if legal is None else legal
    f, t, pr = m
    piece = board.sq[f]
    pt = (piece - 1) % 6 + 1
    if pt == K and abs(t - f) == 2:
        s = "O-O" if t > f else "O-O-O"
    else:
        capture = board.sq[t] != 0 or (pt == P and (f & 7) != (t & 7))
        if pt == P:
            s = (sq_name(f)[0] + "x" if capture else "") + sq_name(t) + (("=" + _LETTER[pr]) if pr else "")
        else:
            others = [o for o in legal if o != m and o[1] == t and board.sq[o[0]] == piece]
            dis = ""
            if others:
                if all((o[0] & 7) != (f & 7) for o in others):
                    dis = sq_name(f)[0]
                elif all((o[0] >> 3) != (f >> 3) for o in others):
                    dis = sq_name(f)[1]
                else:
                    dis = sq_name(f)
            s = _LETTER[pt] + dis + ("x" if capture else "") + sq_name(t)
    board.make(m)
    try:
        if board.in_check():
            s += "#" if not board.legal_moves() else "+"
    finally:
        board.unmake()
    return s
