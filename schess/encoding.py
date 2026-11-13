"""Position / move encoding shared by preprocessing, training and play.

Everything is encoded from the side-to-move's perspective: if Black is to move the board is
mirrored vertically and colours are swapped, so the network always sees "us" as White moving up.
"""
import numpy as np

NUM_MOVES = 4096 + 72  # from*64+to, plus 72 knight/bishop/rook underpromotions
_FLIP = np.arange(64) ^ 56


def encode_position(board):
    """-> (squares int8[64], meta uint8). meta = castling(4 bits: ownK,ownQ,oppK,oppQ) | (ep_file+1) << 4"""
    a = np.array(board.sq, dtype=np.int8)
    c = board.castling
    if board.turn == 1:
        a = a[_FLIP]
        a = np.where(a > 0, (a + 5) % 12 + 1, 0).astype(np.int8)
        c = (c >> 2) | ((c & 3) << 2)
    ep = (board.ep & 7) + 1 if board.ep >= 0 else 0
    return a, np.uint8(c | (ep << 4))


def encode_move(m, turn):
    f, t, pr = m
    if turn == 1:
        f, t = f ^ 56, t ^ 56
    if pr and pr != 5:  # underpromotion (queen promotion is the plain from-to move)
        return 4096 + (pr - 2) * 24 + (f & 7) * 3 + ((t & 7) - (f & 7) + 1)
    return f * 64 + t
