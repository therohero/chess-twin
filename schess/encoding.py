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


# ---------------------------------------------------------------- move history (optional network input)
# History = the last K moves before the position, as int16 from*64+to in the CURRENT mover's perspective
# (newest first, -1 = no move: start of the game or of the recorded data). Promotion piece is dropped.

def history_moves(board, k):
    """-> int16[k] from board.stack (the moves played on this Board object)."""
    out = np.full(k, -1, np.int16)
    flip = 56 if board.turn == 1 else 0
    for j in range(min(k, len(board.stack))):
        f, t, _ = board.stack[-1 - j][0]
        out[j] = (f ^ flip) * 64 + (t ^ flip)
    return out


def _decode_from_to(idx):
    """Encoded move index -> (from, to) squares in that move's own (mover's) perspective. Vectorised."""
    idx = idx.astype(np.int32)
    f, t = idx // 64, idx % 64
    up = idx >= 4096                                  # underpromotion: always rank 7 -> rank 8 for the mover
    if up.any():
        r = (idx[up] - 4096) % 24
        ff, dt = r // 3, r % 3 - 1
        f[up], t[up] = 48 + ff, 56 + ff + dt
    return f, t


def sequence_history(moves, games, k):
    """History for consecutive plies of games (rows of a shard / one player's game list).

    moves: encoded moves (each in its own mover's perspective), games: game id per row. Row i's history is
    rows i-1, i-2, ... of the same game, so every ply of each game must be present and in order."""
    n = len(moves)
    out = np.full((n, k), -1, np.int16)
    f, t = _decode_from_to(np.asarray(moves))
    games = np.asarray(games)
    for j in range(1, k + 1):
        if n <= j:
            break
        same = games[j:] == games[:-j]
        flip = 56 if j % 2 else 0                     # odd distance = the opponent's move: other perspective
        val = ((f[:-j] ^ flip) * 64 + (t[:-j] ^ flip)).astype(np.int16)
        out[j:, j - 1] = np.where(same, val, -1)
    return out
