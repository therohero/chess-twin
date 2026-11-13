"""Complete chess rules: move generation, make/unmake, check/mate/draw detection, FEN.

Squares are 0..63 with a1=0, b1=1, ..., h8=63 (square = rank*8 + file).
Pieces: 0 empty, 1..6 = white P N B R Q K, 7..12 = black P N B R Q K.
A move is a tuple (from_sq, to_sq, promo) where promo is 0 or a piece type 2..5 (N,B,R,Q).
"""

WHITE, BLACK = 0, 1
P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_FEN_CHARS = {c: i + 1 for i, c in enumerate("PNBRQKpnbrqk")}
_CHAR_OF = {v: k for k, v in _FEN_CHARS.items()}


def sq_name(s):
    return "abcdefgh"[s & 7] + str((s >> 3) + 1)


def name_sq(n):
    return (int(n[1]) - 1) * 8 + "abcdefgh".index(n[0])


def move_uci(m):
    return sq_name(m[0]) + sq_name(m[1]) + ("" if not m[2] else " nbrq"[m[2] - 1])


# ---------------------------------------------------------------- tables
def _build_tables():
    knight = [[] for _ in range(64)]
    king = [[] for _ in range(64)]
    rays = [[] for _ in range(64)]
    pawn_from = [[[] for _ in range(64)] for _ in range(2)]  # squares from which a pawn attacks s
    pawn_to = [[[] for _ in range(64)] for _ in range(2)]    # squares a pawn on s attacks
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for s in range(64):
        r, f = divmod(s, 8)
        for dr, df in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]:
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                knight[s].append((r + dr) * 8 + f + df)
        for dr, df in dirs:
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                king[s].append((r + dr) * 8 + f + df)
            ray, rr, ff = [], r + dr, f + df
            while 0 <= rr < 8 and 0 <= ff < 8:
                ray.append(rr * 8 + ff)
                rr += dr
                ff += df
            rays[s].append(ray)
        for color, d in ((WHITE, 1), (BLACK, -1)):
            for df in (-1, 1):
                if 0 <= r + d < 8 and 0 <= f + df < 8:
                    t = (r + d) * 8 + f + df
                    pawn_to[color][s].append(t)
                    pawn_from[color][t].append(s)
    return knight, king, rays, pawn_from, pawn_to


KNIGHT, KING, RAYS, PAWN_FROM, PAWN_TO = _build_tables()

CASTLE_MASK = [15] * 64
CASTLE_MASK[0] = 15 & ~2
CASTLE_MASK[7] = 15 & ~1
CASTLE_MASK[4] = 15 & ~3
CASTLE_MASK[56] = 15 & ~8
CASTLE_MASK[63] = 15 & ~4
CASTLE_MASK[60] = 15 & ~12


class Board:
    def __init__(self, fen=START_FEN):
        self.set_fen(fen)

    # ------------------------------------------------------------ FEN
    def set_fen(self, fen):
        parts = fen.split()
        self.sq = [0] * 64
        for i, row in enumerate(parts[0].split("/")):
            r, f = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    self.sq[r * 8 + f] = _FEN_CHARS[ch]
                    f += 1
        self.turn = WHITE if parts[1] == "w" else BLACK
        c = parts[2] if len(parts) > 2 else "-"
        self.castling = (1 if "K" in c else 0) | (2 if "Q" in c else 0) | (4 if "k" in c else 0) | (8 if "q" in c else 0)
        self.ep = name_sq(parts[3]) if len(parts) > 3 and parts[3] != "-" else -1
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1
        self.stack = []
        self.keys = [self._key()]

    def fen(self):
        rows = []
        for r in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.sq[r * 8 + f]
                if p == 0:
                    empty += 1
                else:
                    row += (str(empty) if empty else "") + _CHAR_OF[p]
                    empty = 0
            rows.append(row + (str(empty) if empty else ""))
        c = "".join(ch for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if self.castling & bit) or "-"
        ep = sq_name(self.ep) if self.ep >= 0 else "-"
        return f"{'/'.join(rows)} {'w' if self.turn == WHITE else 'b'} {c} {ep} {self.halfmove} {self.fullmove}"

    def _key(self):
        return (bytes(self.sq), self.turn, self.castling, self.ep)

    def copy(self):
        return Board(self.fen())

    # ------------------------------------------------------------ attacks
    def is_attacked(self, s, by):
        sq = self.sq
        off = 6 * by
        for q in PAWN_FROM[by][s]:
            if sq[q] == P + off:
                return True
        for q in KNIGHT[s]:
            if sq[q] == N + off:
                return True
        for q in KING[s]:
            if sq[q] == K + off:
                return True
        rook, bishop, queen = R + off, B + off, Q + off
        for i, ray in enumerate(RAYS[s]):
            for q in ray:
                p = sq[q]
                if p:
                    if p == queen or p == (rook if i < 4 else bishop):
                        return True
                    break
        return False

    def king_square(self, color):
        return self.sq.index(K + 6 * color)

    def in_check(self):
        return self.is_attacked(self.king_square(self.turn), 1 - self.turn)

    # ------------------------------------------------------------ move generation
    def pseudo_moves(self):
        sq, us, ep = self.sq, self.turn, self.ep
        moves = []
        add = moves.append
        for s in range(64):
            p = sq[s]
            if not p or (p - 1) // 6 != us:
                continue
            t = (p - 1) % 6 + 1
            if t == P:
                r = s >> 3
                d = 8 if us == WHITE else -8
                promo_rank = 6 if us == WHITE else 1
                start_rank = 1 if us == WHITE else 6
                f1 = s + d
                if sq[f1] == 0:
                    if r == promo_rank:
                        for pr in (Q, R, B, N):
                            add((s, f1, pr))
                    else:
                        add((s, f1, 0))
                        if r == start_rank and sq[f1 + d] == 0:
                            add((s, f1 + d, 0))
                for q in PAWN_TO[us][s]:
                    tp = sq[q]
                    if tp and (tp - 1) // 6 != us:
                        if r == promo_rank:
                            for pr in (Q, R, B, N):
                                add((s, q, pr))
                        else:
                            add((s, q, 0))
                    elif q == ep:
                        add((s, q, 0))
            elif t == N or t == K:
                for q in (KNIGHT[s] if t == N else KING[s]):
                    tp = sq[q]
                    if tp == 0 or (tp - 1) // 6 != us:
                        add((s, q, 0))
                if t == K:
                    base = 0 if us == WHITE else 56
                    if s == base + 4 and self.castling & (3 << (2 * us)):
                        them = 1 - us
                        if self.castling & (1 << (2 * us)) and sq[base + 5] == 0 and sq[base + 6] == 0 \
                                and sq[base + 7] == R + 6 * us \
                                and not self.is_attacked(base + 4, them) and not self.is_attacked(base + 5, them) \
                                and not self.is_attacked(base + 6, them):
                            add((s, base + 6, 0))
                        if self.castling & (2 << (2 * us)) and sq[base + 1] == 0 and sq[base + 2] == 0 \
                                and sq[base + 3] == 0 and sq[base] == R + 6 * us \
                                and not self.is_attacked(base + 4, them) and not self.is_attacked(base + 3, them) \
                                and not self.is_attacked(base + 2, them):
                            add((s, base + 2, 0))
            else:
                lo, hi = (4, 8) if t == B else (0, 4) if t == R else (0, 8)
                rays = RAYS[s]
                for i in range(lo, hi):
                    for q in rays[i]:
                        tp = sq[q]
                        if tp == 0:
                            add((s, q, 0))
                        else:
                            if (tp - 1) // 6 != us:
                                add((s, q, 0))
                            break
        return moves

    def is_legal_pseudo(self, m):
        """Is a pseudo-legal move legal (does it leave own king safe)?"""
        us = self.turn
        self.make(m)
        ok = not self.is_attacked(self.king_square(us), 1 - us)
        self.unmake()
        return ok

    def legal_moves(self):
        return [m for m in self.pseudo_moves() if self.is_legal_pseudo(m)]

    # ------------------------------------------------------------ make / unmake
    def make(self, m):
        f, t, pr = m
        sq, us = self.sq, self.turn
        p, cap = sq[f], sq[t]
        pt = (p - 1) % 6 + 1
        ep_capture = pt == P and cap == 0 and (f & 7) != (t & 7)
        self.stack.append((m, p, cap, self.castling, self.ep, self.halfmove, ep_capture))
        sq[f] = 0
        sq[t] = (pr + 6 * us) if pr else p
        if ep_capture:
            sq[t - 8 if us == WHITE else t + 8] = 0
        elif pt == K and t - f == 2:
            sq[f + 1], sq[f + 3] = sq[f + 3], 0
        elif pt == K and f - t == 2:
            sq[f - 1], sq[f - 4] = sq[f - 4], 0
        self.castling &= CASTLE_MASK[f] & CASTLE_MASK[t]
        self.ep = (f + t) // 2 if pt == P and abs(t - f) == 16 else -1
        self.halfmove = 0 if (pt == P or cap or ep_capture) else self.halfmove + 1
        if us == BLACK:
            self.fullmove += 1
        self.turn = 1 - us
        self.keys.append(self._key())

    def unmake(self):
        m, p, cap, castling, ep, halfmove, ep_capture = self.stack.pop()
        self.keys.pop()
        f, t, pr = m
        sq = self.sq
        self.turn = us = 1 - self.turn
        if us == BLACK:
            self.fullmove -= 1
        self.castling, self.ep, self.halfmove = castling, ep, halfmove
        sq[f], sq[t] = p, cap
        pt = (p - 1) % 6 + 1
        if ep_capture:
            sq[t - 8 if us == WHITE else t + 8] = P + 6 * (1 - us)
        elif pt == K and t - f == 2:
            sq[f + 3], sq[f + 1] = sq[f + 1], 0
        elif pt == K and f - t == 2:
            sq[f - 4], sq[f - 1] = sq[f - 1], 0

    # ------------------------------------------------------------ game state
    def insufficient_material(self):
        minors = []
        for s, p in enumerate(self.sq):
            if not p:
                continue
            t = (p - 1) % 6 + 1
            if t in (P, R, Q):
                return False
            if t in (N, B):
                minors.append((t, ((s >> 3) + (s & 7)) & 1))
        if len(minors) <= 1:
            return True
        return all(t == B for t, _ in minors) and len({c for _, c in minors}) == 1

    def outcome(self, legal=None):
        """None if game continues, else (result, reason); result is '1-0', '0-1' or '1/2-1/2'."""
        if legal is None:
            legal = self.legal_moves()
        if not legal:
            if self.in_check():
                return ("0-1" if self.turn == WHITE else "1-0"), "checkmate"
            return "1/2-1/2", "stalemate"
        if self.insufficient_material():
            return "1/2-1/2", "insufficient material"
        if self.halfmove >= 100:
            return "1/2-1/2", "fifty-move rule"
        if self.keys.count(self.keys[-1]) >= 3:
            return "1/2-1/2", "threefold repetition"
        return None


def perft(board, depth):
    if depth == 0:
        return 1
    moves = board.legal_moves()
    if depth == 1:
        return len(moves)
    n = 0
    for m in moves:
        board.make(m)
        n += perft(board, depth - 1)
        board.unmake()
    return n
