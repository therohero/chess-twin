"""Streaming PGN reader (plain, .bz2, .zst) and SAN parser built on schess.rules."""
import bz2
import io
import re
import shutil
import subprocess

from .rules import Board, K, N, B, R, Q, P

_HEADER = re.compile(r'\[(\w+)\s+"(.*)"\]')
_CLEAN = re.compile(r"\{[^}]*\}|\([^)]*\)|\$\d+|;[^\n]*")
_MOVENUM = re.compile(r"^\d+\.+")
_SAN = re.compile(r"^([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=?([NBRQ]))?$")
_PIECE = {"N": N, "B": B, "R": R, "Q": Q, "K": K}
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def open_text(path):
    if path.endswith(".zst"):
        if shutil.which("zstd"):
            proc = subprocess.Popen(["zstd", "-dc", "--long=31", path], stdout=subprocess.PIPE)
            return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        import zstandard  # pip install zstandard
        raw = open(path, "rb")
        reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_games(f):
    """Yield (headers dict, movetext str) for each game in a text stream."""
    headers, text, in_moves = {}, [], False
    for line in f:
        if line.startswith("["):
            if in_moves:
                yield headers, " ".join(text)
                headers, text, in_moves = {}, [], False
            m = _HEADER.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        elif line.strip():
            in_moves = True
            text.append(line.strip())
    if in_moves:
        yield headers, " ".join(text)


def movetext_to_sans(text):
    sans = []
    for tok in _CLEAN.sub(" ", text).split():
        tok = _MOVENUM.sub("", tok)
        if not tok or tok in RESULTS:
            continue
        sans.append(tok)
    return sans


def parse_san(board, san):
    """Return the legal move (from, to, promo) matching `san`, or raise ValueError."""
    s = san.rstrip("+#!?")
    us = board.turn
    if s in ("O-O", "0-0", "O-O-O", "0-0-0"):
        base = 0 if us == 0 else 56
        want = (base + 4, base + (6 if len(s) == 3 else 2), 0)
        for m in board.pseudo_moves():
            if m == want and board.is_legal_pseudo(m):
                return m
        raise ValueError(f"illegal castling {san}")
    mt = _SAN.match(s)
    if not mt:
        raise ValueError(f"bad SAN {san}")
    piece, ff, rr, cap, dest, promo = mt.groups()
    ptype = _PIECE[piece] if piece else P
    to = (int(dest[1]) - 1) * 8 + "abcdefgh".index(dest[0])
    promo_t = _PIECE[promo] if promo else 0
    sq = board.sq
    cands = []
    for m in board.pseudo_moves():
        f, t, pr = m
        if t != to or pr != promo_t or (sq[f] - 1) % 6 + 1 != ptype:
            continue
        if ff and (f & 7) != "abcdefgh".index(ff):
            continue
        if rr and (f >> 3) != int(rr) - 1:
            continue
        if ptype == P and not cap and (f & 7) != (t & 7):
            continue
        if board.is_legal_pseudo(m):
            cands.append(m)
    if len(cands) != 1:
        raise ValueError(f"{'ambiguous' if cands else 'illegal'} SAN {san} in {board.fen()}")
    return cands[0]
