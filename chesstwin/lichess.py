"""Download a user's games from Lichess and turn them into (position -> move the user played) samples."""
import re

import numpy as np
import requests

from schess.encoding import encode_move, encode_position
from schess.pgn import iter_games, movetext_to_sans, parse_san
from schess.rules import Board, WHITE, BLACK

API = "https://lichess.org/api/games/user/{}"
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,28}[A-Za-z0-9]$")
TIME_CONTROLS = ("bullet", "blitz", "rapid", "classical")


class LichessError(Exception):
    """An error with a message that is safe and useful to show the user."""


class Cancelled(InterruptedError):
    pass


def valid_username(name):
    return bool(USERNAME_RE.match(name or ""))


def stream_games(username, max_games=500, time_controls=None, color=None, token=None, rated_only=False,
                 session=None, cancel=None):
    """Yield (headers, movetext) from the Lichess export API (newest games first)."""
    params = {"max": max_games, "moves": "true", "tags": "true", "clocks": "false", "evals": "false",
              "opening": "false", "ongoing": "false", "finished": "true", "sort": "dateDesc"}
    if time_controls:
        params["perfType"] = ",".join(time_controls)
    if color in ("white", "black"):
        params["color"] = color
    if rated_only:
        params["rated"] = "true"
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": "chess-twin"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = (session or requests).get(API.format(username), params=params, headers=headers, stream=True, timeout=(10, 60))
    except requests.RequestException as e:
        raise LichessError(f"Could not reach Lichess ({e.__class__.__name__}). Check your internet connection.") from e
    try:
        if resp.status_code == 404:
            raise LichessError(f'Lichess user "{username}" was not found.')
        if resp.status_code == 429:
            raise LichessError("Lichess is rate-limiting requests. Wait a minute and try again "
                               "(or set a LICHESS_TOKEN to raise the limit).")
        if resp.status_code != 200:
            raise LichessError(f"Lichess returned an unexpected error (HTTP {resp.status_code}).")
        resp.encoding = "utf-8"   # the PGN content-type has no charset, so requests would otherwise yield bytes
        lines = ((ln or "") for ln in resp.iter_lines(decode_unicode=True))
        for h, text in iter_games(_checked(lines, cancel)):
            yield h, text
    except requests.RequestException as e:
        raise LichessError(f"The connection to Lichess dropped ({e.__class__.__name__}).") from e
    finally:
        resp.close()


def _checked(lines, cancel):
    for ln in lines:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        yield ln


def game_samples(headers, text, username):
    """-> (squares[N,64] int8, meta[N] uint8, moves[N] int16) for the moves `username` played, or None if unusable."""
    if headers.get("Variant", "Standard") not in ("Standard", "From Position") or "FEN" in headers:
        return None
    u = username.lower()
    if headers.get("White", "").lower() == u:
        me = WHITE
    elif headers.get("Black", "").lower() == u:
        me = BLACK
    else:
        return None
    board = Board()
    sqs, metas, moves = [], [], []
    try:
        for san in movetext_to_sans(text):
            m = parse_san(board, san)
            if board.turn == me:
                sq, meta = encode_position(board)
                sqs.append(sq); metas.append(meta); moves.append(encode_move(m, board.turn))
            board.make(m)
    except (ValueError, KeyError, IndexError):
        pass   # keep whatever parsed before a bad token
    if not sqs:
        return None
    return np.stack(sqs), np.array(metas, np.uint8), np.array(moves, np.int16)


def collect(games, username, max_games, progress=None, cancel=None):
    """Consume a games iterator -> dict(squares, meta, move, game, games, skipped)."""
    S, M, V, G = [], [], [], []
    n_games = skipped = n_pos = 0
    for h, text in games:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        s = game_samples(h, text, username)
        if s is None:
            skipped += 1
        else:
            S.append(s[0]); M.append(s[1]); V.append(s[2]); G.append(np.full(len(s[2]), n_games, np.int32))
            n_games += 1
            n_pos += len(s[2])
        if progress:
            progress(n_games + skipped, max_games, n_games, n_pos)
    if not S:
        raise LichessError("No usable games were found for these filters. Try more games or other time controls.")
    return {"squares": np.concatenate(S), "meta": np.concatenate(M), "move": np.concatenate(V),
            "game": np.concatenate(G), "games": n_games, "skipped": skipped}
