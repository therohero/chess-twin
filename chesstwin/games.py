"""Interactive game sessions (state lives in memory; the browser keeps the game id)."""
import random
import uuid
from collections import OrderedDict

from schess.rules import Board, WHITE, BLACK, move_uci, sq_name, name_sq

from .san import to_san


class IllegalMove(ValueError):
    pass


class Game:
    def __init__(self, profile, human):
        self.id = uuid.uuid4().hex[:12]
        self.profile, self.human = profile, human
        self.board = Board()
        self.sans = []
        self.ucis = []
        self.resigned = False
        self.analysis = None      # filled by the Stockfish analysis (chesstwin.analysis.analyse_game)
        self.learned = False      # True once the twin was fine-tuned on this game

    @property
    def bot_to_move(self):
        return not self.resigned and self.board.turn != self.human and self.result() is None

    def result(self):
        if self.resigned:
            return ("0-1" if self.human == WHITE else "1-0"), "resignation"
        return self.board.outcome()

    def apply(self, m):
        legal = self.board.legal_moves()
        if m not in legal:
            raise IllegalMove(move_uci(m))
        self.sans.append(to_san(self.board, m, legal))
        self.ucis.append(move_uci(m))
        self.analysis = None
        self.board.make(m)

    def parse(self, uci):
        try:
            f, t = name_sq(uci[:2]), name_sq(uci[2:4])
            pr = " nbrq".index(uci[4]) + 1 if len(uci) > 4 else 0
        except (ValueError, IndexError):
            raise IllegalMove(uci) from None
        return (f, t, pr)

    def undo(self):
        """Take back to the human's previous turn."""
        n = 2 if self.board.turn == self.human else 1
        for _ in range(n):
            if self.board.stack:
                self.board.unmake()
                self.sans.pop()
                self.ucis.pop()
        self.analysis = None
        self.resigned = False

    def state(self):
        legal = self.board.legal_moves()
        res = self.result()
        dests, promos = {}, []
        if res is None and self.board.turn == self.human:
            for f, t, pr in legal:
                dests.setdefault(sq_name(f), set()).add(sq_name(t))
                if pr:
                    promos.append(sq_name(f) + sq_name(t))
        last = self.board.stack[-1][0] if self.board.stack else None
        return {
            "id": self.id, "profile": self.profile, "human": "white" if self.human == WHITE else "black",
            "fen": self.board.fen(), "turn": "white" if self.board.turn == WHITE else "black",
            "legal": {k: sorted(v) for k, v in dests.items()}, "promotions": sorted(set(promos)),
            "moves": self.sans, "last": [sq_name(last[0]), sq_name(last[1])] if last else None,
            "check": sq_name(self.board.king_square(self.board.turn)) if self.board.in_check() else None,
            "result": {"score": res[0], "reason": res[1]} if res else None,
            "bot_to_move": self.bot_to_move, "analysed": self.analysis is not None, "learned": self.learned,
        }


class GameStore:
    def __init__(self, cap=50):
        self.games = OrderedDict()
        self.cap = cap

    def create(self, profile, color):
        human = random.choice([WHITE, BLACK]) if color == "random" else (BLACK if color == "black" else WHITE)
        g = Game(profile, human)
        self.games[g.id] = g
        while len(self.games) > self.cap:
            self.games.popitem(last=False)
        return g

    def get(self, gid):
        return self.games[gid]
