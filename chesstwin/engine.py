"""Move selection from a profile's policy network.

Two knobs: the sampling temperature (strength/humanness) and an optional blunder check: a small material
search (`schess.search`) that removes candidate moves losing more than `margin` centipawns compared with the
best move it finds, so the twin keeps its own style but stops hanging pieces. search=0 is the pure twin.
"""
import threading
from collections import OrderedDict

import torch

from schess.encoding import encode_move, encode_position, history_moves
from schess.model import load_model
from schess.search import root_scores

SEARCH_MARGIN = 150      # centipawns: allow pawn sacrifices / gambits, reject losing a piece
SEARCH_NODES = 100_000   # budget per move (roughly 1-2 s in pure Python)


class Engine:
    def __init__(self, store, device):
        self.store, self.device = store, device
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def _net(self, pid):
        path = self.store.model_path(pid)
        if path is None:
            raise KeyError(pid)
        key = (str(path), path.stat().st_mtime_ns)   # reload automatically after a retrain
        if key not in self._cache:
            self._cache[key] = load_model(path, self.device)
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def invalidate(self):
        with self._lock:
            self._cache.clear()

    def choose(self, pid, board, temperature=1.0, legal=None, search=0):
        """temperature 0 = always the model's top choice; 1 = sample like the player would; >1 = looser.
        search = plies of blunder check (0 = off, 2 = catches hanging pieces and mate-in-one, 3 = also forks)."""
        legal = board.legal_moves() if legal is None else legal
        sq, meta = encode_position(board)
        with self._lock, torch.no_grad():
            net = self._net(pid)
            logits = net(torch.from_numpy(sq)[None].to(self.device),
                         torch.tensor([int(meta)], dtype=torch.uint8, device=self.device),
                         torch.from_numpy(history_moves(board))[None].to(self.device))[0]
            idx = torch.tensor([encode_move(m, board.turn) for m in legal], device=self.device)
            scores = logits[idx].float().cpu()
        if search > 0 and len(legal) > 1:
            legal, scores = self._safe(board, legal, scores, search)
        if temperature < 0.05:
            return legal[int(scores.argmax())]
        return legal[int(torch.multinomial(torch.softmax(scores / temperature, 0), 1))]

    @staticmethod
    def _safe(board, legal, scores, depth):
        """Keep only the moves whose search score is within SEARCH_MARGIN of the best one."""
        order = scores.argsort(descending=True).tolist()          # the net's favourites first
        b = board.copy()
        b.keys = list(board.keys)                                 # keep the history for repetition checks
        values = root_scores(b, [legal[i] for i in order], depth, SEARCH_MARGIN, SEARCH_NODES)
        if not values:
            return legal, scores
        best = max(values.values())
        keep = [i for i in order if values.get(legal[i], -float("inf")) >= best - SEARCH_MARGIN]
        return [legal[i] for i in keep], scores[keep]
