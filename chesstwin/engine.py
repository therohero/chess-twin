"""Move selection from a profile's policy network, with a strength/humanness knob (sampling temperature)."""
import threading
from collections import OrderedDict

import torch

from schess.encoding import encode_move, encode_position
from schess.model import load_model


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

    def choose(self, pid, board, temperature=1.0, legal=None):
        """temperature 0 = always the model's top choice; 1 = sample like the player would; >1 = looser."""
        legal = board.legal_moves() if legal is None else legal
        sq, meta = encode_position(board)
        with self._lock, torch.no_grad():
            net = self._net(pid)
            logits = net(torch.from_numpy(sq)[None].to(self.device),
                         torch.tensor([int(meta)], dtype=torch.uint8, device=self.device))[0]
            idx = torch.tensor([encode_move(m, board.turn) for m in legal], device=self.device)
            scores = logits[idx].float()
            if temperature < 0.05:
                return legal[int(scores.argmax())]
            return legal[int(torch.multinomial(torch.softmax(scores / temperature, 0), 1))]
