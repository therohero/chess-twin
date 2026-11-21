import random

from .encoding import encode_move, encode_position, history_moves


class RandomBot:
    name = "random"

    def choose(self, board):
        return random.choice(board.legal_moves())


class Bot:
    """Wraps a trained ChessNet. temperature=0 -> always best move, >0 -> sample from the policy."""

    def __init__(self, path, device=None, temperature=0.0):
        import torch
        self.torch = torch
        from .model import load_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_model(path, self.device)
        self.temperature = temperature
        self.name = path

    def choose(self, board):
        torch = self.torch
        legal = board.legal_moves()
        sq, meta = encode_position(board)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(sq)[None].to(self.device),
                              torch.tensor([int(meta)], dtype=torch.uint8, device=self.device),
                              torch.from_numpy(history_moves(board))[None].to(self.device))[0]
        idx = torch.tensor([encode_move(m, board.turn) for m in legal], device=self.device)
        scores = logits[idx].float()
        if self.temperature <= 0:
            return legal[int(scores.argmax())]
        probs = torch.softmax(scores / self.temperature, 0)
        return legal[int(torch.multinomial(probs, 1))]
