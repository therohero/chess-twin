import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_MOVES

IN_CHANNELS = 17


def make_planes(squares, meta):
    """squares: int8/long [B,64] (perspective-normalised), meta: uint8/long [B] -> float [B,17,8,8]"""
    b = squares.shape[0]
    oh = F.one_hot(squares.long(), 13)[..., 1:].float()            # [B,64,12]
    planes = oh.permute(0, 2, 1).reshape(b, 12, 8, 8)               # square = rank*8+file
    meta = meta.long()
    castle = torch.stack([((meta >> i) & 1) for i in range(4)], 1).float()
    castle = castle[:, :, None, None].expand(b, 4, 8, 8)
    ep = torch.zeros(b, 1, 8, 8, device=squares.device)
    file = (meta >> 4) - 1
    has = file >= 0
    if has.any():
        idx = has.nonzero(as_tuple=True)[0]
        ep[idx, 0, 5, file[idx]] = 1.0
    return torch.cat([planes, castle, ep], 1)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class ChessNet(nn.Module):
    """Policy network: position -> logits over NUM_MOVES moves (from the mover's perspective)."""

    def __init__(self, channels=128, blocks=8):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks}
        self.stem = nn.Sequential(nn.Conv2d(IN_CHANNELS, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(32 * 64, NUM_MOVES))

    def forward(self, squares, meta):
        return self.head(self.blocks(self.stem(make_planes(squares, meta))))


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ChessNet(**ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net
