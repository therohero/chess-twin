"""Play against the trained bot.

  python play.py --model checkpoints/model.pt [--color white|black|random] [--temperature 0.3]
  python play.py --random          # no model yet: play against a random mover

Click a piece, then a highlighted square.  Keys: N new game, F flip board, U undo, Esc quit.
"""
import argparse
import os
import random
import sys

import pygame

from schess.rules import Board, WHITE, BLACK, Q, R, B, N, move_uci

SQ = 80
BOARD_PX = SQ * 8
BAR = 44
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
SEL, LAST, DOT = (246, 246, 105), (205, 210, 106), (20, 20, 20)
NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
PROMO_CHOICES = [Q, R, B, N]


def load_images():
    here = os.path.dirname(os.path.abspath(__file__))
    imgs = {}
    for code in range(1, 13):
        color = "white" if code <= 6 else "black"
        path = os.path.join(here, "assets", f"{color}-{NAMES[(code - 1) % 6 + 1]}.png")
        imgs[code] = pygame.transform.smoothscale(pygame.image.load(path).convert_alpha(), (SQ, SQ))
    return imgs


class App:
    def __init__(self, bot, human_color):
        self.bot = bot
        self.screen = pygame.display.set_mode((BOARD_PX, BOARD_PX + BAR))
        pygame.display.set_caption("SL Chess")
        self.font = pygame.font.SysFont(None, 28)
        self.small = pygame.font.SysFont(None, 18)
        self.imgs = load_images()
        self.new_game(human_color)

    def new_game(self, human_color):
        self.board = Board()
        self.human = human_color
        self.flipped = human_color == BLACK
        self.selected = None
        self.promo = None          # (from, to) awaiting piece choice
        self.last = None
        self.result = None
        self.bot_ready = False
        self.legal = self.board.legal_moves()

    # ---- coordinates
    def to_px(self, s):
        r, f = divmod(s, 8)
        col, row = (7 - f, r) if self.flipped else (f, 7 - r)
        return col * SQ, row * SQ

    def from_px(self, x, y):
        col, row = x // SQ, y // SQ
        if self.flipped:
            return row * 8 + (7 - col)
        return (7 - row) * 8 + col

    # ---- game flow
    def play(self, m):
        self.board.make(m)
        self.last = m
        self.selected = None
        self.legal = self.board.legal_moves()
        out = self.board.outcome(self.legal)
        if out:
            self.result = out
        self.bot_ready = False

    def undo(self):
        n = 2 if self.board.turn == self.human else 1
        for _ in range(n):
            if self.board.stack:
                self.board.unmake()
        self.last = self.board.stack[-1][0] if self.board.stack else None
        self.legal = self.board.legal_moves()
        self.result = self.board.outcome(self.legal)
        self.selected = self.promo = None
        self.bot_ready = False

    def click(self, x, y):
        if self.result or self.board.turn != self.human or y >= BOARD_PX:
            return
        if self.promo:
            f, t = self.promo
            cx = (BOARD_PX - 4 * SQ) // 2
            if BOARD_PX // 2 - SQ // 2 <= y < BOARD_PX // 2 + SQ // 2 and cx <= x < cx + 4 * SQ:
                self.play((f, t, PROMO_CHOICES[(x - cx) // SQ]))
            self.promo = None
            return
        s = self.from_px(x, y)
        if self.selected is not None:
            tm = [m for m in self.legal if m[0] == self.selected and m[1] == s]
            if tm:
                if tm[0][2]:
                    self.promo = (self.selected, s)
                else:
                    self.play(tm[0])
                return
        p = self.board.sq[s]
        if p and (p - 1) // 6 == self.human and any(m[0] == s for m in self.legal):
            self.selected = s
        else:
            self.selected = None

    def bot_move(self):
        self.play(self.bot.choose(self.board))

    # ---- drawing
    def draw(self):
        scr = self.screen
        for s in range(64):
            x, y = self.to_px(s)
            r, f = divmod(s, 8)
            color = LIGHT if (r + f) % 2 else DARK
            if (self.last and s in self.last[:2]) or s == self.selected:
                color = SEL if s == self.selected else LAST
            pygame.draw.rect(scr, color, (x, y, SQ, SQ))
        if self.board.in_check():
            x, y = self.to_px(self.board.king_square(self.board.turn))
            pygame.draw.rect(scr, (220, 70, 70), (x, y, SQ, SQ))
        for s in range(64):
            p = self.board.sq[s]
            if p:
                scr.blit(self.imgs[p], self.to_px(s))
        if self.selected is not None:
            for m in self.legal:
                if m[0] == self.selected:
                    x, y = self.to_px(m[1])
                    pygame.draw.circle(scr, DOT, (x + SQ // 2, y + SQ // 2), 9)
        if self.promo:
            cx = (BOARD_PX - 4 * SQ) // 2
            cy = BOARD_PX // 2 - SQ // 2
            pygame.draw.rect(scr, (60, 60, 60), (cx - 6, cy - 6, 4 * SQ + 12, SQ + 12), border_radius=8)
            for i, pt in enumerate(PROMO_CHOICES):
                pygame.draw.rect(scr, LIGHT, (cx + i * SQ, cy, SQ, SQ))
                scr.blit(self.imgs[pt + 6 * self.human], (cx + i * SQ, cy))
        # status bar
        pygame.draw.rect(scr, (38, 36, 33), (0, BOARD_PX, BOARD_PX, BAR))
        if self.result:
            res, why = self.result
            msg = f"{res}  {why}"
            if res != "1/2-1/2":
                msg += "  - you win!" if (res == "1-0") == (self.human == WHITE) else "  - bot wins"
        elif self.board.turn == self.human:
            msg = "Your move" + ("  (check!)" if self.board.in_check() else "")
        else:
            msg = "Bot is thinking..."
        scr.blit(self.font.render(msg, True, (235, 235, 235)), (12, BOARD_PX + 6))
        scr.blit(self.small.render("N new   F flip   U undo", True, (150, 150, 150)), (12, BOARD_PX + 28))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(*e.pos)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_n:
                        self.new_game(1 - self.human)
                    elif e.key == pygame.K_f:
                        self.flipped = not self.flipped
                    elif e.key == pygame.K_u:
                        self.undo()
            self.draw()
            if not self.result and self.board.turn != self.human:
                if self.bot_ready:          # one frame has been drawn with "thinking" status
                    self.bot_move()
                else:
                    self.bot_ready = True
            clock.tick(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/model.pt")
    ap.add_argument("--random", action="store_true", help="play a random-move bot (no model needed)")
    ap.add_argument("--color", default="white", choices=["white", "black", "random"])
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = strongest move, higher = more varied")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.random:
        from schess.bot import RandomBot
        bot = RandomBot()
    else:
        if not os.path.exists(args.model):
            sys.exit(f"{args.model} not found. Train first (preprocess.py -> train.py) or use --random.")
        from schess.bot import Bot
        bot = Bot(args.model, args.device, args.temperature)
    color = random.choice([WHITE, BLACK]) if args.color == "random" else (WHITE if args.color == "white" else BLACK)
    pygame.init()
    App(bot, color).run()
    pygame.quit()


if __name__ == "__main__":
    main()
