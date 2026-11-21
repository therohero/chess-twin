"""Analyse a chess game with Stockfish: move quality, accuracy per side, best moves and an evaluation curve.

    python analyze.py game.pgn                      # first game in the file
    python analyze.py games.pgn --game 3 --depth 16
    python analyze.py --moves "e2e4 e7e5 g1f3"      # moves in UCI notation
    python analyze.py game.pgn --json > report.json

Stockfish is found via --stockfish, $CHESSTWIN_STOCKFISH or PATH. The same analysis powers the "Analyse game"
button in Chess Twin (chesstwin/analysis.py).
"""
import argparse
import json
import sys

from chesstwin.analysis import LABELS, StockfishError, UciEngine, analyse_game
from schess.pgn import iter_games, movetext_to_sans, open_text, parse_san
from schess.rules import Board, move_uci

MARK = {"best": "  ", "excellent": "  ", "good": "  ", "inaccuracy": "?!", "mistake": "? ", "blunder": "??"}


def load_ucis(args):
    if args.moves:
        return args.moves.replace(",", " ").split(), {}
    with open_text(args.pgn) as f:
        for n, (headers, text) in enumerate(iter_games(f), 1):
            if n == args.game:
                break
        else:
            sys.exit(f"{args.pgn} has fewer than {args.game} games.")
    if "FEN" in headers or headers.get("Variant", "Standard") not in ("Standard", "From Position"):
        sys.exit("Only standard games from the initial position are supported.")
    board, ucis = Board(), []
    for san in movetext_to_sans(text):
        m = parse_san(board, san)
        ucis.append(move_uci(m))
        board.make(m)
    return ucis, headers


def fmt_eval(cp):
    return f"#{'+' if cp > 0 else '-'}" if abs(cp) >= 9000 else f"{cp / 100:+.2f}"


def report(a, headers):
    w, b = headers.get("White", "White"), headers.get("Black", "Black")
    print(f"{w} (White) vs {b} (Black)" + (f"   {headers['Result']}" if "Result" in headers else ""))
    print(f"Depth {a['depth']}" if not a["movetime"] else f"{a['movetime']} ms per position", "\n")
    print(f"{'':16}{'accuracy':>9}{'avg loss':>10}   " + "  ".join(f"{k[:5]:>5}" for k in LABELS))
    for color, name in (("white", w), ("black", b)):
        s = a[color]
        acc = f"{s['accuracy']:.1f}" if s["accuracy"] is not None else "-"
        loss = f"{s['avg_loss']:.2f}" if s["avg_loss"] is not None else "-"
        print(f"{name[:15]:16}{acc:>9}{loss:>10}   " + "  ".join(f"{s['counts'][k]:>5}" for k in LABELS))
    print("\nMoves (eval is from White's view; '->' shows the engine's preferred move when you deviated)")
    for m, ev in zip(a["moves"], a["evals"][1:]):
        num = f"{(m['ply'] + 1) // 2}{'.' if m['color'] == 'white' else '...'}"
        hint = f"  -> {m['best_san']}" if m["label"] in ("inaccuracy", "mistake", "blunder") and m["best_san"] else ""
        print(f"{num:>6} {m['san']:<8}{MARK[m['label']]} {m['label']:<11}{fmt_eval(ev):>7}  loss {m['loss']:5.1f}{hint}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="?", help="PGN file (.pgn, .pgn.zst, .pgn.bz2)")
    ap.add_argument("--game", type=int, default=1, help="which game of the PGN to analyse (default 1)")
    ap.add_argument("--moves", help="UCI moves from the start position instead of a PGN, e.g. 'e2e4 e7e5'")
    ap.add_argument("--depth", type=int, default=14, help="search depth per position (default 14)")
    ap.add_argument("--movetime", type=int, help="milliseconds per position instead of a fixed depth")
    ap.add_argument("--stockfish", help="path to the Stockfish binary")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print the full analysis as JSON")
    args = ap.parse_args()
    if not args.pgn and not args.moves:
        ap.error("give a PGN file or --moves")
    ucis, headers = load_ucis(args)
    try:
        with UciEngine(args.stockfish, threads=args.threads) as eng:
            def progress(p, msg):
                if not args.json:
                    print(f"\r{msg}   ", end="", file=sys.stderr, flush=True)
            a = analyse_game(ucis, eng, args.depth, args.movetime, progress)
            print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
    except StockfishError as e:
        sys.exit(str(e))
    print(json.dumps({"headers": headers, **a}, indent=1)) if args.json else report(a, headers)


if __name__ == "__main__":
    main()
