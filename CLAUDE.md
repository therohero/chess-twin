# Chess Twin – project notes for Claude

Play chess against an AI fine-tuned on your Lichess games. Python (FastAPI + PyTorch) backend, dependency-free vanilla-JS frontend, no build step. User-facing docs live in `README.md`; this file is the developer cheat sheet.

## Layout
- `schess/` – self-contained chess core: `rules.py` (move gen, FEN, outcome; squares a1=0..h8=63, move = `(from, to, promo)`), `pgn.py` (PGN reader, SAN parser), `encoding.py` (position/move encoding, always from the side-to-move's view), `model.py` (residual CNN policy net, 4168 moves), `search.py` (material alpha-beta + quiescence for the blunder check).
- `chesstwin/` – the web app: `server.py` (REST API), `games.py` (in-memory games), `engine.py` (move choice by temperature), `training.py` (fine-tune a twin; `learn_from_game` = per-game weighted update), `lichess.py`, `profiles.py`, `jobs.py`, `analysis.py` (Stockfish/UCI client, move classification, weights), `static/` (UI).
- Root scripts: `preprocess.py` + `train.py` (base model pipeline), `play.py` (old desktop UI), `analyze.py` (Stockfish analysis of a PGN/UCI game), `analyze_model.py` (how good is a model: top-k accuracy + Stockfish-rated self-play).
- `models/base.pt` – shipped base model (tracked in git). `data/` (user twins) and `data_base/` (training shards) are not for commits.

## Commands
```bash
python -m pytest                 # all tests (~20 s); needs no network and no real Stockfish
python test_rules.py             # perft checks of the rules engine
python -m chesstwin --no-browser # run the app on :8000
```

## Stockfish analysis feature
- Binary lookup: `CHESSTWIN_STOCKFISH` → `stockfish` on PATH → `/usr/games`. Tests use `tests/fake_stockfish.py` (1-ply material engine on the repo's own rules) wrapped in a temp executable – real Stockfish is never required.
- One search per position (N+1 for N moves). Move loss = drop in the mover's win% (Lichess model); labels best ≤0.5, excellent ≤2, good ≤5, inaccuracy ≤10, mistake ≤20, else blunder.
- API: `POST/GET /api/games/{id}/analysis` (background job, polled), `POST /api/games/{id}/learn`. Games keep `ucis`; any move/undo clears `game.analysis`.
- Learning: weights depend on who was more accurate (`analysis.weighting`): player worse → mistakes up / good moves down; player better → good moves up / slips down; |gap| < 3 → neutral. Small update (12 AdamW steps at 1e-4, BN frozen), once per game, twins only.

## Blunder check (search)
- The net has no value head, so the search is material-only: `root_scores` (negamax depth 2/3 + capture quiescence, check evasions searched, node budget). Scores are absolute material from the mover's view, not deltas.
- `Engine.choose(..., search=d)` keeps moves within `SEARCH_MARGIN` (150 cp) of the best, then samples by policy/temperature as usual. API: `bot-move` body `search` 0..3; UI "Blunder check" Off/Light(2)/Strong(3). `analyze_model.py --search`.

## Move history & twin accuracy
- `ChessNet(history=K)` takes the last K moves as from/to planes (`schess.encoding`: `history_moves(board)` live, `sequence_history(moves, games, K)` from shard rows). Every sample dict carries `hist` (HISTORY_MAX=8 columns); the net slices what it needs, history-free checkpoints ignore it. Always pass `hist` when calling a net (`net(sq, meta, hist)`).
- Measured (A/B, 18M positions, 1 epoch): no history 44.1% top-1 → K=4 45.0% → K=8 45.2%.
- Twin fine-tuning (`training.py`): ~6 epochs at 4e-4, epoch chosen by validation **log-loss** (not top-1; "Like you" samples from the probabilities), validation = every 10th game. Tried and rejected: recency weighting, label smoothing, frozen early layers, lower LR.

## Gotchas
- `ProfileStore.save()` **moves** the model file it is given. Never pass it `models/base.pt` (a test once deleted the real base model this way) – copy first.
- The generic opponent (`profile == "base"`) can be analysed but never learns.
- Frontend: CSS `.view` fades in, so instant headless screenshots can look blank (disable animations when screenshotting).
- Keep the backend the single source of truth for legal moves; the frontend only displays what `state()` returns.
- Don't commit personal data: `data/`, `data_base/`, `*.pgn*`, `checkpoints/` are git-ignored (add `data_base/` if it shows up as untracked).

## Conventions
- **Committing: never ask whether or when to commit.** When a coherent piece of work is finished and `python -m pytest` passes, commit it right away (on the current branch, `main` is fine), split into logical commits. Still never commit personal data (see Gotchas), and don't push unless asked.
- Code and comments in English; small modules, docstring at the top explaining the purpose.
- New behaviour needs a test in `tests/`; API tests use `TestClient(create_app(data_dir=tmp_path, base_model=BASE, device="cpu"))`.
- Update `README.md` (user docs) when adding user-visible features or env vars.
