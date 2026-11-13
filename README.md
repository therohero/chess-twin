# Chess Twin

Play chess against an AI version of **yourself**.

Enter your Lichess username → Chess Twin downloads your games, fine-tunes a small neural network on *your* moves, and lets you play it in the browser. Everything runs on your own machine (GPU if you have one, CPU otherwise). Your games and your twin never leave your computer.

## Quick start

```bash
git clone <this repo> && cd chess-twin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m chesstwin            # opens http://localhost:8000
```

Then type your Lichess username and press **Create my twin**. With 500 games this takes about 1–3 minutes on a CPU.

Options: `python -m chesstwin --port 9000 --data-dir ~/my-twins --no-browser`.

### Docker (optional)

```bash
docker compose up --build                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU
```

Open http://localhost:8000. Twins are stored in the `chesstwin-data` volume, so they survive restarts and rebuilds.

* The default image is CPU-only and small. The GPU variant uses a CUDA 12.4 PyTorch image (works on older cards such as the GTX 1060) and needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.
* **Automatic CPU fallback:** at startup the app runs a tiny CUDA test; if there is no GPU, or CUDA is present but unusable, it quietly uses the CPU. The top-right badge in the UI shows which one is active. (The GPU compose file itself requires the NVIDIA runtime; use the plain compose file on machines without one.)

## Using it

1. **Setup** – username, plus optional filters: time controls, which colour's games to learn from, number of games, rated only, and *fine-tune* (recommended) vs *train from scratch*.
2. **Progress** – three clear steps (download → learn your style → save) with a live progress bar. You can cancel at any time.
3. **Play** – click or drag pieces. Controls: undo, flip, resign, new game (pick a colour), switch opponent.
4. **Playing style slider** – changes how the AI picks among its candidate moves, live, even mid-game:

| Slider | What it does |
|---|---|
| **Strongest** (0) | always the model's single favourite move |
| **Like you** (middle, default) | samples moves with the same odds *you* would – your habits and your occasional mistakes |
| **Looser** (right) | flatter odds: more variety and more slips |

Under the hood this is the sampling temperature of the policy network (`slider / 50`).

A built-in **Generic opponent** (the base model) is always available, so you can try the board before creating a twin. Re-entering a username retrains that twin with fresh games.

## How it works

* **Model** – a residual CNN (64 channels × 4 blocks) that maps a position to a probability for each of 4,168 moves (`schess/model.py`). It is a *pure imitation* policy: no search, no evaluation. It plays like a person, not like Stockfish.
* **Base model** – `models/base.pt`, trained on a large database of strong Lichess games with `preprocess.py` + `train.py` (reaches ~44% top-1 move prediction on held-out positions).
* **Personalising** – the app streams your games from the [Lichess export API](https://lichess.org/api#tag/Games/operation/apiGamesUser), keeps only positions where *you* were to move, holds out your last ~10% of games for validation, and fine-tunes with a small learning rate. The checkpoint with the best held-out accuracy is kept, and that includes the untouched base model, so fine-tuning can never make the twin worse at predicting you. The result card shows *your twin vs. the generic model* on held-out moves.
* **Rules** – a self-contained, perft-tested rules engine (`schess/rules.py`); the backend is the single source of truth for legal moves.
* **Stack** – FastAPI + PyTorch backend, dependency-free vanilla JS frontend (own board renderer, no CDN, works offline). No build step.

```
chesstwin/      web app: server.py (API), lichess.py, training.py, engine.py, games.py, static/ (UI)
schess/         chess rules, SAN/PGN parsing, move encoding, the network
models/base.pt  base policy used for fine-tuning and as the generic opponent
preprocess.py / train.py / play.py   original pipeline: build a dataset, train a base model, desktop UI
data/           created at runtime: data/profiles/<username>/{model.pt,meta.json}
tests/          pytest suite
```

## Configuration

| Env var | Meaning |
|---|---|
| `CHESSTWIN_DATA` | where twins are stored (default `./data`) |
| `CHESSTWIN_DEVICE` | `cpu` or `cuda` to force a device (default: auto) |
| `CHESSTWIN_BASE_MODEL` | path to a different base model |
| `LICHESS_TOKEN` | optional Lichess API token; raises download rate limits |

## Tips & limits

* More games make a better twin; a few hundred is enough to notice your openings. Training from scratch needs thousands of games and is mostly for experimenting.
* Bullet games contain lots of time-pressure moves; they are off by default.
* The twin only knows what it has seen. It imitates move *choice* and will not calculate deep tactics, so it can blunder in ways that are very "you".
* Games are kept in server memory; restarting the app abandons games in progress (twins are saved).
* Anyone who can reach the port can use the app (there is no login), so keep the default localhost binding unless you trust your network.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest          # API, training, SAN, Lichess streaming (against a local HTTP server)
python test_rules.py      # perft checks of the rules engine
```

Retraining the base model: `python preprocess.py games.pgn.zst --out data_base --min-elo 1800` then `python train.py --data data_base --channels 64 --blocks 4`, and copy the checkpoint to `models/base.pt`.

## License

* **Code and base model:** [CC0 1.0 Universal](LICENSE) – public-domain dedication, do whatever you like.
* **Piece images** (`chesstwin/static/pieces/`, `assets/`): by Uray M. János from [greenchess.net](https://greenchess.net/info.php?item=downloads), licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) – *not* CC0. Keep the attribution and share modified versions under the same license. See `ATTRIBUTION.md` next to the images.

## Notes

* Not affiliated with Lichess. Please respect the [Lichess API terms](https://lichess.org/terms-of-service); the app downloads one user's games per request, one request at a time.
