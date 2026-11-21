"""FastAPI app: REST API + static single-page UI."""
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from schess.rules import WHITE

from . import __version__, analysis as sf, config, lichess
from .device import describe, pick_device
from .engine import Engine
from .games import GameStore, IllegalMove
from .jobs import Busy, JobManager
from .profiles import BASE_ID, ProfileStore
from .training import learn_from_game, train_profile

log = logging.getLogger("chesstwin")
STATIC = Path(__file__).parent / "static"


class TrainRequest(BaseModel):
    username: str
    max_games: int = Field(500, ge=50, le=5000)
    time_controls: list[str] = ["blitz", "rapid", "classical"]
    color: str = "any"                 # any | white | black  (which of your games to learn from)
    rated_only: bool = False
    mode: str = "finetune"             # finetune | fresh


class NewGame(BaseModel):
    profile: str = BASE_ID
    color: str = "white"               # white | black | random


class MoveReq(BaseModel):
    uci: str


class AnalysisReq(BaseModel):
    depth: int = Field(sf.DEFAULT_DEPTH, ge=6, le=24)


class BotReq(BaseModel):
    temperature: float = Field(1.0, ge=0.0, le=3.0)
    search: int = Field(0, ge=0, le=3)   # blunder-check plies (0 = pure twin)


def create_app(data_dir=None, base_model=None, device=None, games_source=None):
    """games_source(username, max_games, time_controls, color, rated_only, cancel) -> iterator of (headers, movetext).
    Injectable so tests never touch the network."""
    data_dir = Path(data_dir or config.DATA_DIR)
    app = FastAPI(title="Chess Twin", version=__version__)
    device = pick_device(device or config.DEVICE)
    store = ProfileStore(data_dir, base_model or config.BASE_MODEL)
    engine = Engine(store, device)
    games, jobs, analysis_jobs = GameStore(), JobManager(), JobManager()
    log.info("Chess Twin using %s", describe(device))

    def default_source(username, max_games, time_controls, color, rated_only, cancel):
        return lichess.stream_games(username, max_games, time_controls, color, config.LICHESS_TOKEN, rated_only, cancel=cancel)
    source = games_source or default_source

    # ------------------------------------------------------------------ status / profiles
    @app.get("/api/status")
    def status():
        return {"version": __version__, "device": device, "device_name": describe(device),
                "base_model": store.base_model.exists(), "stockfish": bool(sf.find_stockfish(config.STOCKFISH)), "busy": bool(jobs.job and jobs.job.status == "running")}

    @app.get("/api/profiles")
    def profiles():
        return store.list()

    @app.delete("/api/profiles/{pid}")
    def delete_profile(pid: str):
        if not store.delete(pid):
            raise HTTPException(404, "No such profile")
        engine.invalidate()
        return {"ok": True}

    # ------------------------------------------------------------------ training job
    @app.post("/api/train")
    def train(req: TrainRequest):
        user = req.username.strip()
        if not lichess.valid_username(user):
            raise HTTPException(422, "That doesn't look like a Lichess username (2-30 letters, digits, _ or -).")
        if req.mode not in ("finetune", "fresh") or req.color not in ("any", "white", "black"):
            raise HTTPException(422, "Invalid options.")
        tcs = [t for t in req.time_controls if t in lichess.TIME_CONTROLS] or list(lichess.TIME_CONTROLS)
        if req.mode == "finetune" and not store.base_model.exists():
            raise HTTPException(500, f"Base model not found at {store.base_model}. Choose 'train from scratch' or restore it.")

        def run(job):
            job.update("download", 0.0, f"Contacting Lichess for {user}'s games...")

            def prog(seen, total, n_games, n_pos):
                job.update(progress=seen / max(1, total), message=f"Downloaded {n_games} games ({n_pos:,} moves)",
                           games=n_games, positions=n_pos)
            stream = source(user, req.max_games, tcs, None if req.color == "any" else req.color, req.rated_only, job.cancel)
            data = lichess.collect(stream, user, req.max_games, prog, job.cancel)
            job.update("train", 0.0, f"Learning from {len(data['move']):,} of your moves on {describe(device)}...",
                       games=data["games"], positions=len(data["move"]))
            tmp = Path(tempfile.mkdtemp(dir=data_dir)) / "model.pt"
            stats = train_profile(data, tmp, store.base_model if store.base_model.exists() else None,
                                  req.mode, device, lambda p, m: job.update("train", p, m), job.cancel)
            job.update("save", 0.5, "Saving your twin...")
            meta = {**stats, "games": data["games"], "mode": req.mode, "time_controls": tcs,
                    "color": req.color, "max_games": req.max_games, "device": device}
            pid = store.save(user, tmp, meta)
            tmp.parent.rmdir()
            engine.invalidate()
            return {"profile": pid, **meta}
        try:
            jobs.start(req.model_dump(), run)
        except Busy:
            raise HTTPException(409, "A twin is already being trained.") from None
        return jobs.snapshot()

    @app.get("/api/train")
    def train_status():
        return jobs.snapshot() or {"status": "idle"}

    @app.post("/api/train/cancel")
    def train_cancel():
        return {"cancelled": jobs.cancel()}

    # ------------------------------------------------------------------ games
    def _game(gid):
        try:
            return games.get(gid)
        except KeyError:
            raise HTTPException(404, "Game not found (the server may have restarted). Start a new game.") from None

    @app.post("/api/games")
    def new_game(req: NewGame):
        if store.model_path(req.profile) is None:
            raise HTTPException(404, "Unknown opponent.")
        return games.create(req.profile, req.color).state()

    @app.get("/api/games/{gid}")
    def get_game(gid: str):
        return _game(gid).state()

    @app.post("/api/games/{gid}/move")
    def move(gid: str, req: MoveReq):
        g = _game(gid)
        if g.result() or g.board.turn != g.human:
            raise HTTPException(409, "It's not your turn.")
        try:
            g.apply(g.parse(req.uci))
        except IllegalMove:
            raise HTTPException(422, "Illegal move.") from None
        return g.state()

    @app.post("/api/games/{gid}/bot-move")
    def bot_move(gid: str, req: BotReq = BotReq()):
        g = _game(gid)
        if g.bot_to_move:
            g.apply(engine.choose(g.profile, g.board, req.temperature, search=req.search))
        return g.state()

    @app.post("/api/games/{gid}/undo")
    def undo(gid: str):
        g = _game(gid)
        g.undo()
        return g.state()

    @app.post("/api/games/{gid}/resign")
    def resign(gid: str):
        g = _game(gid)
        if not g.result():
            g.resigned = True
        return g.state()

    # ------------------------------------------------------------------ analysis (Stockfish) + learning from a game
    def _analysis_payload(g):
        a = g.analysis
        w = sf.weighting(a, "white" if g.human == WHITE else "black")
        return {"status": "done", "human": "white" if g.human == WHITE else "black", "weighting": w,
                "can_learn": g.profile != BASE_ID and not g.learned, "learned": g.learned, **a}

    @app.post("/api/games/{gid}/analysis")
    def start_analysis(gid: str, req: AnalysisReq = AnalysisReq()):
        g = _game(gid)
        if g.analysis is not None:
            return _analysis_payload(g)
        if len(g.ucis) < 2:
            raise HTTPException(422, "Play a few moves first.")
        if not sf.find_stockfish(config.STOCKFISH):
            raise HTTPException(503, "Stockfish is not installed. Install it (e.g. `apt install stockfish`) "
                                     "or set CHESSTWIN_STOCKFISH to its path, then restart Chess Twin.")
        ucis = list(g.ucis)

        def run(job):
            with sf.UciEngine(config.STOCKFISH) as eng:
                result = sf.analyse_game(ucis, eng, req.depth, None, lambda p, m: job.update("analyse", p, m), job.cancel)
            if g.ucis == ucis:       # ignore the result if the game was changed (undo) meanwhile
                g.analysis = result
            return {"game": gid}
        try:
            analysis_jobs.start({"game": gid}, run, done_message="Analysis complete.")
        except Busy:
            raise HTTPException(409, "Another analysis is still running.") from None
        return analysis_jobs.snapshot()

    @app.get("/api/games/{gid}/analysis")
    def get_analysis(gid: str):
        g = _game(gid)
        if g.analysis is not None:
            return _analysis_payload(g)
        j = analysis_jobs.snapshot()
        if j and j["params"].get("game") == gid:
            return j
        return {"status": "idle"}

    @app.post("/api/games/{gid}/learn")
    def learn(gid: str):
        g = _game(gid)
        if g.analysis is None:
            raise HTTPException(409, "Analyse the game first.")
        if g.profile == BASE_ID or store.model_path(g.profile) is None:
            raise HTTPException(409, "Only your own twin can learn from games - the generic opponent stays as it is.")
        if g.learned:
            raise HTTPException(409, "Your twin already learned from this game.")
        samples = sf.training_samples(g.analysis, "white" if g.human == WHITE else "black")
        if samples is None:
            raise HTTPException(422, "There are no moves of yours to learn from.")
        if jobs.job and jobs.job.status == "running":
            raise HTTPException(409, "A twin is being trained right now. Try again in a moment.")
        tmp = Path(tempfile.mkdtemp(dir=data_dir)) / "model.pt"
        try:
            stats = learn_from_game(samples, store.model_path(g.profile), tmp, device)
            meta = store.meta(g.profile)
            store.update(g.profile, tmp, games_learned=meta.get("games_learned", 0) + 1)
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
        g.learned = True
        engine.invalidate()
        return {**stats, "weighting": sf.weighting(g.analysis, "white" if g.human == WHITE else "black")}

    # ------------------------------------------------------------------ UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
