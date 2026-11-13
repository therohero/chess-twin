"""One background job at a time (download + train), observable by polling."""
import threading
import time


class Busy(Exception):
    pass


class Job:
    PHASES = ("download", "train", "save")

    def __init__(self, params):
        self.params = params
        self.status = "running"          # running | done | error | cancelled
        self.phase = "download"
        self.progress = 0.0              # 0..1 within the current phase
        self.message = "Starting..."
        self.detail = {}
        self.log = []
        self.result = None
        self.error = None
        self.started = time.time()
        self.cancel = threading.Event()

    def update(self, phase=None, progress=None, message=None, **detail):
        if phase:
            self.phase = phase
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message and message != self.message:
            self.message = message
            self.log.append(message)
            del self.log[:-100]
        self.detail.update(detail)

    def snapshot(self):
        return {"status": self.status, "phase": self.phase, "progress": self.progress, "message": self.message,
                "detail": self.detail, "log": self.log[-12:], "result": self.result, "error": self.error,
                "params": self.params, "elapsed": round(time.time() - self.started, 1)}


class JobManager:
    def __init__(self):
        self.job = None
        self._lock = threading.Lock()

    def start(self, params, fn):
        """fn(job) runs in a thread; it should call job.update(...) and return the result dict."""
        with self._lock:
            if self.job and self.job.status == "running":
                raise Busy()
            job = self.job = Job(params)

        def run():
            try:
                job.result = fn(job)
                job.status = "done"
                job.update("save", 1.0, "Your twin is ready!")
            except (InterruptedError, KeyboardInterrupt):
                job.status = "cancelled"
                job.message = "Cancelled."
            except Exception as e:  # noqa: BLE001 - surfaced to the UI
                job.status = "error"
                job.error = str(e) or e.__class__.__name__
        threading.Thread(target=run, daemon=True, name="chesstwin-job").start()
        return job

    def cancel(self):
        if self.job and self.job.status == "running":
            self.job.cancel.set()
            return True
        return False

    def snapshot(self):
        return self.job.snapshot() if self.job else None
