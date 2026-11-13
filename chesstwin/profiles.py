"""Persisted opponents: one folder per profile with model.pt + meta.json. Plus the built-in generic base model."""
import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

BASE_ID = "base"
_SLUG = re.compile(r"[^a-z0-9_-]")


def slug(name):
    return _SLUG.sub("", name.lower())


class ProfileStore:
    def __init__(self, data_dir, base_model):
        self.root = Path(data_dir) / "profiles"
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_model = Path(base_model)
        self._lock = threading.Lock()

    def model_path(self, pid):
        if pid == BASE_ID:
            return self.base_model
        p = self.root / slug(pid) / "model.pt"
        return p if p.exists() else None

    def _read(self, d):
        try:
            return json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            return None

    def list(self):
        out = []
        if self.base_model.exists():
            out.append({"id": BASE_ID, "name": "Generic opponent", "kind": "base",
                        "description": "Trained on thousands of strong Lichess games - no account needed."})
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "model.pt").exists():
                meta = self._read(d)
                if meta:
                    out.append({"id": d.name, "name": meta.get("username", d.name), "kind": "twin", **meta})
        out[1:] = sorted(out[1:], key=lambda p: p.get("trained_at", ""), reverse=True)
        return out

    def save(self, username, tmp_model, meta):
        pid = slug(username)
        with self._lock:
            d = self.root / pid
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_model), d / "model.pt")
            meta = {**meta, "username": username, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return pid

    def delete(self, pid):
        d = self.root / slug(pid)
        if pid == BASE_ID or not d.is_dir() or d.parent != self.root:
            return False
        shutil.rmtree(d)
        return True
