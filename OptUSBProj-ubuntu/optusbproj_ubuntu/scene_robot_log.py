"""Buffered JSONL recording and diagnostic throttling, independent of Qt."""
from datetime import datetime
import json
from pathlib import Path
import time


class SceneLog:
    def __init__(self, enabled, config, *, folder=None):
        self.file = None
        self.path = None
        if enabled:
            folder = Path(folder) if folder is not None else (
                Path(__file__).resolve().parents[1] / "recordings/coppeliasim")
            folder.mkdir(parents=True, exist_ok=True)
            self.path = folder / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
            self.file = self.path.open("w", encoding="utf-8", buffering=1)
            try:
                self.write({"config": config}, ensure_ascii=False)
            except Exception:
                self.file.close()
                raise

    def write(self, row, *, ensure_ascii=True):
        if self.file is not None:
            self.file.write(json.dumps(row, ensure_ascii=ensure_ascii) + "\n")

    def close(self):
        if self.file is not None:
            self.file.close()


class SceneDiagnostics:
    def __init__(self, report, log, *, clock=time.monotonic):
        self.report, self.log, self.clock = report, log, clock
        self.last_event, self.last_at = None, 0.

    def emit(self, event, message, **values):
        moment = self.clock()
        if event == self.last_event and moment - self.last_at < .5:
            return
        self.last_event, self.last_at = event, moment
        self.log.write({"event": event, "message": message,
                        "mono_s": moment, **values}, ensure_ascii=False)
        self.report({"diagnostic": message})
