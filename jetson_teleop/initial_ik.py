"""Board-only initial solve: no Qt widgets, serial access or motion authority."""
import threading
import time
from PySide6.QtCore import QThread, Signal


class InitialIKWorker(QThread):
    prepared = Signal(object)
    failed = Signal(str)

    def __init__(self, factory, pose, parent=None):
        super().__init__(parent)
        self.factory, self.pose = factory, pose
        self.cancelled = threading.Event()

    def shutdown(self):
        self.cancelled.set()

    def run(self):
        try:
            started = time.monotonic()
            solver = self.factory()
            result = solver.solve(self.pose)
            if not self.cancelled.is_set():
                self.prepared.emit((solver, result, (time.monotonic()-started)*1000))
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(str(exc))
