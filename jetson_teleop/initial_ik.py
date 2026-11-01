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
        self.solver = None

    def shutdown(self):
        self.cancelled.set()
        from .process_ik import ProcessIK
        if isinstance(self.solver, ProcessIK):
            self.solver.cancelled.set()
            if not self.isRunning():self.solver.close()

    def run(self):
        delivered = False
        try:
            started = time.monotonic()
            solver = self.factory()
            self.solver = solver
            from .process_ik import ProcessIK
            if isinstance(solver, ProcessIK):solver.cancelled = self.cancelled
            result = solver.solve(self.pose)
            if not self.cancelled.is_set():
                self.prepared.emit((solver, result, (time.monotonic()-started)*1000))
                delivered = True
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(str(exc))
        finally:
            from .process_ik import ProcessIK
            if isinstance(self.solver, ProcessIK) and (not delivered or self.cancelled.is_set()):
                self.solver.close()
