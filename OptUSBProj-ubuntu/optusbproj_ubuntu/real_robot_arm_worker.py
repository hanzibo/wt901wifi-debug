"""Latest-only IK worker; independent of the serial planner's braking clock."""
import threading
import time
from PySide6.QtCore import QThread, Signal


class ArmIKWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, solver, parent=None, recorder=None):
        super().__init__(parent)
        self.solver = solver
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False
        self.recorder = recorder
        self.sequence = 0

    def diagnose(self, event, **data):
        if self.recorder is not None:
            self.recorder.record(event, **data)

    def submit(self, side, deadline, diagnostic=None):
        with self.condition:
            if not self.stopping:
                self.sequence += 1
                replaced = self.pending is not None
                self.pending = (side, deadline, time.monotonic(), self.sequence, diagnostic)
                self.diagnose('arm_ik_submit', sequence=self.sequence, deadline=deadline,
                              replaced_pending=replaced, input_timing=diagnostic)
                self.condition.notify()

    def shutdown(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify()

    def run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.stopping or self.pending is not None)
                    if self.stopping:
                        return
                    side, deadline, submitted, sequence, diagnostic = self.pending
                    self.pending = None
                before = time.monotonic()
                if before > deadline:
                    self.diagnose('arm_ik_discard', stage='before_solve', sequence=sequence,
                                  deadline=deadline, queue_ms=(before-submitted)*1000,
                                  expired_ms=(before-deadline)*1000, input_timing=diagnostic)
                    continue
                result = self.solver.solve(side)
                after = time.monotonic()
                with self.condition:
                    if self.stopping:
                        return
                # Keep ORIGINAL input expiry, not the completion time of IK.
                if after <= deadline:
                    result.update(deadline=deadline, ik_ms=(after-before)*1000,
                                  timing=dict(sequence=sequence, submitted_at=submitted,
                                      queue_ms=(before-submitted)*1000, solve_finished_at=after,
                                      input_timing=diagnostic))
                    self.result.emit(result)
                else:
                    self.diagnose('arm_ik_discard', stage='after_solve', sequence=sequence,
                                  deadline=deadline, queue_ms=(before-submitted)*1000,
                                  ik_ms=(after-before)*1000, expired_ms=(after-deadline)*1000,
                                  input_timing=diagnostic)
        except Exception as exc:
            self.failed.emit(str(exc))
