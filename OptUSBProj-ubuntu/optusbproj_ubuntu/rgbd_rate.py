"""Phase-preserving output sampling, independent of SDK and Qt."""
import math


class PhaseRateLimiter:
    def __init__(self, fps: float, start_s: float):
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(start_s):
            raise ValueError("invalid frame rate / start")
        self.period = 1./fps
        self.deadline = float(start_s)
        self.tolerance = min(.01, .1*self.period)

    def accept(self, now_s: float) -> bool:
        if not math.isfinite(now_s):
            raise ValueError("invalid frame timestamp")
        if now_s+self.tolerance < self.deadline:
            return False
        # Advance the original phase, not 'now + period': tiny receipt jitter
        # must not discard every other frame when source FPS equals output FPS.
        periods = max(1, math.floor((now_s-self.deadline+self.tolerance)/self.period)+1)
        self.deadline += periods*self.period
        return True
