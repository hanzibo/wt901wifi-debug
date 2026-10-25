"""Read-only rolling statistics for valid packets read by the application."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceiveStats:
    rate_hz: float | None
    age_ms: float
    packets_in_window: int


class ReceiveRateTracker:
    """Count every parsed packet, including multiple packets in one GUI poll.

    A fixed window avoids treating a delayed batch as a high instantaneous rate.
    Startup has no numerical rate until a complete observation window is ready.
    Timestamps measure application reads, not radio or kernel arrival times.
    """

    def __init__(self, window_s: float = 2.0):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = window_s
        self._samples = {}
        self._first_seen = {}
        self._last_seen = {}

    def _prune(self, samples, now):
        while samples and samples[0] <= now - self.window_s:
            samples.popleft()

    def record(self, device_id: str, now: float):
        samples = self._samples.setdefault(device_id, deque())
        self._first_seen.setdefault(device_id, now)
        self._last_seen[device_id] = now
        self._prune(samples, now)
        samples.append(now)

    def snapshot(self, now: float) -> dict[str, ReceiveStats]:
        result = {}
        for device_id, samples in self._samples.items():
            self._prune(samples, now)
            rate = (len(samples) / self.window_s
                    if now - self._first_seen[device_id] >= self.window_s else None)
            result[device_id] = ReceiveStats(
                rate, max(0.0, now - self._last_seen[device_id]) * 1000, len(samples))
        return result
