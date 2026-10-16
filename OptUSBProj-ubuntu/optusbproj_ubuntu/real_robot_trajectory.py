"""Exact acceleration-only stop at Ruckig's degenerate stopping boundary.

No IO or robot feedback. This branch is valid only with unlimited jerk, zero
terminal velocity/acceleration and symmetric acceleration bounds. All other
targets still use Ruckig; callers must retain their full position-extrema checks.
"""
import math
from types import SimpleNamespace


class AccelerationStop:
    """Monotonic maximum-deceleration trajectory with Ruckig's sampling API."""
    def __init__(self, position, velocity, acceleration):
        self.position = position
        self.velocity = velocity
        self.deceleration = -math.copysign(acceleration, velocity)
        self.duration = abs(velocity) / acceleration
        self.end = position + velocity * abs(velocity) / (2 * acceleration)
        self.position_extrema = [SimpleNamespace(min=min(position, self.end),
                                                max=max(position, self.end))]

    def at_time(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('轨迹采样时间必须为非负有限值')
        if seconds >= self.duration:
            return [self.end], [0.], [0.]
        q = self.position + self.velocity * seconds + .5 * self.deceleration * seconds**2
        extent = self.position_extrema[0]
        # Roundoff only; the analytic path is monotonic by construction.
        q = min(extent.max, max(extent.min, q))
        return [q], [self.velocity + self.deceleration * seconds], [self.deceleration]


def stopping_boundary(position, velocity, current_acceleration, target, speed, acceleration):
    """Return an exact stop only when target equals the stop point to roundoff.

    This is not a catch-all fallback for invalid inputs or planning failures.
    Never round the target to a protocol angle, enlarge limits or reset velocity.
    """
    values = (position, velocity, current_acceleration, target, speed, acceleration)
    if not all(math.isfinite(x) for x in values) or speed <= 0 or acceleration <= 0:
        return None
    if abs(velocity) > speed or abs(current_acceleration) > acceleration:
        return None
    trajectory = AccelerationStop(position, velocity, acceleration)
    tolerance = max(1e-12, 32 * math.ulp(max(1., abs(position), abs(target))))
    if abs(target - trajectory.end) <= tolerance:
        # Preserve the requested endpoint exactly (including 0°). Only the
        # roundoff-sized terminal residue is removed; keep both endpoints in
        # extrema so this cannot hide an out-of-envelope theoretical stop.
        extent = trajectory.position_extrema[0]
        extent.min, extent.max = min(extent.min, target), max(extent.max, target)
        trajectory.end = target
        return trajectory
    return None
