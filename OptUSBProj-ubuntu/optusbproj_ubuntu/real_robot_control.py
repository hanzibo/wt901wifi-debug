"""Command-side single-elbow control. No robot state/feedback is inferred here."""
from dataclasses import dataclass
import math

from .real_robot_protocol import encode, decode
from .real_robot_trajectory import stopping_boundary

MAX_COMMAND_SPEED = 250.
MAX_COMMAND_ACCELERATION = 250.
L4_MIN_ANGLE = 0.
L4_MAX_ANGLE = 100.

def elbow_angle(directions):
    import numpy as np
    u, f = (np.asarray(v, dtype=float) for v in directions)
    if u.shape != (3,) or f.shape != (3,) or not np.isfinite([u, f]).all():
        raise ValueError('骨段方向无效')
    if min(np.linalg.norm(u), np.linalg.norm(f)) < 1e-8:
        raise ValueError('骨段长度为零')
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, f))), float(u @ f)))


@dataclass(frozen=True)
class Limits:
    frequency: float = 20.
    speed: float = 10.
    acceleration: float = 20.
    gain: float = 1.
    sign: int = 1
    lower: float = L4_MIN_ANGLE
    upper: float = L4_MAX_ANGLE

    def __post_init__(self):
        values = (self.frequency, self.speed, self.acceleration, self.gain, self.lower, self.upper)
        if not all(math.isfinite(v) and not isinstance(v, bool) for v in values):
            raise ValueError('控制参数必须为有限数值')
        if not (1 <= self.frequency <= 100 and 0 < self.speed <= MAX_COMMAND_SPEED and
                0 < self.acceleration <= MAX_COMMAND_ACCELERATION and 0 < self.gain <= 3 and
                self.sign in (-1, 1) and L4_MIN_ANGLE <= self.lower < self.upper <= L4_MAX_ANGLE):
            raise ValueError('控制参数超出受限测试范围')


class ElbowControl:
    """Latest-target online planning with acceleration limits, not torque control.

    Uses acceleration-constrained Ruckig (no finite jerk bound). Position extrema
    are checked explicitly: a retarget cannot drive outside the testing envelope.
    """
    def __init__(self, baseline, theta0, limits=Limits()):
        import ruckig
        self.r = ruckig
        self.angles, self.fingers = decode(baseline)
        self.baseline = baseline
        self.limits = limits
        self.q0 = self.q = self.angles[9]
        if not limits.lower <= self.q0 <= limits.upper or not math.isfinite(theta0) or not 0 <= theta0 <= 180:
            raise ValueError('接管参考不在允许范围内')
        if any(q != 90. for index, q in enumerate(self.angles) if index != 9) or any(self.fingers):
            raise ValueError('仅允许 L4 改变；其他角度必须为 90°，手指量必须为 0')
        self.theta0 = theta0
        self.theta = theta0
        self.v = self.a = 0.
        self.target = self.q
        self.saturated = False
        self.reason = ''
        self.mode = 'active'
        self.first = True
        self.planning_method = 'ruckig'
        self.planner = ruckig.Ruckig(1, 1 / limits.frequency)

    def set_target(self, theta):
        if self.mode != 'active':
            return
        if not math.isfinite(theta) or not 0 <= theta <= 180:
            raise ValueError('屈肘角无效')
        self.theta = theta
        raw = self.q0 + self.limits.sign * self.limits.gain * (theta - self.theta0)
        self.target = min(self.limits.upper, max(self.limits.lower, raw))
        self.saturated = raw != self.target

    def brake(self, reason):
        if self.mode not in ('active', 'gap'):
            return
        self.reason = reason
        # Exact stopping distance for acceleration-constrained command state.
        self.target = self.q + self.v * abs(self.v) / (2 * self.limits.acceleration)
        if not self.limits.lower - 1e-8 <= self.target <= self.limits.upper + 1e-8:
            raise RuntimeError('制动轨迹越界；指令状态不可信')
        self.target = min(self.limits.upper, max(self.limits.lower, self.target))
        self.mode = 'braking'

    def _plan(self):
        stop = stopping_boundary(self.q, self.v, self.a, self.target,
                                 self.limits.speed, self.limits.acceleration)
        self.planning_method = 'analytic_stop' if stop is not None else 'ruckig'
        if stop is not None:
            return stop
        inp = self.r.InputParameter(1)
        inp.current_position = [self.q]
        inp.current_velocity = [self.v]
        inp.current_acceleration = [self.a]
        inp.target_position = [self.target]
        inp.target_velocity = [0.]
        inp.target_acceleration = [0.]
        inp.max_velocity = [self.limits.speed]
        inp.max_acceleration = [self.limits.acceleration]
        trajectory = self.r.Trajectory(1)
        try:
            result = self.planner.calculate(inp, trajectory)
        except self.r.RuckigError as exc:
            raise RuntimeError(f'L4 轨迹规划失败（不是串口错误）：{exc}') from exc
        if int(result) < 0:
            raise RuntimeError(f'轨迹生成失败：{result}')
        return trajectory

    def step(self):
        if self.first:
            self.first = False
            return self.baseline
        trajectory = self._plan()
        extent = trajectory.position_extrema[0]
        if extent.min < self.limits.lower - 1e-7 or extent.max > self.limits.upper + 1e-7:
            self.brake('目标变化会造成越界，已制动；需重新接管')
            trajectory = self._plan()
        (q,), (v,), (a,) = trajectory.at_time(1 / self.limits.frequency)
        if not self.limits.lower - 1e-7 <= q <= self.limits.upper + 1e-7:
            raise RuntimeError('轨迹输出越界')
        self.q, self.v, self.a = q, v, a
        if self.mode == 'braking' and abs(v) < 1e-8 and abs(q - self.target) < 1e-8:
            self.mode = 'paused'
        angles = self.angles.copy()
        angles[9] = q
        frame = encode(angles, self.fingers)
        # Whitelist exact bytes: L4 centidegrees plus checksum, nothing else.
        if frame[:20] != self.baseline[:20] or frame[22:52] != self.baseline[22:52]:
            raise RuntimeError('非 L4 通道发生变化')
        return frame

    def status(self):
        return dict(mode=self.mode, theta=self.theta, theta0=self.theta0, q0=self.q0,
                    target=self.target, command=self.q, velocity=self.v,
                    acceleration=self.a, saturated=self.saturated, reason=self.reason,
                    planning_method=self.planning_method)
