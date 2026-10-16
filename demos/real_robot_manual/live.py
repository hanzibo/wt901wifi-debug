"""Single-axis latest-target command planning, never physical limit detection."""
import math
import ruckig
from .protocol import encode, decode, ARM_LIMITS


class LiveTrajectory:
    live = True
    steps = None
    duration = None

    def __init__(self, baseline, axis, lower, upper, speed, acceleration, frequency_hz):
        self.start = decode(baseline)
        self.baseline = baseline
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < 15:
            raise ValueError('请选择一个角度通道')
        values = (lower, upper, speed, acceleration, frequency_hz)
        if not all(math.isfinite(v) and not isinstance(v, bool) for v in values):
            raise ValueError('实时控制参数须为有限数值')
        if not (0 <= lower < upper <= 327.67 and 0 < speed <= 180 and
                0 < acceleration <= 720 and 1 <= frequency_hz <= 100):
            raise ValueError('实时控制参数超出软件范围')
        if axis < 12 and not (ARM_LIMITS[axis][0] <= lower < upper <= ARM_LIMITS[axis][1]):
            raise ValueError('实时测试区间超出该轴原软件范围')
        self.axis, self.lower, self.upper = axis, lower, upper
        self.q = self.start[0][axis]
        if not lower <= self.q <= upper:
            raise ValueError('已确认起点不在测试区间内')
        self.v = self.a = 0.
        self.target = self.q
        self.frequency_hz = frequency_hz
        self.period = 1/frequency_hz
        self.limits = (speed, acceleration)
        self.first = True
        self.planner = ruckig.Ruckig(1, self.period)

    def set_target(self, target):
        if not math.isfinite(target) or not self.lower <= target <= self.upper:
            raise ValueError('实时目标超出本次测试区间')
        self.target = target

    def _plan(self, target):
        inp = ruckig.InputParameter(1)
        inp.current_position = [self.q]
        inp.current_velocity = [self.v]
        inp.current_acceleration = [self.a]
        inp.target_position = [target]
        inp.target_velocity = inp.target_acceleration = [0.]
        inp.max_velocity = [self.limits[0]]
        inp.max_acceleration = [self.limits[1]]
        trajectory = ruckig.Trajectory(1)
        if int(self.planner.calculate(inp, trajectory)) < 0:
            raise RuntimeError('实时指令轨迹生成失败')
        return trajectory

    def frame(self, step):
        if step == 0:
            return self.baseline
        if self.first:
            self.first = False
            return self.baseline
        trajectory = self._plan(self.target)
        extent = trajectory.position_extrema[0]
        if extent.min < self.lower-1e-7 or extent.max > self.upper+1e-7:
            # Reversal may overshoot the latest target. First stop inside the
            # envelope; re-evaluate the latest target on the next serial tick.
            stop = self.q+self.v*abs(self.v)/(2*self.limits[1])
            if not self.lower-1e-7 <= stop <= self.upper+1e-7:
                raise RuntimeError('制动路径越界，指令状态不可信')
            trajectory = self._plan(min(self.upper, max(self.lower, stop)))
        (self.q,), (self.v,), (self.a,) = trajectory.at_time(self.period)
        if not self.lower-1e-7 <= self.q <= self.upper+1e-7:
            raise RuntimeError('实时输出越界')
        angles = self.start[0].copy()
        angles[self.axis] = min(self.upper, max(self.lower, self.q))
        return encode(angles, self.start[1])
