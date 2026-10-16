"""Synchronized quintic command trajectory; not a physical feedback controller."""
import math
from .protocol import decode, encode


class Trajectory:
    period = .02

    def __init__(self, start, target, speed, acceleration, finger_speed, finger_acceleration,
                 frequency_hz=50.):
        if isinstance(frequency_hz, bool) or not math.isfinite(frequency_hz) or not 1 <= frequency_hz <= 100:
            raise ValueError('发送频率必须为1至100Hz的有限数值')
        self.frequency_hz = frequency_hz
        self.period = 1/frequency_hz
        self.start = decode(start)
        self.target = decode(target)
        limits = (speed, acceleration, finger_speed, finger_acceleration)
        self.limits = limits
        if any(not math.isfinite(v) or v <= 0 for v in limits):
            raise ValueError('速度和加速度必须为有限正数')
        angle_delta = max(abs(a-b) for a,b in zip(self.start[0],self.target[0]))
        finger_delta = max(abs(a-b) for a,b in zip(self.start[1],self.target[1]))
        # Max derivatives of 10s^3-15s^4+6s^5 are 1.875 and 10/sqrt(3).
        self.duration = max(self.period, 1.875*angle_delta/speed,
            math.sqrt((10/math.sqrt(3))*angle_delta/acceleration),
            1.875*finger_delta/finger_speed,
            math.sqrt((10/math.sqrt(3))*finger_delta/finger_acceleration))
        self.steps = math.ceil(self.duration/self.period)
        self.duration = self.steps*self.period

    def frame(self, step):
        s = max(0., min(1., step/self.steps))
        blend = s*s*s*(10+s*(-15+6*s))
        angles = [a+(b-a)*blend for a,b in zip(self.start[0],self.target[0])]
        fingers = [round(a+(b-a)*blend) for a,b in zip(self.start[1],self.target[1])]
        return encode(angles, fingers)
