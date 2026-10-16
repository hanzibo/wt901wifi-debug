"""Monotonic, one-shot command-angle end gesture; not physical pose feedback."""
from dataclasses import dataclass
import math
from .real_robot_protocol import decode

GESTURE_INDICES = (1, 3, 7, 9)  # R2 R4 L2 L4 in the protocol, not scene radians.
GESTURE_KINDS = {'lateral_raise': '侧举', 'front_raise': '前举'}
ALL_GESTURE_INDICES = (0, 1, 3, 6, 7, 9)


def gesture_indices(kind='lateral_raise', *, bilateral=True):
    if not isinstance(kind, str) or kind not in GESTURE_KINDS:
        raise ValueError('不支持的停止姿态类型')
    joints = (1, 3) if kind == 'lateral_raise' else (0, 1, 3)
    return tuple(offset+i for offset in ((0, 6) if bilateral else (6,)) for i in joints)


def axis_name(index):
    return ('R' if index < 6 else 'L')+str(index % 6+1)


@dataclass(frozen=True)
class GestureConfig:
    angles: tuple
    tolerances: tuple = (5.,)*4
    seconds: float = 7.
    indices: tuple = GESTURE_INDICES
    kind: str = 'lateral_raise'

    def __post_init__(self):
        if self.indices not in (gesture_indices(self.kind), gesture_indices(self.kind, bilateral=False)):
            raise ValueError('结束手势检测轴与停止姿态类型不匹配')
        if len(self.angles) != len(self.indices) or len(self.tolerances) != len(self.indices):
            raise ValueError('结束手势参考角和容差须与检测轴数一致')
        if not all(math.isfinite(x) for x in (*self.angles,*self.tolerances,self.seconds)):
            raise ValueError('结束手势配置含非有限值')
        if any(x <= 0 for x in self.tolerances) or self.seconds <= 0:
            raise ValueError('容差与保持时间须大于零')


class EndGesture:
    def __init__(self, config):
        self.config = config
        self.started = None
        self.previous = None
        self.elapsed = 0.
        self.triggered = False

    def reset(self):
        self.started = self.previous = None
        self.elapsed = 0.

    def update(self, frame, now, *, following, fresh, enabled, max_gap):
        if self.triggered:
            return False
        q,_ = decode(frame)
        inside = (following and fresh and all(enabled[i] for i in self.config.indices)
                  and all(abs(q[i]-a) <= t for i,a,t in
                          zip(self.config.indices,self.config.angles,self.config.tolerances)))
        if not inside or not math.isfinite(now):
            self.reset()
            return False
        if self.previous is not None and (now < self.previous or now-self.previous > max_gap):
            self.reset()
        if self.started is None:
            self.started = now
        self.previous = now
        self.elapsed = now-self.started
        if self.elapsed >= self.config.seconds:
            self.triggered = True
            return True
        return False
