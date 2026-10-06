"""Read-only two-IMU gesture projection, not a robot or anatomical estimator."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def proper(value):
    r = np.asarray(value, float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-6)):
        raise ValueError('姿态／安装映射必须是SO(3)旋转')
    return r.copy()


class PairBuffer:
    """Host-time alignment only; device clocks still drive IMU integration."""
    def __init__(self):
        self.rows = {}

    def append(self, device, stamp, rotation):
        rows = self.rows.setdefault(device, deque(maxlen=150))
        if rows and stamp <= rows[-1][0]:
            return
        rows.append((stamp, proper(rotation)))

    def pair(self, hand, finger, now, max_age=.3, max_gap=.1):
        if not hand or not finger or hand == finger:
            raise ValueError('请选择两台不同的在线设备')
        buffers = [self.rows.get(d, ()) for d in (hand, finger)]
        if any(not rows or now - rows[-1][0] > max_age for rows in buffers):
            raise ValueError('设备数据过期／未收到数据；显示保持，不继续估计')
        stamp = min(rows[-1][0] for rows in buffers)
        result = []
        for rows in buffers:
            if stamp < rows[0][0]:
                raise ValueError('等待两设备同期样本')
            for i, (t, r) in enumerate(rows):
                if t == stamp:
                    result.append(r.copy()); break
                if t > stamp:
                    t0, r0 = rows[i-1]
                    if t - t0 > max_gap:
                        raise ValueError('同期插值间隔过大，暂停估计')
                    result.append(Slerp([0., t-t0], Rotation.from_matrix([r0, r]))([stamp-t0]).as_matrix()[0])
                    break
            else:
                raise ValueError('无法找到同期姿态')
        return stamp, *result


def static_mean(rotations):
    if len(rotations) < 10:
        raise ValueError('至少需要10个不同时间的同期样本')
    rotations = Rotation.from_matrix(rotations)
    mean = rotations.mean()
    spread = np.rad2deg((mean.inv() * rotations).magnitude())
    if np.percentile(spread, 90) > 3.:
        raise ValueError('静止窗口波动超过3°，请支撑手并重采')
    return mean.as_matrix()


@dataclass
class GestureResult:
    angle_deg: float
    residual_deg: float
    closure: float | None


class DiscreteClosure:
    """Two-threshold hysteresis with a continuous dwell; no guessed initial state."""
    def __init__(self, open_threshold=.3, closed_threshold=.7, dwell=.15):
        if not 0 <= open_threshold < closed_threshold <= 1 or not math.isfinite(dwell) or dwell < 0:
            raise ValueError('打开阈值必须小于闭合阈值，范围0–100%；确认时间不能为负')
        self.open_threshold = open_threshold
        self.closed_threshold = closed_threshold
        self.dwell = dwell
        self.state = None
        self.pending = None
        self.since = None

    def pause(self):
        self.pending = self.since = None

    def update(self, value, stamp):
        if not math.isfinite(value) or not 0 <= value <= 1 or not math.isfinite(stamp):
            raise ValueError('无效离散开合输入')
        target = False if value <= self.open_threshold else True if value >= self.closed_threshold else None
        if target is None or target == self.state:
            self.pause()
        else:
            if target != self.pending or self.since is None or stamp < self.since:
                self.pending, self.since = target, stamp
            if stamp-self.since >= self.dwell:
                self.state = target
                self.pause()
        return self.state


class GestureCalibration:
    def __init__(self, hand_open, finger_open, installation):
        self.hand_open = proper(hand_open)
        self.finger_open = proper(finger_open)
        self.installation = proper(installation)
        self.axis = None
        self.closed_deg = None

    def vector(self, hand, finger):
        dh = self.hand_open.T @ proper(hand)
        df = self.finger_open.T @ proper(finger)
        r = dh.T @ self.installation @ df @ self.installation.T
        vector = Rotation.from_matrix(r).as_rotvec()
        if np.linalg.norm(vector) >= np.deg2rad(170.):
            raise ValueError('相对旋转接近对数分支边界，请重采开合参考')
        return vector

    def set_closed(self, hand, finger):
        v = self.vector(hand, finger)
        angle = np.linalg.norm(v)
        if angle < np.deg2rad(10.):
            raise ValueError('开合幅度小于10°，请检查动作／绑定后重采闭合')
        self.axis = v / angle
        self.closed_deg = float(np.rad2deg(angle))

    def evaluate(self, hand, finger):
        v = self.vector(hand, finger)
        if self.axis is None:
            return GestureResult(float(np.rad2deg(np.linalg.norm(v))), 0., None)
        angle = float(self.axis @ v)
        residual = float(np.rad2deg(np.linalg.norm(v - angle*self.axis)))
        return GestureResult(float(np.rad2deg(angle)), residual,
                             float(np.clip(np.rad2deg(angle)/self.closed_deg, 0., 1.)))

    def snapshot(self):
        return {key: None if value is None else value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()}
