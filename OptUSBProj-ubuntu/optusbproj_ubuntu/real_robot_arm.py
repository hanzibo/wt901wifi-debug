"""Left-arm command planning and relative IK. No serial or simulator access.

Scene geometry is a nominal model, not a measured physical calibration. Command
states must never be labelled as actual robot joint positions.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .real_robot_protocol import decode, encode, ARM_LIMITS
from .scene_robot_model import SceneArm
from .real_robot_trajectory import stopping_boundary


@dataclass(frozen=True)
class ArmLimits:
    frequency: float = 20.
    speed: tuple = (100.,) * 6
    acceleration: tuple = (150.,) * 6
    lower: tuple = (0., 90., 0., 0., 0., 51.)
    upper: tuple = (226., 233., 180., 119., 180., 122.)
    enabled: tuple = (False,) * 6
    signs: tuple = (1,) * 6
    side: str = 'left'

    @property
    def indices(self):
        return tuple(range(6, 12)) if self.side == 'left' else tuple(range(6))

    def __post_init__(self):
        if self.side not in ('left', 'right'):
            raise ValueError('未知机器人侧别')
        if not np.isfinite(self.frequency) or not 1 <= self.frequency <= 100:
            raise ValueError('发送频率须为 1–100 Hz')
        for name in ('speed', 'acceleration', 'lower', 'upper', 'enabled', 'signs'):
            value = tuple(getattr(self, name))
            if len(value) != 6 or not np.isfinite(value).all():
                raise ValueError(f'{name} 须为六个有限值')
            object.__setattr__(self, name, value)
        if any(type(x) is not bool for x in self.enabled):
            raise ValueError('输出许可须为布尔值')
        if any(x not in (-1, 1) for x in self.signs):
            raise ValueError('方向须为 ±1')
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            historical = ARM_LIMITS[self.indices[i]]
            # Old PC ranges are an outer software envelope, NOT certified limits.
            cap = historical[1]
            if not historical[0] <= lo < hi <= cap:
                raise ValueError(f'{self.side} {i+1} 范围超出原 PC 软件包络（不是实测机械限位）')
            if not 0 < self.speed[i] <= 250 or not 0 < self.acceleration[i] <= 250:
                raise ValueError('速度 / 加速度须大于 0 且不超过 250')

    def to_sim(self, command):
        return np.deg2rad((np.asarray(command)-90.) / np.asarray(self.signs))

    def to_command(self, q):
        return 90. + np.asarray(self.signs) * np.rad2deg(q)


class ArmControl:
    """Independent per-axis acceleration-limited trajectories, latest target only."""
    def __init__(self, baseline, limits, *, preserve_fingers=False):
        import ruckig
        self.r, self.limits, self.baseline = ruckig, limits, baseline
        self.angles, self.fingers = decode(baseline)
        self.indices = limits.indices
        self.size = len(self.indices)
        self.names = tuple(('R' if i < 6 else 'L')+str(i%6+1) for i in self.indices)
        if any(x != 90. for i, x in enumerate(self.angles) if i not in self.indices) or (any(self.fingers) and not preserve_fingers):
            raise ValueError('未参与的手臂 / 头腰必须为 90°；手指量必须为 0')
        if not any(limits.enabled):
            raise ValueError('至少明确勾选一个关节输出')
        self.q0 = np.array([self.angles[i] for i in self.indices])
        if np.any(self.q0 < limits.lower) or np.any(self.q0 > limits.upper):
            raise ValueError('接管起点不在六轴配置范围内；调整范围或重新上电')
        self.q = self.q0.copy()
        self.target = self.q0.copy()
        self.requested = self.q0.copy()
        self.v, self.a = np.zeros(self.size), np.zeros(self.size)
        self.saturated = [False] * self.size
        self.mode, self.reason, self.first = 'active', '', True
        self.planning_methods = ['ruckig'] * self.size
        self.planner = ruckig.Ruckig(1, 1 / limits.frequency)

    def set_target(self, values):
        if self.mode != 'active':
            return
        values = np.asarray(values, dtype=float)
        if values.shape != (self.size,) or not np.isfinite(values).all():
            raise ValueError('关节目标维数或数值无效')
        self.requested = values.copy()
        clipped = np.clip(values, self.limits.lower, self.limits.upper)
        self.target = np.where(self.limits.enabled, clipped, self.q0)
        self.saturated = (values != clipped).tolist()

    def brake(self, reason):
        if self.mode not in ('active', 'gap'):
            return
        target = self.q + self.v * np.abs(self.v) / (2*np.asarray(self.limits.acceleration))
        if np.any(target < np.asarray(self.limits.lower)-1e-7) or np.any(target > np.asarray(self.limits.upper)+1e-7):
            raise RuntimeError('六轴制动轨迹越界；指令状态不可信')
        self.target = np.clip(target, self.limits.lower, self.limits.upper)
        self.mode, self.reason = 'braking', reason

    def _plan(self, i):
        stop = stopping_boundary(self.q[i], self.v[i], self.a[i], self.target[i],
                                 self.limits.speed[i], self.limits.acceleration[i])
        self.planning_methods[i] = 'analytic_stop' if stop is not None else 'ruckig'
        if stop is not None:
            return stop
        inp = self.r.InputParameter(1)
        inp.current_position, inp.current_velocity, inp.current_acceleration = [self.q[i]], [self.v[i]], [self.a[i]]
        inp.target_position, inp.target_velocity, inp.target_acceleration = [self.target[i]], [0.], [0.]
        inp.max_velocity, inp.max_acceleration = [self.limits.speed[i]], [self.limits.acceleration[i]]
        trajectory = self.r.Trajectory(1)
        try:
            result = self.planner.calculate(inp, trajectory)
        except self.r.RuckigError as exc:
            raise RuntimeError(f'{self.names[i]} 轨迹规划失败（不是串口错误）：{exc}') from exc
        if int(result) < 0:
            raise RuntimeError(f'{self.names[i]} 轨迹生成失败：{result}')
        return trajectory

    def step(self):
        if self.first:
            self.first = False
            return self.baseline
        plans = [self._plan(i) for i in range(self.size)]
        if any(p.position_extrema[0].min < self.limits.lower[i]-1e-7 or
               p.position_extrema[0].max > self.limits.upper[i]+1e-7 for i, p in enumerate(plans)):
            self.brake('目标反转预测越界；六轴制动，需重新接管')
            plans = [self._plan(i) for i in range(self.size)]
        for i, p in enumerate(plans):
            (q,), (v,), (a,) = p.at_time(1/self.limits.frequency)
            if not self.limits.lower[i]-1e-7 <= q <= self.limits.upper[i]+1e-7:
                raise RuntimeError(f'{self.names[i]} 轨迹输出越界')
            # Remove only numerical epsilon at an endpoint (e.g. -1e-14 at 0°).
            self.q[i], self.v[i], self.a[i] = float(np.clip(q, self.limits.lower[i], self.limits.upper[i])), v, a
        if self.mode == 'braking' and np.max(np.abs(self.v)) < 1e-8 and np.max(np.abs(self.q-self.target)) < 1e-8:
            self.mode = 'paused'
        angles = self.angles.copy()
        for i, q in zip(self.indices, self.q):
            angles[i] = float(q)
        frame = encode(angles, self.fingers)
        for i in range(15):
            if not (i in self.indices and self.limits.enabled[self.indices.index(i)]):
                if frame[2+2*i:4+2*i] != self.baseline[2+2*i:4+2*i]:
                    raise RuntimeError('未授权关节字节发生变化')
        if frame[:2] != self.baseline[:2] or frame[32:52] != self.baseline[32:52]:
            raise RuntimeError('手指 / 保留字节发生变化')
        return frame

    def status(self):
        return dict(mode=self.mode, kind='left_arm' if self.indices == tuple(range(6,12)) else 'bilateral_arm',
                    names=self.names, q0=self.q0.tolist(),
                    target=self.target.tolist(), requested=self.requested.tolist(),
                    command=self.q.tolist(), velocity=self.v.tolist(), acceleration=self.a.tolist(),
                    saturated=self.saturated, enabled=list(self.limits.enabled), reason=self.reason,
                    planning_methods=self.planning_methods.copy())


def load_profile(side='left'):
    if side not in ('left', 'right'):
        raise ValueError('未知机器人侧别')
    data = json.loads(Path(__file__).with_name(f'real_robot_{side}_profile.json').read_text())
    C = np.asarray(data['axes'], dtype=float)
    if data['version'] != 1 or C.shape != (3, 3) or not np.allclose(C.T@C, np.eye(3), atol=1e-6) or np.linalg.det(C) < .99:
        raise ValueError('机器人名义坐标系配置无效')
    arm = data['arm']
    if np.asarray(arm['q']).shape != (6,) or np.asarray(arm['joints']).shape != (6, 12) or np.asarray(arm['palm']).shape != (12,):
        raise ValueError('机器人名义几何维数无效')
    if not all(np.isfinite(arm[k]).all() for k in ('q', 'joints', 'palm')):
        raise ValueError('机器人名义几何包含无效数值')
    for raw in [*arm['joints'], arm['palm']]:
        R = np.asarray(raw).reshape(3, 4)[:, :3]
        if not np.allclose(R.T@R, np.eye(3), atol=1e-5) or np.linalg.det(R) < .99:
            raise ValueError('机器人名义几何旋转无效')
    return arm, C, data['provenance']


class RelativeArmIK:
    """Anchor all tasks to the confirmed command pose, not hanging CAD neutral.

Six targets are solved for comparison. The output mask is enforced separately;
with a partial mask the physical commanded pose need not meet the full IK task.
"""
    def __init__(self, reference, baseline, limits, profile=None):
        arm, self.C, self.provenance = load_profile() if profile is None else profile
        self.model = SceneArm(arm, np.full(6, -np.pi), np.full(6, np.pi))
        self.limits, self.reference = limits, reference
        self.q0 = limits.to_sim(decode(baseline)[0][6:12])
        if np.any(limits.to_command(self.q0) < limits.lower) or np.any(limits.to_command(self.q0) > limits.upper):
            raise ValueError('机器人起点超出设定范围')
        sim_a, sim_b = limits.to_sim(limits.lower), limits.to_sim(limits.upper)
        self.model.lower, self.model.upper = np.minimum(sim_a, sim_b), np.maximum(sim_a, sim_b)
        # The reference of exported screw geometry stays unchanged!
        self.model.neutral = self.q0.copy()
        self.previous = self.q0.copy()
        self.robot_points = self.model.points(self.q0)
        self.palm0 = self.model.tool_frame(self.q0)[:3, :3]
        self.segment0 = {k:self.model.segment_frame(self.q0, k)[:3, :3] for k in ('upper_arm', 'forearm')}
        self._validate(reference)

    @staticmethod
    def _validate(side):
        for key, value, shape in [('points', side['points'], (3, 3)), ('hand', side['hand'], (3, 3)),
                                  *[(k, side['rotations'][k], (3, 3)) for k in ('upper_arm', 'forearm')]]:
            v = np.asarray(value)
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError(f'人体 {key} 输入无效')
            if key != 'points' and (not np.allclose(v.T@v, np.eye(3), atol=1e-5) or np.linalg.det(v) < .99):
                raise ValueError('人体朝向不是旋转矩阵')
        s,e,w = side['points']
        if min(np.linalg.norm(e-s), np.linalg.norm(w-e)) < 1e-6:
            raise ValueError('人体骨段长度无效')

    def solve(self, side):
        self._validate(side)
        s0,e0,w0 = self.reference['points']
        s,e,w = side['points']
        su = self.model.lengths[0]/np.linalg.norm(e0-s0)
        sf = self.model.lengths[1]/np.linalg.norm(w0-e0)
        du = su*self.C@((e-s)-(e0-s0))
        df = sf*self.C@((w-e)-(w0-e0))
        positions = (self.robot_points[1]+du, self.robot_points[2]+du+df)
        segments = {k:self.C@side['rotations'][k]@self.reference['rotations'][k].T@self.C.T@self.segment0[k]
                    for k in self.segment0}
        palm = self.C@side['hand']@self.reference['hand'].T@self.C.T@self.palm0
        q, error, orientation = self.model.fit(side['directions'], self.previous, np.full(6, 2*np.pi),
                                               palm, positions=positions, segment_rotations=segments,
                                               solver_mode='staged')
        self.previous = q
        return dict(target=self.limits.to_command(q).tolist(), position_error_mm=error*1000,
                    orientation_error_deg=orientation)
