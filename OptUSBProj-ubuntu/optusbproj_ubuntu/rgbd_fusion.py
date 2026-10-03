"""Read-only, offline six-axis propagation + gravity-axis visual correction.

Only segment pointing direction is observed. Axial twist is NOT observable
from two endpoints, and no full wrist orientation is inferred here.
"""
from dataclasses import dataclass
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from wt901_viz.so3_filter import AdaptiveTiltFilter
from wt901_viz.yaw_guard import PacketClock
from .fusion_sync import OrientationSeries

SEGMENTS = {"right_upper_arm": (1, 3), "right_forearm": (3, 5)}


def unit(vector):
    a = np.asarray(vector, float)
    n = float(np.linalg.norm(a))
    if not np.isfinite(a).all() or n < 1e-8:
        raise ValueError("invalid direction")
    return a/n


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1., 1.))))


def gravity_alignment(source, target):
    a, b = unit(source), unit(target)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = float(a @ b)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        reference = np.eye(3)[np.argmin(np.abs(a))]
        return Rotation.from_rotvec(unit(np.cross(a, reference))*math.pi).as_matrix()
    return Rotation.from_rotvec(cross/sine*math.atan2(sine, cosine)).as_matrix()


@dataclass
class MountingFit:
    camera_from_imu_world: np.ndarray
    bone_axis_sensor: np.ndarray
    median_error_deg: float
    p90_error_deg: float
    jacobian_singular_values: np.ndarray
    horizontal_samples: int

    @property
    def accepted(self):
        return bool(self.median_error_deg < 12 and self.p90_error_deg < 25
                and self.horizontal_samples >= 10
                and self.jacobian_singular_values[-1] > .03)


def fit_mounting(rotations, directions, up_world, up_camera):
    """Fit fixed world heading and sensor bone direction on a TRAINING prefix.

    Q = Exp([up_C] yaw) Q_gravity; d_C = Q R_WS b_S.
    b uses two spherical coordinates; no redundant axial-twist parameter.
    """
    matrices = np.asarray(rotations, float)
    dirs = np.array([unit(d) for d in directions])
    if len(dirs) < 15 or matrices.shape != (len(dirs), 3, 3):
        raise ValueError("标定有效方向不足 15 帧")
    up = unit(up_camera)
    q0 = gravity_alignment(up_world, up)
    horizontal = np.linalg.norm(dirs-(dirs @ up)[:, None]*up, axis=1)
    if np.count_nonzero(horizontal >= .35) < 10:
        raise ValueError("标定前段缺少足够的非竖直动作；不能估计航向对齐")

    def unpack(x):
        q = Rotation.from_rotvec(up*x[0]).as_matrix() @ q0
        b = np.array([math.sin(x[1])*math.cos(x[2]), math.sin(x[1])*math.sin(x[2]), math.cos(x[1])])
        return q, b

    def residual(x):
        q, b = unpack(x)
        predicted = np.einsum('ij,njk,k->ni', q, matrices, b)
        return (predicted-dirs).ravel()

    fits = []
    for yaw in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        q = Rotation.from_rotvec(up*yaw).as_matrix() @ q0
        initial_b = unit(matrices[0].T @ q.T @ dirs[0])
        x = [yaw, math.acos(np.clip(initial_b[2], -1, 1)), math.atan2(initial_b[1], initial_b[0])]
        fits.append(least_squares(residual, x, loss='soft_l1', f_scale=.08, max_nfev=200))
    result = min(fits, key=lambda f: f.cost)
    q, b = unpack(result.x)
    predicted = np.einsum('ij,njk,k->ni', q, matrices, b)
    errors = np.degrees(np.arccos(np.clip(np.sum(predicted*dirs, axis=1), -1, 1)))
    return MountingFit(q, b, float(np.median(errors)), float(np.percentile(errors, 90)),
                       np.linalg.svd(result.jac, compute_uv=False), int(np.count_nonzero(horizontal >= .35)))


class HeadingCorrector:
    """World-gravity left correction; no Euler yaw, tilt change or axial fit."""
    def __init__(self, up_camera, *, tau_s=3., max_rate_deg_s=3., max_innovation_deg=30., min_innovation_deg=0.):
        self.up = unit(up_camera)
        self.tau_s = tau_s
        self.max_rate = math.radians(max_rate_deg_s)
        self.max_innovation = math.radians(max_innovation_deg)
        self.min_innovation = math.radians(min_innovation_deg)
        self.offset_rad = 0.
        self.last_update_s = None

    def rotation(self):
        return Rotation.from_rotvec(self.up*self.offset_rad).as_matrix()

    def update(self, time_s, predicted_direction, observed_direction):
        if self.last_update_s is not None and time_s <= self.last_update_s:
            raise ValueError("correction timestamps must increase")
        dt = .1 if self.last_update_s is None else min(.25, time_s-self.last_update_s)
        self.last_update_s = time_s
        p, d = unit(self.rotation() @ unit(predicted_direction)), unit(observed_direction)
        ph = p-self.up*(p @ self.up)
        dh = d-self.up*(d @ self.up)
        if min(np.linalg.norm(ph), np.linalg.norm(dh)) < .35:
            return False, "近竖直肢段：航向不可观测", None
        tilt_difference = abs(math.asin(np.clip(p @ self.up, -1, 1))-math.asin(np.clip(d @ self.up, -1, 1)))
        if tilt_difference > math.radians(15):
            return False, "视觉/IMU 倾斜不一致", None
        ph, dh = unit(ph), unit(dh)
        innovation = math.atan2(float(self.up @ np.cross(ph, dh)), float(ph @ dh))
        if abs(innovation) > self.max_innovation:
            return False, "航向创新过大，拒绝强行纠偏", math.degrees(innovation)
        if abs(innovation)<self.min_innovation:
            return False,"小残差死区：不追随视觉噪声",math.degrees(innovation)
        delta = float(np.clip((1-math.exp(-dt/self.tau_s))*innovation, -self.max_rate*dt, self.max_rate*dt))
        candidate = self.offset_rad+delta
        if abs(candidate) > math.radians(30):
            return False, "累计纠偏超过初版安全范围", math.degrees(innovation)
        self.offset_rad = candidate
        return True, "低频重力轴纠偏", math.degrees(innovation)


def load_six_axis(path: Path, device_id: str, *, neutral_ns=None):
    clock, estimator = PacketClock(), AdaptiveTiltFilter()
    times, matrices, samples = [], [], []
    measurements = []
    with (path/'imu.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['device_id'] != device_id:
                continue
            packet = SimpleNamespace(**{name: float(row[name]) for name in
                                      ('hour','minute','second','millisecond')})
            elapsed = clock.elapsed(packet)
            result = estimator.update(elapsed,
                                      tuple(float(row['gyro_'+a]) for a in 'xyz'),
                                      tuple(float(row['acc_'+a]) for a in 'xyz'))
            stamp = int(row['host_recv_mono_ns'])
            times.append(stamp)
            matrices.append(result.rotation)
            samples.append({'host_recv_mono_ns': stamp, 'device_elapsed_s': elapsed,
                            'rotation_imu_world': result.rotation.tolist(),
                            'so3_distance': result.health.distance_to_so3,
                            'bias_ready': result.bias_ready})
            measurements.append((elapsed, tuple(float(row['gyro_'+a]) for a in 'xyz'),
                                 tuple(float(row['acc_'+a]) for a in 'xyz')))
    bootstrap = {'method':'existing stillness gravity initialization'}
    if len(times) >= 50 and estimator.neutral_up is None and neutral_ns is not None:
        # Some devices never satisfy the original strict 0.8 deg/s stillness
        # rule. A DECLARED prefix window can bootstrap gravity, but NOT bias.
        vectors = []
        for stamp, matrix, (_, gyro, acc) in zip(times, matrices, measurements):
            norm = np.linalg.norm(acc)
            if neutral_ns[0] <= stamp <= neutral_ns[1] and abs(norm-1) < .12:
                vectors.append(unit(matrix @ unit(acc)))
        if len(vectors) >= 20:
            up = unit(np.mean(vectors, axis=0))
            spread = float(np.percentile([angle_deg(v,up) for v in vectors],90))
            if spread <= 5:
                bootstrap = {'method':'declared neutral prefix gravity bootstrap, no forced gyro-bias fit',
                             'spread_p90_deg':spread,'samples':len(vectors),'activated_ns':neutral_ns[1]}
                estimator = AdaptiveTiltFilter()
                matrices, samples = [], []
                for stamp, (elapsed, gyro, acc) in zip(times, measurements):
                    if stamp >= neutral_ns[1] and estimator.neutral_up is None:
                        estimator.neutral_up = up.copy()
                    result = estimator.update(elapsed,gyro,acc)
                    matrices.append(result.rotation)
                    samples.append({'host_recv_mono_ns':stamp,'device_elapsed_s':elapsed,
                                    'rotation_imu_world':result.rotation.tolist(),
                                    'so3_distance':result.health.distance_to_so3,'bias_ready':result.bias_ready})
    if len(times) < 50 or estimator.neutral_up is None:
        raise ValueError(f'{device_id} 缺少足够 IMU 数据或静止重力初始化')
    return OrientationSeries(np.array(times), np.array(matrices), max_gap_ms=100), unit(estimator.neutral_up), samples, bootstrap


def visual_direction(row, part, expected_length=None):
    a, b = SEGMENTS[part]
    if not row['valid'][a] or not row['valid'][b]:
        return None, "视觉端点无效"
    indices = [5,6,7,8,9,10,11,12]
    if min(row['confidence'][indices[a]], row['confidence'][indices[b]]) < .6:
        return None, "二维分数低于融合门槛"
    if min(row['support'][a], row['support'][b]) < .7:
        return None, "深度支持低于融合门槛"
    xyz = np.asarray(row['points_camera_m'], float)
    d = xyz[b]-xyz[a]
    length = np.linalg.norm(d)
    if not np.isfinite(length) or length < .08:
        return None, "无效骨段"
    if expected_length is not None and abs(length/expected_length-1) > .25:
        return None, "骨长偏离标定中值超过25%"
    return unit(d), "视觉可用"


def load_live_series(path,device_id,neutral_ns):
    import json
    with (path/'runtime_pose.jsonl').open() as stream:
        rows=[json.loads(line) for line in stream]
    rows=[r for r in rows if r['device_id']==device_id and 'gyro_tilt' in r['orientation_by_mode']]
    if len(rows)<50:raise ValueError('实时姿态记录不足，请重新进行相机引导标定')
    if len({r['epoch'] for r in rows})!=1:raise ValueError('积分参考系在录制中改变，请重录')
    times=np.array([r['host_recv_mono_ns'] for r in rows])
    matrices=np.array([r['orientation_by_mode']['gyro_tilt'] for r in rows])
    prefix=[r for r in rows if neutral_ns[0]<=r['host_recv_mono_ns']<=neutral_ns[1]]
    ups=[r['up_world'] for r in prefix if r['up_world'] is not None]
    bootstrap={'method':'live integrator gravity reference'}
    guide_path=path/'camera_guide.json'
    locked=json.loads(guide_path.read_text()).get('gravity_reference',{}).get(device_id) if guide_path.exists() else None
    if locked is not None:
        if locked['epoch']!=rows[0]['epoch']:raise ValueError('锁定重力参考与实时积分参考系不一致，请重录')
        up=unit(locked['up_world'])
        if up.shape!=(3,) or not np.isfinite(up).all():raise ValueError('锁定重力参考无效')
        later=[r for r in rows if r['host_recv_mono_ns']>=locked['locked_at_ns']]
        if not later:raise ValueError('缺少重力参考锁定后的实时记录，请重录')
        if any(r['up_world'] is None or angle_deg(up,r['up_world'])>.1 for r in later):
            raise ValueError('锁定后实时重力参考发生变化，请重录')
        bootstrap=dict(locked,method='guided locked live gravity reference')
    elif ups:up=unit(ups[-1])
    else:
        with (path/'imu.csv').open() as stream:
            acc={(int(r['host_recv_mono_ns']),r['device_id']):[float(r['acc_'+a]) for a in 'xyz'] for r in csv.DictReader(stream)}
        vectors=[]
        for r in prefix:
            a=np.array(acc[(r['host_recv_mono_ns'],device_id)])
            if abs(np.linalg.norm(a)-1)<.12:vectors.append(unit(np.array(r['orientation_by_mode']['gyro_tilt']) @ unit(a)))
        if len(vectors)<20:raise ValueError('实时重力参考尚未建立，中立窗口数据不足')
        up=unit(np.mean(vectors,axis=0));spread=float(np.percentile([angle_deg(v,up) for v in vectors],90))
        if spread>5:raise ValueError('实时中立窗口重力方向不稳定，请重录')
        bootstrap={'method':'live-prefix gravity candidate; activate without resetting live R', 'spread_p90_deg':spread}
    samples=[dict(r,rotation_imu_world=matrices[i].tolist(),so3_distance=float(np.linalg.norm(matrices[i].T@matrices[i]-np.eye(3)))) for i,r in enumerate(rows)]
    return OrientationSeries(times,matrices,max_gap_ms=100),up,samples,bootstrap
