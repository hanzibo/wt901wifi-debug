"""Candidate sensor-to-model basis from two instructed arm movements.

Neutral stance is model zero. With the elbow straight, a forward arm raise
is rotation about model -Y; a lateral raise is about +X (left) or -X (right).
This estimates the continuous basis B in B @ (R0.T @ Rt) @ B.T. It does not
estimate camera extrinsics, absolute position, or a biomechanical joint center.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .fusion_sync import load_imu


@dataclass(frozen=True)
class HoldWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not 0 <= self.start_s < self.end_s:
            raise ValueError("标定时间窗应为非负、递增秒数")


@dataclass(frozen=True)
class CalibrationCandidate:
    part: str
    device_id: str
    basis_matrix: list[list[float]]
    neutral_samples: int
    forward_samples: int
    lateral_samples: int
    forward_angle_deg: float
    lateral_angle_deg: float
    axis_separation_deg: float
    forward_axis_error_deg: float
    lateral_axis_error_deg: float
    hold_spread_deg: dict[str, float]
    gyro_p90_deg_s: dict[str, float]
    status: str
    warnings: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BoneCalibrationCandidate:
    """Conservative basis based on limb direction, not pure joint-axis motion."""

    part: str
    device_id: str
    basis_matrix: list[list[float]]
    forward_bone_error_deg: float
    lateral_bone_error_deg: float
    forward_elevation_deg: float
    lateral_elevation_deg: float
    hold_spread_deg: dict[str, float]
    gyro_p90_deg_s: dict[str, float]
    status: str
    warnings: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _hold(times_ns: np.ndarray, gyro: np.ndarray, matrices: np.ndarray,
          origin_ns: int, window: HoldWindow
          ) -> tuple[np.ndarray, int, float, float]:
    seconds = (times_ns - origin_ns) / 1e9
    selected = (seconds >= window.start_s) & (seconds <= window.end_s)
    count = int(np.count_nonzero(selected))
    if count < 15:
        raise ValueError(f"{window.start_s:g}–{window.end_s:g} 秒窗口只有 {count} 包；至少需要 15 包")
    rotations = Rotation.from_matrix(matrices[selected])
    mean = rotations.mean()
    spread = np.rad2deg(np.percentile((mean.inv() * rotations).magnitude(), 90))
    gyro_p90 = float(np.percentile(np.linalg.norm(gyro[selected], axis=1), 90))
    return mean.as_matrix(), count, float(spread), gyro_p90


def _axis_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(np.dot(left, right), -1, 1))))


def fit_arm_basis(session_path, part: str, device_id: str,
                  neutral: HoldWindow, forward: HoldWindow,
                  lateral: HoldWindow) -> CalibrationCandidate:
    if part not in ("left_upper_arm", "left_forearm",
                    "right_upper_arm", "right_forearm"):
        raise ValueError("第一版功能标定仅支持左右上臂与前臂")
    if any(a.end_s > b.start_s for a, b in
           ((neutral, forward), (forward, lateral))):
        raise ValueError("中立、前举、侧举窗口不能重叠，且须依次排列")
    times, gyro, matrices = load_imu(session_path, device_id)
    if len(times) < 45:
        raise ValueError(f"{device_id} 的录制数据不足")
    import json
    origin = int(json.loads((session_path / "manifest.json").read_text(
        encoding="utf-8"))["started_mono_ns"])
    hold_data = [_hold(times, gyro, matrices, origin, window)
                 for window in (neutral, forward, lateral)]
    (r0, n0, spread0, gyro0), (r1, n1, spread1, gyro1), (
        r2, n2, spread2, gyro2) = hold_data
    # Directions in the sensor's neutral *local* frame.
    forward_vector = Rotation.from_matrix(r0.T @ r1).as_rotvec()
    lateral_vector = Rotation.from_matrix(r0.T @ r2).as_rotvec()
    forward_angle = float(np.rad2deg(np.linalg.norm(forward_vector)))
    lateral_angle = float(np.rad2deg(np.linalg.norm(lateral_vector)))
    if min(forward_angle, lateral_angle) < 20:
        raise ValueError("动作不足 20°；请更明显地抬臂并在终点保持静止")
    if max(forward_angle, lateral_angle) > 135:
        raise ValueError("动作超过 135°；可能选错时间窗或发生姿态翻转")
    source = np.column_stack((forward_vector / np.linalg.norm(forward_vector),
                              lateral_vector / np.linalg.norm(lateral_vector)))
    side = 1.0 if part.startswith("left_") else -1.0
    target = np.column_stack(((0., -1., 0.), (side, 0., 0.)))
    separation = _axis_error(source[:, 0], source[:, 1])
    if not 35 <= separation <= 145:
        raise ValueError(f"两次动作轴仅相隔 {separation:.1f}°；无法稳定确定三维安装方向")
    u, _, vt = np.linalg.svd(target @ source.T)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    basis = u @ correction @ vt
    errors = [_axis_error(basis @ source[:, index], target[:, index])
              for index in range(2)]
    spreads = dict(neutral=spread0, forward=spread1, lateral=spread2)
    gyros = dict(neutral=gyro0, forward=gyro1, lateral=gyro2)
    warnings = []
    if any(value > 6 for value in spreads.values()):
        warnings.append("静止窗口姿态波动超过 6°；请避开运动过程并保持至少 1 秒")
    if any(value > 15 for value in gyros.values()):
        warnings.append("静止窗口角速度偏大；可能未完全停稳")
    if max(errors) > 12:
        warnings.append("动作轴与理想前举/侧举轴偏差较大；可能弯肘、耸肩或绑带滑动")
    return CalibrationCandidate(
        part, device_id, basis.tolist(), n0, n1, n2,
        forward_angle, lateral_angle, separation, errors[0], errors[1],
        spreads, gyros, "待复测" if warnings else "候选有效", warnings)


def fit_arm_bone_basis(session_path, part: str, device_id: str,
                       neutral: HoldWindow, forward: HoldWindow,
                       lateral: HoldWindow) -> BoneCalibrationCandidate:
    """Fit mounting from three known *bone directions* and gravity.

    Neutral arm points down. Front/side raises target horizontal +X and
    +/-Y. Only a heading rotation remains unknown after using the IMU's
    gravity-aligned world Z. This tolerates shoulder axial rotation better
    than fitting full 3-D rotation axes, but requires a true arm-down neutral.
    """
    if part not in ("left_upper_arm", "left_forearm",
                    "right_upper_arm", "right_forearm"):
        raise ValueError("骨段方向标定仅支持左右上臂与前臂")
    if any(a.end_s > b.start_s for a, b in
           ((neutral, forward), (forward, lateral))):
        raise ValueError("三个窗口不能重叠，且须依次排列")
    import json
    times, gyro, matrices = load_imu(session_path, device_id)
    if len(times) < 45:
        raise ValueError(f"{device_id} 的录制数据不足")
    origin = int(json.loads((session_path / "manifest.json").read_text(
        encoding="utf-8"))["started_mono_ns"])
    holds = [_hold(times, gyro, matrices, origin, window)
             for window in (neutral, forward, lateral)]
    r0, r1, r2 = (item[0] for item in holds)
    down = np.array([0., 0., -1.])
    sensor_bone_axis = r0.T @ down
    observed = [r1 @ sensor_bone_axis, r2 @ sensor_bone_axis]
    side = 1. if part.startswith("left_") else -1.
    target = [np.array([1., 0., 0.]), np.array([0., side, 0.])]
    # Optimize one heading angle around gravity. The neutral vertical axis
    # remains exact; there is no spurious tilt introduced by the fit.
    cosine = sum(float(np.dot(a, b)) for a, b in zip(observed, target))
    sine = sum(float(np.dot(np.cross([0., 0., 1.], a), b))
               for a, b in zip(observed, target))
    if np.hypot(cosine, sine) < 0.2:
        raise ValueError("前举/侧举方向无法确定水平航向")
    heading = np.arctan2(sine, cosine)
    h = Rotation.from_rotvec([0, 0, heading]).as_matrix()
    basis = h @ r0
    aligned = [h @ vector for vector in observed]
    errors = [_axis_error(a, b) for a, b in zip(aligned, target)]
    spreads = dict(zip(("neutral", "forward", "lateral"),
                       (item[2] for item in holds)))
    gyros = dict(zip(("neutral", "forward", "lateral"),
                     (item[3] for item in holds)))
    warnings = []
    if any(value > 6 for value in spreads.values()) or any(
            value > 15 for value in gyros.values()):
        warnings.append("至少一个姿势未停稳")
    if max(errors) > 25:
        warnings.append("骨段方向偏离目标超过 25°；请核对姿势或重录")
    elif max(errors) > 12:
        warnings.append("骨段方向有 12–25° 偏差；只建议在模型中人工预览")
    return BoneCalibrationCandidate(
        part, device_id, basis.tolist(), errors[0], errors[1],
        float(np.rad2deg(np.arcsin(np.clip(aligned[0][2], -1, 1)))),
        float(np.rad2deg(np.arcsin(np.clip(aligned[1][2], -1, 1)))),
        spreads, gyros, "需人工预览" if warnings else "候选有效", warnings)
