"""Pure, testable fitting and quality checks for guided wearing calibration.

The signed-axis mapping is a prior, not a replacement for a fresh fit after
each donning. A calibration is expressed in the sensor's *neutral local* frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from wt901_viz.orientation import rotation_matrix

from .axis_mapping import mapping_matrix


STEPS = ("neutral", "forward", "lateral", "check_forward", "check_neutral")
ARM_PARTS = ("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")
CALIBRATABLE_PARTS = ("torso",) + ARM_PARTS


@dataclass(frozen=True)
class GuidedResult:
    part: str
    device_id: str
    basis_matrix: list[list[float]]
    neutral_matrix: list[list[float]]
    fit_error_deg: dict[str, float]
    check_error_deg: dict[str, float]
    hold_spread_deg: dict[str, float]
    gyro_p90_deg_s: dict[str, float]
    sample_counts: dict[str, int]
    status: str
    warnings: list[str]


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _hold(rows: list[dict]) -> tuple[np.ndarray, float, float]:
    if len(rows) < 30:
        raise ValueError(f"静止窗口只有 {len(rows)} 包，至少需要 30 包")
    rotations = Rotation.from_matrix(np.asarray([
        rotation_matrix(row["roll"], row["pitch"], row["yaw"])
        for row in rows], dtype=float))
    mean = rotations.mean()
    spread = float(np.rad2deg(np.percentile(
        (mean.inv() * rotations).magnitude(), 90)))
    gyro = np.asarray([[row["gyro_x"], row["gyro_y"], row["gyro_z"]]
                       for row in rows], dtype=float)
    gyro_p90 = float(np.percentile(np.linalg.norm(gyro, axis=1), 90))
    return mean.as_matrix(), spread, gyro_p90


def _unreliable_arrivals(rows: list[dict]) -> bool:
    """Ignore isolated UDP receive jitter, but reject sustained/long outages."""
    if not all("received_ns" in row for row in rows):
        return False
    gaps_ms = np.diff([row["received_ns"] for row in rows]) / 1e6
    return bool(len(gaps_ms) and
                (float(np.max(gaps_ms)) > 500.0 or
                 int(np.count_nonzero(gaps_ms > 150.0)) > 2))


def fit_guided_arm(part: str, device_id: str, samples: dict[str, list[dict]],
                   manual_mapping: tuple[str, str, str]) -> GuidedResult:
    """Fit arm bone direction from five labelled static holds.

    Gravity fixes model vertical. Two raised-arm directions identify heading;
    the manual map weakly regularizes that heading, rather than forcing an
    exact signed permutation. Independent repeat/return holds test the fit.
    This does *not* claim to identify anatomical twist about the bone axis.
    """
    if part not in ARM_PARTS:
        raise ValueError("引导式骨段标定目前仅支持左右上臂、前臂")
    missing = set(STEPS) - set(samples)
    if missing:
        raise ValueError(f"缺少动作窗口：{', '.join(sorted(missing))}")
    b0 = mapping_matrix(manual_mapping)
    holds = {step: _hold(samples[step]) for step in STEPS}
    r0, rf, rs, rfc, rnc = (holds[step][0] for step in STEPS)
    down = np.array([0.0, 0.0, -1.0])
    sensor_bone = r0.T @ down
    observed = [rf @ sensor_bone, rs @ sensor_bone]
    side = 1.0 if part.startswith("left_") else -1.0
    targets = [np.array([1.0, 0.0, 0.0]), np.array([0.0, side, 0.0])]
    # The signed map is a *weak* heading prior; large, consistent observations
    # are allowed to correct mounting offsets without being pulled back to 90°.
    prior_world = b0 @ r0.T
    prior_heading = np.arctan2(prior_world[1, 0], prior_world[0, 0])
    cosine = sum(float(np.dot(a[:2], b[:2])) for a, b in zip(observed, targets))
    sine = sum(float(np.dot(np.cross([0.0, 0.0, 1.0], a)[:2], b[:2]))
               for a, b in zip(observed, targets))
    evidence = float(np.hypot(cosine, sine))
    if evidence < 0.3:
        raise ValueError("前举/侧举没有提供足够的水平指向信息")
    prior_weight = 0.15
    heading = np.arctan2(sine + prior_weight * np.sin(prior_heading),
                         cosine + prior_weight * np.cos(prior_heading))
    world_to_model = Rotation.from_rotvec([0, 0, heading]).as_matrix()
    basis = world_to_model @ r0
    predicted = [world_to_model @ value for value in observed]
    fit_errors = {name: _angle(value, target) for name, value, target in
                  zip(("forward", "lateral"), predicted, targets)}
    check_forward = _angle(world_to_model @ (rfc @ sensor_bone), targets[0])
    check_neutral = float(np.rad2deg(Rotation.from_matrix(r0.T @ rnc).magnitude()))
    check_errors = {"check_forward": check_forward,
                    "check_neutral": check_neutral}
    spreads = {step: holds[step][1] for step in STEPS}
    gyros = {step: holds[step][2] for step in STEPS}
    warnings = []
    for step in STEPS:
        # A raised arm can have small physiological tremor. Its orientation
        # spread is the more direct measure of whether the hold is usable;
        # reject substantial angular motion, not a modest gyro P90 alone.
        if spreads[step] > 5.0 or gyros[step] > 20.0:
            warnings.append(f"{step} 未充分静止：姿态波动 {spreads[step]:.1f}°，"
                            f"角速度 P90 {gyros[step]:.1f}°/s")
        rows = samples[step]
        if _unreliable_arrivals(rows):
            warnings.append(f"{step} 回传中断（>500 ms 或多次 >150 ms）")
        if all(all(name in row for name in ("acc_x", "acc_y", "acc_z"))
               for row in rows):
            magnitudes = np.linalg.norm(np.asarray([
                [row["acc_x"], row["acc_y"], row["acc_z"]] for row in rows]),
                axis=1)
            if abs(float(np.median(magnitudes)) - 1.0) > 0.2:
                warnings.append(f"{step} 加速度模长偏离 1 g；请检查是否停稳或加计校准")
    if max(fit_errors.values()) > 25.0:
        warnings.append("标定动作与标准前举/侧举方向偏差超过 25°")
    if check_forward > 25.0 or check_neutral > 15.0:
        warnings.append("独立复测未通过：请检查动作、绑带滑动或航向漂移")
    return GuidedResult(
        part=part, device_id=device_id, basis_matrix=basis.tolist(),
        neutral_matrix=r0.tolist(), fit_error_deg=fit_errors,
        check_error_deg=check_errors, hold_spread_deg=spreads,
        gyro_p90_deg_s=gyros,
        sample_counts={step: len(samples[step]) for step in STEPS},
        status="通过" if not warnings else "需重录", warnings=warnings)


def fit_guided_torso(device_id: str, samples: dict[str, list[dict]],
                     manual_mapping: tuple[str, str, str]) -> GuidedResult:
    """Fit torso axes from forward flexion and a rightward side bend.

    Unlike the arm protocol this uses *motion axes*, not bone pointing.
    It assumes the pelvis stays approximately fixed while the torso bends.
    """
    missing = set(STEPS) - set(samples)
    if missing:
        raise ValueError(f"缺少动作窗口：{', '.join(sorted(missing))}")
    b0 = mapping_matrix(manual_mapping)
    holds = {step: _hold(samples[step]) for step in STEPS}
    r0, rf, rs, rfc, rnc = (holds[step][0] for step in STEPS)
    delta_f = Rotation.from_matrix(r0.T @ rf).as_rotvec()
    delta_s = Rotation.from_matrix(r0.T @ rs).as_rotvec()
    angles = [float(np.rad2deg(np.linalg.norm(vector)))
              for vector in (delta_f, delta_s)]
    if min(angles) < 15.0 or max(angles) > 105.0:
        raise ValueError("躯干前屈/侧屈幅度须在 15–105°；建议做 25–60°，当前幅度不足或过大")
    source = np.column_stack((delta_f / np.linalg.norm(delta_f),
                              delta_s / np.linalg.norm(delta_s)))
    separation = _angle(source[:, 0], source[:, 1])
    if not 40.0 <= separation <= 140.0:
        raise ValueError("前屈与侧屈的转轴太接近，无法确定躯干三维映射")
    # Model +Z is up; leaning forward rotates about +Y, bending right about +X.
    target = np.column_stack(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
    u, _, vt = np.linalg.svd(target @ source.T + 0.15 * b0)
    correction = np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))])
    basis = u @ correction @ vt
    fit_errors = {name: _angle(basis @ source[:, index], target[:, index])
                  for index, name in enumerate(("forward", "lateral"))}
    repeated = Rotation.from_matrix(r0.T @ rfc).as_rotvec()
    repeated_angle = float(np.rad2deg(np.linalg.norm(repeated)))
    check_forward = (180.0 if repeated_angle < 15.0 else
                     _angle(basis @ (repeated / np.linalg.norm(repeated)), target[:, 0]))
    check_neutral = float(np.rad2deg(Rotation.from_matrix(r0.T @ rnc).magnitude()))
    checks = {"check_forward": check_forward, "check_neutral": check_neutral}
    spreads = {step: holds[step][1] for step in STEPS}
    gyros = {step: holds[step][2] for step in STEPS}
    warnings = []
    for step in STEPS:
        if spreads[step] > 5.0 or gyros[step] > 10.0:
            warnings.append(f"{step} 未充分静止：姿态波动 {spreads[step]:.1f}°，"
                            f"角速度 P90 {gyros[step]:.1f}°/s")
        if _unreliable_arrivals(samples[step]):
            warnings.append(f"{step} 回传中断（>500 ms 或多次 >150 ms）")
    if max(fit_errors.values()) > 20.0:
        warnings.append("前屈/侧屈转轴与标准姿势偏差超过 20°")
    if check_forward > 20.0 or check_neutral > 15.0:
        warnings.append("独立复测未通过：请保持骨盆不动并检查绑带")
    return GuidedResult(
        part="torso", device_id=device_id, basis_matrix=basis.tolist(),
        neutral_matrix=r0.tolist(), fit_error_deg=fit_errors,
        check_error_deg=checks, hold_spread_deg=spreads,
        gyro_p90_deg_s=gyros,
        sample_counts={step: len(samples[step]) for step in STEPS},
        status="通过" if not warnings else "需重录", warnings=warnings)


def reference_motion_deg(samples: dict[str, list[dict]]) -> dict[str, float]:
    """How much a supposedly fixed parent segment moved from its neutral hold."""
    matrices = {step: _hold(samples[step])[0] for step in STEPS}
    neutral = matrices["neutral"]
    return {step: float(np.rad2deg(Rotation.from_matrix(
        neutral.T @ matrices[step]).magnitude()))
            for step in STEPS if step != "neutral"}
