"""Pure, testable fitting and quality checks for guided wearing calibration.

The signed-axis mapping is a prior, not a replacement for a fresh fit after
each donning. A calibration is expressed in the sensor's *neutral local* frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from wt901_viz.orientation import rotation_matrix

from .axis_mapping import mapping_matrix


STEPS = ("neutral", "forward", "lateral", "check_forward", "check_neutral")
WRIST_STEPS = STEPS + ("check_long_axis",)
ARM_PARTS = ("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")
HAND_PARTS = ("left_hand", "right_hand")
CALIBRATABLE_PARTS = ("torso",) + ARM_PARTS + HAND_PARTS


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
    if any(row.get("orientation_mode") == "gyro_tilt" and
           "orientation_matrix" not in row for row in rows):
        raise ValueError("陀螺＋加计尚未建立重力竖直参考；请先静止约 1 秒再重录")
    rotations = Rotation.from_matrix(np.asarray([
        row["orientation_matrix"] if "orientation_matrix" in row else
        rotation_matrix(row["roll"], row["pitch"], row["yaw"])
        for row in rows], dtype=float))
    mean = rotations.mean()
    spread = float(np.rad2deg(np.percentile(
        (mean.inv() * rotations).magnitude(), 90)))
    gyro = np.asarray([[row["gyro_x"], row["gyro_y"], row["gyro_z"]]
                       for row in rows], dtype=float)
    gyro_p90 = float(np.percentile(np.linalg.norm(gyro, axis=1), 90))
    return mean.as_matrix(), spread, gyro_p90


def neutral_matrices_by_mode(rows: list[dict]) -> dict[str, np.ndarray]:
    """Same physical hold, independently averaged in both attitude streams."""
    if len(rows) < 30:
        raise ValueError("同步中立位采样不足")
    modes = set.intersection(*(set(row.get("orientation_by_mode", {})) for row in rows))
    return {mode: Rotation.from_matrix(np.asarray([
        row["orientation_by_mode"][mode] for row in rows], dtype=float
    )).mean().as_matrix() for mode in modes}


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
    For either hand, the straight palm/finger long axis follows the extended
    arm during the same raises; no forearm-reference subtraction is used.
    """
    if part not in ARM_PARTS + HAND_PARTS:
        raise ValueError("引导式骨段标定目前仅支持左右上臂、前臂及双手")
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


def fit_guided_left_wrist(device_id: str, samples: dict[str, list[dict]],
                          manual_mapping: tuple[str, str, str],
                          forearm_samples: dict[str, list[dict]],
                          forearm_basis: np.ndarray) -> GuidedResult:
    """Fit a left-hand mounting from wrist flexion and thumb-side deviation.

    In the neutral model the dorsum points +X, thumb points -Y, and fingers
    point -Z. Palm flexion rotates around +Y; thumb-side deviation around -X.
    Both sensors use their first static hold as neutral. Forearm movement
    between holds is removed in the calibrated model frame; each hold itself
    must still be sufficiently stationary.
    """
    missing = set(WRIST_STEPS) - set(samples)
    if missing:
        raise ValueError(f"缺少手腕动作窗口：{', '.join(sorted(missing))}；请按新增的长轴验证流程重录")
    missing_forearm = set(WRIST_STEPS) - set(forearm_samples)
    if missing_forearm:
        raise ValueError(f"缺少前臂动作窗口：{', '.join(sorted(missing_forearm))}")
    prior = mapping_matrix(manual_mapping)
    holds = {step: _hold(samples[step]) for step in WRIST_STEPS}
    forearm_holds = {step: _hold(forearm_samples[step]) for step in WRIST_STEPS}
    forearm_basis = np.asarray(forearm_basis, dtype=float)
    if (forearm_basis.shape != (3, 3) or not np.isfinite(forearm_basis).all() or
            not np.allclose(forearm_basis.T @ forearm_basis, np.eye(3), atol=1e-3) or
            not np.isclose(np.linalg.det(forearm_basis), 1.0, atol=1e-3)):
        raise ValueError("左前臂轴映射矩阵无效")
    r0 = holds["neutral"][0]
    forearm_neutral = forearm_holds["neutral"][0]
    forearm_motion = {
        step: forearm_basis @ (forearm_neutral.T @ hold[0]) @ forearm_basis.T
        for step, hold in forearm_holds.items()}
    hand_motion = {step: r0.T @ hold[0] for step, hold in holds.items()}
    def relative_wrist(step: str, basis: np.ndarray) -> np.ndarray:
        hand_model = basis @ hand_motion[step] @ basis.T
        return forearm_motion[step].T @ hand_model

    # Build the seed from wrist motion, not hand motion. During a worn trial
    # the forearm can move appreciably even when the requested wrist action is
    # correct; including that motion in the seed can flip the fitted axes.
    delta_f = Rotation.from_matrix(relative_wrist("forward", prior)).as_rotvec()
    delta_s = Rotation.from_matrix(relative_wrist("lateral", prior)).as_rotvec()
    if min(np.linalg.norm(delta_f), np.linalg.norm(delta_s)) < np.deg2rad(5):
        raise ValueError("扣除前臂运动后至少一个手腕动作不足 5°，无法辨识安装轴")
    target = np.column_stack(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    # The observed axes above are in model coordinates under the prior map.
    # Find a small correction to that map, rather than fitting an unrelated
    # mounting from the two training gestures alone.
    observed = np.column_stack((delta_f / np.linalg.norm(delta_f),
                                delta_s / np.linalg.norm(delta_s)))
    u, _, vt = np.linalg.svd(target @ observed.T + 0.15 * np.eye(3))
    correction = np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))])
    seed_rotation = Rotation.from_matrix(u @ correction @ vt).as_rotvec()
    seed_angle = float(np.linalg.norm(seed_rotation))
    if seed_angle > np.deg2rad(20.0):
        seed_rotation *= np.deg2rad(20.0) / seed_angle

    def constrained_basis(vector: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(vector))
        if angle > np.deg2rad(25.0):
            vector = vector * (np.deg2rad(25.0) / angle)
        return Rotation.from_rotvec(vector).as_matrix() @ prior

    def residual(parameters: np.ndarray) -> np.ndarray:
        basis = constrained_basis(parameters[:3])
        flex = Rotation.from_matrix(relative_wrist("forward", basis)).as_rotvec()
        radial = Rotation.from_matrix(relative_wrist("lateral", basis)).as_rotvec()
        offset = float(np.linalg.norm(parameters[:3]))
        return np.concatenate((flex - target[:, 0] * parameters[3],
                               radial - target[:, 1] * parameters[4],
                               0.20 * parameters[:3],
                               [5.0 * max(0.0, offset - np.deg2rad(25.0))]))

    initial_angles = [float(np.linalg.norm(Rotation.from_matrix(
        relative_wrist(step, prior)).as_rotvec()))
        for step in ("forward", "lateral")]
    lower = np.deg2rad([18.0, 8.0])
    upper = np.deg2rad([85.0, 45.0])
    start = np.r_[seed_rotation, np.clip(initial_angles, lower + 1e-5, upper - 1e-5)]
    fit = least_squares(residual, start, bounds=(
        np.r_[[-np.deg2rad(25.0)] * 3, lower],
        np.r_[[np.deg2rad(25.0)] * 3, upper]))
    if not fit.success:
        raise ValueError("手腕双设备映射拟合未收敛，请重新采样")
    basis = constrained_basis(fit.x[:3])
    vectors = {step: Rotation.from_matrix(relative_wrist(step, basis)).as_rotvec()
               for step in ("forward", "lateral", "check_forward", "check_neutral")}
    angles = {step: float(np.rad2deg(np.linalg.norm(vector)))
              for step, vector in vectors.items()}
    if not 18.0 <= angles["forward"] <= 85.0:
        raise ValueError(f"扣除前臂动作后掌屈幅度 {angles['forward']:.1f}°；建议约 25–55°")
    if not 8.0 <= angles["lateral"] <= 45.0:
        raise ValueError(f"扣除前臂动作后拇指侧偏转幅度 {angles['lateral']:.1f}°；建议约 12–25°")
    separation = _angle(vectors["forward"] / np.linalg.norm(vectors["forward"]),
                        vectors["lateral"] / np.linalg.norm(vectors["lateral"]))
    if not 50.0 <= separation <= 130.0:
        raise ValueError("掌屈和拇指侧偏转的相对运动轴太接近；请重做")
    fit_errors = {name: _angle(vectors[step] / np.linalg.norm(vectors[step]),
                               target[:, index])
                  for index, (step, name) in enumerate((("forward", "forward"),
                                                        ("lateral", "lateral")))}
    repeated = vectors["check_forward"]
    check_forward = (180.0 if angles["check_forward"] < 18.0 else
                     _angle(repeated / np.linalg.norm(repeated), target[:, 0]))
    check_neutral = angles["check_neutral"]
    manual_check_neutral = float(np.rad2deg(Rotation.from_matrix(
        relative_wrist("check_neutral", prior)).magnitude()))
    checks = {"check_forward": check_forward, "check_neutral": check_neutral,
              "check_neutral_manual_mapping": manual_check_neutral}
    # This is an independent *forearm pronation/supination* check, not a
    # request to twist the wrist joint. With the wrist straight, both devices
    # should turn together about the model forearm's long (+Z/-Z) axis.
    forearm_axial = Rotation.from_matrix(forearm_motion["check_long_axis"]).as_rotvec()
    hand_axial = Rotation.from_matrix(
        basis @ hand_motion["check_long_axis"] @ basis.T).as_rotvec()
    axial_angle = float(np.rad2deg(np.linalg.norm(forearm_axial)))
    hand_axial_angle = float(np.rad2deg(np.linalg.norm(hand_axial)))
    axial_axis_error = (_angle(forearm_axial / np.linalg.norm(forearm_axial),
                               hand_axial / np.linalg.norm(hand_axial))
                        if min(axial_angle, hand_axial_angle) >= 1.0 else 180.0)
    axial_relative = float(np.rad2deg(Rotation.from_matrix(
        relative_wrist("check_long_axis", basis)).magnitude()))
    axial_forearm_z = (_angle(np.abs(forearm_axial / np.linalg.norm(forearm_axial)),
                               np.array([0.0, 0.0, 1.0]))
                        if axial_angle >= 1.0 else 180.0)
    checks.update({"check_long_axis_angle": axial_angle,
                   "check_long_axis_hand_angle": hand_axial_angle,
                   "check_long_axis_axis": axial_axis_error,
                   "check_long_axis_relative": axial_relative,
                   "check_long_axis_forearm_z": axial_forearm_z})
    spreads = {step: holds[step][1] for step in WRIST_STEPS}
    gyros = {step: holds[step][2] for step in WRIST_STEPS}
    warnings = []
    for step in WRIST_STEPS:
        if spreads[step] > 5.0 or gyros[step] > 15.0:
            warnings.append(f"{step} 未充分静止：姿态波动 {spreads[step]:.1f}°，"
                            f"角速度 P90 {gyros[step]:.1f}°/s")
        if _unreliable_arrivals(samples[step]):
            warnings.append(f"{step} 回传中断（>500 ms 或多次 >150 ms）")
    if max(fit_errors.values()) > 18.0:
        warnings.append("掌屈/拇指侧偏转方向与模型轴偏差超过 18°")
    prior_error = float(np.rad2deg(Rotation.from_matrix(basis @ prior.T).magnitude()))
    checks["manual_mapping_offset"] = prior_error
    if prior_error >= 24.5:
        warnings.append(
            f"候选轴映射达到手动初始映射约束边界 {prior_error:.1f}°（上限 25°），"
            "拟合不可信；回中立不能仅按候选映射判定动作是否规范，勿应用本结果")
    if check_forward > 18.0 or check_neutral > 12.0:
        warnings.append(
            f"相对前臂复测未通过：候选映射下回中立 {check_neutral:.1f}°，"
            f"手动映射下 {manual_check_neutral:.1f}°（上限 12°）；"
            + ("请先检查拟合质量，再判断回中立动作"
               if prior_error >= 24.5 else "请检查手背绑带及回中立姿势"))
    axial_range_ok = (30.0 <= axial_angle <= 85.0 and
                      30.0 <= hand_axial_angle <= 85.0)
    if not axial_range_ok:
        warnings.append(
            f"长轴验证转动不足或过大：前臂 {axial_angle:.1f}°、手背 "
            f"{hand_axial_angle:.1f}°；请伸直手腕，随前臂共同旋转约 40–60° 后重录")
    if axial_range_ok and axial_forearm_z > 25.0:
        warnings.append(
            f"长轴验证时前臂转轴偏离模型长轴 {axial_forearm_z:.1f}°；"
            "请避免抬臂、屈肘或摆动上臂，重录")
    if axial_range_ok and (axial_axis_error > 20.0 or axial_relative > 12.0):
        warnings.append(
            f"长轴验证未通过：手与前臂转轴相差 {axial_axis_error:.1f}°、"
            f"相对转动 {axial_relative:.1f}°；请保持手腕伸直，检查绑带和航向后重录")
    return GuidedResult(
        part="left_hand", device_id=device_id, basis_matrix=basis.tolist(),
        neutral_matrix=r0.tolist(), fit_error_deg=fit_errors,
        check_error_deg=checks, hold_spread_deg=spreads,
        gyro_p90_deg_s=gyros,
        sample_counts={step: len(samples[step]) for step in WRIST_STEPS},
        status="通过" if not warnings else "需重录", warnings=warnings)


def reference_motion_deg(samples: dict[str, list[dict]]) -> dict[str, float]:
    """How much a supposedly fixed parent segment moved from its neutral hold."""
    matrices = {step: _hold(samples[step])[0] for step in STEPS}
    neutral = matrices["neutral"]
    return {step: float(np.rad2deg(Rotation.from_matrix(
        neutral.T @ matrices[step]).magnitude()))
            for step in STEPS if step != "neutral"}


def reference_hold_quality(samples: dict[str, list[dict]]) -> dict[str, tuple[float, float]]:
    """Return parent orientation spread and gyro P90 for every static hold."""
    return {step: _hold(samples[step])[1:] for step in STEPS}


def reference_neutral_matrix(rows: list[dict]) -> np.ndarray:
    """Return the first stationary parent frame, shared with wrist neutral."""
    return _hold(rows)[0]
