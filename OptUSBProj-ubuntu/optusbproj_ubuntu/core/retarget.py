"""Neutral-preserving point scaling, independent of robot models and I/O.

Each point triple is ordered (shoulder, elbow, wrist). Human points must use
one consistent unit and frame; robot points use one consistent unit and
frame. Scales include robot-unit / human-unit conversion. X1 callers map
human axes into robot axes before calling retarget_bone_points. Its returned
positions have robot units (metres in production), and bend is in degrees.
No fixed bone-length constraint is imposed. Scene-specific frame mapping and
its existing arithmetic remain in scene_robot_model.position_targets.
"""
import numpy as np


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm < 1e-9 or second_norm < 1e-9:
        raise ValueError("Bone direction has zero length")
    cosine = np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def retarget_scales(human_neutral, robot_neutral, scale_mode="uniform"):
    """Neutral-distance scales, including model-unit conversion (not rigid IK lengths)."""
    if scale_mode not in ("uniform", "segments"):
        raise ValueError("未知人体到机器人长度比例模式")
    hs, he, hw = map(np.asarray, human_neutral)
    rs, re, rw = map(np.asarray, robot_neutral)
    if scale_mode == "uniform":
        denominators = (np.linalg.norm(hw - hs),) * 2
        numerators = (np.linalg.norm(rw - rs),) * 2
    else:
        denominators = (np.linalg.norm(he - hs), np.linalg.norm(hw - he))
        numerators = (np.linalg.norm(re - rs), np.linalg.norm(rw - re))
    if any(not np.isfinite(d) or d < 1e-9 for d in denominators) or any(
            not np.isfinite(n) or n < 1e-9 for n in numerators):
        raise ValueError("中立位参考长度无效，无法计算比例")
    return tuple(float(n / d) for n, d in zip(numerators, denominators))

def retarget_bone_points(
    human_neutral: tuple[np.ndarray, np.ndarray, np.ndarray],
    human_current: tuple[np.ndarray, np.ndarray, np.ndarray],
    robot_neutral: tuple[np.ndarray, np.ndarray, np.ndarray],
    *, scale_mode: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map shoulder-relative joint displacements and measure human elbow bend.

    The URDF shoulder/elbow axes have nonzero intervening offsets, so its
    shoulder-to-elbow joint-center distance is *not* a fixed link length over
    all poses.  Enforcing two fixed robot bone lengths made lateral raising
    substantially worse.  Keep neutral poses aligned, use a common reach
    scale by default, or separate neutral-distance scales for comparison;
    neither option enforces fixed robot bone lengths. FK/IK handles the URDF.
    The anatomical straightness cue is independent of that scale.
    """
    h_shoulder_0, h_elbow_0, h_wrist_0 = human_neutral
    h_shoulder, h_elbow, h_wrist = human_current
    r_shoulder_0, r_elbow_0, r_wrist_0 = robot_neutral
    neutral_upper = h_elbow_0 - h_shoulder_0
    neutral_forearm = h_wrist_0 - h_elbow_0
    current_upper = h_elbow - h_shoulder
    current_forearm = h_wrist - h_elbow
    upper_scale, forearm_scale = retarget_scales(human_neutral, robot_neutral, scale_mode)
    upper_displacement = current_upper - neutral_upper
    forearm_displacement = current_forearm - neutral_forearm
    elbow = r_elbow_0 + upper_scale * upper_displacement
    # Preserve the original arithmetic exactly for the default path.
    wrist = (r_wrist_0 + upper_scale * (upper_displacement + forearm_displacement)
             if scale_mode == "uniform" else
             r_wrist_0 + upper_scale * upper_displacement + forearm_scale * forearm_displacement)
    human_bend_deg = _angle_degrees(current_upper, current_forearm)
    return elbow, wrist, human_bend_deg
