"""Offline, visualization-only arm retargeting from a calibrated WT session.

This deliberately emits joint *states*, not hardware commands.  A three-IMU
session without a hand sensor cannot determine the robot hand's full SE(3)
pose, so the objective uses wrist/elbow positions and keeps wrist joints near
their neutral angles.
"""

from __future__ import annotations

import csv
from dataclasses import fields
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from wt901.protocol import AttitudePacket

from .pose import PoseEngine
from .skeleton import PART_BY_KEY


ARM_JOINTS = {
    "right": ("51", "52", "53", "54", "55", "56"),
    "left": ("1", "2", "3", "4", "5", "6"),
}
RIGHT_JOINTS = ARM_JOINTS["right"]
REQUIRED_PARTS = ("right_upper_arm", "right_forearm")
TRAJECTORY_FIELDS = ("time_s",) + RIGHT_JOINTS + (
    "elbow_target_x", "elbow_target_y", "elbow_target_z",
    "elbow_actual_x", "elbow_actual_y", "elbow_actual_z",
    "wrist_target_x", "wrist_target_y", "wrist_target_z",
    "wrist_actual_x", "wrist_actual_y", "wrist_actual_z",
    "human_bend_deg", "robot_bend_deg",
    "elbow_error_m", "wrist_error_m", "fit_success",
)


def _origin_transform(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    result = np.eye(4)
    if origin is not None:
        xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        result[:3, 3] = xyz
    return result


class ArmKinematics:
    """URDF FK for one arm from base_link_0 to L6/R56, in metres/radians."""

    def __init__(self, urdf: Path, side: str):
        if side not in ARM_JOINTS:
            raise ValueError(f"Unknown arm side: {side}")
        self.side = side
        self.joint_names = ARM_JOINTS[side]
        root = ET.parse(urdf).getroot()
        joints = {item.get("name"): item for item in root.findall("joint")}
        self.origins = []
        self.axes = []
        self.lower_limits = []
        self.upper_limits = []
        parent = "base_link_0"
        for name in self.joint_names:
            joint = joints[name]
            if joint.get("type") not in {"continuous", "revolute"}:
                raise ValueError(f"{name} is not a rotary joint")
            if joint.find("parent").get("link") != parent:
                raise ValueError(f"{name} is not contiguous with {parent}")
            parent = joint.find("child").get("link")
            self.origins.append(_origin_transform(joint))
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            self.axes.append(axis / np.linalg.norm(axis))
            limit = joint.find("limit")
            if joint.get("type") == "revolute":
                if limit is None:
                    raise ValueError(f"{name} has no revolute joint limits")
                self.lower_limits.append(float(limit.get("lower")))
                self.upper_limits.append(float(limit.get("upper")))
            else:
                self.lower_limits.append(-np.inf)
                self.upper_limits.append(np.inf)
        expected_tip = "R56" if side == "right" else "L6"
        if parent != expected_tip:
            raise ValueError(f"Unexpected {side} arm tip: {parent}")

    def frames(self, q: np.ndarray) -> dict[str, np.ndarray]:
        if len(q) != 6:
            raise ValueError("Expected six right arm angles")
        transform = np.eye(4)
        result = {}
        for name, origin, axis, angle in zip(self.joint_names, self.origins, self.axes, q):
            joint_rotation = np.eye(4)
            joint_rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ joint_rotation
            result[name] = transform.copy()
        return result

    def points(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = self.frames(q)
        return tuple(frames[self.joint_names[index]][:3, 3]
                     for index in (0, 3, 5))

    def wrist_relative_rotation(self, q: np.ndarray) -> np.ndarray:
        """L6/R56 orientation relative to L4/R54, not the URDF base frame."""
        frames = self.frames(q)
        forearm = frames[self.joint_names[3]][:3, :3]
        hand = frames[self.joint_names[5]][:3, :3]
        return forearm.T @ hand


class RightArmKinematics(ArmKinematics):
    """Compatibility entry point for the original right-only experiment."""

    def __init__(self, urdf: Path):
        super().__init__(urdf, "right")


def _load_calibration(session: Path, part: str, device_id: str) -> dict:
    path = session / f"guided_result_{part}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("part") != part or data.get("device_id") != device_id:
        raise ValueError(f"{path} does not match this session's assignment")
    if data.get("status") != "通过":
        raise ValueError(f"{path} has not passed calibration")
    if Path(data.get("session", "")).resolve() != session.resolve():
        raise ValueError(f"{path} belongs to a different capture session")
    if "neutral_matrix" not in data or "basis_matrix" not in data:
        raise ValueError(f"{path} lacks the neutral/basis snapshot")
    return data


def load_calibrated_session(
        session: Path, sides: tuple[str, ...] = ("right",)) -> tuple[PoseEngine, dict[str, str]]:
    """Reject uncalibrated or cross-session data instead of guessing axes."""
    session = session.resolve()
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assignments = manifest["assignments_at_start"]
    if not sides or len(set(sides)) != len(sides) or any(
            side not in ARM_JOINTS for side in sides):
        raise ValueError("Expected one or two unique arm sides")
    parts = tuple(f"{side}_{segment}" for side in sides
                  for segment in ("upper_arm", "forearm"))
    engine = PoseEngine({part: assignments[part] for part in parts})
    for part in parts:
        device_id = assignments[part]
        if not device_id:
            raise ValueError(f"No device assigned to {part}")
        calibration = _load_calibration(session, part, device_id)
        engine.apply_calibration(
            part, device_id, np.array(calibration["basis_matrix"]),
            np.array(calibration["neutral_matrix"]),
        )
    return engine, {part: assignments[part] for part in parts}


_PACKET_FIELDS = tuple(item.name for item in fields(AttitudePacket))
_INT_FIELDS = {"year", "month", "day", "hour", "minute", "second", "millisecond", "rssi", "version"}


def _packet_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not {"host_recv_mono_ns", *_PACKET_FIELDS}.issubset(reader.fieldnames or []):
            raise ValueError("Not a WT901 fusion-session IMU CSV")
        for row in reader:
            values = {
                key: row[key] if key == "device_id" else
                int(row[key]) if key in _INT_FIELDS else float(row[key])
                for key in _PACKET_FIELDS
            }
            yield int(row["host_recv_mono_ns"]), AttitudePacket(**values)


def _human_points(engine: PoseEngine, side: str = "right") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transforms = engine.transforms()
    return tuple(transforms[PART_BY_KEY[key]][:3, 3].copy() for key in (
        f"{side}_upper_arm", f"{side}_forearm", f"{side}_hand"))


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


def fit_arm(kinematics: ArmKinematics, target_elbow: np.ndarray,
            target_wrist: np.ndarray, previous_q: np.ndarray,
            *, human_bend_deg: float = 180.0,
            max_step_rad: float = np.deg2rad(9),
            target_wrist_rotation: np.ndarray | None = None,
            diagnostics: dict | None = None
            ) -> tuple[np.ndarray, float, float, bool]:
    """Position IK with an optional soft, forearm-relative wrist target."""
    started = time.perf_counter()
    previous_q = np.asarray(previous_q, dtype=float)
    # These are *visualization-only* conservative bounds, not hardware limits.
    bounds = np.deg2rad(np.array([[-110, 110], [-85, 85], [-90, 90],
                                  [-125, 125], [-90, 90], [-45, 45]], dtype=float)).T
    bounds[0] = np.maximum(bounds[0], kinematics.lower_limits)
    bounds[1] = np.minimum(bounds[1], kinematics.upper_limits)
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("URDF 关节限位与预览范围没有有效交集")
    previous_q = np.clip(previous_q, bounds[0], bounds[1])
    lower = np.maximum(bounds[0], previous_q - max_step_rad)
    upper = np.minimum(bounds[1], previous_q + max_step_rad)
    straight_weight = 0.65 * np.clip((45.0 - human_bend_deg) / 35.0, 0.0, 1.0)
    if target_wrist_rotation is not None:
        target_wrist_rotation = np.asarray(target_wrist_rotation, dtype=float)
        if (target_wrist_rotation.shape != (3, 3) or
                not np.isfinite(target_wrist_rotation).all() or
                not np.allclose(target_wrist_rotation.T @ target_wrist_rotation,
                                np.eye(3), atol=1e-3) or
                not np.isclose(np.linalg.det(target_wrist_rotation), 1.0, atol=1e-3)):
            raise ValueError("机器人手腕目标不是有效旋转矩阵")

    def residual(q):
        _, elbow, wrist = kinematics.points(q)
        base = (
            5.0 * (wrist - target_wrist),
            2.5 * (elbow - target_elbow),
            0.18 * (q - previous_q),
            0.08 * q,
            (0.06 if target_wrist_rotation is not None else 0.5) * q[4:],
            np.array([straight_weight * q[3]]),
        )
        if target_wrist_rotation is None:
            return np.concatenate(base)
        actual = kinematics.wrist_relative_rotation(q)
        orientation_error = Rotation.from_matrix(
            target_wrist_rotation.T @ actual).as_rotvec()
        return np.concatenate((*base, 0.30 * orientation_error))

    result = least_squares(residual, previous_q, bounds=(lower - 1e-9, upper + 1e-9),
                           max_nfev=50, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    q = np.clip(result.x, bounds[0], bounds[1])
    _, elbow, wrist = kinematics.points(q)
    elbow_error = float(np.linalg.norm(elbow - target_elbow))
    wrist_error = float(np.linalg.norm(wrist - target_wrist))
    if diagnostics is not None:
        diagnostics.update(elapsed_ms=(time.perf_counter() - started) * 1000,
                           iterations=result.nfev,
                           reason="收敛" if result.success else "迭代预算耗尽",
                           at_limit=bool(np.any(np.minimum(q - bounds[0], bounds[1] - q)
                                                < np.deg2rad(.5))))
    return q, elbow_error, wrist_error, bool(result.success)


fit_right_arm = fit_arm  # Backward-compatible name used by right-only tests/CLI.


def generate_trajectory(session: Path, urdf: Path, output: Path, *, rate_hz: float = 30.0,
                        max_age_s: float = 0.15) -> dict:
    if rate_hz <= 0 or max_age_s <= 0:
        raise ValueError("rate_hz and max_age_s must be positive")
    engine, assignments = load_calibrated_session(session)
    kinematics = RightArmKinematics(urdf)
    zero_q = np.zeros(6)
    robot_shoulder, robot_elbow, robot_wrist = kinematics.points(zero_q)
    # Human and X1 both use x-forward, y-left, z-up.  IK handles the displaced
    # and non-intersecting URDF axes; human bone directions supply straightness.
    human_neutral = _human_points(engine)
    needed = set(assignments.values())
    last_seen = {}
    first_ns = None
    next_ns = None
    interval_ns = round(1e9 / rate_hz)
    previous_q = zero_q.copy()
    rows = []
    held = 0
    gaps = 0
    for stamp_ns, packet in _packet_rows(session / "imu.csv"):
        if packet.device_id not in needed:
            continue
        if first_ns is None:
            first_ns = stamp_ns
            next_ns = stamp_ns
        # Emit all frames due *before* this sample, using only past packets.
        while next_ns is not None and next_ns < stamp_ns:
            if all(device in last_seen for device in needed):
                stale = any((next_ns - last_seen[device]) / 1e9 > max_age_s
                            for device in needed)
                if stale:
                    gaps += 1
                    q = previous_q.copy()
                    _, elbow, wrist = kinematics.points(q)
                    target_elbow, target_wrist = elbow, wrist
                    human_bend = 0.0
                    elbow_error = wrist_error = 0.0
                    success = False
                else:
                    target_elbow, target_wrist, human_bend = retarget_bone_points(
                        human_neutral, _human_points(engine),
                        (robot_shoulder, robot_elbow, robot_wrist))
                    q, elbow_error, wrist_error, success = fit_right_arm(
                        kinematics, target_elbow, target_wrist, previous_q,
                        human_bend_deg=human_bend)
                    if not success or wrist_error > 0.08:
                        q = previous_q.copy()
                        _, elbow, wrist = kinematics.points(q)
                        elbow_error = float(np.linalg.norm(elbow - target_elbow))
                        wrist_error = float(np.linalg.norm(wrist - target_wrist))
                        held += 1
                    previous_q = q
                shoulder, elbow, wrist = kinematics.points(q)
                robot_bend = _angle_degrees(elbow - shoulder, wrist - elbow)
                rows.append([round((next_ns - first_ns) / 1e9, 6), *q,
                             *target_elbow, *elbow, *target_wrist, *wrist,
                             human_bend, robot_bend, elbow_error, wrist_error,
                             int(success and wrist_error <= 0.08)])
            next_ns += interval_ns
        engine.feed(packet)
        last_seen[packet.device_id] = stamp_ns
    if not rows:
        raise ValueError("Session has insufficient synchronized right-arm packets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(TRAJECTORY_FIELDS)
        writer.writerows(rows)
    errors = np.array([[row[-3], row[-2]] for row in rows], dtype=float)
    summary = {
        "input_session": str(session.resolve()), "output": str(output.resolve()),
        "imu_devices": assignments, "frames": len(rows),
        "duration_s": rows[-1][0], "sample_rate_hz": rate_hz,
        "retarget": "neutral-relative joint displacement + human elbow straightness",
        "held_ik_frames": held,
        "stale_input_frames": gaps,
        "elbow_error_median_m": float(np.median(errors[:, 0])),
        "wrist_error_median_m": float(np.median(errors[:, 1])),
        "wrist_error_p95_m": float(np.percentile(errors[:, 1], 95)),
        "mode": "RViz preview only; no physical-joint safety limits",
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
