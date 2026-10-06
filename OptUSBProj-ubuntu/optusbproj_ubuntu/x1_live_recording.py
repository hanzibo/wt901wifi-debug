"""Synchronous, same-tick diagnostics for the visualization-only X1 preview."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from .x1_live import (JOINT_NAMES, MAX_JOINT_SPEED_DEG_S,
                      MAX_JOINT_STEP_DEG, NOMINAL_TICK_S, PARTS, SIDES,
                      LiveRetargeter)


DEVICE_VALUES = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                 "mag_x", "mag_y", "mag_z", "roll", "pitch", "yaw",
                 "battery_v", "rssi")
DEVICE_FIELDS = ("device_id", "packet_count", "recv_mono_ns", "age_ms",
                 "device_time") + DEVICE_VALUES
POINT_NAMES = ("human_shoulder", "human_elbow", "human_wrist",
               "target_elbow", "target_wrist", "robot_shoulder",
               "robot_elbow", "robot_wrist")
HAND_PARTS = tuple(f"{side}_hand" for side in SIDES)
SIDE_FIELDS = (
    "state", "input_stale", "age_ms", "human_bend_deg", "robot_bend_deg",
    "elbow_error_m", "wrist_error_m", "fit_success", "fit_wrist_error_m",
    "backend", "task_mode", "world_orientation_error_deg", "solver_ms",
    "solver_iterations", "solver_reason", "solver_at_limit",
) + tuple(f"target_pose_{i}{j}" for i in range(4) for j in range(4)) + tuple(
    f"relative_target_{i}{j}" for i in range(3) for j in range(3)
) + tuple(f"{name}_{axis}" for name in POINT_NAMES for axis in "xyz") + (
    "candidate_q0", "candidate_q1", "candidate_q2", "candidate_q3",
    "candidate_q4", "candidate_q5",
)
FRAME_FIELDS = (
    "frame", "tick_mono_ns", "wall_ns", "elapsed_s", "dt_s", "send_seq",
    "ack_seq", "ack_recv_mono_ns", "ros_subscribers",
) + tuple(f"{part}_{name}" for part in (*PARTS, *HAND_PARTS)
          for name in DEVICE_FIELDS) + (
    tuple(f"joint_{name}" for name in JOINT_NAMES)
) + tuple(f"{side}_{name}" for side in SIDES for name in SIDE_FIELDS) + tuple(
    f"{part}_{name}" for part in HAND_PARTS for name in (
        "state", "axial_deg", "orientation_error_deg",
        "target_rotvec_x", "target_rotvec_y", "target_rotvec_z",
        "robot_rotvec_x", "robot_rotvec_y", "robot_rotvec_z"))


class X1DiagnosticRecorder:
    """Write exact pose/IK inputs and accepted output from each GUI solver tick.

    This is a sampled diagnostic trace, not a lossless raw WT901 packet capture.
    Packet count and receive timestamp reveal whether a tick reused old data.
    """

    def __init__(self, parent: Path, retargeter: LiveRetargeter, urdf: Path) -> None:
        self.path = parent / (datetime.now().strftime("%Y%m%d-%H%M%S")
                              + "-" + uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.started_mono_ns = time.monotonic_ns()
        self.started_wall_ns = time.time_ns()
        self.previous_tick_ns: int | None = None
        self.count = 0
        self._closed = False
        self.manifest = {
            "format": "x1-live-diagnostic-v2",
            "purpose": "RViz preview diagnostics only; not robot control commands",
            "started_wall_ns": self.started_wall_ns,
            "started_mono_ns": self.started_mono_ns,
            "imu_sampling": "latest packet snapshot at each IK tick; not lossless raw stream",
            "joint_order": list(JOINT_NAMES),
            "parts": {part: {
                "device_id": retargeter.ids[part],
                "basis_matrix": retargeter.bases[part].tolist(),
                "neutral_matrix": retargeter.neutrals[part].tolist(),
            } for part in PARTS},
            **{f"{side}_hand": ({"enabled": True,
                  "device_id": snapshot["device_id"],
                  "mapping": list(snapshot["mapping"]),
                  "calibrated": snapshot["calibrated"],
                  "algorithm_mode": snapshot["algorithm_mode"],
                  "basis_matrix": snapshot["basis"].tolist(),
                  "neutral_matrix": snapshot["neutral"].tolist()}
                if (snapshot := retargeter.hands.get(side)) is not None
                else {"enabled": False}) for side in SIDES},
            "urdf": str(urdf.resolve()),
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "solver": {"tick_interval_s": NOMINAL_TICK_S,
                       "backend": retargeter.backend, "task_mode": retargeter.task_mode,
                       "scale_mode": retargeter.scale_mode,
                       "length_scales": {side: {"upper": values[0], "forearm": values[1]}
                                         for side, values in retargeter.length_scales.items()},
                       "end_effector": {"left": "L6", "right": "R56"},
                       "pose_axes": {side: retargeter.pose_axes[side].tolist() for side in SIDES},
                       "pose_residual_weights": {"rotation": .30, "translation": 5.0},
                       "max_joint_speed_deg_s": MAX_JOINT_SPEED_DEG_S,
                       "max_step_per_tick_deg": MAX_JOINT_STEP_DEG,
                       "accepted_wrist_error_m": 0.08,
                       "fresh_sample_age_s": retargeter.max_age_s,
                       "wrist_preview_limits_deg": {"axial": [-90, 90],
                                                    "radial": [-45, 45]}},
        }
        self._write_manifest()
        self.file = (self.path / "frames.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FRAME_FIELDS)
        self.writer.writeheader()

    def _write_manifest(self) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    def write_frame(self, retargeter: LiveRetargeter, positions: tuple[float, ...],
                    status: dict, *, tick_mono_ns: int, send_seq: int,
                    ack_seq: int | None, ack_recv_mono_ns: int | None,
                    ros_subscribers: int) -> None:
        if self._closed:
            raise RuntimeError("诊断录制已结束")
        if len(positions) != len(JOINT_NAMES):
            raise ValueError("双臂关节角数量不正确")
        row = {
            "frame": self.count,
            "tick_mono_ns": tick_mono_ns,
            "wall_ns": time.time_ns(),
            "elapsed_s": (tick_mono_ns - self.started_mono_ns) / 1e9,
            "dt_s": (tick_mono_ns - self.previous_tick_ns) / 1e9
                    if self.previous_tick_ns is not None else "",
            "send_seq": send_seq,
            "ack_seq": "" if ack_seq is None else ack_seq,
            "ack_recv_mono_ns": "" if ack_recv_mono_ns is None else ack_recv_mono_ns,
            "ros_subscribers": ros_subscribers,
        }
        engine = retargeter.engine
        for part in (*PARTS, *HAND_PARTS):
            if part in HAND_PARTS:
                snapshot = retargeter.hands.get(part.removesuffix("_hand"))
                device = snapshot["device_id"] if snapshot else None
            else:
                device = retargeter.ids[part]
            if device is None:
                continue
            packet = engine.packets.get(device)
            received_at = engine.received_at.get(device)
            prefix = part + "_"
            row[prefix + "device_id"] = device
            row[prefix + "packet_count"] = engine.counts.get(device, 0)
            row[prefix + "recv_mono_ns"] = (
                round(received_at * 1e9) if received_at is not None else "")
            row[prefix + "age_ms"] = (
                (tick_mono_ns / 1e9 - received_at) * 1e3
                if received_at is not None else "")
            row[prefix + "device_time"] = (
                f"{packet.year:04d}-{packet.month:02d}-{packet.day:02d} "
                f"{packet.hour:02d}:{packet.minute:02d}:{packet.second:02d}."
                f"{packet.millisecond:03d}" if packet is not None else "")
            for name in DEVICE_VALUES:
                row[prefix + name] = getattr(packet, name) if packet is not None else ""
        for name, value in zip(JOINT_NAMES, positions):
            row[f"joint_{name}"] = value
        for side in SIDES:
            item = status[side]
            prefix = side + "_"
            for name in ("state", "input_stale", "human_bend_deg",
                         "robot_bend_deg", "elbow_error_m", "wrist_error_m",
                         "fit_success"):
                row[prefix + name] = item[name]
            row[prefix + "age_ms"] = item["age_s"] * 1e3
            for name in ("backend", "task_mode", "world_orientation_error_deg",
                         "solver_ms", "solver_iterations", "solver_reason", "solver_at_limit"):
                row[prefix + name] = item[name] if item[name] is not None else ""
            for name, matrix in (("target_pose", item["target_pose"]),
                                  ("relative_target", item["hand_target_rotation"])):
                if matrix is not None:
                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            row[f"{prefix}{name}_{i}{j}"] = float(matrix[i, j])
            row[prefix + "fit_wrist_error_m"] = (
                item["fit_wrist_error_m"] if item["fit_wrist_error_m"] is not None else "")
            for name in POINT_NAMES:
                for axis, value in zip("xyz", np.asarray(item[name], dtype=float)):
                    row[f"{prefix}{name}_{axis}"] = float(value)
            candidate = item["candidate_q"]
            for index in range(6):
                row[f"{prefix}candidate_q{index}"] = (
                    float(candidate[index]) if candidate is not None else "")
        for side in SIDES:
            hand = status[side]
            for name in ("state", "axial_deg", "orientation_error_deg"):
                value = hand[f"hand_{name}"]
                row[f"{side}_hand_{name}"] = value if value is not None else ""
            for label, matrix in (("target", hand["hand_target_rotation"]),
                                  ("robot", hand["robot_hand_rotation"])):
                if matrix is not None:
                    vector = Rotation.from_matrix(matrix).as_rotvec()
                    for axis, value in zip("xyz", vector):
                        row[f"{side}_hand_{label}_rotvec_{axis}"] = float(value)
        self.writer.writerow(row)
        self.count += 1
        self.previous_tick_ns = tick_mono_ns
        if self.count % 20 == 0:
            self.file.flush()

    def close(self, reason: str = "stopped") -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.file.flush()
        finally:
            self.file.close()
        self.manifest.update({"finished_wall_ns": time.time_ns(),
                              "frames": self.count, "stop_reason": reason})
        self._write_manifest()
