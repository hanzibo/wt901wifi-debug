"""Live, RViz-only bilateral retargeting from the viewer's calibrated pose.

The GUI owns the WT UDP socket.  This module never reads a second device
stream, changes a calibration, or sends a hardware command.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .axis_mapping import mapping_matrix
from .pose import PoseEngine
from .skeleton import PART_BY_KEY
from .x1_kinematics import PoEArmKinematics
from .x1_tasks import ArmTask
from .x1_ik import solve
from .x1_retarget import (ARM_JOINTS, ArmKinematics, _angle_degrees,
                          _human_points, fit_arm,
                          retarget_bone_points, retarget_scales)


SIDES = ("left", "right")
PARTS = tuple(f"{side}_{segment}" for side in SIDES
              for segment in ("upper_arm", "forearm"))
JOINT_NAMES = ARM_JOINTS["left"] + ARM_JOINTS["right"]
PREVIEW_PORT = 13990
NOMINAL_TICK_S = 0.05
MAX_JOINT_SPEED_DEG_S = 120.0
MAX_JOINT_STEP_DEG = 9.0


def readiness(engine: PoseEngine, *, now: float | None = None,
              max_age_s: float = 0.25,
              follow_left_hand: bool = False,
              follow_right_hand: bool = False,
              task_mode: str = "mimic") -> list[str]:
    """List the reasons a live calibrated bilateral preview cannot start."""
    moment = time.monotonic() if now is None else now
    problems = []
    if task_mode not in ("mimic", "pose"):
        problems.append("未知 X1 任务模式")
    if task_mode == "pose" and not (follow_left_hand and follow_right_hand):
        problems.append("完整末端位姿模式需同时启用左右手掌跟随，并提供两台在线手背设备")
    ids = [engine.assignments.get(part, "") for part in PARTS]
    if any(not device for device in ids) or len(set(ids)) != len(ids):
        problems.append("左右上臂和前臂须各分配一台不同设备")
    for part, device in zip(PARTS, ids):
        if not device:
            continue
        if part not in engine.continuous_basis or device not in engine.neutral:
            problems.append(f"{part} 未应用本次穿戴标定")
        age = moment - engine.received_at.get(device, -1e9)
        if age > max_age_s or age < -0.1:
            problems.append(f"{part} 数据不新鲜（{max(0.0, age):.1f} 秒）")
    for side, enabled in (("left", follow_left_hand), ("right", follow_right_hand)):
        if not enabled:
            continue
        label = "左" if side == "left" else "右"
        device = engine.assignments.get(f"{side}_hand", "")
        if not device or device in ids:
            problems.append(f"{label}手掌朝向跟随需要另一台已分配的手背设备")
        elif device not in engine.neutral:
            problems.append(f"{label}手背尚无中立位；请摆正手掌并归零")
        elif (moment - engine.received_at.get(device, -1e9) > max_age_s or
              moment - engine.received_at.get(device, -1e9) < -0.1 or
              device not in engine.current):
            problems.append(f"{label}手背数据不新鲜；请检查连接")
        if device:
            ids.append(device)
    return problems


class LiveRetargeter:
    """The same neutral displacement and warm-start IK as offline replay."""

    def __init__(self, engine: PoseEngine, urdf: Path,
                 *, max_age_s: float = 0.25,
                 follow_left_hand: bool = False,
                 follow_right_hand: bool = False,
                 backend: str = "least_squares",
                 task_mode: str = "mimic", scale_mode: str = "uniform") -> None:
        if scale_mode not in ("uniform", "segments"):
            raise ValueError("未知人体到机器人长度比例模式")
        if backend not in ("least_squares", "jacobian"):
            raise ValueError("未知 X1 IK 后端")
        problems = readiness(engine, max_age_s=max_age_s,
                             follow_left_hand=follow_left_hand,
                             follow_right_hand=follow_right_hand,
                             task_mode=task_mode)
        if problems:
            raise ValueError("；".join(problems))
        self.engine = engine
        self.backend = backend
        self.task_mode = task_mode
        self.scale_mode = scale_mode
        self.max_age_s = max_age_s
        self.follow_left_hand = follow_left_hand
        self.follow_right_hand = follow_right_hand
        self.ids = {part: engine.assignments[part] for part in PARTS}
        self.algorithm_modes = {part: engine.algorithm_modes[part] for part in PARTS}
        self.bases = {part: engine.continuous_basis[part].copy() for part in PARTS}
        self.neutrals = {part: engine.neutral[device].copy()
                         for part, device in self.ids.items()}
        self.models = {side: PoEArmKinematics(urdf, side) for side in SIDES}
        self.robot_zero = {side: self.models[side].points(np.zeros(6)) for side in SIDES}
        self.robot_wrist_zero = {
            side: self.models[side].wrist_relative_rotation(np.zeros(6))
            for side in SIDES}
        self.hand_to_robot_axes_by_side = {
            side: self._wrist_axis_alignment(side) for side in SIDES}
        self.hand_to_robot_axes = self.hand_to_robot_axes_by_side["left"]
        self.pose_axes = {}
        self.robot_hand_zero = {}
        self.previous_pose = {}
        for side in SIDES:
            model = self.models[side]
            frames = model.frames(np.zeros(6))
            home = frames[model.joint_names[5]]
            # Position retargeting already uses aligned STL/URDF base axes:
            # +X forward, +Y left, +Z up. Absolute orientation must use the
            # SAME conversion on both arms, not mirror-local wrist DOF signs.
            self.pose_axes[side] = np.eye(3)
            self.robot_hand_zero[side] = home[:3, :3].copy()
            self.previous_pose[side] = home.copy()
        # PoseEngine without live samples gives the STL model's genuine rest
        # geometry.  Starting the preview in a raised pose must not redefine it.
        rest = PoseEngine()
        self.human_zero = {side: _human_points(rest, side) for side in SIDES}
        self.length_scales = {side: retarget_scales(
            self.human_zero[side], self.robot_zero[side], scale_mode) for side in SIDES}
        self.previous = {side: np.zeros(6) for side in SIDES}
        self.last_step_at: float | None = None
        self.hands = {}
        for side, enabled in (("left", follow_left_hand), ("right", follow_right_hand)):
            if not enabled:
                continue
            part = f"{side}_hand"
            device = engine.assignments[part]
            self.hands[side] = {
                "device_id": device, "algorithm_mode": engine.algorithm_modes[part],
                "mapping": engine.axis_mappings[part],
                "calibrated": part in engine.continuous_basis,
                "basis": engine.continuous_basis.get(
                    part, mapping_matrix(engine.axis_mappings[part])).copy(),
                "neutral": engine.neutral[device].copy(),
            }
        # Preserve the original left-only API for existing diagnostic callers.
        left = self.hands.get("left", {})
        self.hand_id = left.get("device_id")
        self.hand_algorithm_mode = left.get("algorithm_mode")
        self.hand_basis = left.get("basis")
        self.hand_mapping = left.get("mapping")
        self.hand_neutral = left.get("neutral")

    def calibration_changed(self) -> bool:
        arm_changed = any(
            self.engine.assignments.get(part) != device or
            self.engine.algorithm_modes[part] != self.algorithm_modes[part] or
            part not in self.engine.continuous_basis or
            device not in self.engine.neutral or
            not np.allclose(self.engine.continuous_basis[part], self.bases[part]) or
            not np.allclose(self.engine.neutral[device], self.neutrals[part])
            for part, device in self.ids.items()
        )
        for side, snapshot in self.hands.items():
            part = f"{side}_hand"
            device = snapshot["device_id"]
            basis = self.engine.continuous_basis.get(
                part, mapping_matrix(self.engine.axis_mappings[part]))
            if (self.engine.assignments.get(part) != device or
                    self.engine.algorithm_modes[part] != snapshot["algorithm_mode"] or
                    self.engine.axis_mappings[part] != snapshot["mapping"] or
                    (part in self.engine.continuous_basis) != snapshot["calibrated"] or
                    not np.allclose(basis, snapshot["basis"]) or
                    device not in self.engine.neutral or
                    not np.allclose(self.engine.neutral[device], snapshot["neutral"])):
                return True
        return arm_changed

    def left_hand_target(self, human_relative: np.ndarray | None = None) -> np.ndarray:
        return self.hand_target("left", human_relative)

    def hand_target(self, side: str,
                    human_relative: np.ndarray | None = None) -> np.ndarray:
        """Map neutral-relative human wrist motion into X1 forearm axes."""
        if human_relative is None:
            human_relative = self.hand_relative_rotation(side)
        axes = self.hand_to_robot_axes_by_side[side]
        return axes @ human_relative @ axes.T @ self.robot_wrist_zero[side]

    def left_hand_relative_rotation(self) -> np.ndarray:
        return self.hand_relative_rotation("left")

    def absolute_hand_target(self, side: str, wrist: np.ndarray) -> np.ndarray:
        """Fixed neutral-frame conversion, never based on the IK's own forearm."""
        hand = self.engine.transforms(include_natural=False)[PART_BY_KEY[f"{side}_hand"]][:3, :3]
        C = self.pose_axes[side]
        target = np.eye(4)
        target[:3, :3] = C @ hand @ C.T @ self.robot_hand_zero[side]
        target[:3, 3] = wrist
        return target

    def hand_relative_rotation(self, side: str) -> np.ndarray:
        """Hand orientation relative to the calibrated forearm model frame."""
        transforms = self.engine.transforms(include_natural=False)
        forearm = transforms[PART_BY_KEY[f"{side}_forearm"]][:3, :3]
        hand = transforms[PART_BY_KEY[f"{side}_hand"]][:3, :3]
        return forearm.T @ hand

    def left_hand_axial_degrees(self,
                                human_relative: np.ndarray | None = None) -> float:
        return self.hand_axial_degrees("left", human_relative)

    def hand_axial_degrees(self, side: str,
                           human_relative: np.ndarray | None = None) -> float:
        """Signed SO(3) twist about the hand model's neutral +Z long axis."""
        if human_relative is None:
            human_relative = self.hand_relative_rotation(side)
        _, _, z, w = Rotation.from_matrix(human_relative).as_quat()
        return float((np.rad2deg(2.0 * np.arctan2(z, w)) + 180.0) % 360.0 - 180.0)

    def _wrist_axis_alignment(self, side: str) -> np.ndarray:
        """Match model spatial axes, accounting for URDF wrist joint signs."""
        model = self.models[side]
        zero = self.robot_wrist_zero[side]
        axes = []
        for index in (4, 5):
            q = np.zeros(6)
            q[index] = 0.01
            direction = Rotation.from_matrix(
                model.wrist_relative_rotation(q) @ zero.T).as_rotvec()
            axes.append(direction / np.linalg.norm(direction))
        # Axial joints are 5/55; transverse joints are 6/56. Their positive
        # axial directions are opposite in the URDF: joint 5 points toward
        # the elbow, joint 55 away from it. Model +Z points toward the elbow
        # on BOTH hands, so right axial motion must use negative joint 55.
        # Both transverse joints point along world -X at neutral. Therefore
        # model +X maps to NEGATIVE joint 6/56 on both sides. Thumb deviation
        # itself is left -X (positive joint 6), right +X (negative joint 56).
        # The remaining Y flexion axis is
        # not independently observable by these two robot wrist joints.
        x = -axes[1]
        axial_axis = (1.0 if side == "left" else -1.0) * axes[0]
        z = axial_axis - np.dot(axial_axis, x) * x
        if np.linalg.norm(z) < 0.1:
            raise ValueError(f"X1 {side} 两个腕关节轴无法区分，不能映射手掌")
        z /= np.linalg.norm(z)
        return np.column_stack((x, np.cross(z, x), z))

    def accept_published(self, positions) -> None:
        """Synchronize IK start with the bridge's validated output, not ACK."""
        values = np.asarray(positions, dtype=float)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError('无效安全接受角度')
        for i, side in enumerate(SIDES):
            self.previous[side] = values[6*i:6*i+6].copy()

    def step(self, *, now: float | None = None, commit: bool = True) -> tuple[tuple[float, ...], dict]:
        if self.calibration_changed():
            raise ValueError("设备分配、穿戴标定或中立位已改变；请停止并重新启动 X1 预览")
        moment = time.monotonic() if now is None else now
        if not np.isfinite(moment):
            raise ValueError("X1 预览时钟无效")
        dt_s = (NOMINAL_TICK_S if self.last_step_at is None else
                max(0.0, moment - self.last_step_at))
        # A long GUI stall must not accumulate a large one-frame joint jump.
        max_step_rad = np.deg2rad(min(MAX_JOINT_STEP_DEG,
                                      MAX_JOINT_SPEED_DEG_S * dt_s))
        self.last_step_at = moment
        status = {}
        positions = []
        for side in SIDES:
            ids = (self.ids[f"{side}_upper_arm"], self.ids[f"{side}_forearm"])
            age = max(moment - self.engine.received_at.get(device, -1e9)
                      for device in ids)
            if age > 1.0 or age < -0.1:
                raise ValueError(f"{side} 臂设备持续断流；X1 预览已停止")
            human_shoulder, human_elbow, human_wrist = _human_points(self.engine, side)
            target_elbow, target_wrist, bend = retarget_bone_points(
                self.human_zero[side], (human_shoulder, human_elbow, human_wrist),
                self.robot_zero[side], scale_mode=self.scale_mode)
            candidate = None
            fit_success = False
            fit_wrist_error = None
            hand_target = None
            hand_age = None
            hand_state = "关闭"
            hand_axial_deg = None
            pose_target = None
            ik_result = None
            legacy_diagnostics = {}
            hand_snapshot = self.hands.get(side)
            if hand_snapshot is not None:
                hand_id = hand_snapshot["device_id"]
                hand_age = moment - self.engine.received_at.get(hand_id, -1e9)
                if (hand_age <= self.max_age_s and hand_age >= -0.1 and
                        hand_id in self.engine.current):
                    human_relative = self.hand_relative_rotation(side)
                    hand_target = self.hand_target(side, human_relative)
                    hand_axial_deg = self.hand_axial_degrees(side, human_relative)
                    hand_state = "跟随"
                    pose_target = self.absolute_hand_target(side, target_wrist)
                    self.previous_pose[side] = pose_target.copy()
                else:
                    # A stale hand must not freeze the entire arm. Keep its
                    # last robot wrist orientation while position IK continues.
                    hand_target = self.models[side].wrist_relative_rotation(
                        self.previous[side])
                    hand_state = "手背断流，保持腕部"
                    if self.task_mode == "pose":
                        pose_target = self.previous_pose[side].copy()
                        if hand_age > 1.0 or hand_age < -0.1:
                            raise ValueError(f"{side} 手背持续断流；完整位姿预览已停止")
            if age > self.max_age_s or (self.task_mode == "pose" and hand_state != "跟随"):
                q = self.previous[side]
                state = "短时断流，保持该侧"
            else:
                fit_kwargs = {"human_bend_deg": bend, "max_step_rad": max_step_rad}
                if hand_target is not None:
                    fit_kwargs["target_wrist_rotation"] = hand_target
                if self.backend == "least_squares" and self.task_mode == "mimic":
                    fitted, _, error, success = fit_arm(
                        self.models[side], target_elbow, target_wrist,
                        self.previous[side], diagnostics=legacy_diagnostics, **fit_kwargs)
                else:
                    task = ArmTask(self.models[side], target_elbow, target_wrist,
                                   self.previous[side], bend, hand_target,
                                   pose_target if self.task_mode == "pose" else None)
                    ik_result = solve(task, self.backend, max_step_rad)
                    fitted, error, success = (ik_result.q, ik_result.position_error_m,
                                              ik_result.success)
                candidate = fitted
                fit_success = success
                fit_wrist_error = error
                if success and (error <= 0.08 or self.task_mode == "pose"):
                    q = fitted
                    if commit:
                        self.previous[side] = q
                    state = "跟随"
                else:
                    q = self.previous[side]
                    state = "IK 未收敛，保持该侧"
            robot_shoulder, robot_elbow, robot_wrist = self.models[side].points(q)
            actual_hand = (self.models[side].wrist_relative_rotation(q)
                           if hand_target is not None else None)
            hand_error = (float(np.rad2deg(Rotation.from_matrix(
                hand_target.T @ actual_hand).magnitude()))
                if hand_target is not None else None)
            world_error = None
            if pose_target is not None:
                actual_pose = self.models[side].frames(q)[self.models[side].joint_names[5]]
                world_error = float(np.rad2deg(Rotation.from_matrix(
                    pose_target[:3, :3].T @ actual_pose[:3, :3]).magnitude()))
                if self.task_mode == "pose" and state == "跟随" and (
                        np.linalg.norm(actual_pose[:3, 3] - pose_target[:3, 3]) > .005 or
                        world_error > 2.):
                    state = "跟随（目标尚未达到）"
            status[side] = {
                "state": state, "age_s": age, "input_stale": age > self.max_age_s,
                "human_shoulder": human_shoulder,
                "human_elbow": human_elbow, "human_wrist": human_wrist,
                "target_elbow": target_elbow, "target_wrist": target_wrist,
                "robot_shoulder": robot_shoulder,
                "robot_elbow": robot_elbow, "robot_wrist": robot_wrist,
                "human_bend_deg": bend,
                "robot_bend_deg": _angle_degrees(robot_elbow - robot_shoulder,
                                                   robot_wrist - robot_elbow),
                "elbow_error_m": float(np.linalg.norm(robot_elbow - target_elbow)),
                "wrist_error_m": float(np.linalg.norm(robot_wrist - target_wrist)),
                "fit_success": fit_success,
                "fit_wrist_error_m": fit_wrist_error,
                "candidate_q": candidate,
                "hand_state": hand_state, "hand_age_s": hand_age,
                "hand_axial_deg": hand_axial_deg,
                "hand_target_rotation": hand_target,
                "robot_hand_rotation": actual_hand,
                "hand_orientation_error_deg": hand_error,
                "backend": self.backend, "task_mode": self.task_mode,
                "scale_mode": self.scale_mode, "length_scales": self.length_scales[side],
                "target_pose": pose_target,
                "world_orientation_error_deg": world_error,
                "solver_ms": ik_result.elapsed_ms if ik_result else legacy_diagnostics.get("elapsed_ms"),
                "solver_iterations": ik_result.iterations if ik_result else legacy_diagnostics.get("iterations"),
                "solver_reason": ik_result.reason if ik_result else legacy_diagnostics.get("reason", "未求解"),
                "solver_at_limit": ik_result.at_limit if ik_result else legacy_diagnostics.get("at_limit"),
            }
            positions.extend(float(value) for value in q)
        return tuple(positions), status


class PreviewSender:
    """Loopback-only, nonblocking transport to the separate ROS environment."""

    def __init__(self, port: int = PREVIEW_PORT) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.setblocking(False)
        self.target = ("127.0.0.1", port)
        self.sequence = 0
        self.last_ack_at: float | None = None
        self.last_ack_seq: int | None = None
        self.last_ros_recv_mono_ns: int | None = None
        self.subscribers = 0
        self.safety = None
        self.safety_revision = -1

    def send(self, positions: tuple[float, ...]) -> None:
        self.sequence += 1
        payload = {"version": 1, "type": "frame", "seq": self.sequence,
                   "sent_mono_ns": time.monotonic_ns(),
                   "joint_names": JOINT_NAMES, "positions": positions}
        self.socket.sendto(json.dumps(payload).encode("utf-8"), self.target)

    def poll_ack(self) -> None:
        while True:
            try:
                packet, peer = self.socket.recvfrom(8192)
            except BlockingIOError:
                return
            if peer != self.target:
                continue
            try:
                reply = json.loads(packet)
                if reply.get('version') == 1 and reply.get('type') == 'safety_status':
                    data = reply.get('safety', {})
                    revision = data.get('revision')
                    if (type(revision) is int and revision >= self.safety_revision and
                            type(reply.get('sent_mono_ns')) is int and
                            abs(time.monotonic_ns()-reply['sent_mono_ns']) < 500_000_000):
                        self.safety, self.safety_revision = data, revision
                    continue
                if reply.get("version") != 1 or reply.get("type") != "ack":
                    continue
                self.subscribers = int(reply.get("subscribers", 0))
                sequence = reply.get("seq")
                if type(sequence) is not int or sequence < 1 or sequence > self.sequence:
                    continue
                self.last_ack_seq = sequence
                stamp = reply.get("recv_mono_ns")
                self.last_ros_recv_mono_ns = stamp if type(stamp) is int else None
                self.last_ack_at = time.monotonic()
            except (ValueError, TypeError, AttributeError):
                continue

    def close(self) -> None:
        self.socket.close()
