#!/usr/bin/env python3
"""Drive one temporary CoppeliaSim revolute joint from one WT901 orientation.

The simulator must already be open with its simulation stopped. This demo
does not start a simulation or touch an existing robot model.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from wt901.gui_udp import GuiUdpSource
from wt901_viz.orientation import rotation_matrix

from optusbproj_ubuntu.axis_mapping import mapping_matrix
from sim_connectivity import _load_client


AXIS_CHOICES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def twist_angle_deg(rotation: np.ndarray, axis: str) -> float:
    """Signed twist around a model axis, not an Euler-angle component.

    Swing-twist decomposition projects the rotation quaternion's vector part
    onto the hinge axis. The result is wrapped to [-180, 180] degrees.
    """
    if axis not in AXIS_CHOICES:
        raise ValueError(f"无效关节轴：{axis}")
    vector = np.zeros(3)
    vector["XYZ".index(axis[1])] = 1.0 if axis[0] == "+" else -1.0
    quaternion = Rotation.from_matrix(rotation).as_quat()  # x, y, z, w
    axial = float(np.dot(quaternion[:3], vector))
    scalar = float(quaternion[3])
    if math.hypot(axial, scalar) < 1e-8:
        raise ValueError("姿态接近绕垂直关节轴的 180° 翻转，扭转角不确定")
    angle = math.degrees(2.0 * math.atan2(axial, scalar))
    return (angle + 180.0) % 360.0 - 180.0


class SingleJointMapper:
    def __init__(self, axis: str = "+Z", mapping: tuple[str, str, str] =
                 ("+X", "+Y", "+Z"), limit_deg: float = 90.0,
                 gain: float = 1.0) -> None:
        if axis not in AXIS_CHOICES:
            raise ValueError(f"无效关节轴：{axis}")
        if not 0 < limit_deg <= 180:
            raise ValueError("角度限制应在 0～180° 之间")
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("增益必须为正数")
        self.axis = axis
        self.basis = mapping_matrix(mapping)
        self.limit_deg = limit_deg
        self.gain = gain
        self.neutral: np.ndarray | None = None

    def zero(self, packet: object) -> None:
        self.neutral = rotation_matrix(packet.roll, packet.pitch, packet.yaw)

    def angle_deg(self, packet: object) -> float:
        current = rotation_matrix(packet.roll, packet.pitch, packet.yaw)
        if self.neutral is None:
            self.neutral = current.copy()
            return 0.0
        delta_sensor = self.neutral.T @ current
        delta_model = self.basis @ delta_sensor @ self.basis.T
        angle = self.gain * twist_angle_deg(delta_model, self.axis)
        return float(np.clip(angle, -self.limit_deg, self.limit_deg))


class TemporaryJoint:
    """One visible, non-dynamic Z-axis hinge in the currently stopped scene."""

    def __init__(self, sim: object) -> None:
        self.sim = sim
        self.handles: list[int] = []

    def __enter__(self) -> "TemporaryJoint":
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError("请先停止 CoppeliaSim 仿真；演示不会修改运行中的场景")
        try:
            joint = sim.createJoint(sim.joint_revolute, sim.jointmode_kinematic,
                                    0, [0.12, 0.07])
            self.handles.append(joint)
            sim.setObjectAlias(joint, "WT901_temporary_revolute_joint")
            sim.setObjectPosition(joint, [0.0, 0.0, 0.5])
            bar = sim.createPrimitiveShape(sim.primitiveshape_cuboid,
                                           [0.5, 0.07, 0.07])
            self.handles.append(bar)
            sim.setObjectAlias(bar, "WT901_temporary_joint_link")
            sim.setObjectParent(bar, joint, False)
            sim.setObjectPosition(bar, [0.25, 0.0, 0.0], joint)
            sim.setJointPosition(joint, 0.0)
            return self
        except Exception:
            self.close()
            raise

    def command_deg(self, angle: float) -> float:
        target = math.radians(angle)
        self.sim.setJointPosition(self.handles[0], target)
        actual = float(self.sim.getJointPosition(self.handles[0]))
        if not math.isclose(actual, target, abs_tol=1e-3):
            raise RuntimeError(f"关节角回读异常：目标 {angle:+.1f}°，实际 {math.degrees(actual):+.1f}°")
        return math.degrees(actual)

    def close(self) -> None:
        for handle in reversed(self.handles):
            self.sim.removeObjects([handle])
        self.handles.clear()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一台 WT901 → CoppeliaSim 临时单自由度旋转关节")
    parser.add_argument("--device-id", required=True, help="准确的 WT 设备 ID，例如 WT5500011361")
    parser.add_argument("--axis", choices=AXIS_CHOICES, default="+Z",
                        help="从 IMU 相对姿态提取的模型关节轴，默认 +Z")
    parser.add_argument("--mapping", default="+X,+Y,+Z",
                        help="设备 X/Y/Z 正轴映射到模型轴，逗号分隔；默认 +X,+Y,+Z")
    parser.add_argument("--limit-deg", type=float, default=90.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="演示时长；0 表示运行到 Ctrl+C")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--coppelia-dir", type=Path,
                        default=Path(os.environ["COPPELIASIM_ROOT"])
                        if "COPPELIASIM_ROOT" in os.environ else None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args(argv)
    if args.seconds < 0:
        parser.error("--seconds 不能为负数")
    try:
        mapping = tuple(part.strip() for part in args.mapping.split(","))
        mapper = SingleJointMapper(args.axis, mapping, args.limit_deg, args.gain)
        client_type = _load_client(args.coppelia_dir)
        client = client_type(host=args.host, port=args.port)
        sim = client.require("sim")
        with GuiUdpSource(port=args.listen_port) as source, TemporaryJoint(sim) as joint:
            print(f"临时关节已建立；等待 {args.device_id} 数据。"
                  f"模型轴 {args.axis}，限制 ±{args.limit_deg:g}°。"
                  "首包自动归零；请保持设备静止直到显示归零消息。", flush=True)
            started = time.monotonic()
            last_packet_at = 0.0
            last_command_at = 0.0
            last_report_at = 0.0
            was_stale = False
            while args.seconds == 0 or time.monotonic() - started < args.seconds:
                packets = [packet for packet in source.poll()
                           if packet.device_id == args.device_id]
                now = time.monotonic()
                if packets:
                    last_packet_at = now
                    was_stale = False
                    packet = packets[-1]
                    if mapper.neutral is None:
                        mapper.zero(packet)
                        print("收到目标设备首包，已设为 0° 中立位。", flush=True)
                    if now - last_command_at >= 0.05:
                        try:
                            command = mapper.angle_deg(packet)
                        except ValueError as exc:
                            print(f"跳过不确定姿态：{exc}", flush=True)
                        else:
                            actual = joint.command_deg(command)
                            last_command_at = now
                            if now - last_report_at >= 1.0:
                                print(f"关节目标/回读 {command:+.1f}°/{actual:+.1f}°",
                                      flush=True)
                                last_report_at = now
                elif last_packet_at and now - last_packet_at > 0.5 and not was_stale:
                    print("目标设备已断流 >0.5 秒；停止发送新关节角，保留末次位置。",
                          flush=True)
                    was_stale = True
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("用户结束演示。", flush=True)
    except Exception as exc:
        parser.exit(1, f"单关节演示失败：{exc}\n")
    print("演示结束；临时关节和连杆已删除，场景未保存。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
