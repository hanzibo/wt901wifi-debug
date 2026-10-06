#!/usr/bin/env python3
"""Replay precomputed right-only or dual-arm WT901 trajectory in RViz only."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

from x1_common import JOINT_NAMES


RIGHT_JOINTS = ("51", "52", "53", "54", "55", "56")
LEFT_JOINTS = ("1", "2", "3", "4", "5", "6")
DEFAULT_TRAJECTORY = (Path(__file__).resolve().parents[2] / "OptUSBProj-ubuntu"
                      / "calibrations" / "20260926-174220-4304e9"
                      / "x1_right_trajectory.csv")
DEFAULT_DUAL_TRAJECTORY = DEFAULT_TRAJECTORY.with_name("x1_dual_trajectory.csv")


def load_trajectory(path: Path) -> tuple[tuple[str, ...], list[tuple[float, tuple[float, ...]]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        if "time_s" not in columns:
            raise ValueError("轨迹缺少时间戳")
        for side, group in (("左", LEFT_JOINTS), ("右", RIGHT_JOINTS)):
            if columns.intersection(group) and not set(group).issubset(columns):
                raise ValueError(f"轨迹的{side}臂六关节列不完整")
        joint_names = tuple(name for group in (LEFT_JOINTS, RIGHT_JOINTS)
                            if set(group).issubset(columns) for name in group)
        if not joint_names:
            raise ValueError("轨迹不含完整的左臂或右臂六关节列")
        frames = []
        previous = -1.0
        for row in reader:
            t = float(row["time_s"])
            q = tuple(float(row[name]) for name in joint_names)
            if not math.isfinite(t) or t < 0 or t <= previous or not all(
                    math.isfinite(value) and abs(value) <= math.pi for value in q):
                raise ValueError("轨迹包含无效或非递增时间/关节角")
            frames.append((t, q))
            previous = t
    if not frames:
        raise ValueError("轨迹为空")
    return joint_names, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path)
    parser.add_argument("--dual", action="store_true", help="使用默认双臂轨迹；不影响右臂默认回放")
    parser.add_argument("--speed", type=float, default=1.0, help="0.1–2.0 倍速")
    parser.add_argument("--dry-run", action="store_true", help="仅检查文件，不导入 ROS")
    args = parser.parse_args()
    if not 0.1 <= args.speed <= 2.0:
        parser.error("--speed 应在 0.1 到 2.0 之间")
    if args.dual and args.trajectory is not None:
        parser.error("--dual 与显式轨迹路径不能同时使用")
    path = args.trajectory or (DEFAULT_DUAL_TRAJECTORY if args.dual else DEFAULT_TRAJECTORY)
    try:
        joint_names, frames = load_trajectory(path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    arms = "双臂" if len(joint_names) == 12 else ("左臂" if joint_names[0] == "1" else "右臂")
    print(f"轨迹：{arms}、{len(frames)} 帧 / {frames[-1][0]:.2f} 秒；"
          "仅向 /joint_states 发布 RViz 预览。")
    if args.dry_run:
        return

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("x1_imu_arm_replay")
    publisher = node.create_publisher(JointState, "/joint_states", 10)
    names = list(JOINT_NAMES)
    positions = [0.0] * len(names)
    indices = [names.index(name) for name in joint_names]

    def publish(q: tuple[float, ...]) -> None:
        for index, value in zip(indices, q):
            positions[index] = value
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.name = names
        message.position = positions.copy()
        publisher.publish(message)

    try:
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("10 秒内没有 /joint_states 订阅者；请先启动 RViz display.launch.py")
            rclpy.spin_once(node, timeout_sec=0.05)
        # Hold the first pose briefly so RViz can settle before playback.
        for _ in range(15):
            publish(frames[0][1])
            rclpy.spin_once(node, timeout_sec=0.033)
        start = time.monotonic()
        for t, q in frames:
            due = start + t / args.speed
            while rclpy.ok() and time.monotonic() < due:
                rclpy.spin_once(node, timeout_sec=min(0.01, due - time.monotonic()))
            if not rclpy.ok():
                break
            publish(q)
        print("回放结束；保持末帧 2 秒。")
        end = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < end:
            publish(frames[-1][1])
            rclpy.spin_once(node, timeout_sec=0.033)
    except KeyboardInterrupt:
        print("已停止回放。")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
