#!/usr/bin/env python3
"""Publish calibrated X1 bilateral joint *states* to RViz; never command hardware.

The WT901/IK GUI runs in OptUSBProj-fusion.  This ROS 2 process runs in
x1_ros2_py and accepts only short-lived, validated loopback UDP frames.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time

from x1_common import JOINT_NAMES


LEFT = ("1", "2", "3", "4", "5", "6")
RIGHT = ("51", "52", "53", "54", "55", "56")
ARMS = LEFT + RIGHT
DEFAULT_PORT = 13990


def parse_frame(raw: bytes, *, now_ns: int | None = None) -> tuple[int, tuple[float, ...]]:
    """Reject malformed, stale or non-bilateral messages before ROS publishing."""
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get("version") != 1 or
            payload.get("type") != "frame" or
            tuple(payload.get("joint_names", ())) != ARMS):
        raise ValueError("无效的 X1 双臂预览协议或关节列表")
    seq = payload.get("seq")
    timestamp = payload.get("sent_mono_ns")
    if type(seq) is not int or seq <= 0 or type(timestamp) is not int:
        raise ValueError("无效的帧序号或时间戳")
    moment = time.monotonic_ns() if now_ns is None else now_ns
    age_ns = moment - timestamp
    if age_ns < -100_000_000 or age_ns > 500_000_000:
        raise ValueError("已过期或时间异常的预览帧")
    values = payload.get("positions")
    if not isinstance(values, (list, tuple)) or len(values) != len(ARMS):
        raise ValueError("双臂关节角数量不正确")
    if any(type(value) not in (int, float) or not math.isfinite(value) or
           abs(value) > math.pi for value in values):
        raise ValueError("关节角含非有限值或超出预览范围")
    return seq, tuple(float(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口必须在 1024–65535 之间")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", args.port))
    receiver.setblocking(False)
    rclpy.init()
    node = Node("x1_imu_live_preview")
    publisher = node.create_publisher(JointState, "/joint_states", 10)
    names = list(JOINT_NAMES)
    positions = [0.0] * len(names)
    indices = tuple(names.index(name) for name in ARMS)
    current: tuple[float, ...] | None = None
    last_received = 0.0
    last_sequence = 0
    peer_address = None

    def publish_latest() -> None:
        if current is None or time.monotonic() - last_received > 0.3:
            return  # A stopped or disconnected GUI must not keep publishing.
        for index, value in zip(indices, current):
            positions[index] = value
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.name = names
        message.position = positions.copy()
        publisher.publish(message)

    node.create_timer(1 / 30, publish_latest)
    print(f"仅 RViz 预览：监听 127.0.0.1:{args.port}；不连接 X1 硬件。", flush=True)
    try:
        while rclpy.ok():
            while True:
                try:
                    raw, peer = receiver.recvfrom(8192)
                except BlockingIOError:
                    break
                try:
                    sequence, angles = parse_frame(raw)
                except (ValueError, TypeError, UnicodeDecodeError):
                    continue
                if peer != peer_address:
                    last_sequence = 0
                    peer_address = peer
                if sequence <= last_sequence:
                    continue
                last_sequence = sequence
                current = angles
                last_received = time.monotonic()
                ack = {"version": 1, "type": "ack", "seq": sequence,
                       "recv_mono_ns": time.monotonic_ns(),
                       "subscribers": publisher.get_subscription_count()}
                receiver.sendto(json.dumps(ack).encode("utf-8"), peer)
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        print("实时预览结束；RViz 保持末次姿态。", flush=True)
    finally:
        receiver.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
