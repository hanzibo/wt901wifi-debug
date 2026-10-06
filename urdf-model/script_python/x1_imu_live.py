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
GRIPPER_PORT = 13991


def parse_gripper(raw: bytes, *, now_ns=None):
    payload = json.loads(raw)
    moment = time.monotonic_ns() if now_ns is None else now_ns
    if (not isinstance(payload, dict) or payload.get('version') != 1 or
            payload.get('type') != 'gripper' or payload.get('joint') not in ('7', '57')):
        raise ValueError('无效夹爪协议')
    seq, stamp, angle = (payload.get(key) for key in ('seq', 'sent_mono_ns', 'position'))
    if (type(seq) is not int or seq <= 0 or type(stamp) is not int or
            not -100_000_000 <= moment-stamp <= 300_000_000 or
            type(angle) not in (int, float) or not math.isfinite(angle) or not 0 <= angle <= 1):
        raise ValueError('过期或无效夹爪数据')
    return seq, payload['joint'], float(angle)


class GripperPreview:
    """Hold independently of IK; enforce URDF 1 rad/s and fresh input."""
    def __init__(self):
        self.positions = {'7': 0., '57': 0.}
        self.targets = self.positions.copy()
        self.received = {'7': -1e9, '57': -1e9}
        self.sequences = {}
        self.updated = None

    def accept(self, raw, peer, now):
        seq, joint, angle = parse_gripper(raw, now_ns=round(now*1e9))
        if seq <= self.sequences.get(peer, 0):
            return False
        self.sequences[peer] = seq
        self.targets[joint] = angle
        self.received[joint] = now
        return True

    def step(self, now):
        dt = 0. if self.updated is None else max(0., min(.1, now-self.updated))
        self.updated = now
        fresh = []
        for joint in self.positions:
            if 0 <= now-self.received[joint] <= .3:
                error = self.targets[joint]-self.positions[joint]
                self.positions[joint] += max(-dt, min(dt, error))
                fresh.append(joint)
        return fresh


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
    parser.add_argument('--gripper-port', type=int, default=GRIPPER_PORT)
    args = parser.parse_args()
    if (not 1024 <= args.port <= 65535 or not 1024 <= args.gripper_port <= 65535 or
            args.port == args.gripper_port):
        parser.error("端口必须在 1024–65535 之间")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", args.port))
    receiver.setblocking(False)
    gripper_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        gripper_receiver.bind(('127.0.0.1', args.gripper_port))
    except OSError:
        receiver.close()
        gripper_receiver.close()
        raise
    gripper_receiver.setblocking(False)
    grippers = GripperPreview()
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
    published_count = 0

    def gripper_status():
        now = time.monotonic()
        return dict(version=1, type='gripper_status', sent_mono_ns=time.monotonic_ns(),
                    joints={j: dict(position=grippers.positions[j], target=grippers.targets[j],
                                    age_s=None if grippers.received[j] < 0 else now-grippers.received[j])
                            for j in grippers.positions},
                    subscribers=publisher.get_subscription_count(), published_count=published_count,
                    arms_fresh=current is not None and now-last_received <= .3)

    def publish_latest() -> None:
        nonlocal published_count
        now = time.monotonic()
        fresh_grippers = grippers.step(now)
        arms_fresh = current is not None and now-last_received <= .3
        if not arms_fresh and not fresh_grippers:
            return
        if arms_fresh:
            for index, value in zip(indices, current):
                positions[index] = value
        for joint, angle in grippers.positions.items():
            positions[names.index(joint)] = angle
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        # Refresh the complete TF chain even for gripper-only preview. Arm
        # positions are HELD, never extrapolated or treated as fresh IK input.
        message.name = names
        message.position = positions.copy()
        publisher.publish(message)
        published_count += 1

    node.create_timer(1 / 30, publish_latest)
    print(f"仅 RViz 预览：双臂 127.0.0.1:{args.port}；手机夹爪 {args.gripper_port}；不连接 X1 硬件。", flush=True)
    try:
        while rclpy.ok():
            while True:
                try:
                    raw, peer = gripper_receiver.recvfrom(2048)
                except BlockingIOError:
                    break
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and payload.get('type') == 'gripper_status_query':
                        stamp = payload.get('sent_mono_ns')
                        if (payload.get('version') == 1 and type(stamp) is int and
                                abs(time.monotonic_ns()-stamp) < 500_000_000):
                            gripper_receiver.sendto(json.dumps(gripper_status()).encode(), peer)
                    elif grippers.accept(raw, peer, time.monotonic()):
                        gripper_receiver.sendto(json.dumps(gripper_status()).encode(), peer)
                except (ValueError, TypeError, UnicodeDecodeError):
                    continue
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
        gripper_receiver.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
