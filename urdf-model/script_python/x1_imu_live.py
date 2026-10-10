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
import signal
import threading
import time
from pathlib import Path

from x1_common import JOINT_NAMES
from x1_collision_gate import CollisionGate
from x1_angle_log import AngleLogger


LEFT = ("1", "2", "3", "4", "5", "6")
RIGHT = ("51", "52", "53", "54", "55", "56")
ARMS = LEFT + RIGHT
DEFAULT_PORT = 13990
GRIPPER_PORT = 13991


def limited_step(start, goal, dt):
    """Common fraction preserves the certified straight joint-space path."""
    dt = max(0., min(float(dt), .075))
    alpha = 1.
    for name in start:
        delta = abs(goal[name]-start[name])
        limit = min(math.radians(9), math.radians(120)*dt) if name in ARMS else dt
        if delta > 0:
            alpha = min(alpha, limit/delta)
    return {name: value+alpha*(goal[name]-value) for name,value in start.items()}, alpha


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
        self.owners = {}
        self.updated = None

    def accept(self, raw, peer, now):
        seq, joint, angle = parse_gripper(raw, now_ns=round(now*1e9))
        if seq <= self.sequences.get(peer, 0):
            return False
        # Do not let a phone and IMU popup fight over the same gripper.
        if self.owners.get(joint, peer) != peer and 0 <= now-self.received[joint] <= .3:
            return False
        self.sequences[peer] = seq
        self.owners[joint] = peer
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
    parser.add_argument('--collision-python', default=None,
                        help='Python executable for isolated optional collision worker')
    parser.add_argument('--collision-start-enabled', action='store_true',
                        help='Start fail-closed BEFORE accepting any gripper/arm input')
    parser.add_argument('--angle-log-root', default=str(Path(__file__).resolve().parents[2] /
        'OptUSBProj-ubuntu/recordings/x1_bridge'))
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
    angle_log = AngleLogger(args.angle_log_root)
    log_request = None
    positions = [0.0] * len(names)
    indices = tuple(names.index(name) for name in ARMS)
    current: tuple[float, ...] | None = None
    last_received = 0.0
    last_sequence = 0
    peer_address = None
    published_count = 0
    last_publish_at = time.monotonic()
    gate = CollisionGate(args.collision_python)
    if args.collision_start_enabled:
        gate.configure(True, dict(zip(names, positions)))
    configuration_id = None
    configuration_peer = None
    feedback_seq = 0

    def safety_status():
        return dict(gate.status(), configuration_id=configuration_id,
                    published_count=published_count, accepted_arm_seq=feedback_seq,
                    output_speed_limit=True)

    def configure(raw, peer, sock):
        nonlocal configuration_id, configuration_peer, log_request
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get('type') == 'angle_log_config':
            stamp, request_id = payload.get('sent_mono_ns'), payload.get('request_id')
            if (payload.get('version') != 1 or type(stamp) is not int or
                    abs(time.monotonic_ns()-stamp) > 500_000_000 or
                    type(payload.get('enabled')) is not bool or
                    not isinstance(request_id, str) or not 1 <= len(request_id) <= 100):
                raise ValueError('Invalid angle log configuration')
            if log_request != (peer, request_id):
                if payload['enabled']:
                    angle_log.start(names)
                else:
                    angle_log.close()
                log_request = (peer, request_id)
            sock.sendto(json.dumps(dict(version=1, type='angle_log_config_ack',
                request_id=request_id, angle_log=angle_log.status())).encode(), peer)
            return True
        if not isinstance(payload, dict) or payload.get('type') != 'collision_config':
            return False
        stamp, request_id = payload.get('sent_mono_ns'), payload.get('request_id')
        if (payload.get('version') != 1 or type(stamp) is not int or
                abs(time.monotonic_ns()-stamp) > 500_000_000 or
                type(payload.get('enabled')) is not bool or
                not isinstance(request_id, str) or not 1 <= len(request_id) <= 100):
            raise ValueError('Invalid collision configuration')
        same = (configuration_id == request_id and configuration_peer == peer)
        busy = ((current is not None and time.monotonic()-last_received <= .3) or
                any(0 <= time.monotonic()-t <= .3 for t in grippers.received.values()))
        error = None
        adopt_startup = configuration_id is None and gate.enabled == payload['enabled']
        if not same and busy and not adopt_startup:
            error = '请先停止双臂和夹爪输入，再切换保护'
        elif not same:
            configuration_id, configuration_peer = request_id, peer
            if not adopt_startup:
                gate.configure(payload['enabled'], dict(zip(names, positions)))
        sock.sendto(json.dumps(dict(version=1, type='collision_config_ack',
            request_id=request_id, error=error, safety=safety_status())).encode(), peer)
        return True

    def safety_feedback(seq):
        if peer_address is not None:
            receiver.sendto(json.dumps(dict(version=1, type='safety_status',
                seq=seq, safety=safety_status(), published_count=published_count,
                sent_mono_ns=time.monotonic_ns())).encode(), peer_address)

    def gripper_status():
        now = time.monotonic()
        return dict(version=1, type='gripper_status', sent_mono_ns=time.monotonic_ns(),
                    joints={j: dict(position=grippers.positions[j], target=grippers.targets[j],
                                    age_s=None if grippers.received[j] < 0 else now-grippers.received[j])
                            for j in grippers.positions},
                    subscribers=publisher.get_subscription_count(), published_count=published_count,
                    arms_fresh=current is not None and now-last_received <= .3,
                    angle_log=angle_log.status(),
                    safety=safety_status())

    def publish_latest() -> None:
        nonlocal published_count, feedback_seq, last_publish_at
        now = time.monotonic()
        previous_pending = dict(gate.pending) if gate.pending else None
        previous_result = gate.last_result
        previous_state = gate.state
        commit = gate.poll()
        if gate.last_result is not previous_result or gate.state != previous_state:
            angle_log.emit('check_result', pending=previous_pending, safety=safety_status())
        if gate.pending and previous_pending is None:
            angle_log.emit('check_submit', pending=gate.pending, initial=True)
        if commit is not None:
            accepted, metadata = commit
            # Prefix of exactly the verified segment, not per-joint clipping
            # or an unverified interpolation path. Keep the gate ledger equal
            # to what we will actually publish before any next submission.
            accepted, fraction = limited_step(dict(zip(names, positions)), accepted,
                                               now-last_publish_at)
            gate.accepted = accepted.copy()
            gate.last_result = dict(gate.last_result, output_limit_alpha=fraction)
            positions[:] = [accepted[name] for name in names]
            for joint in grippers.positions:
                grippers.positions[joint] = accepted[joint]
            feedback_seq = metadata if type(metadata) is int else 0
        fresh_grippers = grippers.step(now)
        arms_fresh = current is not None and now-last_received <= .3
        if not arms_fresh and not fresh_grippers and commit is None:
            return
        candidate = positions.copy()
        if arms_fresh:
            for index, value in zip(indices, current):
                candidate[index] = value
        for joint, angle in grippers.positions.items():
            candidate[names.index(joint)] = angle
        if gate.enabled:
            # GripperPreview integrates from accepted positions only.
            for joint in grippers.positions:
                grippers.positions[joint] = positions[names.index(joint)]
            if gate.validated and gate.state != '故障' and (arms_fresh or fresh_grippers):
                goal, fraction = limited_step(dict(zip(names, positions)), dict(zip(names, candidate)),
                                               min(1/30, now-last_publish_at))
                if gate.submit(goal, last_sequence if arms_fresh else 0):
                    angle_log.emit('check_submit', pending=gate.pending, candidate=candidate,
                                   input_limit_alpha=fraction)
            if not gate.validated:
                safety_feedback(0)
                return
        else:
            positions[:] = candidate
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        # Refresh the complete TF chain even for gripper-only preview. Arm
        # positions are HELD, never extrapolated or treated as fresh IK input.
        message.name = names
        message.position = positions.copy()
        publisher.publish(message)
        last_publish_at = now
        published_count += 1
        angle_log.emit('publish', positions=positions, candidate=candidate,
                       input_seq=last_sequence, accepted_arm_seq=feedback_seq if gate.enabled else last_sequence,
                       published_count=published_count, arms_fresh=arms_fresh,
                       safety=safety_status())
        if gate.enabled:
            safety_feedback(feedback_seq)

    node.create_timer(1 / 30, publish_latest)
    stopping = threading.Event()
    previous_term = None
    if threading.current_thread() is threading.main_thread():
        previous_term = signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    print(f"仅 RViz 预览：双臂 127.0.0.1:{args.port}；手机夹爪 {args.gripper_port}；不连接 X1 硬件。", flush=True)
    try:
        while rclpy.ok() and not stopping.is_set():
            while True:
                try:
                    raw, peer = gripper_receiver.recvfrom(2048)
                except BlockingIOError:
                    break
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and configure(raw, peer, gripper_receiver):
                        continue
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
                    if configure(raw, peer, receiver):
                        continue
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
                angle_log.emit('receive', seq=sequence, positions=angles,
                    joint_names=list(ARMS), source_sent_mono_ns=json.loads(raw)['sent_mono_ns'])
                ack = {"version": 1, "type": "ack", "seq": sequence,
                       "recv_mono_ns": time.monotonic_ns(),
                       "subscribers": publisher.get_subscription_count()}
                receiver.sendto(json.dumps(ack).encode("utf-8"), peer)
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        print("实时预览结束；RViz 保持末次姿态。", flush=True)
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        angle_log.close()
        gate.close()
        receiver.close()
        gripper_receiver.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
