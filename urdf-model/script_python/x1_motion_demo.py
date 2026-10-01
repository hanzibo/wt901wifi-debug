#!/usr/bin/env python3
"""Headless all-joint motion demo for RViz visualization only."""

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from x1_common import JOINT_NAMES


class X1MotionDemo(Node):
    def __init__(self, rate_hz: float, speed: float, amplitude_deg: float) -> None:
        super().__init__("x1_python_motion_demo")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.speed = speed
        self.amplitude_rad = math.radians(amplitude_deg)
        self.started_at = time.monotonic()
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_motion)

    def publish_positions(self, positions: list[float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = positions
        self.publisher.publish(message)

    def publish_motion(self) -> None:
        elapsed = time.monotonic() - self.started_at
        positions = []
        for index, name in enumerate(JOINT_NAMES):
            wave = math.sin(elapsed * self.speed + index * math.pi / 8.0)
            if name in {"57", "7"}:
                position = 0.5 + 0.45 * wave
            elif name == "102":
                position = 0.35 * wave
            else:
                position = self.amplitude_rad * wave
            positions.append(position)
        self.publish_positions(positions)

    def publish_zero(self) -> None:
        self.publish_positions([0.0] * len(JOINT_NAMES))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a smooth 16-joint motion to RViz via /joint_states."
    )
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate in Hz")
    parser.add_argument("--speed", type=float, default=0.8, help="Wave speed multiplier")
    parser.add_argument("--amplitude-deg", type=float, default=30.0)
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Stop after N seconds; 0 means run until Ctrl+C",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0:
        raise SystemExit("--rate must be greater than zero")
    if args.amplitude_deg < 0.0 or args.amplitude_deg > 90.0:
        raise SystemExit("--amplitude-deg must be between 0 and 90")

    rclpy.init()
    node = X1MotionDemo(args.rate, args.speed, args.amplitude_deg)
    deadline = time.monotonic() + args.duration if args.duration > 0.0 else None
    try:
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
