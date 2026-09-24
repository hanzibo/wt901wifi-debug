#!/usr/bin/env python3
"""Estimate a WT901WIFI pose relative to its position and orientation at startup.

The rotation is derived from the device's fused attitude. Translation is
dead reckoning from acceleration and will drift; it is not an absolute position.
Keep the device still for the first two seconds to set the origin and bias.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import sys
import time

import numpy as np

from wt901_viz.orientation import rotation_matrix

G = 9.80665
UP = np.array([0.0, 0.0, 1.0])


def packet_time(packet) -> datetime:
    return datetime(
        packet.year, packet.month, packet.day, packet.hour,
        packet.minute, packet.second, packet.millisecond * 1000,
    )


def specific_force(packet) -> np.ndarray:
    """Accelerometer output in body axes, converted from g to m/s²."""
    return G * np.array([packet.acc_x, packet.acc_y, packet.acc_z])


def attitude(packet) -> np.ndarray:
    """Body axes expressed in the navigation frame, assuming ZYX Euler order."""
    return rotation_matrix(packet.roll, packet.pitch, packet.yaw, "ZYX")


class PoseEstimator:
    def __init__(self, stationary_seconds: float = 2.0):
        self.stationary_seconds = stationary_seconds
        self.calibration = deque()
        self.device_id = None
        self.r0 = None
        self.bias_body = np.zeros(3)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.last_time = None
        self.last_acceleration = None
        self.skipped_intervals = 0
        self.stationary = False
        self.still_since = None

    def _calibrate(self, packet) -> bool:
        stamp = packet_time(packet)
        self.calibration.append((stamp, packet))
        while self.calibration and (stamp - self.calibration[0][0]).total_seconds() > self.stationary_seconds:
            self.calibration.popleft()
        if len(self.calibration) < 2 or (stamp - self.calibration[0][0]).total_seconds() < self.stationary_seconds * 0.9:
            return False

        samples = [item[1] for item in self.calibration]
        max_gyro = max(np.linalg.norm([p.gyro_x, p.gyro_y, p.gyro_z]) for p in samples)
        max_norm_error = max(abs(np.linalg.norm(specific_force(p)) - G) for p in samples)
        if max_gyro > 3.0 or max_norm_error > 0.1 * G:
            self.calibration.clear()
            return False

        self.r0 = attitude(packet)
        # Bias belongs to the sensor axes; keep it in the body frame so a
        # later rotation does not turn the calibration error into motion.
        self.bias_body = np.mean(
            [specific_force(p) - attitude(p).T @ (G * UP) for p in samples], axis=0,
        )
        self.last_time = stamp
        self.last_acceleration = self._world_acceleration(packet)
        self.stationary = True
        self.still_since = stamp
        self.calibration.clear()
        return True

    def _world_acceleration(self, packet) -> np.ndarray:
        # Navigation frame is assumed Z-up. Specific force includes +g at rest.
        a_nav = attitude(packet) @ (specific_force(packet) - self.bias_body) - G * UP
        return self.r0.T @ a_nav

    def _looks_still(self, packet, acceleration: np.ndarray) -> bool:
        gyro = np.linalg.norm([packet.gyro_x, packet.gyro_y, packet.gyro_z])
        return gyro < 1.5 and np.linalg.norm(acceleration) < 0.15

    def update(self, packet) -> np.ndarray | None:
        if self.device_id is None:
            self.device_id = packet.device_id
        if packet.device_id != self.device_id:
            return None
        if self.r0 is None:
            if not self._calibrate(packet):
                return None
        else:
            stamp = packet_time(packet)
            dt = (stamp - self.last_time).total_seconds()
            acceleration = self._world_acceleration(packet)
            if self._looks_still(packet, acceleration):
                if self.still_since is None:
                    self.still_since = stamp
                if (stamp - self.still_since).total_seconds() >= 0.4:
                    self.stationary = True
            else:
                self.still_since = None
                self.stationary = False
            if 0 < dt <= 0.5:
                if self.stationary:
                    self.velocity[:] = 0.0
                    # Slowly track temperature dependent accelerometer bias
                    # only while the sensor appears stationary.
                    observed_bias = specific_force(packet) - attitude(packet).T @ (G * UP)
                    weight = min(dt / 5.0, 0.05)
                    self.bias_body += weight * (observed_bias - self.bias_body)
                    self.last_acceleration = np.zeros(3)
                else:
                    new_velocity = self.velocity + (self.last_acceleration + acceleration) * (dt / 2)
                    self.position += (self.velocity + new_velocity) * (dt / 2)
                    self.velocity = new_velocity
                    self.last_acceleration = acceleration
            else:
                # A duplicate timestamp or reception gap cannot be integrated safely.
                self.skipped_intervals += 1
                self.last_acceleration = acceleration
            self.last_time = stamp

        transform = np.eye(4)
        transform[:3, :3] = self.r0.T @ attitude(packet)
        transform[:3, 3] = self.position
        return transform


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901WIFI 相对初始位姿估计（平移会漂移）")
    parser.add_argument("--tcp", action="store_true", help="监听 TCP；默认监听 UDP")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--device-id", help="只处理指定的 12 位设备 ID")
    parser.add_argument("--hz", type=float, default=5.0, help="矩阵显示刷新率，默认 5 Hz")
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz 必须大于 0")

    from wt901.net_source import TcpAttitudeSource, UdpAttitudeSource

    source_class = TcpAttitudeSource if args.tcp else UdpAttitudeSource
    estimator = PoseEstimator()
    next_print = 0.0
    ready = False
    print(f"监听 {'TCP' if args.tcp else 'UDP'} :{args.listen_port}；请保持设备静止约 2 秒", file=sys.stderr)
    try:
        with source_class(args.listen_port) as source:
            while True:
                packets = source.poll()
                if not packets:
                    time.sleep(0.005)
                    continue
                latest = None
                for packet in packets:
                    if args.device_id and packet.device_id != args.device_id:
                        continue
                    transform = estimator.update(packet)
                    if transform is not None:
                        latest = (packet, transform)
                if latest is None:
                    continue
                if not ready:
                    print(f"以 {latest[0].device_id} 的当前位姿建立世界坐标系", file=sys.stderr)
                    ready = True
                now = time.monotonic()
                if now < next_print:
                    continue
                next_print = now + 1 / args.hz
                packet, transform = latest
                print(f"\n{packet.device_id}  {packet.time_str}  估算位置(m) "
                      f"{np.array2string(estimator.position, precision=3)}  "
                      f"状态 {'静止' if estimator.stationary else '运动/未确认静止'}  "
                      f"跳过时间间隔 {estimator.skipped_intervals}")
                print("T_world_body =")
                print(np.array2string(transform, precision=4, suppress_small=True))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
