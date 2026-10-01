#!/usr/bin/env python3
"""Experimental 6-axis yaw guard around the ZYX pitch singularity.

At |pitch| >= 80 deg, stop trusting the device's Euler yaw and integrate the
full body-frame gyro vector on SO(3). After |pitch| <= 75 deg, keep the yaw
offset needed to reconnect the gyro pose to the device's roll/pitch output.
Also track a continuous *world-vertical angular displacement*. That scalar is
not the ZYX Euler yaw at high pitch and must not be used as a full 3D pose.

This is read-only: it never changes sensor registers or device calibration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import warnings
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from wt901.gui_udp import GuiUdpSource
from wt901.protocol import AttitudePacket


DEFAULT_DEVICE_ID = "WT5500011590"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "recordings" / "yaw_guard"
RESULT_FIELDS = (
    "sample", "elapsed_s", "phase", "device_id", "device_time",
    "roll_deg", "pitch_deg", "raw_yaw_deg", "raw_yaw_unwrapped_deg",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps", "gyro_norm_dps",
    "guard_active", "entered_guard", "exited_guard", "gyro_gap_clamped",
    "stable_heading_deg", "protected_euler_yaw_deg", "pose_yaw_offset_deg",
    "raw_vs_protected_angle_deg", "step_jump_deg",
    "protected_qx", "protected_qy", "protected_qz", "protected_qw",
)


def wrap_degrees(angle: float) -> float:
    """Wrap an angle difference into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def near_branch(value: float, reference: float) -> float:
    """Choose the 360-degree branch of value closest to reference."""
    return reference + wrap_degrees(value - reference)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> Rotation:
    return Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=True)


@dataclass
class GuardResult:
    raw_yaw_unwrapped_deg: float
    stable_heading_deg: float
    protected_euler_yaw_deg: float
    pose_yaw_offset_deg: float
    raw_vs_protected_angle_deg: float
    step_jump_deg: float
    protected_quat_xyzw: tuple[float, float, float, float]
    guard_active: bool
    entered_guard: bool
    exited_guard: bool
    gyro_gap_clamped: bool


class YawGuard:
    """One device's streaming, stateful yaw-guard experiment.

    R = Rz(yaw) Ry(pitch) Rx(roll), with body-frame gyro integrated by right
    multiplication. A fixed world-Z correction is retained after exiting.
    """

    def __init__(self, enter_deg: float = 80.0, exit_deg: float = 75.0,
                 max_gyro_gap_s: float = 0.1) -> None:
        if not 0 < exit_deg < enter_deg < 90:
            raise ValueError("require 0 < exit_deg < enter_deg < 90")
        self.enter_deg = enter_deg
        self.exit_deg = exit_deg
        self.max_gyro_gap_s = max_gyro_gap_s
        self.active = False
        self.pose_yaw_offset_deg = 0.0
        self.heading_offset_deg = 0.0
        self.rotation: Rotation | None = None
        self.stable_heading_deg = 0.0
        self.raw_yaw_unwrapped_deg = 0.0
        self.previous_yaw_deg: float | None = None
        self.previous_gyro: np.ndarray | None = None
        self.previous_time_s: float | None = None
        self.events: list[dict] = []

    def update(self, time_s: float, roll_deg: float, pitch_deg: float,
               yaw_deg: float, gyro_xyz_dps: tuple[float, float, float]) -> GuardResult:
        gyro = np.asarray(gyro_xyz_dps, dtype=float)
        if not np.all(np.isfinite(gyro)) or not all(np.isfinite(
                (time_s, roll_deg, pitch_deg, yaw_deg))):
            raise ValueError("non-finite IMU value")
        if self.previous_time_s is not None and time_s <= self.previous_time_s:
            raise ValueError("sample times must increase for one device")

        raw_rotation = rotation_from_rpy(roll_deg, pitch_deg, yaw_deg)
        if self.previous_yaw_deg is None:
            self.raw_yaw_unwrapped_deg = yaw_deg
        else:
            self.raw_yaw_unwrapped_deg += wrap_degrees(yaw_deg - self.previous_yaw_deg)
        entered = False
        exited = False
        gap_clamped = False
        step_jump = 0.0
        if self.rotation is None:
            # Starting inside the guard is possible, but its first pose is not
            # independently validated; begin recordings below the threshold.
            self.rotation = raw_rotation
            self.stable_heading_deg = self.raw_yaw_unwrapped_deg
            self.active = abs(pitch_deg) >= self.enter_deg
            if self.active:
                entered = True
                self.events.append({"kind": "entered", "time_s": time_s,
                                    "pitch_deg": pitch_deg, "initial_pose_untrusted": True})
        else:
            dt = time_s - self.previous_time_s  # type: ignore[operator]
            if dt > self.max_gyro_gap_s:
                dt = self.max_gyro_gap_s
                gap_clamped = True
            average_gyro = (self.previous_gyro + gyro) * 0.5  # type: ignore[operator]
            entering = not self.active and abs(pitch_deg) >= self.enter_deg
            if entering:
                self.active = True
                entered = True
                self.events.append({"kind": "entered", "time_s": time_s,
                                    "pitch_deg": pitch_deg,
                                    "raw_yaw_deg": yaw_deg,
                                    "stable_heading_deg": self.stable_heading_deg})
            if self.active:
                previous_rotation = self.rotation
                body_step_rad = np.deg2rad(average_gyro * dt)
                self.rotation = previous_rotation * Rotation.from_rotvec(body_step_rad)
                # A continuous scalar, not Euler yaw: integrate angular rate
                # projected on the original world vertical axis.
                world_gyro_rad_s = previous_rotation.apply(np.deg2rad(average_gyro))
                self.stable_heading_deg += np.rad2deg(world_gyro_rad_s[2] * dt)
                if abs(pitch_deg) <= self.exit_deg:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        gyro_yaw = float(self.rotation.as_euler("xyz", degrees=True)[2])
                    self.pose_yaw_offset_deg = near_branch(
                        gyro_yaw - yaw_deg, self.pose_yaw_offset_deg)
                    self.heading_offset_deg = (self.stable_heading_deg
                                               - self.raw_yaw_unwrapped_deg)
                    rejoined = Rotation.from_euler(
                        "z", self.pose_yaw_offset_deg, degrees=True) * raw_rotation
                    step_jump = float(np.rad2deg((self.rotation.inv() * rejoined).magnitude()))
                    self.rotation = rejoined
                    self.active = False
                    exited = True
                    self.events.append({"kind": "exited", "time_s": time_s,
                                        "pitch_deg": pitch_deg,
                                        "retained_pose_yaw_offset_deg":
                                            self.pose_yaw_offset_deg,
                                        "rejoin_jump_deg": step_jump})
            else:
                self.rotation = Rotation.from_euler(
                    "z", self.pose_yaw_offset_deg, degrees=True) * raw_rotation
                self.stable_heading_deg = (self.raw_yaw_unwrapped_deg
                                           + self.heading_offset_deg)

        self.previous_time_s = time_s
        self.previous_yaw_deg = yaw_deg
        self.previous_gyro = gyro
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            protected_yaw = float(self.rotation.as_euler("xyz", degrees=True)[2])
        raw_difference = float(np.rad2deg(
            (raw_rotation.inv() * self.rotation).magnitude()))
        quat = tuple(float(value) for value in self.rotation.as_quat())
        return GuardResult(
            raw_yaw_unwrapped_deg=self.raw_yaw_unwrapped_deg,
            stable_heading_deg=self.stable_heading_deg,
            protected_euler_yaw_deg=protected_yaw,
            pose_yaw_offset_deg=self.pose_yaw_offset_deg,
            raw_vs_protected_angle_deg=raw_difference,
            step_jump_deg=step_jump,
            protected_quat_xyzw=quat,
            guard_active=self.active,
            entered_guard=entered,
            exited_guard=exited,
            gyro_gap_clamped=gap_clamped,
        )


def device_clock_seconds(packet: AttitudePacket) -> float:
    return (packet.hour * 3600 + packet.minute * 60 + packet.second
            + packet.millisecond / 1000)


class DeviceClock:
    """Unwrap the packet's time-of-day; fall back to host time after glitches."""

    def __init__(self) -> None:
        self.start_tick: float | None = None
        self.last_tick: float | None = None
        self.day_offset = 0.0
        self.last_elapsed = 0.0
        self.last_host_s: float | None = None

    def elapsed(self, packet: AttitudePacket, host_s: float) -> float:
        tick = device_clock_seconds(packet)
        if self.last_tick is not None and tick + self.day_offset < self.last_tick - 43200:
            self.day_offset += 86400
        tick += self.day_offset
        if self.start_tick is None:
            self.start_tick = tick
            elapsed = 0.0
        else:
            elapsed = tick - self.start_tick
            if elapsed <= self.last_elapsed or elapsed - self.last_elapsed > 1:
                host_dt = host_s - self.last_host_s  # type: ignore[operator]
                elapsed = self.last_elapsed + max(0.000001, min(host_dt, 1.0))
                self.start_tick = tick - elapsed
        self.last_tick = tick
        self.last_elapsed = elapsed
        self.last_host_s = host_s
        return elapsed


def output_row(sample: int, elapsed_s: float, phase: str, packet: AttitudePacket,
               result: GuardResult) -> dict:
    qx, qy, qz, qw = result.protected_quat_xyzw
    return {
        "sample": sample, "elapsed_s": elapsed_s, "phase": phase,
        "device_id": packet.device_id, "device_time": packet.time_str,
        "roll_deg": packet.roll, "pitch_deg": packet.pitch,
        "raw_yaw_deg": packet.yaw,
        "raw_yaw_unwrapped_deg": result.raw_yaw_unwrapped_deg,
        "gyro_x_dps": packet.gyro_x, "gyro_y_dps": packet.gyro_y,
        "gyro_z_dps": packet.gyro_z,
        "gyro_norm_dps": float(np.linalg.norm(
            (packet.gyro_x, packet.gyro_y, packet.gyro_z))),
        "guard_active": int(result.guard_active),
        "entered_guard": int(result.entered_guard),
        "exited_guard": int(result.exited_guard),
        "gyro_gap_clamped": int(result.gyro_gap_clamped),
        "stable_heading_deg": result.stable_heading_deg,
        "protected_euler_yaw_deg": result.protected_euler_yaw_deg,
        "pose_yaw_offset_deg": result.pose_yaw_offset_deg,
        "raw_vs_protected_angle_deg": result.raw_vs_protected_angle_deg,
        "step_jump_deg": result.step_jump_deg,
        "protected_qx": qx, "protected_qy": qy,
        "protected_qz": qz, "protected_qw": qw,
    }


def _packet_from_csv(row: dict) -> AttitudePacket:
    return AttitudePacket(
        device_id=row["device_id"],
        year=int(row["year"]), month=int(row["month"]), day=int(row["day"]),
        hour=int(row["hour"]), minute=int(row["minute"]), second=int(row["second"]),
        millisecond=int(row["millisecond"]),
        acc_x=float(row["acc_x"]), acc_y=float(row["acc_y"]), acc_z=float(row["acc_z"]),
        gyro_x=float(row["gyro_x"]), gyro_y=float(row["gyro_y"]),
        gyro_z=float(row["gyro_z"]),
        mag_x=float(row["mag_x"]), mag_y=float(row["mag_y"]),
        mag_z=float(row["mag_z"]),
        roll=float(row["roll"]), pitch=float(row["pitch"]), yaw=float(row["yaw"]),
        temperature=float(row["temperature"]), battery_v=float(row["battery_v"]),
        rssi=int(row["rssi"]), version=int(row["version"]),
    )


def replay_csv(path: Path, device_id: str) -> list[tuple[float, str, AttitudePacket]]:
    samples = []
    clock = DeviceClock()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["device_id"] != device_id:
                continue
            packet = _packet_from_csv(row)
            host_s = int(row["host_recv_mono_ns"]) / 1e9
            elapsed = clock.elapsed(packet, host_s)
            samples.append((elapsed, row.get("phase", "replay"), packet))
    return samples


def live_samples(device_id: str, port: int, seconds: float,
                 prepare_seconds: float, enter_deg: float,
                 exit_deg: float) -> list[tuple[float, str, AttitudePacket]]:
    source = GuiUdpSource(port=port)
    try:
        print(f"等待 {device_id} 的 UDP 数据，监听端口 {port}…", flush=True)
        endpoint = None
        while endpoint is None:
            source.poll_timed()
            endpoint = source.seen_endpoints().get(device_id)
            time.sleep(0.02)
        print(f"识别成功：{device_id}，{endpoint[0]}:{endpoint[1]}", flush=True)
        input("把设备摆好后按回车；随后有 5 秒准备时间，自动录制。")
        start_ns = time.monotonic_ns() + round(prepare_seconds * 1e9)
        deadline_ns = start_ns + round(seconds * 1e9)
        print(f"准备 {prepare_seconds:g} 秒；随后录制 {seconds:g} 秒。", flush=True)
        clock = DeviceClock()
        guard = YawGuard(enter_deg, exit_deg)
        samples: list[tuple[float, str, AttitudePacket]] = []
        last_announce = -1
        last_display_s = -1.0
        while time.monotonic_ns() < deadline_ns:
            now_ns = time.monotonic_ns()
            if now_ns < start_ns:
                remaining = int(np.ceil((start_ns - now_ns) / 1e9))
                if remaining != last_announce:
                    print(f"准备：{remaining} 秒", flush=True)
                    last_announce = remaining
                source.poll_timed()  # drain pre-start data
                time.sleep(0.01)
                continue
            for received_ns, packet in source.poll_timed():
                if received_ns < start_ns or packet.device_id != device_id:
                    continue
                elapsed = clock.elapsed(packet, received_ns / 1e9)
                samples.append((elapsed, "live", packet))
                result = guard.update(elapsed, packet.roll, packet.pitch, packet.yaw,
                                      (packet.gyro_x, packet.gyro_y, packet.gyro_z))
                if result.entered_guard:
                    print(f"进入保护区：t={elapsed:.2f}s，Pitch={packet.pitch:+.1f}°", flush=True)
                if result.exited_guard:
                    print(f"离开保护区：t={elapsed:.2f}s，"
                          f"保留航向修正 {result.pose_yaw_offset_deg:+.1f}°", flush=True)
                if elapsed - last_display_s >= 0.25:
                    print(f"t={elapsed:5.1f}s Pitch={packet.pitch:+6.1f}° "
                          f"原始Yaw={packet.yaw:+7.1f}° "
                          f"连续竖轴航向={result.stable_heading_deg:+7.1f}° "
                          f"保护姿态Yaw={result.protected_euler_yaw_deg:+7.1f}° "
                          f"{'[陀螺保护]' if result.guard_active else '[设备+偏置]'}",
                          flush=True)
                    last_display_s = elapsed
            time.sleep(0.005)
        print(f"录制结束：{len(samples)} 包。", flush=True)
        return samples
    finally:
        source.close()


def create_plot(rows: list[dict], path: Path, enter_deg: float, exit_deg: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([row["elapsed_s"] for row in rows])
    pitch = np.array([row["pitch_deg"] for row in rows])
    raw = np.array([row["raw_yaw_unwrapped_deg"] for row in rows])
    stable = np.array([row["stable_heading_deg"] for row in rows])
    difference = np.array([row["raw_vs_protected_angle_deg"] for row in rows])
    guard = np.array([row["guard_active"] for row in rows], dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, pitch, color="#a855f7", label="Raw pitch")
    axes[0].axhline(enter_deg, color="#7c3aed", linestyle="--", linewidth=0.9)
    axes[0].axhline(exit_deg, color="#7c3aed", linestyle=":", linewidth=0.9)
    axes[0].set_ylabel("Pitch (deg)")
    axes[1].plot(t, raw, color="#ef4444", linewidth=1, label="Raw yaw (unwrapped)")
    axes[1].plot(t, stable, color="#0f766e", linewidth=1.5,
                 label="Integrated vertical heading")
    axes[1].set_ylabel("Heading-like angle (deg)")
    axes[1].legend(loc="best")
    axes[2].plot(t, difference, color="#2563eb", label="Raw vs protected SO(3)")
    axes[2].set_ylabel("Pose difference (deg)")
    axes[2].set_xlabel("Elapsed time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        if guard.any():
            axis.fill_between(t, 0, 1, where=guard, transform=axis.get_xaxis_transform(),
                              color="#facc15", alpha=0.12, label="Guard active")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(samples: list[tuple[float, str, AttitudePacket]], output_dir: Path,
                 device_id: str, enter_deg: float, exit_deg: float,
                 source_description: str) -> Path:
    if not samples:
        raise ValueError(f"没有 {device_id} 的数据")
    path = output_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    guard = YawGuard(enter_deg=enter_deg, exit_deg=exit_deg)
    rows = []
    for sample, (elapsed, phase, packet) in enumerate(samples):
        result = guard.update(elapsed, packet.roll, packet.pitch, packet.yaw,
                              (packet.gyro_x, packet.gyro_y, packet.gyro_z))
        rows.append(output_row(sample, elapsed, phase, packet, result))
    with (path / "guarded.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    create_plot(rows, path / "comparison.png", enter_deg, exit_deg)
    from_first = Rotation.from_quat(
        [rows[0][key] for key in ("protected_qx", "protected_qy",
                                  "protected_qz", "protected_qw")])
    from_last = Rotation.from_quat(
        [rows[-1][key] for key in ("protected_qx", "protected_qy",
                                   "protected_qz", "protected_qw")])
    first_raw = rotation_from_rpy(rows[0]["roll_deg"], rows[0]["pitch_deg"],
                                  rows[0]["raw_yaw_deg"])
    last_raw = rotation_from_rpy(rows[-1]["roll_deg"], rows[-1]["pitch_deg"],
                                 rows[-1]["raw_yaw_deg"])
    summary = {
        "format": "wt901-gyro-yaw-guard-experiment-v1",
        "source": source_description,
        "device_id": device_id,
        "enter_abs_pitch_deg": enter_deg,
        "exit_abs_pitch_deg": exit_deg,
        "rotation_order": "Rz(yaw) * Ry(pitch) * Rx(roll)",
        "method": "integrate 3-axis body gyro quaternion inside guard; retain world-Z yaw offset on exit",
        "stable_heading_warning": "World-vertical gyro integral is not ZYX Euler yaw near pitch 90 degrees",
        "samples": len(rows),
        "guard_events": guard.events,
        "gyro_gap_clamped_samples": sum(row["gyro_gap_clamped"] for row in rows),
        "raw_start_to_end_angle_deg": float(np.rad2deg(
            (first_raw.inv() * last_raw).magnitude())),
        "protected_start_to_end_angle_deg": float(np.rad2deg(
            (from_first.inv() * from_last).magnitude())),
        "final_pose_yaw_offset_deg": rows[-1]["pose_yaw_offset_deg"],
        "final_raw_vs_protected_angle_deg": rows[-1]["raw_vs_protected_angle_deg"],
        "max_rejoin_jump_deg": max(row["step_jump_deg"] for row in rows),
    }
    (path / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"输出目录：{path}")
    print(f"保护区进入/退出：{sum(e['kind'] == 'entered' for e in guard.events)}/"
          f"{sum(e['kind'] == 'exited' for e in guard.events)} 次；"
          f"最终航向修正 {summary['final_pose_yaw_offset_deg']:+.1f}°")
    print(f"起点→终点旋转：原始 {summary['raw_start_to_end_angle_deg']:.1f}°，"
          f"保护后 {summary['protected_start_to_end_angle_deg']:.1f}°；"
          f"最大重接跳变 {summary['max_rejoin_jump_deg']:.1f}°")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901 Pitch 80° 附近陀螺航向保护实验")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="现有 samples.csv 文件或其录制目录")
    mode.add_argument("--live", action="store_true", help="实时监听 UDP 1399")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--enter-deg", type=float, default=80.0)
    parser.add_argument("--exit-deg", type=float, default=75.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--seconds", type=float, default=42.0,
                        help="实时录制持续秒数，默认 42")
    parser.add_argument("--prepare-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.input:
        source = args.input / "samples.csv" if args.input.is_dir() else args.input
        samples = replay_csv(source, args.device_id)
        description = str(source.resolve())
    else:
        try:
            samples = live_samples(args.device_id, args.listen_port,
                                   args.seconds, args.prepare_seconds,
                                   args.enter_deg, args.exit_deg)
        except OSError as exc:
            parser.exit(1, f"无法监听 UDP {args.listen_port}：{exc}\n请先关闭其他 GUI。\n")
        description = f"live UDP {args.listen_port}"
    run_analysis(samples, args.output_dir, args.device_id,
                 args.enter_deg, args.exit_deg, description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
