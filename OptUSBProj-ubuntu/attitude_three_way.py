#!/usr/bin/env python3
"""Offline comparison: WT RPY, gyro-only SO(3), and wrist GUI yaw guard.

The gyro-only estimator starts at identity and never reads RPY. All three
outputs are subsequently expressed relative to their own initial static
window so their trajectories can be compared in a common neutral frame.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from wt901_viz.yaw_guard import PacketClock, YawGuard


OUTPUT_FIELDS = (
    "sample", "elapsed_s", "phase", "pitch_deg", "gyro_norm_dps", "guard_active",
    "raw_neutral_angle_deg", "gyro_neutral_angle_deg", "guard_neutral_angle_deg",
    "raw_vs_gyro_deg", "raw_vs_guard_deg", "gyro_vs_guard_deg",
    "raw_qx", "raw_qy", "raw_qz", "raw_qw",
    "gyro_qx", "gyro_qy", "gyro_qz", "gyro_qw",
    "guard_qx", "guard_qy", "guard_qz", "guard_qw",
)


@dataclass(frozen=True)
class Sample:
    packet: object
    phase: str
    host_mono_ns: int


def read_samples(path: Path, device_id: str | None) -> tuple[str, list[Sample]]:
    if path.is_dir():
        path = path / "samples.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or "device_id" not in rows[0]:
        raise ValueError("samples.csv 为空或没有 device_id 列")
    available = sorted({row["device_id"] for row in rows})
    if device_id is None:
        if len(available) != 1:
            raise ValueError(f"录制包含多台设备，请指定 --device-id：{', '.join(available)}")
        device_id = available[0]
    if device_id not in available:
        raise ValueError(f"未找到 {device_id}；可选：{', '.join(available)}")
    required = ("hour", "minute", "second", "millisecond", "roll", "pitch", "yaw",
                "gyro_x", "gyro_y", "gyro_z")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"录制缺少列：{', '.join(missing)}")
    result = []
    for row in rows:
        if row["device_id"] != device_id:
            continue
        from types import SimpleNamespace
        packet = SimpleNamespace(**{field: int(row[field]) for field in required[:4]},
                                 **{field: float(row[field]) for field in required[4:]})
        result.append(Sample(packet, row.get("phase", ""),
                             int(row.get("host_recv_mono_ns") or 0)))
    return device_id, result


def _angle_degrees(rotation: Rotation) -> float:
    return float(np.rad2deg(rotation.magnitude()))


def _reference_rotation(rotations: Rotation, indices: np.ndarray) -> Rotation:
    return rotations[indices].mean()


def compare(samples: list[Sample], enter_deg: float = 80.0,
            exit_deg: float = 75.0, reference_seconds: float = 2.0) -> tuple[list[dict], dict]:
    if len(samples) < 2:
        raise ValueError("至少需要两包数据")
    clock = PacketClock()
    guard = YawGuard(enter_deg, exit_deg)
    times = []
    raw = []
    gyro = []
    protected = []
    guard_flags = []
    gyro_norms = []
    dt_values = []
    bad_device_ticks = 0
    raw_device_gaps = 0
    previous_device_tick = None
    current_gyro = Rotation.identity()
    previous_rate = None
    previous_time = None
    for sample in samples:
        packet = sample.packet
        device_tick = (packet.hour * 3600 + packet.minute * 60 + packet.second
                       + packet.millisecond / 1000.0)
        if previous_device_tick is not None:
            raw_dt = device_tick - previous_device_tick
            if raw_dt < -43200:
                raw_dt += 86400
            if raw_dt <= 0 or raw_dt > 1.0:
                bad_device_ticks += 1
            if raw_dt > 0.1:
                raw_device_gaps += 1
        previous_device_tick = device_tick
        t = clock.elapsed(packet)
        rate = np.array((packet.gyro_x, packet.gyro_y, packet.gyro_z), dtype=float)
        if previous_time is not None:
            dt = t - previous_time
            # This stream is only a diagnostic after a gap; do not silently
            # pretend a long Wi-Fi outage is a normal sample interval.
            current_gyro = current_gyro * Rotation.from_rotvec(
                np.deg2rad((previous_rate + rate) * 0.5 * dt))
            dt_values.append(dt)
        guarded = guard.update(t, packet.roll, packet.pitch, packet.yaw, tuple(rate))
        raw.append(Rotation.from_euler("xyz", (packet.roll, packet.pitch, packet.yaw),
                                       degrees=True))
        gyro.append(current_gyro)
        protected.append(Rotation.from_matrix(guarded.rotation))
        guard_flags.append(guarded.active)
        gyro_norms.append(float(np.linalg.norm(rate)))
        times.append(t)
        previous_rate = rate
        previous_time = t
    times = np.asarray(times)
    initial = np.flatnonzero(times <= reference_seconds)
    if len(initial) < 5:
        raise ValueError("初始静止窗口样本不足，请调整 --reference-seconds")
    series = {}
    for name, values in (("raw", raw), ("gyro", gyro), ("guard", protected)):
        absolute = Rotation.concatenate(values)
        neutral = _reference_rotation(absolute, initial)
        series[name] = neutral.inv() * absolute

    raw_gyro = np.rad2deg((series["raw"].inv() * series["gyro"]).magnitude())
    raw_guard = np.rad2deg((series["raw"].inv() * series["guard"]).magnitude())
    gyro_guard = np.rad2deg((series["gyro"].inv() * series["guard"]).magnitude())
    neutral_angles = {name: np.rad2deg(rotations.magnitude())
                      for name, rotations in series.items()}
    quaternions = {name: rotations.as_quat() for name, rotations in series.items()}
    rows = []
    for i, sample in enumerate(samples):
        row = {
            "sample": i, "elapsed_s": times[i], "phase": sample.phase,
            "pitch_deg": sample.packet.pitch,
            "gyro_norm_dps": gyro_norms[i], "guard_active": int(guard_flags[i]),
            "raw_neutral_angle_deg": neutral_angles["raw"][i],
            "gyro_neutral_angle_deg": neutral_angles["gyro"][i],
            "guard_neutral_angle_deg": neutral_angles["guard"][i],
            "raw_vs_gyro_deg": raw_gyro[i], "raw_vs_guard_deg": raw_guard[i],
            "gyro_vs_guard_deg": gyro_guard[i],
        }
        for name, values in quaternions.items():
            row.update({f"{name}_q{axis}": float(values[i, j])
                        for j, axis in enumerate("xyzw")})
        rows.append(row)

    # Only call a final posture a return-to-neutral when the recording's
    # protocol explicitly says so. Otherwise report endpoint displacement.
    phases = np.array([sample.phase for sample in samples])
    final_neutral = np.flatnonzero(phases == "neutral_after")
    is_return = len(final_neutral) >= 5
    final = final_neutral if is_return else np.flatnonzero(times >= times[-1] - 2.0)
    endpoint = {}
    for name, rotations in series.items():
        endpoint[name] = _angle_degrees(_reference_rotation(rotations, final))
    initial_rates = np.array(gyro_norms)[initial]
    summary = {
        "samples": len(samples), "duration_s": float(times[-1]),
        "reference_window_s": [0.0, reference_seconds],
        "initial_gyro_norm_median_dps": float(np.median(initial_rates)),
        "timing": {
            "source": "device packet time-of-day; duplicate/bad ticks use 20 ms fallback",
            "median_dt_ms": float(np.median(dt_values) * 1000),
            "max_dt_ms": float(np.max(dt_values) * 1000),
            "gaps_over_100_ms": raw_device_gaps,
            "duplicate_or_bad_device_ticks": bad_device_ticks,
        },
        "guard": {"enter_abs_pitch_deg": enter_deg, "exit_abs_pitch_deg": exit_deg,
                  "active_samples": int(np.count_nonzero(guard_flags))},
        "endpoint_kind": "return_to_neutral" if is_return else "end_of_recording_not_known_neutral",
        "endpoint_mean_so3_angle_deg": endpoint,
        "peak_pairwise_difference_deg": {
            "raw_vs_gyro": float(np.max(raw_gyro)),
            "raw_vs_guard": float(np.max(raw_guard)),
            "gyro_vs_guard": float(np.max(gyro_guard)),
        },
        "final_pairwise_difference_deg": {
            "raw_vs_gyro": float(np.median(raw_gyro[final])),
            "raw_vs_guard": float(np.median(raw_guard[final])),
            "gyro_vs_guard": float(np.median(gyro_guard[final])),
        },
        "caveats": [
            "No external ground truth: smaller endpoint error is not proof of accurate intermediate motion.",
            "Gyro-only starts at identity and does not read device RPY; no gyro bias correction is applied.",
            "Each trajectory is expressed relative to its own initial mean orientation.",
            "A timestamp gap makes gyro-only propagation uncertain even if a result is plotted.",
        ],
    }
    return rows, summary


def create_plot(rows: list[dict], path: Path, enter_deg: float,
                exit_deg: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([row["elapsed_s"] for row in rows])
    pitch = np.array([row["pitch_deg"] for row in rows])
    active = np.array([row["guard_active"] for row in rows], dtype=bool)
    colors = {"raw": "#ef4444", "gyro": "#2563eb", "guard": "#16a34a"}
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.5, 1.0))
    ax_pitch = fig.add_subplot(grid[0, 0])
    ax_angle = fig.add_subplot(grid[1, 0], sharex=ax_pitch)
    ax_difference = fig.add_subplot(grid[2, 0], sharex=ax_pitch)
    ax_3d = fig.add_subplot(grid[:, 1], projection="3d")
    ax_pitch.plot(t, pitch, color="#9333ea", linewidth=1, label="WT pitch")
    for sign in (-1, 1):
        ax_pitch.axhline(sign * enter_deg, color="#9333ea", linestyle="--", alpha=0.6)
        ax_pitch.axhline(sign * exit_deg, color="#9333ea", linestyle=":", alpha=0.6)
    ax_pitch.set_ylabel("Pitch (deg)")
    for name, label in (("raw", "WT RPY"), ("gyro", "Gyro-only"),
                        ("guard", "Pitch guard")):
        ax_angle.plot(t, [row[f"{name}_neutral_angle_deg"] for row in rows],
                      label=label, color=colors[name], linewidth=1.2)
    ax_angle.set_ylabel("Rotation from neutral (deg)")
    ax_angle.legend(loc="upper right")
    for key, label, color in (("raw_vs_gyro_deg", "WT vs gyro", "#7c3aed"),
                              ("raw_vs_guard_deg", "WT vs guard", "#ea580c"),
                              ("gyro_vs_guard_deg", "Gyro vs guard", "#0891b2")):
        ax_difference.plot(t, [row[key] for row in rows], label=label,
                           color=color, linewidth=1)
    ax_difference.set_ylabel("SO(3) difference (deg)")
    ax_difference.set_xlabel("Time (s)")
    ax_difference.legend(loc="upper left")
    for axis in (ax_pitch, ax_angle, ax_difference):
        axis.grid(alpha=0.25)
        if active.any():
            axis.fill_between(t, 0, 1, where=active, transform=axis.get_xaxis_transform(),
                              color="#facc15", alpha=0.13)
    for name, label in (("raw", "WT RPY"), ("gyro", "Gyro-only"),
                        ("guard", "Pitch guard")):
        quats = np.array([[row[f"{name}_q{axis}"] for axis in "xyzw"] for row in rows])
        # This is the trajectory of the unit sensor-Y direction, not sensor position.
        tip = Rotation.from_quat(quats).apply((0, 1, 0))
        ax_3d.plot(tip[::2, 0], tip[::2, 1], tip[::2, 2],
                   label=label, color=colors[name], alpha=0.8)
    ax_3d.scatter([0], [1], [0], color="black", s=30, label="Neutral")
    ax_3d.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
              xlabel="X", ylabel="Y", zlabel="Z",
              title="Unit sensor-Y direction (not position)")
    ax_3d.legend(loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="WT 姿态三轨迹离线对比（只读）")
    parser.add_argument("--input", required=True, type=Path,
                        help="录制目录或 samples.csv")
    parser.add_argument("--device-id", help="多设备录制时必填")
    parser.add_argument("--enter-deg", type=float, default=80.0)
    parser.add_argument("--exit-deg", type=float, default=75.0)
    parser.add_argument("--reference-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "recordings" / "attitude_three_way")
    args = parser.parse_args()
    if not 0 < args.reference_seconds:
        parser.error("--reference-seconds 必须大于 0")
    try:
        device_id, samples = read_samples(args.input, args.device_id)
        rows, summary = compare(samples, args.enter_deg, args.exit_deg,
                                args.reference_seconds)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output_dir / (datetime.now().strftime("%Y%m%d-%H%M%S")
                                + "-" + uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary.update({"format": "wt901-attitude-three-way-v1",
                    "device_id": device_id, "input": str(args.input.resolve())})
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    create_plot(rows, output / "comparison.png", args.enter_deg, args.exit_deg)
    print(f"设备：{device_id}；{summary['samples']} 包，{summary['duration_s']:.1f} 秒")
    print(f"时间间隔中位数 {summary['timing']['median_dt_ms']:.1f} ms；"
          f">100 ms 间断 {summary['timing']['gaps_over_100_ms']} 次")
    label = "中立位回归" if summary["endpoint_kind"] == "return_to_neutral" else "末帧偏离起点（不保证已回中立）"
    print(label + " SO(3) 误差：" + " / ".join(
        f"{key} {value:.1f}°" for key, value in
        summary["endpoint_mean_so3_angle_deg"].items()))
    print(f"输出目录：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
