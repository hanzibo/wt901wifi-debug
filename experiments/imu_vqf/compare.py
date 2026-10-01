#!/usr/bin/env python3
"""Offline, same-input comparison of our 6-axis SO(3) filter and VQF 6D.

No magnetometer or vendor R/P/Y enters either filter. Both see one uniformly
resampled gyro+accelerometer stream derived from a single WT901 recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wt901_viz.so3_filter import AdaptiveTiltFilter


G = 9.80665
SAMPLE_FIELDS = ("elapsed_s", "gyro_x", "gyro_y", "gyro_z",
                 "acc_x", "acc_y", "acc_z", "battery_v", "temperature")
DEVICE_TIME_FIELDS = ("hour", "minute", "second", "millisecond")


def quat_wxyz_to_rotation(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("VQF returned invalid quaternion")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                     (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                     (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))))


def angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def tilt_and_heading_deg(relative: np.ndarray, initial_up: np.ndarray) -> tuple[float, float]:
    """Return gravity-axis tilt and a heading proxy (valid near neutral)."""
    up = initial_up / np.linalg.norm(initial_up)
    tilt = angle_between_deg(up, relative @ up)
    basis = np.eye(3)[int(np.argmin(np.abs(up)))]
    horizontal = basis - np.dot(basis, up) * up
    horizontal /= np.linalg.norm(horizontal)
    mapped = relative @ horizontal
    mapped -= np.dot(mapped, up) * up
    mapped /= np.linalg.norm(mapped)
    heading = math.degrees(math.atan2(np.dot(up, np.cross(horizontal, mapped)),
                                      np.dot(horizontal, mapped)))
    return tilt, heading


def read_samples(path: Path, device_id: str | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_format = manifest.get("format")
    if source_format in ("wt901-single-axis-experiment-v1",
                         "wt901-dual-relative-experiment-v1"):
        stages = manifest.get("stages", [])
        if (len(stages) < 3 or stages[0].get("name") != "neutral_before"
                or stages[-1].get("name") != "neutral_after"):
            raise ValueError("旧单设备录制缺少起止中立位阶段")
        if source_format == "wt901-dual-relative-experiment-v1":
            available = (manifest["hand_id"], manifest["forearm_id"])
            if device_id is None:
                device_id = manifest["hand_id"]
            matches = [candidate for candidate in available
                       if candidate == device_id or candidate.endswith(device_id)]
            if len(matches) != 1:
                raise ValueError(f"设备选择 {device_id!r} 不唯一或不在录制中：{available}")
            device_id = matches[0]
        elif device_id is not None and device_id not in (manifest["device_id"],
                                                        manifest["device_id"][-len(device_id):]):
            raise ValueError(f"录制设备为 {manifest['device_id']}，并非 {device_id}")
        manifest = {
            "format": source_format,
            "device_id": device_id or manifest["device_id"],
            "neutral_seconds_each": float(stages[0]["seconds"]),
            "movement_seconds": float(sum(stage["seconds"] for stage in stages[1:-1])),
            "end_neutral_seconds": float(stages[-1]["seconds"]),
            "status": manifest.get("stop_reason"),
        }
    elif source_format == "wt901-imu-long-run-v1":
        if device_id is not None and not manifest["device_id"].endswith(device_id):
            raise ValueError(f"录制设备为 {manifest['device_id']}，并非 {device_id}")
        manifest["end_neutral_seconds"] = manifest["neutral_seconds_each"]
    else:
        raise ValueError("仅支持 long_run、imu_axis、imu_relative 原始录制")
    rows: list[list[float]] = []
    with (path / "samples.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(SAMPLE_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("samples.csv 缺少必要列")
        has_device_time = set(DEVICE_TIME_FIELDS).issubset(reader.fieldnames or ())
        for row in reader:
            if row["device_id"] != manifest["device_id"]:
                if source_format == "wt901-dual-relative-experiment-v1":
                    continue
                raise ValueError("录制中混入另一台设备")
            values = [float(row[key]) for key in SAMPLE_FIELDS]
            if has_device_time:
                values.extend(float(row[key]) for key in DEVICE_TIME_FIELDS)
            if np.isfinite(values).all():
                rows.append(values)
    if len(rows) < 100:
        raise ValueError("有效数据不足 100 包")
    raw = np.asarray(rows)
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    keep = np.r_[True, np.diff(raw[:, 0]) > 1e-6]
    raw = raw[keep]
    host_t = raw[:, 0].copy()
    host_intervals = np.diff(host_t)
    t = host_t.copy()
    manifest["analysis_timebase"] = "host_receive"
    if has_device_time:
        seconds_of_day = (raw[:, 9] * 3600 + raw[:, 10] * 60
                          + raw[:, 11] + raw[:, 12] / 1000)
        rollovers = np.cumsum(np.r_[0, np.diff(seconds_of_day) < -43200])
        device_seconds = seconds_of_day + rollovers * 86400
        device_intervals = np.diff(device_seconds)
        plausible = ((device_intervals > 0.005) & (device_intervals < 0.1))
        if len(device_intervals) and np.mean(plausible) >= 0.99:
            # WT901 may have no valid calendar date, but its H:M:S:ms field
            # is a stable relative clock. Preserve the host phase offset.
            t = host_t[0] + device_seconds - device_seconds[0]
            raw[:, 0] = t
            manifest["analysis_timebase"] = "device_hms_ms"
    if t[-1] - t[0] < 2:
        raise ValueError("数据不足 2 秒")
    return t, raw, np.diff(t), host_intervals, manifest


def resample(t: np.ndarray, raw: np.ndarray, intervals: np.ndarray,
             nominal_hz: float | None, timebase: str = "host_receive"
             ) -> tuple[np.ndarray, np.ndarray, float]:
    regular = intervals[(intervals >= 0.005) & (intervals <= 0.1)]
    if len(regular) < 50 and nominal_hz is None:
        raise ValueError("无法从接收时间推断回传频率；用 --hz 指定")
    if nominal_hz is not None:
        hz = nominal_hz
    elif timebase == "device_hms_ms":
        # Missing packets lower count/duration, not the IMU's actual rate.
        hz = 1.0 / float(np.median(regular))
    else:
        # Host intervals are biased by Wi-Fi batching; use the full span.
        hz = (len(t) - 1) / float(t[-1] - t[0])
    hz = min(200.0, max(10.0, hz))
    ts = 1.0 / hz
    grid = np.arange(t[0], t[-1], ts)
    # Apply one interpolation to the raw sensor axes, then feed *both* filters.
    out = np.column_stack([np.interp(grid, t, raw[:, column])
                           for column in range(1, 7)])
    return grid, out, hz


def run_estimators(grid: np.ndarray, measurements: np.ndarray, hz: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from vqf import VQF
    except ImportError as exc:
        raise RuntimeError("缺少 VQF：请在 wt901 环境执行 python -m pip install vqf") from exc
    local = AdaptiveTiltFilter()
    published = VQF(1.0 / hz)
    own = np.empty((len(grid), 3, 3))
    vqf = np.empty_like(own)
    own_bias = np.empty((len(grid), 3))
    vqf_bias = np.empty_like(own_bias)
    for index, (time_s, sample) in enumerate(zip(grid, measurements)):
        gyro_dps, acc_g = sample[:3], sample[3:]
        estimate = local.update(float(time_s - grid[0]), tuple(gyro_dps), tuple(acc_g))
        published.update(np.deg2rad(gyro_dps), acc_g * G)  # No mag: 6D only.
        own[index] = estimate.rotation
        vqf[index] = quat_wxyz_to_rotation(published.getQuat6D())
        own_bias[index] = estimate.gyro_bias_dps
        vqf_bias[index] = np.rad2deg(published.getBiasEstimate()[0])
    return own, vqf, own_bias, vqf_bias


def nearest_rotation_mean(rotations: np.ndarray) -> np.ndarray:
    """Chordal SO(3) mean; adequate for a nearly static neutral window."""
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    fix = np.diag((1.0, 1.0, 1.0 if np.linalg.det(u @ vt) > 0 else -1.0))
    return u @ fix @ vt


def window_indices(grid: np.ndarray, begin: float, end: float) -> np.ndarray:
    return np.flatnonzero((grid >= begin) & (grid < end))


def pose_metrics(rotation: np.ndarray, start_ref: np.ndarray,
                 end_indices: np.ndarray, initial_up: np.ndarray) -> dict:
    final_ref = nearest_rotation_mean(rotation[end_indices])
    relative = start_ref.T @ final_ref
    tilt, heading = tilt_and_heading_deg(relative, initial_up)
    sample_angles = [angle_deg(start_ref.T @ rotation[index]) for index in end_indices]
    return {
        "return_so3_angle_deg": angle_deg(relative),
        "return_tilt_deg": tilt,
        "return_heading_proxy_deg": heading,
        "end_neutral_relative_angle_median_deg": float(np.median(sample_angles)),
        "end_neutral_relative_angle_p95_deg": float(np.percentile(sample_angles, 95)),
    }


def make_svg(path: Path, rows: list[dict[str, float]], movement_end: float) -> None:
    width, height = 1100, 480
    series = (("local_so3_deg", "现有算法", "#1f77b4"),
              ("vqf_so3_deg", "VQF 6D", "#e67e22"),
              ("between_deg", "两算法差异", "#c0392b"))
    ymax = max(5.0, float(np.percentile([row[key] for row in rows for key, _, _ in series], 99)))
    tmax = max(row["elapsed_s"] for row in rows)
    def x(value: float) -> float:
        return 70 + 980 * value / max(tmax, 1)
    def y(value: float) -> float:
        return 420 - 350 * min(value, ymax) / ymax
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="70" y="30" font-size="18">起始中立位相对转角（仅为运动幅度，非真值误差）</text>',
             '<path d="M70 70 V420 H1050" fill="none" stroke="#777"/>',
             f'<text x="20" y="76" font-size="13">{ymax:.0f}°</text>',
             '<text x="28" y="422" font-size="13">0°</text>',
             f'<text x="950" y="450" font-size="13">{tmax/60:.1f} min</text>']
    if 0 <= movement_end <= tmax:
        parts.append(f'<path d="M{x(movement_end):.1f} 70 V420" stroke="#aaa" stroke-dasharray="5 5"/>')
    for key, label, color in series:
        points = " ".join(f'{x(row["elapsed_s"]):.1f},{y(row[key]):.1f}' for row in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>')
        parts.append(f'<text x="{70+series.index((key,label,color))*230}" y="465" fill="{color}" font-size="14">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze(path: Path, output: Path, nominal_hz: float | None,
            device_id: str | None = None) -> dict:
    t, raw, intervals, host_intervals, manifest = read_samples(path, device_id)
    grid, measurements, hz = resample(t, raw, intervals, nominal_hz,
                                      manifest["analysis_timebase"])
    own, published, own_bias, vqf_bias = run_estimators(grid, measurements, hz)
    neutral_s = float(manifest["neutral_seconds_each"])
    move_s = float(manifest["movement_seconds"])
    # Last 2 s of the initial pose, and last 2 s of the final pose.
    start = window_indices(grid, max(t[0], neutral_s - 2), neutral_s)
    total = neutral_s + move_s + float(manifest["end_neutral_seconds"])
    end = window_indices(grid, max(neutral_s + move_s, total - 2), min(total, grid[-1] + 1e-6))
    if len(start) < 5 or len(end) < 5:
        raise ValueError("起止静止窗口不足；需完整录制起始和结束中立位")
    up_start = np.mean(measurements[start, 3:], axis=0)
    up_end = np.mean(measurements[end, 3:], axis=0)
    own_ref = nearest_rotation_mean(own[start])
    vqf_ref = nearest_rotation_mean(published[start])
    own_metrics = pose_metrics(own, own_ref, end, up_start)
    vqf_metrics = pose_metrics(published, vqf_ref, end, up_start)
    start_gyro_p95 = float(np.percentile(np.linalg.norm(measurements[start, :3], axis=1), 95))
    end_gyro_p95 = float(np.percentile(np.linalg.norm(measurements[end, :3], axis=1), 95))
    # Compare physical *relative* rotations, not their unrelated world frames.
    every_second = np.unique(np.searchsorted(grid, np.arange(math.ceil(grid[0]), grid[-1], 1.0)))
    every_second = every_second[every_second < len(grid)]
    rows = []
    for index in every_second:
        local_rel = own_ref.T @ own[index]
        vqf_rel = vqf_ref.T @ published[index]
        own_tilt, own_heading = tilt_and_heading_deg(local_rel, up_start)
        vqf_tilt, vqf_heading = tilt_and_heading_deg(vqf_rel, up_start)
        rows.append({
            "elapsed_s": round(float(grid[index]), 3),
            "local_so3_deg": angle_deg(local_rel),
            "vqf_so3_deg": angle_deg(vqf_rel),
            "between_deg": angle_deg(local_rel.T @ vqf_rel),
            "local_tilt_deg": own_tilt,
            "vqf_tilt_deg": vqf_tilt,
            "local_heading_proxy_deg": own_heading,
            "vqf_heading_proxy_deg": vqf_heading,
            "local_bias_x_dps": float(own_bias[index, 0]),
            "local_bias_y_dps": float(own_bias[index, 1]),
            "local_bias_z_dps": float(own_bias[index, 2]),
            "vqf_bias_x_dps": float(vqf_bias[index, 0]),
            "vqf_bias_y_dps": float(vqf_bias[index, 1]),
            "vqf_bias_z_dps": float(vqf_bias[index, 2]),
            "battery_v": float(np.interp(grid[index], t, raw[:, 7])),
            "temperature_c": float(np.interp(grid[index], t, raw[:, 8])),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison_1hz.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_svg(output / "comparison.svg", rows, neutral_s + move_s)
    gap_count = int(np.sum(intervals > 0.2))
    summary = {
        "format": "wt901-vqf-comparison-v1",
        "source_format": manifest["format"],
        "timebase": manifest["analysis_timebase"],
        "source": str(path),
        "device_id": manifest["device_id"],
        "recording_status": manifest["status"],
        "duration_s": float(t[-1] - t[0]),
        "received_samples": len(t),
        "uniform_samples": len(grid),
        "uniform_hz": hz,
        "longest_host_receive_gap_s": float(np.max(host_intervals)),
        "host_receive_gaps_over_200ms": int(np.sum(host_intervals > 0.2)),
        "longest_sample_time_gap_s": float(np.max(intervals)),
        "sample_gaps_over_200ms": gap_count,
        "battery_start_v": float(raw[0, 7]),
        "battery_end_v": float(raw[-1, 7]),
        "temperature_start_c": float(raw[0, 8]),
        "temperature_end_c": float(raw[-1, 8]),
        "raw_acc_start_end_direction_diff_deg": angle_between_deg(up_start, up_end),
        "start_neutral_gyro_p95_dps": start_gyro_p95,
        "end_neutral_gyro_p95_dps": end_gyro_p95,
        "local_filter": own_metrics | {"final_bias_dps": own_bias[-1].tolist()},
        "vqf_6d": vqf_metrics | {"final_bias_dps": vqf_bias[-1].tolist()},
        "eligible_for_return_comparison": bool(manifest["status"] == "completed"
                                                and np.max(intervals) <= 1.0
                                                and angle_between_deg(up_start, up_end) < 5
                                                and start_gyro_p95 < 5
                                                and end_gyro_p95 < 5),
        "cautions": [
            "No external orientation ground truth: return residual also includes imperfect physical repositioning.",
            "Gravity checks tilt, not physical yaw return; use a fixture or external heading reference for yaw accuracy.",
            "A monotonic device H:M:S:ms clock is preferred; otherwise host receive time is used.",
            "Both filters use the same uniformly interpolated raw gyro/acc stream.",
            "Gaps over 200 ms are counted; a gap over 1 s disqualifies the return comparison.",
            "Heading proxy is meaningful near neutral, not during large tilt.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="long_run.py 创建的录制目录")
    parser.add_argument("--output", type=Path, help="默认写到录制目录的 comparison/ 子目录")
    parser.add_argument("--hz", type=float, help="覆盖按电脑收包间隔推断的频率")
    parser.add_argument("--device-id", help="双设备录制中选择手或前臂的完整 ID/唯一尾号；默认手背")
    args = parser.parse_args()
    if args.hz is not None and not 10 <= args.hz <= 200:
        parser.error("--hz 须在 10~200 Hz")
    recording = args.recording.resolve()
    if args.output is not None:
        destination = args.output.resolve()
    elif args.device_id is not None:
        destination = recording / f"comparison_{args.device_id}"
    else:
        destination = recording / "comparison"
    try:
        result = analyze(recording, destination, args.hz, args.device_id)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"分析失败：{exc}", file=sys.stderr)
        return 1
    print(f"设备 {result['device_id']}，{result['duration_s']/60:.1f} 分钟；"
          f"VQF 与现有算法同用 {result['uniform_hz']:.2f} Hz 原始六轴数据；"
          f"时间轴 {result['timebase']}")
    print(f"接收间隔最长 {result['longest_host_receive_gap_s']:.3f}s；"
          f"设备数据间隔最长 {result['longest_sample_time_gap_s']:.3f}s；"
          f">200ms 数据间隔 {result['sample_gaps_over_200ms']} 个；"
          f"起止加计方向差 {result['raw_acc_start_end_direction_diff_deg']:.2f}°")
    for name, label in (("local_filter", "现有算法"), ("vqf_6d", "VQF 6D")):
        values = result[name]
        print(f"{label}：回到中立位 SO(3) 残差 {values['return_so3_angle_deg']:.2f}°；"
              f"倾斜 {values['return_tilt_deg']:.2f}°；"
              f"航向代理 {values['return_heading_proxy_deg']:+.2f}°")
    if not result["eligible_for_return_comparison"]:
        print("注意：录制不完整、存在 >1s 断流、起止倾斜不一致或中立位未静止；"
              "本次回零比较不宜作为算法结论。")
    print(f"结果：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
