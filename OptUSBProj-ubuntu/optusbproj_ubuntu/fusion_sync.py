"""Offline camera/IMU timing diagnostics and SO(3) interpolation.

All times here are host monotonic timestamps. A UDP arrival is not the sensor's
sample time, and a VideoCapture timestamp is not the exposure time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from wt901_viz.orientation import rotation_matrix


# COCO-WholeBody and MediaPipe Pose shoulder/elbow/wrist indexes.
ARM_POINTS = {
    "rtmw3d": {"left_upper_arm": (5, 7), "left_forearm": (7, 9),
               "right_upper_arm": (6, 8), "right_forearm": (8, 10)},
    "mediapipe": {"left_upper_arm": (11, 13), "left_forearm": (13, 15),
                  "right_upper_arm": (12, 14), "right_forearm": (14, 16)},
}


@dataclass(frozen=True)
class LagEstimate:
    part: str
    device_id: str
    lag_ms: int | None
    correlation: float | None
    overlap_seconds: float
    visual_samples: int
    quality: str
    reason: str


class OrientationSeries:
    """Interpolate absolute rotations without interpolating wrapped Euler angles.

    ``lag_ms`` follows the analysis convention: IMU(t + lag) matches camera(t).
    No extrapolation is permitted; gaps larger than max_gap_ms are rejected.
    """

    def __init__(self, times_ns: np.ndarray, matrices: np.ndarray,
                 *, max_gap_ms: float = 100.0) -> None:
        times = np.asarray(times_ns, dtype=np.int64)
        mats = np.asarray(matrices, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or
                mats.shape != (len(times), 3, 3) or
                np.any(np.diff(times) <= 0) or max_gap_ms <= 0):
            raise ValueError("姿态时间序列必须严格递增，且至少包含两帧旋转矩阵")
        self.times_ns = times
        self.max_gap_ns = round(max_gap_ms * 1e6)
        self._origin_ns = int(times[0])
        self._slerp = Slerp((times - self._origin_ns).astype(np.float64),
                            Rotation.from_matrix(mats))

    def at(self, camera_ns: int, *, lag_ms: float = 0.0) -> np.ndarray | None:
        target = int(camera_ns) + round(lag_ms * 1e6)
        if target == self.times_ns[-1]:
            return self._slerp([float(target - self._origin_ns)]).as_matrix()[0]
        index = int(np.searchsorted(self.times_ns, target, side="right"))
        if index == 0 or index == len(self.times_ns):
            return None
        if self.times_ns[index] - self.times_ns[index - 1] > self.max_gap_ns:
            return None
        return self._slerp([float(target - self._origin_ns)]).as_matrix()[0]


def load_imu(path: Path, device_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, gyro, matrices = [], [], []
    with (path / "imu.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["device_id"] != device_id:
                continue
            times.append(int(row["host_recv_mono_ns"]))
            gyro.append([float(row[f"gyro_{axis}"]) for axis in "xyz"])
            matrices.append(rotation_matrix(*(
                float(row[name]) for name in ("roll", "pitch", "yaw"))))
    return (np.asarray(times, dtype=np.int64), np.asarray(gyro, dtype=float),
            np.asarray(matrices, dtype=float).reshape(-1, 3, 3))


def load_visual_motion(path: Path, part: str, *, min_confidence: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Projected limb-vector speed; invalid endpoints do not bridge gaps."""
    frames = []
    with (path / "camera.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            points = ARM_POINTS.get(row["backend"], {}).get(part)
            if points is None or row["keypoints_xy"] is None:
                continue
            xy = np.asarray(row["keypoints_xy"], dtype=float)
            confidence = np.asarray(row["confidence"], dtype=float)
            a, b = points
            if (min(confidence[a], confidence[b]) < min_confidence or
                    not np.isfinite(xy[[a, b]]).all()):
                frames.append((int(row["captured_mono_ns"]), None))
                continue
            vector = xy[b] - xy[a]
            if np.linalg.norm(vector) < 15:
                frames.append((int(row["captured_mono_ns"]), None))
                continue
            frames.append((int(row["captured_mono_ns"]), vector))
    lengths = [np.linalg.norm(vector) for _, vector in frames if vector is not None]
    if not lengths:
        return np.empty(0, dtype=np.int64), np.empty(0)
    scale = float(np.median(lengths))
    times, speeds = [], []
    for (t0, v0), (t1, v1) in zip(frames[:-1], frames[1:]):
        dt = (t1 - t0) / 1e9
        if v0 is None or v1 is None or not 0.02 <= dt <= 0.15:
            continue
        times.append((t0 + t1) // 2)
        speeds.append(float(np.linalg.norm(v1 - v0) / scale / dt))
    return np.asarray(times, dtype=np.int64), np.asarray(speeds, dtype=float)


def estimate_lag(visual_times_ns: np.ndarray, visual_speed: np.ndarray,
                 imu_times_ns: np.ndarray, imu_gyro: np.ndarray, *,
                 part: str, device_id: str, max_lag_ms: int = 400,
                 step_ms: int = 10) -> LagEstimate:
    """Cross-correlate activity, returning an uncertainty-aware diagnostic.

    Positive lag means IMU activity is later than visual activity in the
    recorded host time coordinates. This cannot recover absolute sensor delay.
    """
    def unavailable(reason: str) -> LagEstimate:
        return LagEstimate(part, device_id, None, None, 0.0,
                           len(visual_times_ns), "不可估计", reason)

    vt = np.asarray(visual_times_ns, dtype=np.int64)
    vs = np.asarray(visual_speed, dtype=float)
    it = np.asarray(imu_times_ns, dtype=np.int64)
    ig = np.asarray(imu_gyro, dtype=float)
    if (len(vt) < 20 or len(it) < 50 or len(vt) != len(vs) or
            ig.shape != (len(it), 3)):
        return unavailable("有效视觉/IMU 运动样本不足")
    if np.any(np.diff(vt) <= 0) or np.any(np.diff(it) <= 0):
        return unavailable("时间戳未严格递增")
    if max_lag_ms < 0 or step_ms <= 0:
        raise ValueError("搜索时间参数无效")
    origin = min(int(vt[0]), int(it[0]))
    visual_t = (vt - origin) / 1e9
    imu_t = (it - origin) / 1e9
    # A common interior interval makes every candidate use the same frames.
    margin = max_lag_ms / 1000 + 0.04
    keep = ((visual_t >= imu_t[0] + margin) &
            (visual_t <= imu_t[-1] - margin) & np.isfinite(vs))
    if np.count_nonzero(keep) < 20:
        return unavailable("共同有效时间段过短")
    vtime = visual_t[keep]
    v = gaussian_filter1d(vs[keep], 1.0)
    if vtime[-1] - vtime[0] < 3 or np.percentile(v, 90) - np.percentile(v, 10) < 0.15:
        return unavailable("视觉动作幅度不足，无法可靠估计延迟")
    activity = np.linalg.norm(ig, axis=1)
    # Smooth IMU activity (~40 ms) to compare with camera's slower cadence.
    dt = float(np.median(np.diff(imu_t)))
    if dt <= 0:
        return unavailable("IMU 采样间隔无效")
    activity = gaussian_filter1d(activity, max(0.5, 0.04 / dt))
    if np.percentile(activity, 90) - np.percentile(activity, 10) < 3:
        return unavailable("IMU 动作幅度不足，无法可靠估计延迟")
    lags = np.arange(-max_lag_ms, max_lag_ms + 1, step_ms)
    correlations = []
    for lag in lags:
        i = np.interp(vtime + lag / 1000, imu_t, activity)
        if np.std(i) < 1e-6 or np.std(v) < 1e-6:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(v, i)[0, 1]))
    correlations = np.asarray(correlations)
    if not np.isfinite(correlations).any():
        return unavailable("运动信号无有效相关性")
    best = int(np.nanargmax(correlations))
    peak = float(correlations[best])
    lag = int(lags[best])
    quality = "可参考" if peak >= 0.55 else "低可信"
    reason = ("仅根据 2D 肢段运动与 IMU 角速度相关性估计；"
              "不是硬件采样延迟或精确同步")
    if abs(lag) >= max_lag_ms - step_ms:
        quality = "低可信"
        reason = "最佳值落在搜索边界；应扩大动作片段或重新录制"
    elif peak < 0.55:
        reason = "两种运动信号相关性较弱；建议录制多次明显、分开的抬臂动作"
    return LagEstimate(part, device_id, lag, peak,
                       float(vtime[-1] - vtime[0]), int(len(vtime)), quality, reason)


def analyze_session(path: Path, assignments: dict[str, str], *,
                    max_lag_ms: int = 400) -> list[LagEstimate]:
    results = []
    for part, device_id in assignments.items():
        if part not in ARM_POINTS["rtmw3d"] or not device_id:
            continue
        vt, vs = load_visual_motion(path, part)
        it, gyro, _ = load_imu(path, device_id)
        results.append(estimate_lag(vt, vs, it, gyro, part=part,
                                    device_id=device_id, max_lag_ms=max_lag_ms))
    return results


def aligned_orientations(path: Path, device_lags_ms: dict[str, float], *,
                         max_gap_ms: float = 100.0) -> list[dict]:
    """Sample each device's SO(3) orientation at every camera frame time.

    A missing/out-of-gap rotation is ``None``. The result retains capture
    timestamps and explicitly states the lag used for each device. It does
    not apply neutral-pose or sensor-to-body calibration.
    """
    series = {}
    for device_id in device_lags_ms:
        times, _, matrices = load_imu(path, device_id)
        if len(times) >= 2:
            series[device_id] = OrientationSeries(times, matrices,
                                                   max_gap_ms=max_gap_ms)
    output = []
    with (path / "camera.jsonl").open(encoding="utf-8") as source:
        for line in source:
            frame = json.loads(line)
            stamp = int(frame["captured_mono_ns"])
            orientations = {}
            for device_id, lag in device_lags_ms.items():
                matrix = (series[device_id].at(stamp, lag_ms=lag)
                          if device_id in series else None)
                orientations[device_id] = (None if matrix is None else matrix.tolist())
            output.append({"frame": frame["frame"], "captured_mono_ns": stamp,
                           "lag_ms": device_lags_ms.copy(),
                           "orientation_matrix": orientations})
    return output
