from __future__ import annotations

import sys
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.fusion_sync import (
    OrientationSeries, aligned_orientations, estimate_lag)


class FusionSyncTests(unittest.TestCase):
    def test_slerp_across_euler_wrap(self):
        times = np.array([1_000_000_000, 1_100_000_000], dtype=np.int64)
        matrices = Rotation.from_euler("z", [[179], [-179]], degrees=True).as_matrix()
        series = OrientationSeries(times, matrices, max_gap_ms=120)
        middle = series.at(1_050_000_000)
        self.assertIsNotNone(middle)
        self.assertAlmostEqual(abs(Rotation.from_matrix(middle).as_euler(
            "xyz", degrees=True)[2]), 180, places=3)
        self.assertIsNone(series.at(999_999_999))
        self.assertIsNotNone(series.at(1_100_000_000))

    def test_lag_shifts_camera_time_before_interpolation(self):
        times = np.array([1_000_000_000, 1_100_000_000], dtype=np.int64)
        matrices = Rotation.from_euler("x", [[0], [90]], degrees=True).as_matrix()
        series = OrientationSeries(times, matrices, max_gap_ms=120)
        middle = series.at(1_000_000_000, lag_ms=50)
        self.assertAlmostEqual(Rotation.from_matrix(middle).as_euler(
            "xyz", degrees=True)[0], 45, places=3)
        self.assertIsNone(OrientationSeries(times, matrices, max_gap_ms=90).at(
            1_050_000_000))

    def test_recovers_known_host_axis_lag(self):
        # Separated motion bursts avoid a single broad correlation peak.
        def motion(t):
            return (70 * np.exp(-((t - 2.2) / .18) ** 2)
                    + 105 * np.exp(-((t - 4.9) / .26) ** 2)
                    + 85 * np.exp(-((t - 7.7) / .15) ** 2))

        video_t = np.arange(0, 10, 1 / 20)
        imu_t = np.arange(0, 10, 1 / 50)
        lag = .12
        video_speed = motion(video_t)
        gyro = np.column_stack((motion(imu_t - lag), np.zeros_like(imu_t),
                                np.zeros_like(imu_t)))
        result = estimate_lag((video_t * 1e9).astype(np.int64), video_speed,
                              (imu_t * 1e9).astype(np.int64), gyro,
                              part="right_upper_arm", device_id="test")
        self.assertLessEqual(abs(result.lag_ms - 120), 20)
        self.assertGreater(result.correlation, .8)

    def test_static_motion_is_not_reported_as_a_measured_lag(self):
        video_t = (np.arange(200) * 50_000_000).astype(np.int64)
        imu_t = (np.arange(500) * 20_000_000).astype(np.int64)
        result = estimate_lag(video_t, np.zeros(200), imu_t,
                              np.zeros((500, 3)), part="right_forearm",
                              device_id="test")
        self.assertIsNone(result.lag_ms)
        self.assertEqual(result.quality, "不可估计")

    def test_align_session_preserves_missing_samples(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / "imu.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=(
                    "device_id", "host_recv_mono_ns", "roll", "pitch", "yaw",
                    "gyro_x", "gyro_y", "gyro_z"))
                writer.writeheader()
                for stamp, yaw in ((1_000_000_000, 0), (1_100_000_000, 90)):
                    writer.writerow(dict(device_id="WT", host_recv_mono_ns=stamp,
                                         roll=0, pitch=0, yaw=yaw,
                                         gyro_x=0, gyro_y=0, gyro_z=0))
            with (path / "camera.jsonl").open("w", encoding="utf-8") as output:
                for stamp in (1_050_000_000, 1_200_000_000):
                    output.write(json.dumps({"frame": "frame.jpg",
                                             "captured_mono_ns": stamp}) + "\n")
            rows = aligned_orientations(path, {"WT": 0}, max_gap_ms=120)
            self.assertEqual(len(rows), 2)
            self.assertIsNotNone(rows[0]["orientation_matrix"]["WT"])
            self.assertIsNone(rows[1]["orientation_matrix"]["WT"])


if __name__ == "__main__":
    unittest.main()
