"""Pure helpers of the single-IMU long-run and offline comparison."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.imu_vqf.compare import (
    analyze, angle_deg, nearest_rotation_mean, quat_wxyz_to_rotation, read_samples,
    run_estimators,
    resample, tilt_and_heading_deg,
)
from experiments.imu_vqf.long_run import clock_text, phase_at, progress_text, select_device
from wt901_viz.so3_filter import rotation_step


class LongRunTests(unittest.TestCase):
    def test_phase_boundaries(self):
        self.assertEqual(phase_at(0, 10, 60), "neutral_start")
        self.assertEqual(phase_at(10, 10, 60), "movement")
        self.assertEqual(phase_at(70, 10, 60), "neutral_end")
        self.assertIsNone(phase_at(80, 10, 60))

    def test_auto_select_highest_voltage_and_override(self):
        candidates = {"WT5500011590": SimpleNamespace(battery_v=3.96),
                      "WT5500011381": SimpleNamespace(battery_v=3.80)}
        self.assertEqual(select_device(candidates, None), "WT5500011590")
        self.assertEqual(select_device(candidates, "11381"), "WT5500011381")
        with self.assertRaises(RuntimeError):
            select_device(candidates, "missing")

    def test_progress_reports_elapsed_stage_and_link(self):
        self.assertEqual(clock_text(3661.9), "01:01:01")
        text = progress_text(75.2, 10, 1800, 3725, .05, 3.92)
        self.assertIn("已用 01:15 / 30:20", text)
        self.assertIn("运动剩余 28:55", text)
        self.assertIn("包 3725", text)
        self.assertIn("在线", text)
        self.assertIn("3.92 V", text)
        self.assertIn("断流 3.0s", progress_text(5, 10, 1800, 0, 3, 3.8))


class ComparisonMathTests(unittest.TestCase):
    def test_quaternion_rotation_convention(self):
        quarter = np.array((np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))
        matrix = quat_wxyz_to_rotation(quarter)
        np.testing.assert_allclose(matrix @ (1, 0, 0), (0, 1, 0), atol=1e-12)
        self.assertAlmostEqual(angle_deg(matrix), 90)

    def test_neutral_mean_and_yaw_tilt(self):
        rot = rotation_step(np.deg2rad((0, 0, 30)))
        mean = nearest_rotation_mean(np.stack((rot, rot, rot)))
        np.testing.assert_allclose(mean, rot, atol=1e-12)
        tilt, heading = tilt_and_heading_deg(rot, np.array((0, 0, 1)))
        self.assertAlmostEqual(tilt, 0)
        self.assertAlmostEqual(heading, 30)

    def test_uniform_interpolation_is_shared(self):
        times = np.arange(0, 2, .02)
        raw = np.column_stack((times, np.ones((len(times), 6))))
        grid, samples, hz = resample(times, raw, np.diff(times), None)
        self.assertAlmostEqual(hz, 50)
        np.testing.assert_allclose(samples, 1)
        self.assertGreater(len(grid), 90)

    def test_device_clock_beats_batched_host_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "manifest.json").write_text(json.dumps({
                "format": "wt901-imu-long-run-v1", "device_id": "WT5500011590",
                "neutral_seconds_each": 1, "movement_seconds": 1,
                "status": "completed"}), encoding="utf-8")
            columns = ("device_id", "elapsed_s", "gyro_x", "gyro_y", "gyro_z",
                       "acc_x", "acc_y", "acc_z", "battery_v", "temperature",
                       "hour", "minute", "second", "millisecond")
            with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for index in range(151):
                    writer.writerow({"device_id": "WT5500011590", "elapsed_s": .021 * index,
                                     "gyro_x": 0, "gyro_y": 0, "gyro_z": 0,
                                     "acc_x": 0, "acc_y": 0, "acc_z": 1,
                                     "battery_v": 3.9, "temperature": 25,
                                     "hour": 0, "minute": 0, "second": index // 50,
                                     "millisecond": index % 50 * 20})
            t, raw, intervals, host_intervals, meta = read_samples(folder)
            self.assertEqual(meta["analysis_timebase"], "device_hms_ms")
            self.assertAlmostEqual(float(np.median(intervals)), .02)
            self.assertAlmostEqual(float(np.median(host_intervals)), .021)
            _, _, hz = resample(t, raw, intervals, None, meta["analysis_timebase"])
            self.assertAlmostEqual(hz, 50)

    @unittest.skipIf(importlib.util.find_spec("vqf") is None, "VQF optional dependency absent")
    def test_installed_vqf_six_axis_api(self):
        grid = np.arange(0, 4, .02)
        measurements = np.tile((0, 0, 0, 0, 0, 1), (len(grid), 1))
        local, vqf, local_bias, vqf_bias = run_estimators(grid, measurements, 50)
        np.testing.assert_allclose(local[-1], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(vqf[-1], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(local_bias[-1], 0, atol=1e-8)
        np.testing.assert_allclose(vqf_bias[-1], 0, atol=1e-8)

    def test_complete_report_from_synthetic_recording(self):
        class FakeVQF:
            def __init__(self, ts):
                self.ts = ts

            def update(self, gyro, acc):
                self.gyro = gyro
                self.acc = acc

            def getQuat6D(self):
                return np.array((1., 0., 0., 0.))

            def getBiasEstimate(self):
                return np.zeros(3), 0.

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "manifest.json").write_text(json.dumps({
                "format": "wt901-imu-long-run-v1",
                "device_id": "WT5500011590",
                "neutral_seconds_each": 2,
                "movement_seconds": 3,
                "status": "completed",
            }), encoding="utf-8")
            with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "device_id", "elapsed_s", "gyro_x", "gyro_y", "gyro_z",
                    "acc_x", "acc_y", "acc_z", "battery_v", "temperature"))
                writer.writeheader()
                for time_s in np.arange(0, 7, .02):
                    writer.writerow({"device_id": "WT5500011590", "elapsed_s": time_s,
                                     "gyro_x": 0, "gyro_y": 0, "gyro_z": 0,
                                     "acc_x": 0, "acc_y": 0, "acc_z": 1,
                                     "battery_v": 3.9, "temperature": 25})
            with patch.dict(sys.modules, {"vqf": SimpleNamespace(VQF=FakeVQF)}):
                result = analyze(folder, folder / "comparison", None)
            self.assertTrue(result["eligible_for_return_comparison"])
            self.assertLess(result["local_filter"]["return_so3_angle_deg"], .01)
            self.assertLess(result["vqf_6d"]["return_so3_angle_deg"], .01)
            self.assertTrue((folder / "comparison" / "comparison.svg").exists())


if __name__ == "__main__":
    unittest.main()
