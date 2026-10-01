"""Dual-IMU time pairing and neutral-relative orientation checks."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT.parent), str(PROJECT)]

from imu_relative_experiment import (DualRecorder, FOREARM_ID, HAND_ID,
                                     TOTAL_SECONDS, make_relative_rows, stage_at)
from wt901.protocol import AttitudePacket


def packet(device_id: str, rotation: Rotation) -> AttitudePacket:
    roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
    return AttitudePacket(
        device_id=device_id,
        year=2026, month=9, day=29, hour=10, minute=0, second=0, millisecond=0,
        acc_x=0.0, acc_y=0.0, acc_z=1.0,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        mag_x=30.0, mag_y=20.0, mag_z=10.0,
        roll=float(roll), pitch=float(pitch), yaw=float(yaw),
        temperature=25.0, battery_v=3.8, rssi=-50, version=13038,
    )


class RelativeExperimentTests(unittest.TestCase):
    def test_stage_boundaries(self):
        self.assertEqual(stage_at(0)[0], "neutral_before")
        self.assertEqual(stage_at(5)[0], "raise")
        self.assertEqual(stage_at(13)[0], "twist_out")
        self.assertIsNone(stage_at(TOTAL_SECONDS))

    def test_joint_motion_cancels_and_wrist_motion_remains(self):
        start = 1_000_000_000
        fixed_mount = Rotation.from_euler("xyz", [14, -9, 18], degrees=True)
        forearm_samples = []
        hand_samples = []
        for index in range(351):
            t = index * 0.02
            common = Rotation.from_euler(
                "z", 45 * np.clip((t - 5) / 0.5, 0, 1), degrees=True)
            wrist = Rotation.from_euler(
                "x", 30 * np.clip((t - 6) / 0.5, 0, 1), degrees=True)
            forearm_samples.append((start + round(t * 1e9), packet(FOREARM_ID, common)))
            hand_samples.append((start + round((t + 0.01) * 1e9),
                                 packet(HAND_ID, common * wrist * fixed_mount)))
        rows, stats = make_relative_rows(hand_samples, forearm_samples, start)
        self.assertGreater(stats["valid_pairs"], 300)
        self.assertEqual(stats["rejected_pairs"], 1)  # last hand lacks right bracket
        self.assertGreater(stats["neutral_baseline_pairs"], 100)
        before_wrist = min(rows, key=lambda row: abs(row["elapsed_s"] - 5.7))
        after_wrist = min(rows, key=lambda row: abs(row["elapsed_s"] - 6.8))
        self.assertLess(before_wrist["neutral_delta_angle_deg"], 0.01)
        self.assertAlmostEqual(before_wrist["forearm_neutral_delta_angle_deg"], 45,
                               delta=0.01)
        self.assertAlmostEqual(before_wrist["hand_neutral_delta_angle_deg"], 45,
                               delta=0.01)
        self.assertAlmostEqual(after_wrist["neutral_delta_angle_deg"], 30, delta=0.01)
        self.assertAlmostEqual(stats["pair_offset_median_ms"], 10, delta=0.01)

    def test_rejects_long_receive_gap(self):
        start = 1_000_000_000
        identity = Rotation.identity()
        hand = [(start + 50_000_000, packet(HAND_ID, identity))]
        forearm = [(start, packet(FOREARM_ID, identity)),
                   (start + 100_000_000, packet(FOREARM_ID, identity))]
        rows, stats = make_relative_rows(hand, forearm, start)
        self.assertEqual(rows, [])
        self.assertEqual(stats["rejected_pairs"], 1)

    def test_recorder_saves_both_raw_streams_and_relative_file(self):
        with tempfile.TemporaryDirectory() as directory:
            start = 1_000_000_000
            recorder = DualRecorder(Path(directory), HAND_ID, FOREARM_ID, start)
            identity = Rotation.identity()
            for index in range(300):
                stamp = start + index * 20_000_000
                recorder.append(packet(FOREARM_ID, identity), stamp)
                recorder.append(packet(HAND_ID, identity), stamp + 10_000_000)
            stats = recorder.close()
            with (recorder.path / "samples.csv").open(newline="", encoding="utf-8") as file:
                raw = list(csv.DictReader(file))
            with (recorder.path / "relative.csv").open(newline="", encoding="utf-8") as file:
                relative = list(csv.DictReader(file))
            manifest = json.loads((recorder.path / "manifest.json").read_text())
            self.assertEqual(len(raw), 600)
            self.assertEqual({row["role"] for row in raw}, {"hand", "forearm"})
            self.assertEqual(len(relative), 299)
            self.assertEqual(stats["valid_pairs"], 299)
            self.assertEqual(manifest["stop_reason"], "completed")
            self.assertEqual(manifest["sample_counts"], {"hand": 300, "forearm": 300})
            self.assertAlmostEqual(float(relative[100]["neutral_delta_angle_deg"]), 0)


if __name__ == "__main__":
    unittest.main()
