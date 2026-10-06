"""Offline three-way comparison without an IMU or a display."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from attitude_three_way import Sample, compare


class AttitudeThreeWayTests(unittest.TestCase):
    def test_high_pitch_false_yaw_and_return_to_neutral(self):
        samples = []
        for index in range(401):
            t = index * 0.02
            if t < 2:
                pitch, gyro_y = 0.0, 0.0
                phase = "neutral_before"
            elif t < 3:
                pitch, gyro_y = 85.0 * (t - 2), 85.0
                phase = "raise"
            elif t < 5:
                pitch, gyro_y = 85.0, 0.0
                phase = "hold"
            elif t < 6:
                pitch, gyro_y = 85.0 * (6 - t), -85.0
                phase = "lower"
            else:
                pitch, gyro_y = 0.0, 0.0
                phase = "neutral_after"
            false_yaw = float(60.0 * np.clip((t - 3.2) / 1.5, 0, 1))
            total_ms = index * 20
            packet = SimpleNamespace(hour=0, minute=0,
                                     second=total_ms // 1000,
                                     millisecond=total_ms % 1000,
                                     roll=0.0, pitch=pitch, yaw=false_yaw,
                                     gyro_x=0.0, gyro_y=gyro_y, gyro_z=0.0)
            samples.append(Sample(packet, phase, index * 20_000_000))
        rows, summary = compare(samples)
        self.assertEqual(summary["endpoint_kind"], "return_to_neutral")
        self.assertAlmostEqual(summary["endpoint_mean_so3_angle_deg"]["raw"],
                               60.0, delta=1.0)
        self.assertLess(summary["endpoint_mean_so3_angle_deg"]["gyro"], 2.0)
        self.assertLess(summary["endpoint_mean_so3_angle_deg"]["guard"], 2.0)
        self.assertEqual(summary["timing"]["gaps_over_100_ms"], 0)
        self.assertTrue(any(row["guard_active"] for row in rows))

    def test_no_claim_of_return_without_explicit_phase(self):
        packet = SimpleNamespace(hour=0, minute=0, second=0, millisecond=0,
                                 roll=0.0, pitch=0.0, yaw=0.0,
                                 gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        samples = []
        for index in range(20):
            samples.append(Sample(SimpleNamespace(**{
                **vars(packet), "millisecond": index * 20}), "unknown", 0))
        _rows, summary = compare(samples, reference_seconds=0.1)
        self.assertEqual(summary["endpoint_kind"], "end_of_recording_not_known_neutral")


if __name__ == "__main__":
    unittest.main()
