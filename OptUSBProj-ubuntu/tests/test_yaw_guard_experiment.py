"""Stateful gyro guard near the ZYX pitch singularity."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT.parent), str(PROJECT)]

from yaw_guard_experiment import YawGuard, wrap_degrees


class YawGuardTests(unittest.TestCase):
    def test_angle_unwrap(self):
        self.assertAlmostEqual(wrap_degrees(-358), 2)
        self.assertAlmostEqual(wrap_degrees(358), -2)

    def test_bad_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            YawGuard(enter_deg=80, exit_deg=80)

    def test_stationary_yaw_error_inside_guard_is_not_accepted_after_exit(self):
        guard = YawGuard()
        outputs = []
        for t in np.arange(0, 6.0001, 0.02):
            if t < 1:
                pitch, rate = 0.0, 0.0
            elif t < 2:
                pitch, rate = 85 * (t - 1), 85.0
            elif t < 4:
                pitch, rate = 85.0, 0.0
            elif t < 5:
                pitch, rate = 85 * (5 - t), -85.0
            else:
                pitch, rate = 0.0, 0.0
            # The device's fused yaw erroneously drifts by 60 degrees while
            # pitch is high, and keeps the erroneous offset afterward.
            bad_yaw = 60 * np.clip((t - 2.2) / 1.5, 0, 1)
            result = guard.update(float(t), 0.0, pitch, float(bad_yaw),
                                  (0.0, rate, 0.0))
            outputs.append((t, result))
        high = min(outputs, key=lambda item: abs(item[0] - 3.8))[1]
        final = outputs[-1][1]
        self.assertTrue(high.guard_active)
        self.assertAlmostEqual(high.stable_heading_deg, 0, delta=0.5)
        self.assertAlmostEqual(final.pose_yaw_offset_deg, -60, delta=1.5)
        self.assertAlmostEqual(final.protected_euler_yaw_deg, 0, delta=1.5)
        self.assertAlmostEqual(final.raw_vs_protected_angle_deg, 60, delta=1.5)
        self.assertEqual([event["kind"] for event in guard.events],
                         ["entered", "exited"])

    def test_full_body_gyro_integration_not_gyro_z_only(self):
        guard = YawGuard()
        guard.update(0.0, 0, 79, 0, (0, 0, 0))
        # Device X angular rate rotates the body even though gyro Z is zero.
        guard.update(0.02, 0, 81, 0, (30, 0, 0))
        result = guard.update(0.04, 0, 82, 0, (30, 0, 0))
        protected = Rotation.from_quat(result.protected_quat_xyzw)
        start = Rotation.from_euler("xyz", [0, 79, 0], degrees=True)
        self.assertGreater(np.rad2deg((start.inv() * protected).magnitude()), 0.5)
        self.assertTrue(result.guard_active)


if __name__ == "__main__":
    unittest.main()
