from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.functional_calibration import (
    HoldWindow, fit_arm_basis, fit_arm_bone_basis)


class FunctionalCalibrationTests(unittest.TestCase):
    def _synthetic_session(self, path: Path, *, lateral_axis: np.ndarray) -> np.ndarray:
        origin = 8_000_000_000
        (path / "manifest.json").write_text(json.dumps({"started_mono_ns": origin}))
        basis = Rotation.from_rotvec([.3, -.42, .15]).as_matrix()
        heading = Rotation.from_rotvec([0., 0., .7]).as_matrix()
        neutral = heading.T @ basis
        poses = ((0., np.eye(3)),
                 (2., Rotation.from_rotvec(np.array([0., -1., 0.]) *
                                             np.deg2rad(90)).as_matrix()),
                 (4., Rotation.from_rotvec(lateral_axis *
                                             np.deg2rad(90)).as_matrix()))
        with (path / "imu.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=(
                "host_recv_mono_ns", "device_id", "gyro_x", "gyro_y", "gyro_z",
                "roll", "pitch", "yaw"))
            writer.writeheader()
            for start, model_delta in poses:
                sensor = neutral @ basis.T @ model_delta @ basis
                euler = Rotation.from_matrix(sensor).as_euler("xyz", degrees=True)
                for sample in range(30):
                    writer.writerow(dict(host_recv_mono_ns=
                                         origin + round((start + sample * .02) * 1e9),
                                         device_id="WT_TEST", gyro_x=0, gyro_y=0,
                                         gyro_z=0, roll=euler[0], pitch=euler[1],
                                         yaw=euler[2]))
        return basis

    def test_recovers_continuous_right_arm_basis(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            expected = self._synthetic_session(path, lateral_axis=np.array([-1., 0., 0.]))
            result = fit_arm_basis(path, "right_upper_arm", "WT_TEST",
                                   HoldWindow(0, .58), HoldWindow(2, 2.58),
                                   HoldWindow(4, 4.58))
            actual = np.array(result.basis_matrix)
            self.assertLess(np.rad2deg(Rotation.from_matrix(
                actual @ expected.T).magnitude()), .01)
            self.assertEqual(result.status, "候选有效")
            self.assertAlmostEqual(result.axis_separation_deg, 90, places=3)

    def test_rejects_same_motion_twice(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self._synthetic_session(path, lateral_axis=np.array([0., -1., 0.]))
            with self.assertRaisesRegex(ValueError, "无法稳定确定"):
                fit_arm_basis(path, "right_forearm", "WT_TEST",
                              HoldWindow(0, .58), HoldWindow(2, 2.58),
                              HoldWindow(4, 4.58))

    def test_bone_method_recovers_mounting_despite_axial_twist(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            expected = self._synthetic_session(path, lateral_axis=np.array([-1., 0., 0.]))
            result = fit_arm_bone_basis(path, "right_upper_arm", "WT_TEST",
                                        HoldWindow(0, .58), HoldWindow(2, 2.58),
                                        HoldWindow(4, 4.58))
            actual = np.array(result.basis_matrix)
            self.assertLess(np.rad2deg(Rotation.from_matrix(
                actual @ expected.T).magnitude()), .01)
            self.assertAlmostEqual(result.forward_bone_error_deg, 0, places=3)
            self.assertAlmostEqual(result.lateral_bone_error_deg, 0, places=3)


if __name__ == "__main__":
    unittest.main()
