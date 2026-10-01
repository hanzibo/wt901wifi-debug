"""Small offline regression checks; no ROS, RViz, camera or device required."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from optusbproj_ubuntu.x1_retarget import (  # noqa: E402
    RightArmKinematics, fit_right_arm, load_calibrated_session,
    retarget_bone_points,
)
from scipy.spatial.transform import Rotation


URDF = ROOT / "urdf-model" / "pkg_robot_model" / "urdf" / "pkg_robot_model.urdf"


class RightArmRetargetTests(unittest.TestCase):
    def setUp(self):
        self.kinematics = RightArmKinematics(URDF)

    def test_urdf_zero_chain_has_expected_neutral_geometry(self):
        shoulder, elbow, wrist = self.kinematics.points(np.zeros(6))
        np.testing.assert_allclose(shoulder, [0, -0.079040617, 0.517029235], atol=1e-6)
        self.assertLess(elbow[2], shoulder[2])
        self.assertLess(wrist[2], elbow[2])

    def test_position_ik_recovers_reachable_elbow_and_wrist(self):
        truth = np.deg2rad([20, -15, 10, 30, 0, 0])
        _, elbow, wrist = self.kinematics.points(truth)
        current = np.zeros(6)
        for _ in range(25):
            current, elbow_error, wrist_error, success = fit_right_arm(
                self.kinematics, elbow, wrist, current)
        self.assertTrue(success)
        self.assertLess(elbow_error, 0.025)
        self.assertLess(wrist_error, 0.01)
        self.assertLess(np.linalg.norm(current[4:]), 0.02)

    def test_soft_wrist_orientation_recovers_reachable_pose(self):
        truth = np.deg2rad([12, -10, 8, 15, 18, -14])
        _, elbow, wrist = self.kinematics.points(truth)
        target = self.kinematics.wrist_relative_rotation(truth)
        current = np.zeros(6)
        for _ in range(35):
            current, _, _, success = fit_right_arm(
                self.kinematics, elbow, wrist, current,
                target_wrist_rotation=target)
        self.assertTrue(success)
        self.assertGreater(np.linalg.norm(current[4:]), np.deg2rad(5))
        error = Rotation.from_matrix(
            target.T @ self.kinematics.wrist_relative_rotation(current))
        self.assertLess(np.rad2deg(error.magnitude()), 8.0)

    def test_retarget_preserves_both_neutral_points_and_reads_straightness(self):
        human_neutral = (np.array([0., 0., 0.]),
                         np.array([0., 0., -0.25]),
                         np.array([0., 0., -0.50]))
        robot_neutral = self.kinematics.points(np.zeros(6))
        elbow, wrist, bend = retarget_bone_points(
            human_neutral, human_neutral, robot_neutral)
        np.testing.assert_allclose(elbow, robot_neutral[1], atol=1e-12)
        np.testing.assert_allclose(wrist, robot_neutral[2], atol=1e-12)
        self.assertAlmostEqual(bend, 0.0)

        forward = (human_neutral[0], np.array([0.25, 0., 0.]),
                   np.array([0.50, 0., 0.]))
        elbow, wrist, bend = retarget_bone_points(human_neutral, forward, robot_neutral)
        self.assertGreater(elbow[0], robot_neutral[1][0])
        self.assertGreater(wrist[0], elbow[0])
        self.assertAlmostEqual(bend, 0.0)

    def test_rejects_cross_session_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            manifest = {"assignments_at_start": {
                "right_upper_arm": "WT_UPPER", "right_forearm": "WT_FOREARM"}}
            (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for part, device in (("right_upper_arm", "WT_UPPER"),
                                 ("right_forearm", "WT_FOREARM")):
                data = {"part": part, "device_id": device, "status": "通过",
                        "session": str(session / "other"),
                        "basis_matrix": np.eye(3).tolist(),
                        "neutral_matrix": np.eye(3).tolist()}
                (session / f"guided_result_{part}.json").write_text(
                    json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different capture session"):
                load_calibrated_session(session)


if __name__ == "__main__":
    unittest.main()
