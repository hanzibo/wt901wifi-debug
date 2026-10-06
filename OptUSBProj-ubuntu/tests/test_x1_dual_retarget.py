"""Bilateral URDF and ROS-independent trajectory format checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path[:0] = [str(ROOT), str(PROJECT), str(ROOT / "urdf-model" / "script_python")]

from optusbproj_ubuntu.x1_retarget import (  # noqa: E402
    ARM_JOINTS, ArmKinematics, fit_arm, load_calibrated_session,
)
from optusbproj_ubuntu.x1_dual_retarget import DUAL_FIELDS  # noqa: E402
from x1_imu_replay import load_trajectory  # noqa: E402


URDF = ROOT / "urdf-model" / "pkg_robot_model" / "urdf" / "pkg_robot_model.urdf"


class DualArmTests(unittest.TestCase):
    def test_left_and_right_zero_points_mirror_across_body(self):
        left = ArmKinematics(URDF, "left")
        right = ArmKinematics(URDF, "right")
        self.assertEqual(left.joint_names, ARM_JOINTS["left"])
        self.assertEqual(right.joint_names, ARM_JOINTS["right"])
        for left_point, right_point in zip(left.points(np.zeros(6)),
                                           right.points(np.zeros(6))):
            np.testing.assert_allclose(left_point[0], right_point[0], atol=1e-4)
            np.testing.assert_allclose(left_point[1], -right_point[1], atol=1e-4)
            np.testing.assert_allclose(left_point[2], right_point[2], atol=1e-4)

    def test_left_ik_reaches_left_forward_pose_without_mirroring_joint_angles(self):
        model = ArmKinematics(URDF, "left")
        truth = np.deg2rad([35, -12, 5, 2, 0, 0])
        _, elbow, wrist = model.points(truth)
        q = np.zeros(6)
        for _ in range(15):
            q, elbow_error, wrist_error, success = fit_arm(
                model, elbow, wrist, q, human_bend_deg=5)
        self.assertTrue(success)
        self.assertLess(wrist_error, 0.02)
        self.assertLess(elbow_error, 0.03)

    def test_bilateral_calibration_checks_all_four_device_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            assignments = {f"{side}_{segment}": f"{side}_{segment}_ID"
                           for side in ("left", "right")
                           for segment in ("upper_arm", "forearm")}
            (session / "manifest.json").write_text(
                json.dumps({"assignments_at_start": assignments}), encoding="utf-8")
            for part, device in assignments.items():
                payload = {"part": part, "device_id": device, "status": "通过",
                           "session": str(session),
                           "basis_matrix": np.eye(3).tolist(),
                           "neutral_matrix": np.eye(3).tolist()}
                (session / f"guided_result_{part}.json").write_text(
                    json.dumps(payload), encoding="utf-8")
            _, loaded = load_calibrated_session(session, ("left", "right"))
            self.assertEqual(loaded, assignments)
            bad = session / "guided_result_left_forearm.json"
            data = json.loads(bad.read_text(encoding="utf-8"))
            data["device_id"] = "wrong"
            bad.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assignment"):
                load_calibrated_session(session, ("left", "right"))

    def test_replay_accepts_right_only_and_dual_but_rejects_partial_left(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            for names in (ARM_JOINTS["right"],
                          ARM_JOINTS["left"] + ARM_JOINTS["right"]):
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(("time_s",) + names)
                    writer.writerow((0.0,) + (0.0,) * len(names))
                    writer.writerow((0.1,) + (0.1,) * len(names))
                read_names, frames = load_trajectory(path)
                self.assertEqual(read_names, names)
                self.assertEqual(len(frames), 2)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("time_s", "1") + ARM_JOINTS["right"])
                writer.writerow((0.0,) + (0.0,) * 7)
            with self.assertRaisesRegex(ValueError, "左臂六关节列不完整"):
                load_trajectory(path)

    def test_dual_schema_has_all_twelve_unique_joint_columns(self):
        self.assertEqual(DUAL_FIELDS[1:13],
                         ARM_JOINTS["left"] + ARM_JOINTS["right"])
        self.assertEqual(len(DUAL_FIELDS), len(set(DUAL_FIELDS)))


if __name__ == "__main__":
    unittest.main()
