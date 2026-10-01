from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.pose import GYRO_TILT, VENDOR_RPY, PoseEngine
from optusbproj_ubuntu.axis_mapping import mapping_matrix
from optusbproj_ubuntu.skeleton import PARTS, PART_BY_KEY, SENSOR_KEYS


def packet(device_id: str, roll: float = 0, pitch: float = 0, yaw: float = 0):
    return SimpleNamespace(device_id=device_id, roll=roll, pitch=pitch, yaw=yaw)


def imu_packet(device_id: str, tick_ms: int, gyro_z: float = 0,
               yaw: float = 0):
    return SimpleNamespace(
        device_id=device_id, roll=0.0, pitch=0.0, yaw=yaw,
        hour=0, minute=0, second=tick_ms // 1000,
        millisecond=tick_ms % 1000,
        gyro_x=0.0, gyro_y=0.0, gyro_z=gyro_z,
        acc_x=0.0, acc_y=0.0, acc_z=1.0)


class PoseEngineTests(unittest.TestCase):
    def test_vendor_and_gyro_tilt_run_in_parallel_and_switch_without_rezero(self):
        engine = PoseEngine({"left_hand": "A", "right_hand": "B"})
        for tick in range(50):
            engine.feed(imu_packet("A", tick * 20))
            engine.feed(imu_packet("B", tick * 20))
        self.assertTrue(engine._tilt_filters["A"].bias_ready)
        engine.zero()
        for tick in range(50, 100):
            engine.feed(imu_packet("A", tick * 20, gyro_z=90, yaw=0))
            engine.feed(imu_packet("B", tick * 20, gyro_z=0, yaw=30))
        np.testing.assert_allclose(engine.rotation_for("left_hand"), np.eye(3), atol=1e-6)
        engine.set_algorithm_mode("left_hand", GYRO_TILT)
        self.assertAlmostEqual(Rotation.from_matrix(
            engine.rotation_for("left_hand")).magnitude(), np.pi / 2, delta=0.08)
        self.assertEqual(engine.algorithm_modes["right_hand"], VENDOR_RPY)
        self.assertAlmostEqual(Rotation.from_matrix(
            engine.rotation_for("right_hand")).magnitude(), np.pi / 6, delta=0.01)
        engine.zero("left_hand")
        np.testing.assert_allclose(engine.rotation_for("left_hand"), np.eye(3), atol=1e-6)
        engine.set_algorithm_mode("left_hand", VENDOR_RPY)
        np.testing.assert_allclose(engine.rotation_for("left_hand"), np.eye(3), atol=1e-6)

    def test_calibrated_switch_requires_paired_neutral(self):
        engine = PoseEngine({"left_hand": "A"})
        engine.feed(imu_packet("A", 0))
        engine.apply_calibration("left_hand", "A", np.eye(3), np.eye(3))
        with self.assertRaisesRegex(ValueError, "同步中立位"):
            engine.set_algorithm_mode("left_hand", GYRO_TILT)
        engine.apply_calibration("left_hand", "A", np.eye(3), np.eye(3),
                                 paired_neutrals={GYRO_TILT: np.eye(3)})
        engine.set_algorithm_mode("left_hand", GYRO_TILT)
        np.testing.assert_allclose(engine.rotation_for("left_hand"), np.eye(3))

    def test_vertical_mount_calibration_uses_gravity_aligned_gauge(self):
        engine = PoseEngine({"left_hand": "A"},
                            algorithm_modes={"left_hand": GYRO_TILT})
        for tick in range(60):
            packet_sample = imu_packet("A", tick * 20)
            packet_sample.acc_y = 1.0
            packet_sample.acc_z = 0.0
            engine.feed(packet_sample)
        aligned = engine.calibration_orientation_for("A", GYRO_TILT)
        self.assertIsNotNone(aligned)
        np.testing.assert_allclose(aligned @ [0, 1, 0], [0, 0, 1], atol=1e-8)
        raw_neutral = engine.current_for("A", GYRO_TILT).copy()
        engine.apply_calibration("left_hand", "A", np.eye(3), aligned,
                                 paired_neutrals={GYRO_TILT: raw_neutral})
        np.testing.assert_allclose(engine.neutral["A"], raw_neutral)

    def test_rest_pose_has_all_fifteen_parts_and_legacy_offsets(self):
        engine = PoseEngine()
        transforms = engine.transforms()
        self.assertEqual(len(transforms), 15)
        self.assertEqual(len(SENSOR_KEYS), 15)
        self.assertEqual(set(SENSOR_KEYS), set(PART_BY_KEY))
        np.testing.assert_allclose(transforms[PART_BY_KEY["torso"]][:3, 3],
                                   [-0.00978, 0, 0.13049])
        self.assertEqual(PARTS[PART_BY_KEY["left_upper_arm"]].parent,
                         PART_BY_KEY["torso"])
        self.assertEqual(PARTS[PART_BY_KEY["right_hand"]].parent,
                         PART_BY_KEY["right_forearm"])

    def test_first_packet_is_neutral_and_child_inherits_parent(self):
        engine = PoseEngine({"torso": "WT5500011381"})
        engine.feed(packet("WT5500011381", yaw=30))
        neutral = engine.transforms()
        np.testing.assert_allclose(neutral[PART_BY_KEY["torso"]][:3, :3], np.eye(3), atol=1e-12)
        engine.feed(packet("WT5500011381", yaw=60))
        moved = engine.transforms()
        self.assertFalse(np.allclose(moved[PART_BY_KEY["torso"]][:3, :3], np.eye(3)))
        np.testing.assert_allclose(moved[PART_BY_KEY["head"]][:3, :3],
                                   moved[PART_BY_KEY["torso"]][:3, :3])
        np.testing.assert_allclose(engine.rotation_for("head"), np.eye(3), atol=1e-12)
        engine.zero("torso")
        np.testing.assert_allclose(engine.transforms()[PART_BY_KEY["torso"]][:3, :3],
                                   np.eye(3), atol=1e-12)

    def test_common_motion_does_not_create_relative_joint_rotation(self):
        engine = PoseEngine({"torso": "WT5500011381", "left_upper_arm": "WT5500011408"})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011408"))
        engine.feed(packet("WT5500011381", roll=40))
        engine.feed(packet("WT5500011408", roll=40))
        np.testing.assert_allclose(engine.rotation_for("left_upper_arm"), np.eye(3), atol=1e-12)

    def test_duplicate_assignment_is_rejected(self):
        engine = PoseEngine({"left_hand": "WT5500011381"})
        with self.assertRaises(ValueError):
            engine.assign("right_hand", "WT5500011381")

    def test_new_slots_drive_head_pelvis_and_legs(self):
        for key in ("head", "pelvis", "left_thigh", "left_calf", "left_foot",
                    "right_thigh", "right_calf", "right_foot"):
            with self.subTest(part=key):
                engine = PoseEngine({key: "WT5500011381"})
                engine.feed(packet("WT5500011381"))
                engine.feed(packet("WT5500011381", yaw=45))
                self.assertFalse(np.allclose(engine.rotation_for(key), np.eye(3)))
                engine.zero(key)
                np.testing.assert_allclose(engine.rotation_for(key), np.eye(3), atol=1e-12)

    def test_unassigned_lower_leg_inherits_moving_thigh(self):
        engine = PoseEngine({"left_thigh": "WT5500011381"})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011381", roll=25))
        transforms = engine.transforms()
        np.testing.assert_allclose(transforms[PART_BY_KEY["left_calf"]][:3, :3],
                                   transforms[PART_BY_KEY["left_thigh"]][:3, :3])
        np.testing.assert_allclose(transforms[PART_BY_KEY["left_foot"]][:3, :3],
                                   transforms[PART_BY_KEY["left_thigh"]][:3, :3])

    def test_reassignment_waits_for_new_neutral_packet(self):
        engine = PoseEngine({"left_hand": "WT5500011381"})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011408", yaw=40))
        engine.assign("left_hand", "WT5500011408")
        self.assertNotIn("WT5500011408", engine.neutral)
        engine.feed(packet("WT5500011408", yaw=40))
        np.testing.assert_allclose(engine.rotation_for("left_hand"), np.eye(3), atol=1e-12)

    def test_flat_sensor_yaw_rotates_about_model_blue_z_by_default(self):
        engine = PoseEngine({"left_hand": "WT5500011381"})
        engine.feed(packet("WT5500011381", yaw=25))
        engine.feed(packet("WT5500011381", yaw=115))
        rotated = engine.rotation_for("left_hand")
        np.testing.assert_allclose(rotated @ [0, 0, -1], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(rotated @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_custom_sensor_y_axis_can_drive_model_z(self):
        engine = PoseEngine({"left_hand": "WT5500011381"},
                            {"left_hand": ("+Y", "+Z", "+X")})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011381", pitch=90))
        rotated = engine.rotation_for("left_hand")
        np.testing.assert_allclose(rotated @ [0, 0, -1], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(rotated @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_custom_sensor_z_axis_can_drive_model_x(self):
        engine = PoseEngine({"left_hand": "WT5500011381"},
                            {"left_hand": ("+Y", "+Z", "+X")})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011381", yaw=90))
        np.testing.assert_allclose(engine.rotation_for("left_hand") @ [0, 0, -1],
                                   [0, 1, 0], atol=1e-12)

    def test_mapping_requires_three_unique_right_handed_axes(self):
        with self.assertRaisesRegex(ValueError, "只能分配"):
            mapping_matrix(("+X", "+X", "+Z"))
        with self.assertRaisesRegex(ValueError, "镜像"):
            mapping_matrix(("-X", "+Y", "+Z"))
        np.testing.assert_allclose(mapping_matrix(("-X", "-Y", "+Z")),
                                   np.diag([-1, -1, 1]))

    def test_continuous_basis_is_temporary_and_validated(self):
        engine = PoseEngine({"right_upper_arm": "WT5500011381"})
        basis = Rotation.from_euler("z", 37, degrees=True).as_matrix()
        engine.set_continuous_basis("right_upper_arm", basis)
        np.testing.assert_allclose(engine.continuous_basis["right_upper_arm"], basis)
        with self.assertRaisesRegex(ValueError, "正交"):
            engine.set_continuous_basis("right_upper_arm", np.diag([-1., 1., 1.]))
        engine.set_axis_mapping("right_upper_arm", ("+X", "+Y", "+Z"))
        self.assertNotIn("right_upper_arm", engine.continuous_basis)
        engine.set_continuous_basis("right_upper_arm", basis)
        engine.assign("right_upper_arm", "WT5500011408")
        self.assertNotIn("right_upper_arm", engine.continuous_basis)

    def test_calibrated_basis_stays_fixed_when_rezeroing(self):
        engine = PoseEngine({"right_upper_arm": "WT_TEST"})
        engine.feed(packet("WT_TEST", roll=20, pitch=10, yaw=15))
        original = engine.current["WT_TEST"].copy()
        engine.apply_calibration("right_upper_arm", "WT_TEST", original, original)
        engine.feed(packet("WT_TEST", roll=-5, pitch=30, yaw=42))
        next_neutral = engine.current["WT_TEST"].copy()
        engine.zero("right_upper_arm")
        np.testing.assert_allclose(engine.continuous_basis["right_upper_arm"],
                                   original, atol=1e-12)
        np.testing.assert_allclose(engine.neutral["WT_TEST"], next_neutral, atol=1e-12)
        np.testing.assert_allclose(engine.rotation_for("right_upper_arm"),
                                   np.eye(3), atol=1e-12)

    def test_axial_twist_rezero_keeps_calibrated_forward_raise_direction(self):
        key = "right_upper_arm"
        device_id = "WT_TEST"
        neutral = Rotation.from_euler("xyz", [10, -8, 20], degrees=True)
        twist = Rotation.from_euler("z", 60, degrees=True)
        forward_raise = Rotation.from_euler("y", 45, degrees=True)
        basis = Rotation.from_euler("xyz", [8, -12, 25], degrees=True).as_matrix()

        def feed_rotation(engine: PoseEngine, orientation: Rotation) -> None:
            roll, pitch, yaw = orientation.as_euler("xyz", degrees=True)
            engine.feed(packet(device_id, roll, pitch, yaw))

        for all_parts in (False, True):
            with self.subTest(all_parts=all_parts):
                engine = PoseEngine({key: device_id})
                feed_rotation(engine, neutral)
                engine.apply_calibration(key, device_id, basis,
                                         engine.current[device_id])
                feed_rotation(engine, neutral * twist)
                engine.zero(None if all_parts else key)
                np.testing.assert_allclose(engine.rotation_for(key),
                                           np.eye(3), atol=1e-12)
                np.testing.assert_allclose(engine.continuous_basis[key],
                                           basis, atol=1e-12)
                feed_rotation(engine, neutral * twist * forward_raise)
                np.testing.assert_allclose(
                    engine.rotation_for(key),
                    basis @ forward_raise.as_matrix() @ basis.T,
                    atol=1e-12)


if __name__ == "__main__":
    unittest.main()
