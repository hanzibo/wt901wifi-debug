from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.guided_calibration import (fit_guided_arm, fit_guided_left_wrist,
                                                  fit_guided_torso, reference_motion_deg,
                                                  reference_neutral_matrix)
from optusbproj_ubuntu.axis_mapping import mapping_matrix


MANUAL = ("+Y", "-X", "+Z")


def _rows(matrix: np.ndarray, *, count: int = 100, gyro: float = 0) -> list[dict]:
    euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
    return [dict(roll=euler[0], pitch=euler[1], yaw=euler[2],
                 gyro_x=gyro, gyro_y=0.0, gyro_z=0.0,
                 received_ns=index * 20_000_000)
            for index in range(count)]


def _samples() -> tuple[dict[str, list[dict]], np.ndarray]:
    # True mounting is a continuous offset from the user's signed-axis map.
    expected = Rotation.from_rotvec([0.12, -0.17, 0.19]).as_matrix() @ mapping_matrix(MANUAL)
    heading = Rotation.from_euler("z", 18, degrees=True).as_matrix()
    neutral = heading.T @ expected

    def pose(model_rotation: np.ndarray) -> np.ndarray:
        return neutral @ expected.T @ model_rotation @ expected

    front = Rotation.from_rotvec([0, -np.pi / 2, 0]).as_matrix()
    side = Rotation.from_rotvec([-np.pi / 2, 0, 0]).as_matrix()
    samples = {
        "neutral": _rows(neutral),
        "forward": _rows(pose(front)),
        "lateral": _rows(pose(side)),
        "check_forward": _rows(pose(front)),
        "check_neutral": _rows(neutral),
    }
    return samples, expected


def _torso_samples() -> tuple[dict[str, list[dict]], np.ndarray]:
    expected = Rotation.from_rotvec([0.10, -0.12, 0.14]).as_matrix() @ mapping_matrix(MANUAL)
    neutral = Rotation.from_euler("z", -12, degrees=True).as_matrix() @ expected

    def pose(model_axis: np.ndarray, degrees: float) -> np.ndarray:
        model_rotation = Rotation.from_rotvec(
            model_axis * np.deg2rad(degrees)).as_matrix()
        return neutral @ expected.T @ model_rotation @ expected

    samples = {
        "neutral": _rows(neutral),
        "forward": _rows(pose(np.array([0., 1., 0.]), 35)),
        "lateral": _rows(pose(np.array([1., 0., 0.]), 32)),
        "check_forward": _rows(pose(np.array([0., 1., 0.]), 28)),
        "check_neutral": _rows(neutral),
    }
    return samples, expected


def _left_wrist_samples(*, moving_forearm: bool = False
                        ) -> tuple[dict[str, list[dict]], dict[str, list[dict]], np.ndarray]:
    expected = Rotation.from_rotvec([0.08, -0.10, 0.12]).as_matrix() @ mapping_matrix(MANUAL)
    neutral = Rotation.from_euler("z", 14, degrees=True).as_matrix() @ expected
    parent_angles = (0, 11, -16, 13, 7) if moving_forearm else (0, 0, 0, 0, 0)
    parent_axes = ("x", "y", "x", "y", "z")
    parents = {step: Rotation.from_euler(axis, angle, degrees=True).as_matrix()
               for step, axis, angle in zip(("neutral", "forward", "lateral",
                                            "check_forward", "check_neutral"),
                                           parent_axes, parent_angles)}
    parents["check_long_axis"] = Rotation.from_euler("z", 48, degrees=True).as_matrix()

    def pose(step: str, axis: tuple[float, float, float], degrees: float) -> np.ndarray:
        motion = Rotation.from_rotvec(np.asarray(axis) * np.deg2rad(degrees)).as_matrix()
        return neutral @ expected.T @ parents[step] @ motion @ expected

    return {
        "neutral": _rows(neutral),
        "forward": _rows(pose("forward", (0, 1, 0), 38)),
        "lateral": _rows(pose("lateral", (-1, 0, 0), 19)),
        "check_forward": _rows(pose("check_forward", (0, 1, 0), 34)),
        "check_neutral": _rows(pose("check_neutral", (0, 1, 0), 0)),
        "check_long_axis": _rows(pose("check_long_axis", (0, 1, 0), 0)),
    }, {step: _rows(parent) for step, parent in parents.items()}, expected


class GuidedCalibrationTests(TestCase):
    def test_left_wrist_two_axes_recover_mounting(self):
        samples, forearm, expected = _left_wrist_samples()
        result = fit_guided_left_wrist("WT_HAND", samples, MANUAL,
                                       forearm, np.eye(3))
        self.assertEqual(result.status, "通过")
        actual = np.asarray(result.basis_matrix)
        self.assertLess(np.rad2deg(Rotation.from_matrix(
            actual @ expected.T).magnitude()), 5.0)
        self.assertLess(result.check_error_deg["check_forward"], 5.0)
        self.assertLess(result.check_error_deg["check_long_axis_relative"], 3.0)
        self.assertLess(result.check_error_deg["manual_mapping_offset"], 15.0)

    def test_left_wrist_rejects_large_change_from_manual_mapping(self):
        samples, forearm, _ = _left_wrist_samples()
        # A mixed radial/axial pose used to pass by tilting the mounting far
        # away from a known-good manual signed-axis mapping.
        true_basis = mapping_matrix(MANUAL)
        row = samples["neutral"][0]
        neutral = Rotation.from_euler(
            "xyz", [row["roll"], row["pitch"], row["yaw"]], degrees=True).as_matrix()
        mixed = Rotation.from_rotvec(np.deg2rad([-12., 0., 18.])).as_matrix()
        samples["lateral"] = _rows(neutral @ true_basis.T @ mixed @ true_basis)
        result = fit_guided_left_wrist("WT_HAND", samples, MANUAL, forearm, np.eye(3))
        self.assertEqual(result.status, "需重录")
        self.assertTrue(any("偏离手动初始映射" in warning for warning in result.warnings))

    def test_left_wrist_long_axis_is_independent_and_required(self):
        samples, forearm, _ = _left_wrist_samples()
        samples.pop("check_long_axis")
        with self.assertRaisesRegex(ValueError, "长轴验证流程重录"):
            fit_guided_left_wrist("WT_HAND", samples, MANUAL, forearm, np.eye(3))
        samples, forearm, _ = _left_wrist_samples()
        # Forearm pronates, but hand sensor falsely remains at the neutral pose.
        samples["check_long_axis"] = samples["neutral"]
        result = fit_guided_left_wrist("WT_HAND", samples, MANUAL, forearm, np.eye(3))
        self.assertEqual(result.status, "需重录")
        self.assertTrue(any("长轴验证" in warning for warning in result.warnings))

    def test_left_wrist_rejects_wrong_repeat_and_missing_motion(self):
        samples, forearm, _ = _left_wrist_samples()
        samples["check_forward"] = samples["lateral"]
        self.assertEqual(fit_guided_left_wrist(
            "WT_HAND", samples, MANUAL, forearm, np.eye(3)).status, "需重录")
        samples, forearm, _ = _left_wrist_samples()
        samples["lateral"] = samples["neutral"]
        with self.assertRaisesRegex(ValueError, "无法辨识安装轴"):
            fit_guided_left_wrist("WT_HAND", samples, MANUAL, forearm, np.eye(3))

    def test_first_forearm_hold_is_shared_neutral(self):
        rotated = Rotation.from_euler("x", 24, degrees=True).as_matrix()
        np.testing.assert_allclose(reference_neutral_matrix(_rows(rotated)),
                                   rotated, atol=1e-8)

    def test_left_wrist_compensates_forearm_motion_between_holds(self):
        samples, forearm, expected = _left_wrist_samples(moving_forearm=True)
        result = fit_guided_left_wrist("WT_HAND", samples, MANUAL,
                                       forearm, np.eye(3))
        self.assertEqual(result.status, "通过")
        self.assertLess(result.check_error_deg["check_neutral"], 3.0)
        self.assertLess(np.rad2deg(Rotation.from_matrix(
            np.asarray(result.basis_matrix) @ expected.T).magnitude()), 6.0)

    def test_torso_uses_bending_axes_not_arm_bone_direction(self):
        samples, expected = _torso_samples()
        result = fit_guided_torso("WT_TORSO", samples, MANUAL)
        self.assertEqual(result.status, "通过")
        actual = np.asarray(result.basis_matrix)
        self.assertLess(np.rad2deg(Rotation.from_matrix(
            actual @ expected.T).magnitude()), 3.0)

    def test_torso_one_short_udp_arrival_gap_does_not_block_fit(self):
        samples, _ = _torso_samples()
        for row in samples["neutral"][50:]:
            row["received_ns"] += 180_000_000
        result = fit_guided_torso("WT_TORSO", samples, MANUAL)
        self.assertEqual(result.status, "通过")

    def test_torso_sustained_gaps_and_movement_still_fail(self):
        samples, _ = _torso_samples()
        for index, row in enumerate(samples["neutral"]):
            row["received_ns"] += (index // 20) * 180_000_000
        for row in samples["lateral"]:
            row["gyro_x"] = 15.0
        result = fit_guided_torso("WT_TORSO", samples, MANUAL)
        self.assertEqual(result.status, "需重录")
        self.assertTrue(any("回传中断" in warning for warning in result.warnings))
        self.assertTrue(any("未充分静止" in warning for warning in result.warnings))

    def test_known_mounting_passes_independent_check(self):
        samples, expected = _samples()
        result = fit_guided_arm("right_upper_arm", "WT_TEST", samples, MANUAL)
        self.assertEqual(result.status, "通过")
        actual = np.asarray(result.basis_matrix)
        self.assertLess(np.rad2deg(Rotation.from_matrix(
            actual @ expected.T).magnitude()), 3.0)
        self.assertLess(result.check_error_deg["check_forward"], 3.0)
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-10)

    def test_bad_independent_repeat_cannot_be_applied(self):
        samples, _ = _samples()
        samples["check_forward"] = samples["lateral"]
        result = fit_guided_arm("right_forearm", "WT_TEST", samples, MANUAL)
        self.assertEqual(result.status, "需重录")
        self.assertGreater(result.check_error_deg["check_forward"], 25)

    def test_too_few_packets_is_rejected(self):
        samples, _ = _samples()
        samples["neutral"] = samples["neutral"][:10]
        with self.assertRaisesRegex(ValueError, "至少需要 30 包"):
            fit_guided_arm("right_upper_arm", "WT_TEST", samples, MANUAL)

    def test_unstable_hold_is_not_accepted(self):
        samples, _ = _samples()
        for row in samples["forward"]:
            row["gyro_x"] = 25.0
        result = fit_guided_arm("right_upper_arm", "WT_TEST", samples, MANUAL)
        self.assertEqual(result.status, "需重录")

    def test_minor_arm_tremor_and_one_short_arrival_gap_do_not_block_fit(self):
        samples, _ = _samples()
        for row in samples["lateral"]:
            row["gyro_x"] = 16.0
        for row in samples["lateral"][50:]:
            row["received_ns"] += 180_000_000
        result = fit_guided_arm("right_forearm", "WT_TEST", samples, MANUAL)
        self.assertEqual(result.status, "通过")

    def test_sustained_arm_packet_gaps_are_rejected(self):
        samples, _ = _samples()
        for index, row in enumerate(samples["lateral"]):
            row["received_ns"] += (index // 20) * 180_000_000
        result = fit_guided_arm("right_forearm", "WT_TEST", samples, MANUAL)
        self.assertEqual(result.status, "需重录")
        self.assertTrue(any("回传中断" in warning for warning in result.warnings))

    def test_stationary_parent_check_finds_torso_compensation(self):
        still = {step: _rows(np.eye(3)) for step in (
            "neutral", "forward", "lateral", "check_forward", "check_neutral")}
        still["lateral"] = _rows(Rotation.from_euler("z", 18, degrees=True).as_matrix())
        motion = reference_motion_deg(still)
        self.assertAlmostEqual(motion["lateral"], 18, places=4)
        self.assertAlmostEqual(motion["forward"], 0, places=4)
