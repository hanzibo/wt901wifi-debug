from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.guided_calibration import (fit_guided_arm, fit_guided_torso,
                                                  reference_motion_deg)
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


class GuidedCalibrationTests(TestCase):
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
