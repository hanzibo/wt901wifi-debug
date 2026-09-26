from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from optusbproj_ubuntu.camera_pose import (
    CAMERA_TO_MODEL, CameraPoseEngine, UPPER_BODY_KEYS, frames_from_landmarks,
)
from optusbproj_ubuntu.skeleton import PART_BY_KEY


def sample_landmarks() -> tuple[np.ndarray, np.ndarray]:
    model_points = np.zeros((33, 3), dtype=float)
    positions = {
        11: (0, 0.20, 0.50), 12: (0, -0.20, 0.50),
        23: (0, 0.10, 0), 24: (0, -0.10, 0),
        13: (0, 0.20, 0.20), 14: (0, -0.20, 0.20),
        15: (0, 0.20, -0.10), 16: (0, -0.20, -0.10),
        17: (0, 0.24, -0.18), 19: (0, 0.16, -0.18),
        18: (0, -0.24, -0.18), 20: (0, -0.16, -0.18),
    }
    for index, xyz in positions.items():
        model_points[index] = xyz
    return model_points @ CAMERA_TO_MODEL, np.ones(33)


class CameraPoseTests(unittest.TestCase):
    def test_neutral_has_seven_right_handed_frames(self):
        points, visibility = sample_landmarks()
        frames = frames_from_landmarks(points, visibility)
        self.assertEqual(set(frames), set(UPPER_BODY_KEYS))
        for frame in frames.values():
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame), 1.0)
        engine = CameraPoseEngine(smoothing=1)
        self.assertEqual(engine.update(points, visibility), set(UPPER_BODY_KEYS))
        for key in UPPER_BODY_KEYS:
            np.testing.assert_allclose(engine.transforms()[PART_BY_KEY[key]][:3, :3],
                                       np.eye(3), atol=1e-12)

    def test_one_arm_moves_without_rotating_other_arm_or_legs(self):
        points, visibility = sample_landmarks()
        engine = CameraPoseEngine(smoothing=1)
        engine.update(points, visibility)
        changed = points.copy()
        # Move the subject's left elbow, wrist and hand toward the camera.
        changed[[13, 15, 17, 19], 2] -= 0.18
        engine.update(changed, visibility)
        transforms = engine.transforms()
        self.assertFalse(np.allclose(
            transforms[PART_BY_KEY["left_upper_arm"]][:3, :3], np.eye(3)))
        for key in ("torso", "right_upper_arm", "right_forearm", "right_hand",
                    "left_thigh", "right_thigh"):
            np.testing.assert_allclose(transforms[PART_BY_KEY[key]][:3, :3],
                                       np.eye(3), atol=1e-12)
        self.assertTrue(engine.zero())
        np.testing.assert_allclose(
            engine.transforms()[PART_BY_KEY["left_upper_arm"]][:3, :3],
            np.eye(3), atol=1e-12)

    def test_occluded_child_follows_parent_and_no_person_holds_last_pose(self):
        points, visibility = sample_landmarks()
        engine = CameraPoseEngine(smoothing=1)
        engine.update(points, visibility)
        changed = points.copy()
        changed[[13, 15, 17, 19], 2] -= 0.18
        engine.update(changed, visibility)
        held = engine.transforms()[PART_BY_KEY["left_upper_arm"]].copy()
        self.assertEqual(engine.update(None, None), set())
        self.assertFalse(engine.zero())
        np.testing.assert_allclose(engine.transforms()[PART_BY_KEY["left_upper_arm"]], held)
        hidden = visibility.copy()
        hidden[15] = 0
        tracked = engine.update(changed, hidden)
        self.assertIn("left_upper_arm", tracked)
        self.assertNotIn("left_forearm", tracked)
        transforms = engine.transforms()
        np.testing.assert_allclose(
            transforms[PART_BY_KEY["left_forearm"]][:3, :3],
            transforms[PART_BY_KEY["left_upper_arm"]][:3, :3])

    def test_bad_landmark_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            frames_from_landmarks(np.zeros((32, 3)), np.ones(32))

    def test_cropped_hips_still_allow_approximate_upper_body_tracking(self):
        points, visibility = sample_landmarks()
        visibility[[23, 24]] = 0
        frames = frames_from_landmarks(points, visibility)
        self.assertEqual(set(frames), set(UPPER_BODY_KEYS))

    def test_forward_raise_through_horizontal_is_continuous(self):
        points, visibility = sample_landmarks()
        neutral_model = points @ CAMERA_TO_MODEL.T
        shoulder = neutral_model[11].copy()
        frames = []
        engine = CameraPoseEngine(smoothing=1)
        engine.update(points, visibility)
        for angle in (89, 90, 91):
            moved = neutral_model.copy()
            swing = Rotation.from_euler("y", -angle, degrees=True).as_matrix()
            for index in (13, 15, 17, 19):
                moved[index] = shoulder + swing @ (neutral_model[index] - shoulder)
            camera_points = moved @ CAMERA_TO_MODEL
            observed = frames_from_landmarks(camera_points, visibility)
            self.assertIn("left_upper_arm", observed)
            self.assertIn("left_forearm", observed)
            frames.append(observed["left_upper_arm"])
            if angle == 90:
                engine.update(camera_points, visibility)
                world = engine.transforms()
                upper_origin = world[PART_BY_KEY["left_upper_arm"]][:3, 3]
                elbow_origin = world[PART_BY_KEY["left_forearm"]][:3, 3]
                direction = elbow_origin - upper_origin
                np.testing.assert_allclose(direction / np.linalg.norm(direction),
                                           [1, 0, 0], atol=1e-12)
        change = Rotation.from_matrix(frames[0].T @ frames[2]).magnitude()
        self.assertLess(np.degrees(change), 3)


if __name__ == "__main__":
    unittest.main()
