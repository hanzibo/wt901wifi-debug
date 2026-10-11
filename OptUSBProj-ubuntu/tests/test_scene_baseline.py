"""Frozen synthetic IK outputs: characterize behavior before moving code."""
import json
from pathlib import Path
import unittest

import numpy as np

from optusbproj_ubuntu.scene_robot_model import SceneArm


class SceneBaselineTests(unittest.TestCase):
    def test_frozen_bilateral_trajectory_in_both_solver_modes(self):
        path = Path(__file__).parent / "fixtures/scene_retarget_baseline.json"
        baseline = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(baseline["version"], 1)
        for side, case in baseline["sides"].items():
            model = SceneArm(case["model"], case["lower"], case["upper"])
            for mode, frames in case["modes"].items():
                previous = np.array(case["initial"])
                step = np.array(case["step_rad"])
                for index, frame in enumerate(frames):
                    with self.subTest(side=side, mode=mode, frame=index):
                        q, error, orientation = model.fit(
                            None, previous, step, np.array(frame["palm"]),
                            positions=(np.array(frame["elbow"]), np.array(frame["wrist"])),
                            segment_rotations={s: np.array(R) for s, R in frame["segments"].items()},
                            segment_weight=.15, solver_mode=mode,
                        )
                        np.testing.assert_allclose(
                            q, frame["expected_q"], atol=baseline["angle_atol_rad"], rtol=0)
                        self.assertAlmostEqual(error, frame["expected_error_m"],
                                               delta=baseline["distance_atol_m"])
                        self.assertAlmostEqual(orientation, frame["expected_orientation_deg"],
                                               delta=baseline["orientation_atol_deg"])
                        self.assertTrue(np.all(np.abs(q - previous) <= step + 1e-8))
                        previous = q.copy()
