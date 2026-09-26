from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optusbproj_ubuntu.camera_pose import CameraPoseEngine
from optusbproj_ubuntu.camera_rtmw3d import RTMWPoseStabilizer, adapt_rtmw3d


def sample_rtmw() -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((1, 133, 3), dtype=float)
    scores = np.zeros((1, 133), dtype=float)
    landmarks = {
        5: (200, 100, 0), 6: (100, 100, 0),
        7: (210, 160, -0.2), 8: (90, 160, -0.2),
        9: (220, 210, -0.2), 10: (80, 210, -0.2),
        11: (180, 260, 0), 12: (120, 260, 0),
        96: (210, 240, -0.2), 108: (230, 240, -0.2),
        117: (70, 240, -0.2), 129: (90, 240, -0.2),
    }
    for idx, xyz in landmarks.items():
        points[0, idx] = xyz
        scores[0, idx] = 0.9
    return points, scores


class RTMWAdapterTests(unittest.TestCase):
    def test_invalid_or_missing_person(self):
        points, scores = adapt_rtmw3d(np.empty((0, 133, 3)),
                                       np.empty((0, 133)))
        self.assertIsNone(points)
        self.assertIsNone(scores)

    def test_rtmw_points_drive_all_upper_body_parts(self):
        points, scores = adapt_rtmw3d(*sample_rtmw())
        self.assertEqual(points.shape, (33, 3))
        np.testing.assert_allclose(points[11], [0.2, 0, 0])
        np.testing.assert_allclose(points[12], [-0.2, 0, 0])
        self.assertAlmostEqual(scores[17], 0.9)
        engine = CameraPoseEngine(min_visibility=0.35)
        self.assertEqual(len(engine.update(points, scores)), 7)

    def test_low_confidence_shoulder_is_rejected(self):
        points, scores = sample_rtmw()
        scores[0, 5] = 0.1
        self.assertEqual(adapt_rtmw3d(points, scores), (None, None))


class RTMWStabilizerTests(unittest.TestCase):
    def test_shoulder_scale_changes_gradually(self):
        stabilizer = RTMWPoseStabilizer()
        points, scores = sample_rtmw()
        stabilizer.update(points, scores, timestamp=0.0)
        self.assertAlmostEqual(stabilizer.shoulder_width, 100.0)
        widened = points.copy()
        widened[0, 5, 0] += 30.0
        stabilizer.update(widened, scores, timestamp=1 / 18)
        self.assertGreater(stabilizer.shoulder_width, 100.0)
        self.assertLess(stabilizer.shoulder_width, 103.0)

    def test_single_depth_spike_is_held_but_sustained_motion_is_accepted(self):
        stabilizer = RTMWPoseStabilizer()
        points, scores = sample_rtmw()
        initial, _ = stabilizer.update(points, scores, timestamp=0.0)
        spike = points.copy()
        spike[0, 9, 2] += 1.0
        held, _ = stabilizer.update(spike, scores, timestamp=1 / 18)
        np.testing.assert_allclose(held[15], initial[15])
        moved, _ = stabilizer.update(spike, scores, timestamp=2 / 18)
        self.assertGreater(moved[15, 2], initial[15, 2])

    def test_brief_low_confidence_is_held_then_expires(self):
        stabilizer = RTMWPoseStabilizer()
        points, scores = sample_rtmw()
        initial, _ = stabilizer.update(points, scores, timestamp=0.0)
        low_scores = scores.copy()
        low_scores[0, 9] = 0.1
        held, held_visibility = stabilizer.update(points, low_scores,
                                                   timestamp=0.06)
        np.testing.assert_allclose(held[15], initial[15])
        self.assertGreaterEqual(held_visibility[15], 0.35)
        expired, visibility = stabilizer.update(points, low_scores,
                                                 timestamp=0.25)
        self.assertTrue(np.isnan(expired[15]).all())
        self.assertEqual(visibility[15], 0.0)

    def test_static_depth_noise_is_reduced(self):
        stabilizer = RTMWPoseStabilizer()
        points, scores = sample_rtmw()
        filtered_depth = []
        raw_depth = []
        for frame in range(40):
            noisy = points.copy()
            noisy[0, 9, 2] += 0.04 if frame % 2 else -0.04
            output, _ = stabilizer.update(noisy, scores, timestamp=frame / 18)
            filtered_depth.append(output[15, 2])
            raw_depth.append(noisy[0, 9, 2])
        self.assertLess(np.std(filtered_depth[10:]),
                        0.5 * np.std(raw_depth[10:]))


if __name__ == "__main__":
    unittest.main()
