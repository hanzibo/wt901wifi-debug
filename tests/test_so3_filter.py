"""SO(3) projection and gyro+accelerometer tilt-correction checks."""

import unittest

import numpy as np

from wt901_viz.so3_filter import AdaptiveTiltFilter, check_so3, rotation_step


class SO3FilterTests(unittest.TestCase):
    def test_projection_repairs_scale_and_reflection(self):
        for matrix in (np.diag((1.0, 1.0, 1.001)),
                       np.diag((1.0, 1.0, -1.0))):
            with self.subTest(matrix=matrix):
                fixed, health = check_so3(matrix)
                self.assertTrue(health.projected)
                self.assertGreater(health.distance_to_so3, 0)
                np.testing.assert_allclose(fixed.T @ fixed, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(fixed), 1.0, places=12)

    def test_normal_rotation_does_not_project(self):
        rotation = np.eye(3)
        for _ in range(500):
            rotation = rotation @ rotation_step(np.deg2rad((0.3, -0.2, 0.1)) * .02)
        fixed, health = check_so3(rotation)
        self.assertFalse(health.projected)
        self.assertLess(health.distance_to_so3, 1e-8)
        np.testing.assert_allclose(fixed, rotation)

    def test_static_bias_and_tilt_correction(self):
        filter_ = AdaptiveTiltFilter()
        for i in range(2001):
            pose = filter_.update(i * .02, (.2, -.12, .1), (0, 0, 1))
        self.assertTrue(pose.bias_ready)
        np.testing.assert_allclose(pose.gyro_bias_dps, (.2, -.12, .1), atol=.01)
        self.assertLess(np.linalg.norm(pose.rotation.T @ (0, 0, 1) - (0, 0, 1)), .01)
        self.assertLess(pose.health.distance_to_so3, 1e-8)

    def test_arbitrary_initial_mount_is_not_forced_to_world_z(self):
        filter_ = AdaptiveTiltFilter()
        for i in range(151):
            pose = filter_.update(i * .02, (0, 0, 0), (1, 0, 0))
        self.assertTrue(pose.bias_ready)
        np.testing.assert_allclose(filter_.neutral_up, (1, 0, 0), atol=1e-7)
        np.testing.assert_allclose(pose.rotation, np.eye(3), atol=1e-7)

    def test_dynamic_acceleration_has_no_tilt_correction(self):
        filter_ = AdaptiveTiltFilter()
        filter_.update(0.0, (0, 90, 0), (0, 0, 1))
        pose = filter_.update(.02, (0, 90, 0), (1, 0, 0))
        self.assertEqual(pose.accelerometer_weight, 0.0)
        expected = rotation_step(np.deg2rad((0, 90, 0)) * .02)
        np.testing.assert_allclose(pose.rotation, expected, atol=1e-10)

    def test_accelerometer_does_not_correct_yaw(self):
        filter_ = AdaptiveTiltFilter()
        filter_.update(0.0, (0, 0, 30), (0, 0, 1))
        pose = filter_.update(.02, (0, 0, 30), (0, 0, 1))
        expected = rotation_step(np.deg2rad((0, 0, 30)) * .02)
        np.testing.assert_allclose(pose.rotation, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
