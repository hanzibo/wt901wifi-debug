import unittest

import numpy as np

from wt901_viz.orientation import rotation_matrix
from wt901_viz.so3 import rotation_vector
from so3_rotvec_demo import arrow_segments


def rodrigues(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


class So3RotvecTests(unittest.TestCase):
    def test_log_reconstructs_rotation_near_zero_and_pi(self):
        for angles in ((0, 0, 0), (90, 0, 0), (0, 0, 179.999),
                       (0, 0, 180), (0, 0, 181), (30, -45, 60),
                       (179.999, 0, 0), (0, 180, 0)):
            with self.subTest(angles=angles):
                matrix = rotation_matrix(*angles)
                vector = rotation_vector(matrix)
                self.assertLessEqual(np.linalg.norm(vector), np.pi + 1e-12)
                np.testing.assert_allclose(rodrigues(vector), matrix, atol=1e-7)

    def test_z_axis_181_degree_rotation_uses_opposite_principal_axis(self):
        vector = rotation_vector(rotation_matrix(0, 0, 181))
        np.testing.assert_allclose(vector, [0, 0, -np.deg2rad(179)], atol=1e-12)

    def test_arrow_origin_stays_at_zero(self):
        np.testing.assert_array_equal(arrow_segments(np.array([1, 2, 3]))[0],
                                      np.zeros(3))


if __name__ == "__main__":
    unittest.main()
