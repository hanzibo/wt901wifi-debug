import unittest

import numpy as np
from PySide6 import QtGui

from wt901_viz.axis_mapping import DEFAULT_AXIS_MAPPING, mapping_matrix
from wrist_demo import model_rotation, qt_matrix, wrist_axis_points


class WristAxisMappingTests(unittest.TestCase):
    def test_default_mapping_preserves_relative_rotation(self):
        neutral = np.eye(3)
        current = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(model_rotation(neutral, current, DEFAULT_AXIS_MAPPING),
                                   current)

    def test_sensor_z_rotation_can_drive_model_y(self):
        current = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mapping = ("+X", "-Z", "+Y")
        expected = mapping_matrix(mapping) @ current @ mapping_matrix(mapping).T
        np.testing.assert_allclose(model_rotation(np.eye(3), current, mapping), expected)
        np.testing.assert_allclose(model_rotation(current, current, mapping), np.eye(3))

    def test_duplicate_and_mirrored_axes_are_rejected(self):
        for mapping in (("+X", "+X", "+Z"), ("+X", "+Y", "-Z")):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                mapping_matrix(mapping)

    def test_visual_axes_are_rooted_at_wrist_pivot(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                points = wrist_axis_points(axis)
                np.testing.assert_array_equal(points[0], np.zeros(3))
                np.testing.assert_allclose(points[1], np.eye(3)[axis] * 0.42,
                                           atol=1e-7)
                for rotation in (np.eye(3),
                                 np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                                 np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])):
                    world_origin = qt_matrix(rotation, 0.8).map(
                        QtGui.QVector3D(*points[0]))
                    np.testing.assert_allclose(
                        (world_origin.x(), world_origin.y(), world_origin.z()),
                        (0, 0.8, 0), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
