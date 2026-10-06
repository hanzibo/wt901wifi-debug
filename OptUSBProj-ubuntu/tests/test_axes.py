"""Geometry checks for the per-part model-axis overlay."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optusbproj_ubuntu.gui import _axis_triad


class AxisTriadTests(unittest.TestCase):
    def test_three_colored_arrows_follow_positive_local_axes(self) -> None:
        length = 0.105
        mesh = _axis_triad(length)
        colors = mesh.point_data["axis_rgb"]
        expected = (
            ((255, 50, 50), 0),
            ((50, 220, 50), 1),
            ((50, 120, 255), 2),
        )
        for rgb, coordinate in expected:
            points = mesh.points[np.all(colors == rgb, axis=1)]
            self.assertGreater(len(points), 0)
            self.assertAlmostEqual(float(points[:, coordinate].max()), length, places=5)


if __name__ == "__main__":
    unittest.main()
