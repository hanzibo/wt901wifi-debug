from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from PySide6 import QtWidgets

from optusbproj_ubuntu.gui import BodyWindow
from optusbproj_ubuntu.skeleton import SENSOR_KEYS


class FakeSource:
    def poll(self):
        return []

    def seen_devices(self, max_age=3.0):
        return {"WT5500011381": ("10.42.0.174", time.monotonic())}

    def seen_endpoints(self, max_age=3.0):
        return {}

    def close(self):
        pass


class TimedZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = BodyWindow(FakeSource(), Path("/tmp/wt901-test-timed-zero.json"),
                                 {"left_hand": "WT5500011381", "right_hand": "WT5500011408"})
        for device_id in ("WT5500011381", "WT5500011408"):
            self.window.engine.feed(SimpleNamespace(device_id=device_id, roll=0, pitch=0, yaw=0))
            self.window.engine.feed(SimpleNamespace(device_id=device_id, roll=0, pitch=0, yaw=30))

    def tearDown(self):
        self.window.close()

    def test_all_parts_have_assignment_and_axis_mapping_controls(self):
        self.assertEqual(set(self.window.combos), set(SENSOR_KEYS))
        self.assertEqual(set(self.window.mapping_buttons), set(SENSOR_KEYS))
        self.assertEqual(set(self.window.status_labels), set(SENSOR_KEYS))
        self.assertIsNotNone(self.window.findChild(QtWidgets.QScrollArea))
        headings = self.window.findChildren(QtWidgets.QLabel, "bodySectionHeading")
        self.assertEqual([heading.text() for heading in headings],
                         ["骨盆 / 躯干 / 头部", "左臂", "右臂", "左腿", "右腿"])
        self.assertTrue(all("color:" in heading.styleSheet() for heading in headings))

    def test_countdown_zeros_only_live_assigned_device(self):
        old_right = self.window.engine.neutral["WT5500011408"].copy()
        self.window.zero_delay.setValue(3)
        self.window._start_timed_zero()
        self.assertIn("还剩", self.window.zero_countdown.text())
        self.window._timed_zero_deadline = time.monotonic() - 0.01
        self.window._update_zero_countdown()
        np.testing.assert_allclose(self.window.engine.neutral["WT5500011381"],
                                   self.window.engine.current["WT5500011381"])
        np.testing.assert_allclose(self.window.engine.neutral["WT5500011408"], old_right)
        self.assertIn("1/2", self.window.zero_countdown.text())
        self.assertFalse(self.window.cancel_zero_button.isEnabled())

    def test_cancel_does_not_change_neutral(self):
        old_left = self.window.engine.neutral["WT5500011381"].copy()
        self.window._start_timed_zero()
        self.window._cancel_timed_zero()
        self.window._update_zero_countdown()
        np.testing.assert_allclose(self.window.engine.neutral["WT5500011381"], old_left)
        self.assertIn("已取消", self.window.zero_countdown.text())

    def test_timed_zero_does_not_rotate_calibrated_axis_mapping(self):
        basis = Rotation.from_euler("xyz", [10, -15, 30], degrees=True).as_matrix()
        self.window.engine.set_continuous_basis("left_hand", basis)
        self.window._start_timed_zero()
        self.window._timed_zero_deadline = time.monotonic() - 0.01
        self.window._update_zero_countdown()
        np.testing.assert_allclose(
            self.window.engine.continuous_basis["left_hand"], basis, atol=1e-12)
        np.testing.assert_allclose(
            self.window.engine.neutral["WT5500011381"],
            self.window.engine.current["WT5500011381"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
