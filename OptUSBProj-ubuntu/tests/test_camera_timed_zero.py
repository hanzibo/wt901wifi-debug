from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from PySide6 import QtWidgets

from optusbproj_ubuntu.camera_gui import CameraWindow
from test_camera_pose import sample_landmarks


class RunningCamera:
    def isRunning(self):
        return True


class CameraTimedZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        model = PROJECT / "assets" / "mediapipe" / "pose_landmarker_lite.task"
        self.window = CameraWindow(model, autostart=False)
        self.window.worker = RunningCamera()
        points, visibility = sample_landmarks()
        self.window.engine.update(points, visibility)
        moved = points.copy()
        moved[[13, 15, 17, 19], 2] -= 0.18
        self.window.engine.update(moved, visibility)
        self.window._last_detection = time.monotonic()

    def tearDown(self):
        self.window.worker = None
        self.window.close()

    def test_countdown_zeros_current_detected_pose(self):
        self.window.zero_delay.setValue(3)
        self.window.start_timed_zero()
        self.assertIn("还剩", self.window.zero_countdown.text())
        self.assertTrue(self.window.cancel_timed_zero_button.isEnabled())
        self.window._timed_zero_deadline = time.monotonic() - 0.01
        self.window._update_timed_zero()
        np.testing.assert_allclose(self.window.engine.neutral["left_upper_arm"],
                                   self.window.engine.latest["left_upper_arm"])
        self.assertIn("定时归零完成", self.window.zero_countdown.text())
        self.assertFalse(self.window.cancel_timed_zero_button.isEnabled())

    def test_no_fresh_detection_keeps_previous_neutral(self):
        previous = self.window.engine.neutral["left_upper_arm"].copy()
        self.window.start_timed_zero()
        self.window.engine.update(None, None)
        self.window._timed_zero_deadline = time.monotonic() - 0.01
        self.window._update_timed_zero()
        np.testing.assert_allclose(self.window.engine.neutral["left_upper_arm"], previous)
        self.assertIn("未执行", self.window.zero_countdown.text())

    def test_cancel_preserves_neutral(self):
        previous = self.window.engine.neutral["left_upper_arm"].copy()
        self.window.start_timed_zero()
        self.window.cancel_timed_zero()
        self.window._update_timed_zero()
        np.testing.assert_allclose(self.window.engine.neutral["left_upper_arm"], previous)
        self.assertIn("已取消", self.window.zero_countdown.text())

    def test_stopped_camera_cannot_schedule_or_zero(self):
        previous = self.window.engine.neutral["left_upper_arm"].copy()
        self.window.worker = None
        self.window.start_timed_zero()
        self.assertIsNone(self.window._timed_zero_deadline)
        self.assertIn("相机尚未运行", self.window.zero_countdown.text())
        self.window.zero()
        np.testing.assert_allclose(self.window.engine.neutral["left_upper_arm"], previous)


if __name__ == "__main__":
    unittest.main()
