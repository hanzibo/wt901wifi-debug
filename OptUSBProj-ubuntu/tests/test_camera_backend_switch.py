from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from PySide6 import QtWidgets

from optusbproj_ubuntu.camera_gui import CameraWindow


class BackendSwitchTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        model = PROJECT / "assets" / "mediapipe" / "pose_landmarker_lite.task"
        self.window = CameraWindow(model, autostart=False)

    def tearDown(self):
        self.window.close()

    def test_switch_while_stopped_resets_neutral(self):
        self.window.engine.neutral["torso"] = object()
        with patch.object(self.window, "_backend_ready", return_value=None):
            self.window.backend_choice.setCurrentIndex(1)
        self.assertEqual(self.window.backend_choice.currentData(), "rtmw3d")
        self.assertTrue(self.window.rtmw_stabilize.isEnabled())
        self.assertTrue(self.window.rtmw_stabilize.isChecked())
        self.assertEqual(self.window.engine.min_visibility, 0.35)
        self.assertEqual(self.window.engine.neutral, {})
        self.assertIn("重新归零", self.window.status.text())

    def test_missing_backend_keeps_current_selection(self):
        with patch.object(self.window, "_backend_ready", return_value="缺少 rtmlib"):
            self.window.backend_choice.setCurrentIndex(1)
        self.assertEqual(self.window.backend_choice.currentData(), "mediapipe")
        self.assertFalse(self.window.rtmw_stabilize.isEnabled())
        self.assertIn("缺少 rtmlib", self.window.status.text())

    def test_running_backend_stops_before_restart(self):
        class FakeWorker:
            stopped = False

            def stop(self):
                self.stopped = True

            def deleteLater(self):
                pass

        worker = FakeWorker()
        self.window.worker = worker
        with patch.object(self.window, "_backend_ready", return_value=None):
            self.window.backend_choice.setCurrentIndex(1)
        self.assertTrue(worker.stopped)
        self.assertTrue(self.window._pending_restart)
        self.assertFalse(self.window.backend_choice.isEnabled())
        with patch.object(self.window, "start_camera") as restart:
            self.window._on_stopped()
        restart.assert_called_once_with()
        self.assertFalse(self.window._pending_restart)
        self.assertTrue(self.window.backend_choice.isEnabled())


if __name__ == "__main__":
    import unittest
    unittest.main()
