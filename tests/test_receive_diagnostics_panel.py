import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets
from wt901.gui_udp import GuiUdpSource
from wt901_viz.receive_diagnostics import ReceiveDiagnosticsPanel


class ReceivePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.source = GuiUdpSource(port=0, host="127.0.0.1")
        self.panel = ReceiveDiagnosticsPanel(self.source, root=self.root.name)

    def finish(self):
        end = time.monotonic()+2
        while self.panel.recorder is not None and time.monotonic() < end:
            self.panel.update_recording()
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.panel.recorder)

    def tearDown(self):
        if self.panel.recorder is not None:
            self.panel.stop()
            self.finish()
        self.panel.close()
        self.source.close()
        self.root.cleanup()

    def test_button_start_stop_and_repeated_recording(self):
        finished = []
        self.panel.finished.connect(lambda: finished.append(True))
        for _ in range(2):
            self.panel.button.click()
            recorder = self.panel.recorder
            self.assertIsNotNone(recorder)
            self.assertEqual(len(self.source._diagnostic_sinks), 1)
            self.assertFalse(self.panel.duration.isEnabled())
            self.panel.button.click()
            self.assertFalse(self.panel.button.isEnabled())
            self.finish()
            self.assertTrue((recorder.path/"summary.md").exists())
            self.assertIn("已保存", self.panel.status.text())
            self.assertEqual(self.source._diagnostic_sinks, [])
            self.assertIsNone(self.source._kernel_stamp_option)
        self.assertEqual(len(finished), 2)

    def test_auto_stop_uses_recording_duration(self):
        self.panel.toggle()
        recorder = self.panel.recorder
        with patch("wt901_viz.receive_diagnostics.time.monotonic", return_value=recorder.started+121):
            self.panel.update_recording()
        self.finish()
        self.assertEqual(recorder.summary["stop_reason"], "duration_complete")

    def test_start_failure_and_write_failure_keep_gui_usable(self):
        with patch("wt901_viz.receive_diagnostics.ReceiveRecorder", side_effect=OSError("read only")):
            self.panel.toggle()
        self.assertIsNone(self.panel.recorder)
        self.assertIn("不能开始", self.panel.status.text())
        self.assertTrue(self.panel.button.isEnabled())
        self.panel.toggle()
        recorder = self.panel.recorder
        recorder.error = "simulated write error"
        self.panel.update_recording()
        self.finish()
        self.assertIn("保存失败", self.panel.status.text())
        self.assertEqual(self.source._diagnostic_sinks, [])

    def test_window_close_waits_for_saved_files(self):
        # Exercise the real window close handler without requiring an OpenGL surface.
        from wrist_demo import WristWindow
        class View(QtWidgets.QWidget):
            def setBackgroundColor(self, *args): pass
            def setCameraPosition(self, **kwargs): pass
            def addItem(self, *args): pass
        with patch("wrist_demo.gl.GLViewWidget", View):
            window = WristWindow(self.source, "WT5500011381", "WT5500011408", "test")
        window.timer.stop()
        window.receive_diagnostics.root = self.panel.root
        window.receive_diagnostics.toggle()
        recorder = window.receive_diagnostics.recorder
        self.assertFalse(window.close())
        self.assertTrue(window._closing_for_receive_diagnostics)
        end = time.monotonic()+2
        while window.receive_diagnostics.recorder is not None and time.monotonic() < end:
            window.receive_diagnostics.update_recording()
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertIsNone(window.receive_diagnostics.recorder)
        self.assertTrue((recorder.path/"summary.json").exists())
        self.assertEqual(recorder.summary["stop_reason"], "window_close")
        self.assertEqual(self.source._diagnostic_sinks, [])
        window.close()
