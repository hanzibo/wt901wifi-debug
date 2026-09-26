from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from PySide6 import QtWidgets

from optusbproj_ubuntu.fusion_capture import CameraObservation
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


class FusionGuiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_replay_opens_without_live_camera_or_udp(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            recorder = FusionSessionRecorder(path, tail_trim_seconds=0)
            recorder.write_camera(CameraObservation(
                recorder.started_mono_ns, "rtmw3d",
                np.zeros((8, 12, 3), dtype=np.uint8), None, None))
            recorder.close()
            source = FusionSessionReplay(recorder.path)
            window = FusionWindow(source, path / "settings.json", {}, {})
            try:
                self.assertTrue(window.replay_mode)
                layout = window.centralWidget().layout()
                panel = layout.itemAt(1).widget()
                self.assertEqual((layout.stretch(0), layout.stretch(1)), (3, 2))
                self.assertEqual((panel.minimumWidth(), panel.maximumWidth()),
                                 (520, 740))
                self.assertIsNone(window.camera_worker)
                self.assertFalse(window.session_button.isEnabled())
                window._poll()
                self.assertEqual(window.camera_count, 1)
            finally:
                window.close()

    def test_candidate_preview_is_replay_only_and_does_not_save_mapping(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            recorder = FusionSessionRecorder(path, tail_trim_seconds=0)
            recorder.close()
            source = FusionSessionReplay(recorder.path)
            settings = path / "settings.json"
            candidate = {"part": "right_upper_arm", "device_id": "WT_TEST",
                         "basis_matrix": np.eye(3).tolist()}
            window = FusionWindow(source, settings,
                                  {"right_upper_arm": "WT_TEST"}, {},
                                  calibration_candidates=[candidate])
            try:
                self.assertIn("双模型", window.windowTitle())
                self.assertIsNotNone(window.candidate_engine)
                self.assertEqual(len(window.candidate_actors), len(window.actors))
                bounds = window.plotter.bounds
                self.assertGreater(bounds[3] - bounds[2], 1.6)
                self.assertNotIn("right_upper_arm", window.engine.continuous_basis)
                self.assertIn("right_upper_arm", window.candidate_engine.continuous_basis)
                self.assertIn("你的映射", window.centralWidget().layout().itemAt(
                    0).widget().findChildren(QtWidgets.QLabel)[0].text())
                self.assertFalse(window.combos["right_upper_arm"].isEnabled())
                self.assertFalse(window.mapping_buttons["right_upper_arm"].isEnabled())
                source.poll_timed = lambda: [(1, SimpleNamespace(
                    device_id="WT_TEST", roll=0, pitch=0, yaw=0))]
                window._poll()
                self.assertEqual(window.engine.counts["WT_TEST"], 1)
                self.assertEqual(window.candidate_engine.counts["WT_TEST"], 1)
                self.assertFalse(settings.exists())
            finally:
                window.close()

    def test_delayed_recording_starts_and_stops_automatically(self):
        class FakeSource:
            def seen_devices(self, max_age=3.0):
                return {}

            def close(self):
                pass

        recorder = Mock()
        recorder.path = Path("/tmp/example-session")
        recorder.imu_count = 40
        recorder.frame_count = 15
        recorder.dropped_frames = 0
        recorder.trimmed_imu_packets = 5
        recorder.trimmed_camera_frames = 3
        with TemporaryDirectory() as directory:
            with patch.object(FusionWindow, "start_camera"), patch(
                    "optusbproj_ubuntu.fusion_gui.FusionSessionRecorder",
                    return_value=recorder) as factory:
                window = FusionWindow(FakeSource(), Path(directory) / "settings.json", {}, {})
                try:
                    window.record_delay.setValue(2)
                    window.record_duration.setValue(5)
                    window._start_timed_session()
                    self.assertIsNone(window.session)
                    self.assertFalse(window.timed_session_button.isEnabled())
                    window._record_start_deadline = time.monotonic() - 0.1
                    window._update_record_schedule()
                    self.assertIs(window.session, recorder)
                    factory.assert_called_once()
                    self.assertIsNotNone(window._record_stop_deadline)
                    window._record_stop_deadline = time.monotonic() - 0.1
                    window._update_record_schedule()
                    self.assertIsNone(window.session)
                    recorder.close.assert_called_once()
                    self.assertIn("末尾 3 秒已剔除", window.record_schedule_status.text())
                    window._start_timed_session()
                    window._cancel_record_schedule()
                    factory.assert_called_once()
                    self.assertIsNone(window._record_start_deadline)
                finally:
                    window.close()
