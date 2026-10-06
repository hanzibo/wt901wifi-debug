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

from PySide6 import QtWidgets, QtGui

from optusbproj_ubuntu.fusion_capture import CameraObservation
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


class FusionGuiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_detached_camera_preview_returns_without_stopping_capture(self):
        class FakeSource:
            def seen_devices(self, max_age=3):
                return {}
            def close(self):
                pass
        with TemporaryDirectory() as directory, patch.object(FusionWindow, "start_camera"):
            window = FusionWindow(FakeSource(), Path(directory)/"settings.json", {}, {}, camera_source="orbbec")
            window.poll_timer.stop()
            try:
                with patch.object(window, "stop_camera") as stop, patch.object(window, "_stop_session") as stop_record:
                    image = QtGui.QImage(640, 480, QtGui.QImage.Format.Format_RGB888)
                    image.fill(QtGui.QColor("red"))
                    preview = window.camera_preview
                    window._display_image(image)
                    for _ in range(2):
                        window._toggle_camera_preview_window()
                        self.app.processEvents()
                        dialog = window.camera_preview_window
                        self.assertTrue(dialog.isVisible())
                        self.assertIs(preview.parentWidget(), dialog)
                        dialog.resize(800, 700)
                        self.app.processEvents()
                        window._display_image(image)
                        self.assertEqual(preview._original.size(), image.size())
                        self.assertAlmostEqual(preview.pixmap().width()/preview.pixmap().height(), 4/3, places=2)
                        dialog.close()
                        self.app.processEvents()
                        self.assertFalse(dialog.isVisible())
                        self.assertIsNot(preview.parentWidget(), dialog)
                        self.assertEqual(preview.height(), 120)
                        self.assertFalse(window.camera_preview_placeholder.isVisible())
                    stop.assert_not_called()
                    stop_record.assert_not_called()
            finally:
                window.close()

    def test_rgbd_ready_gate_preview_and_session_snapshot(self):
        from optusbproj_ubuntu.orbbec_worker import OrbbecWorker
        from test_rgbd_capture import observation
        class FakeSource:
            def seen_devices(self, max_age=3):
                return {}
            def close(self):
                pass
        with TemporaryDirectory() as directory, patch.object(FusionWindow, "start_camera"):
            window = FusionWindow(FakeSource(), Path(directory)/"settings.json", {}, {},
                                  camera_source="orbbec")
            worker = OrbbecWorker()
            recorder = Mock()
            recorder.path = Path(directory)
            recorder.imu_count = recorder.frame_count = recorder.dropped_frames = 0
            recorder.trimmed_imu_packets = recorder.trimmed_camera_frames = 0
            try:
                self.assertFalse(window._start_session())
                o = observation(time.monotonic_ns())
                # Preview metadata absent from the tiny generic recorder fixture.
                o.rgbd.timing["valid_depth_fraction"] = .9
                worker.calibration = o.rgbd.calibration
                worker._latest = o
                window.camera_worker = worker
                window.depth_preview.setChecked(True)
                window._poll_rgbd()
                self.assertIs(window.last_rgbd_observation, o)
                self.assertIn("只采集", window.camera_status.text())
                with patch("optusbproj_ubuntu.fusion_gui.FusionSessionRecorder", return_value=recorder):
                    self.assertTrue(window._start_session())
                self.assertEqual(recorder.camera_calibration, worker.calibration)
                self.assertFalse(window.camera_source_combo.isEnabled())
                window._stop_session()
                recorder.close.assert_called_once()
            finally:
                window.camera_worker = None
                window.close()

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

    def test_rgbd_vision_diagnostics_do_not_change_imu_engine(self):
        from test_rgbd_pose import fixture
        from optusbproj_ubuntu.rgbd_pose import DepthPoseEstimator
        class FakeSource:
            def seen_devices(self, max_age=3):
                return {}
            def close(self):
                pass
        with TemporaryDirectory() as directory, patch.object(FusionWindow, "start_camera"):
            window = FusionWindow(FakeSource(), Path(directory)/"settings.json", {}, {}, camera_source="orbbec")
            try:
                assignments = dict(window.engine.assignments)
                window._render_dirty = False
                obs, xy, scores = fixture()
                result = DepthPoseEstimator().estimate(obs, xy, scores)
                window._on_pose_estimate(result)
                window._show_pose_details()
                self.assertEqual(window.engine.assignments, assignments)
                self.assertFalse(window._render_dirty)
                self.assertIs(window.pose_dialog.canvas.estimate, result)
                self.assertIn("8/8", window.pose_status.text())
                result.captured_mono_ns -= 2_000_000_000
                window.pose_worker = SimpleNamespace(take_latest=lambda: None)
                window._poll_pose()
                self.assertIsNone(window.last_pose_estimate)
                self.assertIsNone(window.pose_dialog.canvas.estimate)
            finally:
                window.pose_worker = None
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
            recorder.path=Path(directory)
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
