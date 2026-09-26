from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from PySide6 import QtWidgets
from scipy.spatial.transform import Rotation
import numpy as np

from optusbproj_ubuntu.gui import BodyWindow
from optusbproj_ubuntu.calibration_gui import (BilateralCalibrationDialog,
                                                bilateral_quality_notes)
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.axis_mapping import mapping_matrix


class FakeSource:
    def __init__(self):
        self.reads = []

    def seen_devices(self, max_age=3.0):
        return {"WT_TEST": ("10.42.0.2", 0.0)}

    def seen_endpoints(self, max_age=3.0):
        return {"WT_TEST": ("10.42.0.2", 9250, 0.0)}

    def read_registers(self, device_id, address):
        self.reads.append((device_id, address))

    def take_replies(self):
        return []

    def poll(self):
        return []

    def close(self):
        pass


class GuidedGuiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_wizard_refreshes_newly_assigned_forearm_on_reopen(self):
        with TemporaryDirectory() as directory:
            window = BodyWindow(FakeSource(), Path(directory) / "settings.json",
                                {"right_upper_arm": "WT_TEST"}, {})
            try:
                window._show_calibration()
                dialog = window.calibration_dialog
                self.assertEqual(dialog.part_combo.findData("right_forearm"), -1)
                dialog.close()

                window.engine.assign("right_forearm", "WT5500011408")
                window._show_calibration()
                index = dialog.part_combo.findData("right_forearm")
                self.assertGreaterEqual(index, 0)
                self.assertIn("WT5500011408", dialog.part_combo.itemText(index))

                window.engine.assign("right_forearm", "")
                dialog.close()
                window._show_calibration()
                self.assertEqual(dialog.part_combo.findData("right_forearm"), -1)
            finally:
                window.close()

    def test_bilateral_capture_keeps_each_part_result_independent(self):
        with TemporaryDirectory() as directory:
            ids = {"left_upper_arm": "WT_LU", "left_forearm": "WT_LF",
                   "right_upper_arm": "WT_RU", "right_forearm": "WT_RF"}
            window = BodyWindow(FakeSource(), Path(directory) / "settings.json",
                                ids, {})
            try:
                for device_id in ids.values():
                    window.engine.feed(SimpleNamespace(
                        device_id=device_id, roll=0, pitch=0, yaw=0))
                window._show_bilateral_calibration()
                dialog = window.calibration_dialog
                self.assertIsInstance(dialog, BilateralCalibrationDialog)
                self.assertEqual(set(dialog.device_checks), set(ids.values()))
                dialog.on_replies([(device_id, (1, 1, 0, 0))
                                   for device_id in ids.values()])

                def make_packet(device_id, model_rotation):
                    neutral = mapping_matrix(("+Y", "-X", "+Z"))
                    sensor = model_rotation @ neutral
                    roll, pitch, yaw = Rotation.from_matrix(sensor).as_euler(
                        "xyz", degrees=True)
                    return SimpleNamespace(
                        device_id=device_id, year=2026, month=9, day=26,
                        hour=12, minute=0, second=0, millisecond=0,
                        acc_x=0.0, acc_y=0.0, acc_z=1.0,
                        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                        mag_x=0.0, mag_y=0.0, mag_z=0.0,
                        roll=roll, pitch=pitch, yaw=yaw,
                        temperature=25.0, battery_v=3.9, rssi=-45, version=1)

                poses = (np.eye(3),
                         Rotation.from_euler("y", -80, degrees=True).as_matrix(),
                         None,
                         Rotation.from_euler("y", -80, degrees=True).as_matrix(),
                         np.eye(3))

                def factory(_parent, **kwargs):
                    return FusionSessionRecorder(Path(directory), **kwargs)

                with patch("optusbproj_ubuntu.calibration_gui.FusionSessionRecorder",
                           side_effect=factory):
                    dialog.start()
                    self.assertEqual(dialog.phase, "prepare")
                    for step_index, pose in enumerate(poses):
                        dialog.deadline = time.monotonic() - 0.01
                        dialog._tick()
                        self.assertEqual(dialog.phase, "hold")
                        for _ in range(35):
                            for key, device_id in ids.items():
                                if key == "right_forearm" and step_index == 2:
                                    continue  # One failed part must not veto the other three.
                                target = pose
                                if target is None:
                                    side = 80 if key.startswith("left_") else -80
                                    target = Rotation.from_euler(
                                        "x", side, degrees=True).as_matrix()
                                dialog.on_packet(time.monotonic_ns(),
                                                 make_packet(device_id, target))
                        dialog.deadline = time.monotonic() - 0.01
                        dialog._tick()
                self.assertEqual(dialog.phase, "done")
                self.assertEqual(len(dialog.results), 3)
                self.assertEqual(set(dialog.failures), {"right_forearm"})
                self.assertTrue(all(result.status == "通过"
                                    for result in dialog.results.values()))
                self.assertEqual(dialog.result_select.count(), 4)
                with (patch("optusbproj_ubuntu.fusion_gui.FusionWindow") as replay_window,
                      patch("optusbproj_ubuntu.fusion_replay.FusionSessionReplay")):
                    dialog.preview()
                    candidates = replay_window.call_args.kwargs["calibration_candidates"]
                    self.assertEqual({item["part"] for item in candidates},
                                     {"left_upper_arm", "left_forearm",
                                      "right_upper_arm"})
                index = dialog.result_select.findData("left_forearm")
                dialog.result_select.setCurrentIndex(index)
                self.assertTrue(dialog.apply_button.isEnabled())
                dialog.apply()
                self.assertIn("left_forearm", window.engine.continuous_basis)
                self.assertIn("left_upper_arm", window.engine.continuous_basis)
                self.assertIn("right_upper_arm", window.engine.continuous_basis)
                self.assertNotIn("right_forearm", window.engine.continuous_basis)
                self.assertFalse(dialog.apply_button.isEnabled())
                dialog.restore()
                self.assertNotIn("left_forearm", window.engine.continuous_basis)
                self.assertIn("left_upper_arm", window.engine.continuous_basis)
                dialog.result_select.setCurrentIndex(
                    dialog.result_select.findData("right_forearm"))
                self.assertTrue(dialog.apply_button.isEnabled())  # Restored left forearm remains pending.
                dialog.quality_notes["left_forearm"] = ["侧举误差偏大"]
                with patch.object(QtWidgets.QMessageBox, "question",
                                  return_value=QtWidgets.QMessageBox.StandardButton.Cancel):
                    dialog.apply()
                self.assertNotIn("left_forearm", window.engine.continuous_basis)
                window.engine.assign("left_forearm", "WT_CHANGED")
                with (patch.object(QtWidgets.QMessageBox, "question",
                                   return_value=QtWidgets.QMessageBox.StandardButton.Yes),
                      patch.object(QtWidgets.QMessageBox, "warning") as warning):
                    dialog.apply()
                warning.assert_called_once()
                self.assertNotIn("left_forearm", window.engine.continuous_basis)
                dialog.close()
                window._show_calibration()
                self.assertNotIsInstance(window.calibration_dialog,
                                         BilateralCalibrationDialog)
            finally:
                window.close()

    def test_bilateral_quality_notes_are_advisory(self):
        notes = bilateral_quality_notes(
            "left_forearm", 20.2, {"forward": 6.1, "lateral": 9.2})
        self.assertEqual(len(notes), 2)
        self.assertIn("20.2°", notes[0])
        self.assertIn("9.2°", notes[1])
        self.assertEqual(bilateral_quality_notes(
            "right_upper_arm", 18.0, {"lateral": 4.9}), [])

    def test_bilateral_skips_offline_parts_but_checks_online_installation(self):
        with TemporaryDirectory() as directory:
            source = FakeSource()
            window = BodyWindow(source, Path(directory) / "settings.json",
                                {"left_upper_arm": "WT_ON",
                                 "right_forearm": "WT_OFF"}, {})
            try:
                window.engine.feed(SimpleNamespace(
                    device_id="WT_ON", roll=0, pitch=0, yaw=0))
                window._show_bilateral_calibration()
                dialog = window.calibration_dialog
                self.assertEqual(set(dialog.device_checks), {"WT_ON"})
                self.assertIn("离线，跳过", dialog.device_status.text())
                dialog.on_replies([("WT_ON", (0, 1, 0, 0))])
                dialog.start()
                self.assertEqual(dialog.phase, "idle")
                self.assertIn("不是垂直", dialog.progress.text())
                dialog.on_replies([("WT_ON", (1, 1, 0, 0))])
                recorder = Mock()
                recorder.path = Path(directory)
                with patch("optusbproj_ubuntu.calibration_gui.FusionSessionRecorder",
                           return_value=recorder):
                    dialog.start()
                    self.assertEqual(dialog._participants,
                                     {"left_upper_arm": "WT_ON"})
                    dialog.abort()
                self.assertTrue((Path(directory) / "steps.json").exists())
            finally:
                window.close()

    def test_wizard_uses_main_source_and_requires_vertical_configuration(self):
        with TemporaryDirectory() as directory:
            source = FakeSource()
            window = BodyWindow(source, Path(directory) / "settings.json",
                                {"right_upper_arm": "WT_TEST"},
                                {"right_upper_arm": ("+Y", "-X", "+Z")})
            try:
                window.engine.feed(SimpleNamespace(
                    device_id="WT_TEST", roll=0, pitch=0, yaw=0))
                window._show_calibration()
                dialog = window.calibration_dialog
                self.assertIn(("WT_TEST", 0x23), source.reads)
                dialog.on_replies([("WT_TEST", (0, 1, 0, 0))])
                dialog.start()
                self.assertIn("垂直安装", dialog.progress.text())
                dialog._read_device()
                dialog.on_replies([("WT_TEST", (1, 1, 0, 0))])
                recorder = Mock()
                recorder.path = Path(directory) / "fake-session"
                with patch("optusbproj_ubuntu.calibration_gui.FusionSessionRecorder",
                           return_value=recorder) as factory:
                    dialog.start()
                    self.assertEqual(dialog.phase, "prepare")
                    factory.assert_called_once()
                    dialog.abort()
                    recorder.close.assert_called_once()
                window.engine.apply_calibration(
                    "right_upper_arm", "WT_TEST", np.eye(3), np.eye(3))
                window._zero_device_after_setting("WT_TEST")
                self.assertNotIn("right_upper_arm", window.engine.continuous_basis)
                self.assertNotIn("WT_TEST", window.engine.neutral)
                window.engine.feed(SimpleNamespace(
                    device_id="WT_TEST", roll=0, pitch=0, yaw=45))
                np.testing.assert_allclose(window.engine.rotation_for("right_upper_arm"),
                                           np.eye(3), atol=1e-12)
            finally:
                window.close()

    def test_guided_capture_fit_apply_and_restore(self):
        with TemporaryDirectory() as directory:
            source = FakeSource()
            mapping = ("+Y", "-X", "+Z")
            window = BodyWindow(source, Path(directory) / "settings.json",
                                {"right_upper_arm": "WT_TEST"},
                                {"right_upper_arm": mapping})
            try:
                neutral = mapping_matrix(mapping)
                window.engine.feed(SimpleNamespace(
                    device_id="WT_TEST", roll=0, pitch=0, yaw=90))
                window._show_calibration()
                dialog = window.calibration_dialog
                dialog.on_replies([("WT_TEST", (1, 1, 0, 0))])
                front = Rotation.from_euler("y", -80, degrees=True).as_matrix()
                side = Rotation.from_euler("x", -80, degrees=True).as_matrix()
                targets = (np.eye(3), front, side, front, np.eye(3))

                def make_packet(model_rotation):
                    sensor = neutral @ neutral.T @ model_rotation @ neutral
                    roll, pitch, yaw = Rotation.from_matrix(sensor).as_euler(
                        "xyz", degrees=True)
                    values = dict(device_id="WT_TEST", year=2026, month=9, day=26,
                                  hour=12, minute=0, second=0, millisecond=0,
                                  acc_x=0.0, acc_y=0.0, acc_z=1.0,
                                  gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                                  mag_x=0.0, mag_y=0.0, mag_z=0.0,
                                  roll=roll, pitch=pitch, yaw=yaw,
                                  temperature=25.0, battery_v=3.9,
                                  rssi=-45, version=1)
                    return SimpleNamespace(**values)

                def factory(_parent, **kwargs):
                    return FusionSessionRecorder(Path(directory), **kwargs)

                with patch("optusbproj_ubuntu.calibration_gui.FusionSessionRecorder",
                           side_effect=factory):
                    dialog.start()
                    for target in targets:
                        dialog.deadline = time.monotonic() - 0.01
                        dialog._tick()  # enter static hold
                        self.assertEqual(dialog.phase, "hold")
                        for _ in range(35):
                            dialog.on_packet(time.monotonic_ns(), make_packet(target))
                        dialog.deadline = time.monotonic() - 0.01
                        dialog._tick()  # finish this step
                self.assertEqual(dialog.phase, "done")
                self.assertEqual(dialog.result.status, "通过")
                self.assertTrue(dialog.result_path.exists())
                self.assertTrue((dialog.result_path.parent / "steps.json").exists())
                dialog.preview()
                self.assertIsNotNone(dialog.preview_window)
                np.testing.assert_allclose(
                    dialog.preview_window.engine.neutral["WT_TEST"],
                    dialog.preview_window.candidate_engine.neutral["WT_TEST"])
                dialog.preview_window.close()
                dialog.apply()
                self.assertIn("right_upper_arm", window.engine.continuous_basis)
                dialog.restore()
                self.assertNotIn("right_upper_arm", window.engine.continuous_basis)
            finally:
                window.close()
