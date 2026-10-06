"""The wrist GUI guard must work in the wt901 environment without SciPy."""

import unittest
from types import SimpleNamespace
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from wt901_viz.yaw_guard import GyroOnly, PacketClock, YawGuard


class WristYawGuardTests(unittest.TestCase):
    def test_bad_thresholds(self):
        with self.assertRaises(ValueError):
            YawGuard(80, 80)

    def test_false_yaw_at_high_pitch_remains_corrected_after_exit(self):
        guard = YawGuard(80, 75)
        maximum_jump = 0.0
        previous = None
        for sample in range(301):
            t = sample * 0.02
            if t < 1:
                pitch, gyro_y = 0.0, 0.0
            elif t < 2:
                pitch, gyro_y = 85 * (t - 1), 85.0
            elif t < 4:
                pitch, gyro_y = 85.0, 0.0
            elif t < 5:
                pitch, gyro_y = 85 * (5 - t), -85.0
            else:
                pitch, gyro_y = 0.0, 0.0
            false_yaw = float(60 * np.clip((t - 2.2) / 1.5, 0, 1))
            pose = guard.update(t, 0, pitch, false_yaw, (0, gyro_y, 0))
            if previous is not None:
                delta = previous.T @ pose.rotation
                maximum_jump = max(maximum_jump, np.degrees(np.arccos(
                    np.clip((np.trace(delta) - 1) / 2, -1, 1))))
            previous = pose.rotation
        self.assertAlmostEqual(pose.yaw_offset_deg, -60, delta=1.5)
        np.testing.assert_allclose(pose.rotation, np.eye(3), atol=0.03)
        self.assertLess(maximum_jump, 5)

    def test_gyro_x_changes_pose_inside_guard(self):
        guard = YawGuard()
        start = guard.update(0, 0, 79, 0, (0, 0, 0))
        guard.update(0.02, 0, 81, 0, (30, 0, 0))
        result = guard.update(0.04, 0, 82, 0, (30, 0, 0))
        self.assertTrue(result.active)
        self.assertGreater(np.linalg.norm(result.rotation - start.rotation), 0.01)

    def test_packet_clock_midnight_and_duplicate(self):
        clock = PacketClock()
        packet = lambda h, m, s, ms: SimpleNamespace(
            hour=h, minute=m, second=s, millisecond=ms)
        self.assertEqual(clock.elapsed(packet(23, 59, 59, 990)), 0.0)
        self.assertAlmostEqual(clock.elapsed(packet(0, 0, 0, 10)), 0.02, places=5)
        self.assertAlmostEqual(clock.elapsed(packet(0, 0, 0, 10)), 0.04, places=5)

    def test_gyro_only_does_not_need_rpy_and_integrates_body_axes(self):
        estimator = GyroOnly()
        np.testing.assert_allclose(estimator.update(0.0, (0, 90, 0)), np.eye(3))
        result = estimator.update(1.0, (0, 90, 0))
        np.testing.assert_allclose(result @ (1, 0, 0), (0, 0, -1), atol=1e-7)

    def test_gyro_integrator_projects_only_after_health_threshold(self):
        estimator = GyroOnly()
        estimator.update(0.0, (0, 0, 0))
        estimator.rotation = np.diag((1.0, 1.0, 1.001))
        result = estimator.update(0.02, (0, 0, 0))
        self.assertTrue(estimator.last_health.projected)
        np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-12)

    def test_gui_mode_switch_keeps_all_four_neutral_references(self):
        from wrist_demo import WristWindow

        class Source:
            def __init__(self):
                self.queue = []

            def poll(self):
                packets, self.queue = self.queue, []
                return packets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        source = Source()
        window = WristWindow(source, "WT5500011590", "WT5500011408", "test")
        window.timer.stop()
        state = window.part_states["left_hand"]
        def packet(ms, yaw):
            return SimpleNamespace(
                device_id=state.device_id, hour=0, minute=0,
                second=0, millisecond=ms, roll=0.0, pitch=0.0, yaw=yaw,
                gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                battery_percent=90, battery_v=3.9, rssi=-40,
                acc_x=0.0, acc_y=0.0, acc_z=1.0,
                mag_x=0.0, mag_y=0.0, mag_z=0.0,
                temperature=25.0, version=1)
        try:
            source.queue = [packet(0, 0.0), packet(20, 30.0)]
            window.poll()
            self.assertEqual(set(state.stream_current), {"raw", "guard", "gyro", "fusion"})
            refs = {key: value.copy() for key, value in state.stream_reference.items()}
            self.assertFalse(np.allclose(state.stream_current["raw"],
                                         state.stream_current["gyro"]))
            for mode in ("guard", "gyro", "fusion", "raw"):
                state.mode_combo.setCurrentIndex(state.mode_combo.findData(mode))
                self.assertEqual(set(state.stream_reference), set(refs))
                for key in refs:
                    np.testing.assert_allclose(state.stream_reference[key], refs[key])
            window.set_neutral(state.device_id)
            for key in state.stream_current:
                np.testing.assert_allclose(state.stream_reference[key],
                                           state.stream_current[key])
            state.guard_enter_spin.setValue(78.0)
            self.assertEqual(state.stream_current, {})
            self.assertEqual(state.stream_reference, {})
            self.assertEqual(state.guard_exit_spin.value(), 75.0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
