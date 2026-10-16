"""Offline command controller + virtual UART tests. Never open robot devices."""
import json
import math
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pty
import random
import select
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from PySide6.QtGui import QCloseEvent
from optusbproj_ubuntu.real_robot_control import ElbowControl, Limits, elbow_angle
from optusbproj_ubuntu.real_robot_protocol import encode, decode
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_gui import RealRobotDialog

APP = QApplication.instance() or QApplication([])
BASELINE = encode([90.] * 15, [0] * 10)


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if predicate():
            return
        time.sleep(.004)
    raise AssertionError('virtual test timed out')


class ControlTests(unittest.TestCase):
    def test_angle(self):
        self.assertAlmostEqual(elbow_angle([[0, 0, -1], [0, 0, -3]]), 0)
        self.assertAlmostEqual(elbow_angle([[0, 0, -1], [1, 0, 0]]), 90)
        self.assertAlmostEqual(elbow_angle([[0, 0, -1], [0, 0, 1]]), 180)
        with self.assertRaises(ValueError):
            elbow_angle([[0, 0, 0], [1, 0, 0]])

    def test_first_frame_relative_and_whitelist(self):
        c = ElbowControl(BASELINE, 80)
        c.set_target(50)
        self.assertEqual(c.step(), BASELINE)
        for _ in range(100):
            frame = c.step()
            self.assertEqual(frame[:20], BASELINE[:20])
            self.assertEqual(frame[22:52], BASELINE[22:52])
        self.assertAlmostEqual(decode(frame)[0][9], 60)
        c.set_target(120)
        self.assertEqual(c.target, 100)
        self.assertTrue(c.saturated)

    def test_invalid_limits_and_baseline(self):
        for args in ({'frequency': 0}, {'speed': float('nan')}, {'acceleration': -1}, {'gain': 4},
                     {'speed': 250.01}, {'acceleration': 250.01}, {'lower': -.01}, {'upper': 100.01}):
            with self.assertRaises(ValueError):
                Limits(**args)
        angles = [90.] * 15
        angles[3] = 89
        with self.assertRaises(ValueError):
            ElbowControl(encode(angles, [0] * 10), 90)

    def test_random_retargets_velocity_acceleration_and_bounds(self):
        rng = random.Random(12)
        for frequency in (5, 20, 50, 100):
            for speed, acceleration, sign in ((10, 20, 1), (60, 60, -1), (250, 250, 1), (250, 250, -1)):
                c = ElbowControl(BASELINE, 90, Limits(frequency=frequency, sign=sign, speed=speed, acceleration=acceleration))
                previous = 0.
                for _ in range(1500):
                    c.set_target(rng.uniform(0, 180))
                    c.step()
                    self.assertTrue(0 - 1e-7 <= c.q <= 100 + 1e-7)
                    self.assertLessEqual(abs(c.v), speed + 1e-7)
                    self.assertLessEqual(abs(c.a), acceleration + 1e-7)
                    self.assertLessEqual(abs(c.v - previous), acceleration / frequency + 1e-7)
                    previous = c.v

    def test_high_limits_reach_both_expanded_endpoints_and_brake(self):
        c = ElbowControl(BASELINE, 90, Limits(speed=250, acceleration=250))
        for theta, endpoint in ((0, 0), (180, 100)):
            c.set_target(theta)
            for _ in range(100):
                frame = c.step()
                self.assertEqual(frame[:20], BASELINE[:20])
                self.assertEqual(frame[22:52], BASELINE[22:52])
            self.assertAlmostEqual(c.q, endpoint)
            self.assertEqual(decode(frame)[0][9], endpoint)
        c.set_target(0)
        for _ in range(8):
            c.step()
        self.assertLess(c.v, -60)
        c.brake('high-speed test')
        for _ in range(100):
            c.step()
            self.assertTrue(0 - 1e-7 <= c.q <= 100 + 1e-7)
        self.assertEqual(c.mode, 'paused')

    def test_brake_latches_and_cannot_resume(self):
        c = ElbowControl(BASELINE, 90)
        c.set_target(60)
        for _ in range(15):
            c.step()
        self.assertLess(c.v, 0)
        c.brake('test gap')
        for _ in range(20):
            c.step()
        self.assertEqual(c.mode, 'paused')
        stopped = c.q
        c.set_target(90)
        for _ in range(10):
            c.step()
        self.assertEqual(c.q, stopped)
        self.assertEqual(c.v, 0)


class VirtualSerialTests(unittest.TestCase):
    def test_idle_start_gap_brake_and_no_auto_resume(self):
        master, slave = pty.openpty()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.jsonl'
            worker = ElbowSerialWorker(os.ttyname(slave), Recorder(path))
            ready, states, frames, errors = [], [], [], []
            worker.ready.connect(lambda: ready.append(True))
            worker.progress.connect(states.append)
            worker.sent.connect(frames.append)
            worker.failed.connect(errors.append)
            worker.start()
            try:
                wait_for(lambda: ready)
                self.assertFalse(select.select([master], [], [], .1)[0])
                worker.begin(BASELINE, 90, Limits(), time.monotonic() + .25)
                wait_for(lambda: frames)
                self.assertEqual(frames[0], BASELINE)
                for _ in range(15):
                    worker.update_target(60, time.monotonic() + .25)
                    APP.processEvents()
                    time.sleep(.03)
                wait_for(lambda: states and states[-1]['mode'] == 'paused')
                self.assertFalse(errors)
                self.assertLess(decode(frames[-1])[0][9], 90)
                count = len(frames)
                worker.update_target(90, time.monotonic() + .25)
                time.sleep(.15)
                APP.processEvents()
                self.assertEqual(len(frames), count)
                received = b''
                while select.select([master], [], [], .01)[0]:
                    received += os.read(master, 4096)
                self.assertEqual(received, b''.join(frames))
            finally:
                worker.shutdown()
                self.assertTrue(worker.wait(2000))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            writes = [r for r in records if r['event'] == 'write']
            self.assertEqual(len(writes), len(frames))
            self.assertTrue(all(r['gap_ms'] >= 49 for r in writes[1:]))
        os.close(master)
        os.close(slave)

    def test_partial_write_is_terminal(self):
        class Partial:
            def __init__(self, **kwargs): pass
            def open(self): pass
            def close(self): pass
            def write(self, frame): return 3
        with tempfile.TemporaryDirectory() as directory:
            worker = ElbowSerialWorker('VIRTUAL', Recorder(Path(directory) / 'partial.jsonl'), serial_factory=Partial)
            ready, errors, sent = [], [], []
            worker.ready.connect(lambda: ready.append(True))
            worker.failed.connect(errors.append)
            worker.sent.connect(sent.append)
            worker.start()
            try:
                wait_for(lambda: ready)
                worker.begin(BASELINE, 90, Limits(), time.monotonic() + .25)
                wait_for(lambda: errors)
                self.assertTrue(worker.wait(2000))
                self.assertEqual(sent, [])
                self.assertIn('不重试', errors[0])
            finally:
                worker.shutdown()
                worker.wait(2000)

    def test_expired_start_sends_nothing(self):
        class Dummy:
            writes = []
            def __init__(self, **kwargs): pass
            def open(self): pass
            def close(self): pass
            def write(self, frame): self.writes.append(frame); return 54
        with tempfile.TemporaryDirectory() as directory:
            worker = ElbowSerialWorker('VIRTUAL', Recorder(Path(directory) / 'expired.jsonl'), serial_factory=Dummy)
            states = []
            worker.progress.connect(states.append)
            worker.begin(BASELINE, 90, Limits(), time.monotonic() - 1)
            worker.start()
            try:
                wait_for(lambda: states)
                self.assertEqual(Dummy.writes, [])
            finally:
                worker.shutdown()
                self.assertTrue(worker.wait(2000))


class DialogTests(unittest.TestCase):
    def test_refresh_ports_preserves_selection_without_opening_serial(self):
        viewer = QWidget()
        viewer.replay_mode = False
        viewer.engine = None
        device = lambda name: SimpleNamespace(device=name, description='mock USB UART')
        with patch('optusbproj_ubuntu.real_robot_gui.comports', return_value=[device('/dev/ttyUSB0')]) as scan, \
             patch('serial.Serial') as serial:
            dialog = RealRobotDialog(viewer)
            dialog.timer.stop()
            try:
                self.assertEqual(dialog.port.currentText(), '')
                scan.return_value = [device('/dev/ttyUSB1'), device('/dev/ttyUSB0')]
                dialog.refresh_button.click()
                self.assertEqual(dialog.port.count(), 2)
                self.assertEqual(dialog.port.currentText(), '')
                dialog.port.setCurrentText('/dev/ttyUSB1')
                dialog.refresh_button.click()
                self.assertEqual(dialog.port.currentText(), '/dev/ttyUSB1')
                dialog.port.setEditText('/dev/serial/by-id/custom-robot')
                scan.return_value = []
                dialog.refresh_button.click()
                self.assertEqual(dialog.port.count(), 0)
                self.assertEqual(dialog.port.currentText(), '/dev/serial/by-id/custom-robot')
                self.assertIn('发现 0 个端口', dialog.status.text())
                serial.assert_not_called()
                self.assertIsNone(dialog.worker)
            finally:
                dialog.close()
                viewer.close()

    def test_refresh_failure_and_connected_guard_preserve_state(self):
        viewer = QWidget()
        viewer.replay_mode = False
        viewer.engine = None
        with patch('optusbproj_ubuntu.real_robot_gui.comports', return_value=[
                SimpleNamespace(device='/dev/ttyUSB0', description='mock USB UART')]) as scan:
            dialog = RealRobotDialog(viewer)
            dialog.timer.stop()
            try:
                dialog.port.setCurrentText('/dev/ttyUSB0')
                scan.side_effect = OSError('mock enumeration error')
                dialog.refresh_button.click()
                self.assertIn('串口扫描失败', dialog.status.text())
                self.assertEqual(dialog.port.count(), 1)
                self.assertEqual(dialog.port.currentText(), '/dev/ttyUSB0')
                scan.reset_mock()
                dialog.worker = SimpleNamespace(isRunning=lambda: True, recorder=Mock())
                dialog.connected = True
                dialog.origin = BASELINE
                dialog.status.setText('保持已有连接')
                dialog._controls()
                self.assertFalse(dialog.refresh_button.isEnabled())
                dialog._refresh_ports()
                scan.assert_not_called()
                self.assertEqual(dialog.status.text(), '保持已有连接')
                self.assertEqual(dialog.origin, BASELINE)
                self.assertTrue(dialog.connected)
            finally:
                dialog.worker = None
                dialog.close()
                viewer.close()

    def test_parent_close_requests_braking_before_other_cleanup(self):
        from optusbproj_ubuntu.gui import BodyWindow
        from optusbproj_ubuntu.fusion_gui import FusionWindow
        for cls in (BodyWindow, FusionWindow):
            requests = []
            def deferred_close():
                requests.append('brake')
                return False
            parent = SimpleNamespace(real_robot_dialog=SimpleNamespace(close=deferred_close))
            event = QCloseEvent()
            cls.closeEvent(parent, event)
            self.assertEqual(requests, ['brake'])
            self.assertFalse(event.isAccepted())

    def test_gui_engage_config_pause_and_escape_braking_with_virtual_uart(self):
        master, slave = pty.openpty()
        viewer = QWidget()
        viewer.replay_mode = False
        viewer.engine = object()
        source = dict(theta=80., generation=0)

        def snapshot(*args):
            theta = math.radians(source['theta'])
            return ({'at': time.monotonic(),
                     'device_ages_ms': {'left_upper_arm': 0, 'left_forearm': 0},
                     'sides': {'left': {'directions': [[0, 0, -1], [math.sin(theta), 0, -math.cos(theta)]]}}},
                    ('configuration', source['generation']))

        with tempfile.TemporaryDirectory() as directory, \
                patch('optusbproj_ubuntu.real_robot_gui.pose_sample', side_effect=snapshot), \
                patch('optusbproj_ubuntu.real_robot_gui.Recorder', side_effect=lambda _: Recorder(Path(directory) / 'gui.jsonl')), \
                patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):
            dialog = RealRobotDialog(viewer)
            dialog.show()
            dialog.port.setCurrentText(os.ttyname(slave))
            dialog._connect()
            try:
                wait_for(lambda: dialog.connected)
                dialog._confirm_origin()
                self.assertEqual(dialog.origin, BASELINE)
                self.assertFalse(select.select([master], [], [], .02)[0])
                dialog.safety.setChecked(True)
                dialog._poll()
                dialog.history.clear()
                now = time.monotonic()
                dialog.history.extend((now - .6 + i * .15, 80.) for i in range(5))
                dialog._capture()
                self.assertIsNotNone(dialog.capture)
                dialog.frequency.setValue(25)
                self.assertIsNone(dialog.capture)
                dialog._capture()
                dialog._start()
                self.assertTrue(dialog.busy)
                wait_for(lambda: dialog.last_frame == BASELINE)
                source['theta'] = 75.
                wait_for(lambda: dialog.last_frame and decode(dialog.last_frame)[0][9] < 89.9)
                source['generation'] = 1
                wait_for(lambda: not dialog.busy)
                self.assertIsNone(dialog.capture)
                paused = dialog.last_frame
                source['theta'] = 70.
                for _ in range(30):
                    APP.processEvents()
                    time.sleep(.01)
                self.assertEqual(dialog.last_frame, paused)
                self.assertFalse(dialog.safety.isChecked())
                dialog.safety.setChecked(True)
                dialog.history.clear()
                now = time.monotonic()
                dialog.history.extend((now - .6 + i * .15, 70.) for i in range(5))
                dialog._capture()
                dialog._start()
                source['theta'] = 65.
                wait_for(lambda: dialog.last_frame != paused)
                dialog.reject()  # Escape must not hide a still-running sender.
                wait_for(lambda: dialog.worker is None)
                self.assertFalse(dialog.isVisible())
                self.assertIsNone(dialog.origin)
            finally:
                if dialog.worker is not None:
                    dialog._disconnect()
                    wait_for(lambda: dialog.worker is None)
                dialog.timer.stop()
                dialog.close()
        viewer.close()
        os.close(master)
        os.close(slave)

    def test_no_io_creation_and_replay_guard(self):
        viewer = QWidget()
        viewer.replay_mode = True
        viewer.engine = object()
        with patch('optusbproj_ubuntu.real_robot_gui.ElbowSerialWorker') as worker:
            dialog = RealRobotDialog(viewer)
            dialog._connect()
            worker.assert_not_called()
            self.assertFalse(dialog.connect_button.isEnabled())
            self.assertEqual(dialog.frequency.value(), 20)
            self.assertEqual(dialog.speed.value(), 10)
            self.assertEqual(dialog.acceleration.value(), 20)
            self.assertEqual(dialog.speed.maximum(), 250)
            self.assertEqual(dialog.acceleration.maximum(), 250)
            dialog.speed.setValue(250)
            dialog.acceleration.setValue(250)
            self.assertEqual(dialog.speed.value(), 250)
            self.assertEqual(dialog.acceleration.value(), 250)
            self.assertIsNone(dialog.origin)
            dialog.timer.stop()
            dialog.close()
        viewer.close()


if __name__ == '__main__':
    unittest.main()
