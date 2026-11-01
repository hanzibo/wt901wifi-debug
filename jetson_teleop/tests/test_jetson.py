"""No robot port: strict config, synthetic UDP, Qt gates and shared offline IK."""
import copy
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
import threading
import pty
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'OptUSBProj-ubuntu')]

from jetson_teleop.config import load, PARTS, MODES
from jetson_teleop.run import smoke
from wt901.gui_udp import GuiUdpSource
from optusbproj_ubuntu.real_robot_checkpoints import create, MODES as SIDES
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QCloseEvent
from jetson_teleop.window import Window, Preview
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.real_robot_protocol import encode
from PySide6.QtCore import QTimer
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker

APP = QApplication.instance() or QApplication([])
# Production run.main warms the JIT before creating the live window. Tests that
# construct Window directly must honor that same initialization boundary.
from jetson_teleop.fast_geometry import warmup
warmup()


def raw_config():
    return dict(version=1, mode=MODES[0], frequency=20,
                assignments={key: f'WT{5500000000+i:010d}' for i, key in enumerate(PARTS)},
                axis_mappings={key: ['+Y', '-X', '+Z'] for key in PARTS},
                signs={s:[1]*6 for s in ('left', 'right')},
                palm_offsets={s:dict(axis='z', degrees=90) for s in ('left', 'right')},
                checkpoints={m: m+'.json' for m in MODES})


def checkpoint(mode):
    joints = {}
    for side in SIDES[mode]:
        limits = ArmLimits(side=side, lower=(0,90,0,0,0,50) if side=='right' else ArmLimits().lower,
                           upper=(235,240,180,118,180,121) if side=='right' else ArmLimits().upper)
        for i in range(6):
            joints[side[0].upper()+str(i+1)] = dict(lower=limits.lower[i], upper=limits.upper[i],
                                                   speed=limits.speed[i], acceleration=limits.acceleration[i])
    gesture = {side[0].upper()+str(i): dict(angle=90, tolerance=5)
               for side in SIDES[mode] for i in (2,4)}
    return create(mode, joints, gesture, input_settings=dict(manual_mapping=True,
                          palm_offset_enabled={side:True for side in SIDES[mode]}))


def packet(device, tick=0, gyro=0):
    frame = bytearray(54)
    frame[:12] = device.encode('ascii')
    frame[12:18] = bytes([26,10,25,12,0,0])
    frame[17] = (tick//1000)%60
    frame[16] = (tick//60000)%60
    struct.pack_into('<H', frame, 18, tick%1000)
    struct.pack_into('<h', frame, 24, 2048)
    struct.pack_into('<h', frame, 26, gyro)
    frame[52:] = b'\r\n'
    return bytes(frame)


class RecordingTests(unittest.TestCase):
    def test_batched_telemetry_flushes_when_idle_and_closes_without_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'batch.jsonl'
            recorder = Recorder(path); recorder.flush_interval = .1
            try:
                for i in range(30): recorder.record('telemetry', sequence=i)
                until = time.monotonic()+2
                while len(path.read_text().splitlines()) < 30 and time.monotonic() < until:
                    time.sleep(.01)
                self.assertEqual(len(path.read_text().splitlines()), 30)
                recorder.record('error', message='test critical flush')
            finally:
                recorder.close()
            self.assertIsNone(recorder.error)
            rows = [json.loads(row) for row in path.read_text().splitlines()]
            self.assertEqual([row['sequence'] for row in rows[:-1]], list(range(30)))
            self.assertEqual(rows[-1]['event'], 'error')


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'config.json'

    def read(self, value):
        self.path.write_text(json.dumps(value), encoding='utf-8')
        return load(self.path)

    def test_valid_and_relative_paths(self):
        result = self.read(raw_config())
        self.assertEqual(result['checkpoints'][MODES[0]].parent, self.path.parent)

    def test_rejects_invalid_or_unsafe_values(self):
        for name, value in [('version',True), ('mode','relative'), ('frequency',float('nan')),
                            ('frequency',101), ('assignments',[]), ('checkpoints',None),
                            ('signs',None), ('palm_offsets',None)]:
            with self.subTest(name=name, value=value):
                raw = raw_config(); raw[name] = value
                with self.assertRaises(ValueError): self.read(raw)
        for raw in ([], None):
            with self.assertRaises(ValueError): self.read(raw)
        raw = raw_config(); raw['assignments'][PARTS[1]] = raw['assignments'][PARTS[0]]
        with self.assertRaises(ValueError): self.read(raw)
        raw = raw_config(); raw['axis_mappings'][PARTS[0]] = ['+X','+Y','-Z']
        with self.assertRaises(ValueError): self.read(raw)

    def test_offline_ik_zero_fingers(self):
        self.assertEqual(smoke(), 0)


class GuiTests(ConfigTests):
    def setUp(self):
        super().setUp()
        for mode in MODES:
            (self.path.parent/(mode+'.json')).write_text(json.dumps(checkpoint(mode)), encoding='utf-8')
        self.source = GuiUdpSource(0, '127.0.0.1')
        self.addCleanup(self.source.close)
        self.window = Window(self.source, self.read(raw_config()), background=False)
        for delay in self.window.zero_delays.values():
            delay.setValue(0)
        self.window.timer.stop(); self.window.control.timer.stop()
        self.addCleanup(self.window.close)

    def feed(self, keys, learn=True):
        # Test only on loopback, random UDP port. Never an IMU configuration command.
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _ in range(45 if learn else 1):
                self.test_tick = getattr(self, 'test_tick', 0)+20
                for key in keys:
                    sender.sendto(packet(self.window.config['assignments'][key], self.test_tick),
                                  self.source.sock.getsockname())
                self.window.poll()
        finally:
            sender.close()

    def zero_both(self):
        self.window.zero('arms')
        self.window.zero('hands')

    def age_devices(self, keys, seconds):
        for key in keys:
            device = self.window.engine.assignments[key]
            ip, stamp = self.source._devices[device]
            self.source._devices[device] = (ip, time.monotonic()-seconds)
            self.window.engine.received_at[device] = time.monotonic()-seconds

    def prepare_async_start(self):
        w = self.window; d = w.control
        w.review.setChecked(True); self.feed(PARTS); self.zero_both()
        d._poll()
        self.assertIsNotNone(d.arm_sample)
        d.arm_capture = copy.deepcopy(d.arm_sample)
        d.worker = Mock(active=False, pending_start=None, quit_requested=False)
        d.worker.return_snapshot.return_value = (None, None, False)
        d.worker.isRunning.return_value = True
        d.origin = d.last_frame = encode([90.]*15, [0]*10)
        d.connected = True; d.safety.setChecked(True)
        d._poll = Mock()
        # This tests the async preparation, not a physical transport.
        self.addCleanup(self.cleanup_async_start)
        return d

    def cleanup_async_start(self):
        d = self.window.control
        d.auto_start.cancel('test cleanup')
        if d.ik_worker is not None:
            d.ik_worker.shutdown(); d.ik_worker.wait(3000)
            APP.processEvents()
        d.worker = None; d.ik_worker = None; d.busy = False

    def wait_gui(self, predicate, timeout=3):
        until = time.monotonic()+timeout
        while not predicate() and time.monotonic() < until:
            APP.processEvents(); time.sleep(.002)
        self.assertTrue(predicate())

    def test_display_throttle_does_not_drop_valid_target_publication(self):
        d = self.prepare_async_start(); d.busy = True
        d.ik_worker = Mock()
        result = dict(target=[90.]*6, deadline=time.monotonic()+1.,
                      orientation_error_deg=None, position_error_mm=0., ik_ms=1.)
        with patch.object(d, 'sender', return_value=d.ik_worker), patch.object(
                d, '_read_arm_sample', return_value=(None,d.arm_capture[2],None)), patch.object(
                    d, '_display_due', return_value=False):
            d._ik_result(copy.deepcopy(result)); d._ik_result(copy.deepcopy(result))
        self.assertEqual(d.worker.update_arm_target.call_count, 2)

    def test_async_auto_rejected_by_original_guards_consumes_authorization(self):
        d = self.prepare_async_start()
        d.auto_start.phase = 'locking'; d.auto_start.authorized = True
        solver = Mock(); solver.solve.return_value = dict(target=[90.]*6)
        with patch('jetson_teleop.fast_geometry.prepare', side_effect=lambda s:s), patch('jetson_teleop.window.SharedArmIK', return_value=solver), patch.object(
                RealRobotModesDialog, '_start') as original:
            d._start(automatic=True)
            self.wait_gui(lambda:d.ik_worker is None)
            original.assert_called_once()
            self.assertFalse(d.auto_start.authorized)
            self.assertFalse(d.auto_start.pending)
            d.worker.begin_control.assert_not_called()

    def test_board_async_automatic_takeover_virtual_uart_both_modes(self):
        w = self.window; d = w.control
        for mode in (2, 3):
            with self.subTest(mode=mode):
                d.mode_select.setCurrentIndex(mode)
                w.review.setChecked(True); self.feed(PARTS); self.zero_both(); d._poll()
                side = copy.deepcopy(d.arm_sample[0])
                sides = side if mode == 3 else {'left':side}
                def sample(*args):
                    return dict(at=time.monotonic(), sides=copy.deepcopy(sides),
                                device_ages_ms={'u':1., 'f':1., 'h':1.}), ('stable',)
                master, slave = pty.openpty()
                path = self.path.parent/f'virtual-{mode}.jsonl'
                try:
                    with patch('optusbproj_ubuntu.real_robot_arm_gui.pose_sample', side_effect=sample), patch.object(
                            QMessageBox, 'warning', return_value=QMessageBox.Yes):
                        d.shared_palm.setChecked(False); d.right_options[0].setChecked(False)
                        for rows in (d.axis_rows, d.right_axis_rows):
                            for i,row in enumerate(rows):
                                row[0].setChecked(i == 3)
                        d.auto_start.widgets['countdown'].setValue(1.)
                        d.auto_start.widgets['settle'].setValue(.1)
                        worker = ElbowSerialWorker(os.ttyname(slave), Recorder(path), d)
                        d.worker = worker
                        for signal, slot in ((worker.ready,d._ready), (worker.progress,d._progress),
                                             (worker.sent,d._sent), (worker.failed,d._failed),
                                             (worker.finished,d._finished)):
                            signal.connect(slot)
                        writes = []; worker.sent.connect(writes.append)
                        d.timer.start(20); worker.start(); self.wait_gui(lambda:d.connected)
                        baseline = encode([90.]*15, [0]*10)
                        d.origin = d.last_frame = baseline; d.safety.setChecked(True)
                        self.wait_gui(lambda:d.arm_sample is not None)
                        d.auto_start.arm(); self.assertEqual(writes, [])
                        self.wait_gui(lambda:d.shared_phase == 'following', timeout=15)
                        self.assertEqual(writes[0], baseline)
                        d._pause('virtual test'); self.wait_gui(lambda:not d.busy and d.ik_worker is None)
                        d._disconnect(); self.wait_gui(lambda:d.worker is None)
                        events = [json.loads(s)['event'] for s in path.read_text().splitlines()]
                        for name in ('initial_ik_started','initial_ik_finished','auto_start_completed'):
                            self.assertEqual(events.count(name), 1)
                finally:
                    d.timer.stop(); d._disconnect()
                    self.wait_gui(lambda:d.worker is None and d.ik_worker is None)
                    os.close(master); os.close(slave)

    def test_initial_ik_background_keeps_gui_responsive_and_reuses_result(self):
        d = self.prepare_async_start()
        threads = []; ticks = []
        def solve(side):
            threads.append(threading.get_ident()); time.sleep(.12)
            return dict(target=[90.]*6)
        solver = Mock(); solver.solve.side_effect = solve
        timer = QTimer(); timer.setInterval(5); timer.timeout.connect(lambda:ticks.append(1))
        with patch('jetson_teleop.fast_geometry.prepare', side_effect=lambda s:s), patch('jetson_teleop.window.SharedArmIK', return_value=solver), patch.object(
                RealRobotModesDialog, '_start') as original:
            timer.start(); d._start()
            self.assertFalse(d.busy)
            d.worker.begin_control.assert_not_called()
            self.wait_gui(lambda:original.called)
            timer.stop()
            self.assertGreater(len(ticks), 3)
            self.assertNotEqual(threads[0], threading.get_ident())
            self.assertIs(original.call_args.kwargs['_prepared'][0], solver)
            self.assertEqual(solver.solve.call_count, 1)

    def test_cancelled_initial_ik_never_enters_original_start(self):
        d = self.prepare_async_start()
        event = threading.Event()
        solver = Mock()
        solver.solve.side_effect = lambda side:(event.wait(.5) or dict(target=[90.]*6))
        with patch('jetson_teleop.fast_geometry.prepare', side_effect=lambda s:s), patch('jetson_teleop.window.SharedArmIK', return_value=solver), patch.object(
                RealRobotModesDialog, '_start') as original:
            d._start(); worker = d.ik_worker
            d._invalidate_capture(); event.set()
            self.wait_gui(lambda:d.ik_worker is None)
            self.assertTrue(worker.cancelled.is_set())
            original.assert_not_called(); d.worker.begin_control.assert_not_called()

    def test_initial_ik_context_change_discards_result(self):
        d = self.prepare_async_start()
        solver = Mock(); solver.solve.side_effect = lambda side:(time.sleep(.05) or dict(target=[90.]*6))
        with patch('jetson_teleop.fast_geometry.prepare', side_effect=lambda s:s), patch('jetson_teleop.window.SharedArmIK', return_value=solver), patch.object(
                RealRobotModesDialog, '_start') as original:
            d._start()
            # Even a programmatic change without a UI signal must be rejected.
            d.origin = None
            self.wait_gui(lambda:d.ik_worker is None)
            original.assert_not_called(); d.worker.begin_control.assert_not_called()

    def test_control_poll_drains_existing_socket_before_pose_read(self):
        with patch.object(self.window, 'acquire') as read, patch.object(
                RealRobotModesDialog, '_poll') as control:
            order = []
            read.side_effect = lambda:order.append('read')
            control.side_effect = lambda:order.append('control')
            self.window.control._poll()
            self.assertEqual(order, ['read','control'])

    def test_no_auto_uart_and_explicit_fresh_zero(self):
        w = self.window; d = w.control
        self.assertEqual(w.preset_error, '')
        self.assertIsNone(d.hand_session)
        self.assertIsNone(d.worker)
        self.assertFalse(d.connect_button.isEnabled())
        w.review.setChecked(True); self.zero_both()
        self.assertFalse(w.zero_confirmed)
        self.feed(PARTS[:3]); self.zero_both()
        self.assertTrue(w.zero_confirmed)
        self.assertTrue(d.connect_button.isEnabled())
        self.assertTrue(all(w.engine.algorithm_modes[k]=='gyro_tilt' for k in PARTS))
        w.review.setChecked(False)
        self.assertFalse(w.zero_confirmed)
        with patch('optusbproj_ubuntu.real_robot_gui.ElbowSerialWorker') as uart:
            d._connect()
            uart.assert_not_called()

    def test_bilateral_requires_all_six_and_invalidates_reference(self):
        w = self.window; d = w.control
        self.feed(PARTS[:3]); w.review.setChecked(True); self.zero_both()
        self.assertTrue(w.zero_confirmed)
        d.mode_select.setCurrentIndex(3)
        self.assertFalse(w.zero_confirmed)
        self.assertFalse(w.review.isChecked())
        w.review.setChecked(True); self.zero_both()
        self.assertFalse(w.zero_confirmed)
        self.feed(PARTS); self.zero_both()
        self.assertTrue(w.zero_confirmed)

    def test_stale_input_does_not_allow_zero(self):
        self.feed(PARTS)
        w = self.window; w.review.setChecked(True)
        self.age_devices(PARTS, 3)
        w.zero(); self.assertFalse(w.zero_confirmed)

    def test_missing_preset_fails_closed(self):
        w = self.window; d = w.control
        w.config['checkpoints'][MODES[1]] = self.path.parent/'missing.json'
        d.mode_select.setCurrentIndex(3)
        self.assertTrue(w.preset_error)
        w.review.setChecked(True); self.feed(PARTS); w.zero()
        self.assertFalse(w.zero_confirmed)
        self.assertFalse(d.connect_button.isEnabled())

    def test_groups_are_independent_and_both_required(self):
        w = self.window
        w.control.mode_select.setCurrentIndex(3)
        w.review.setChecked(True); self.feed(PARTS)
        with patch.object(w.engine, 'zero', wraps=w.engine.zero) as capture:
            w.zero('arms')
            self.assertEqual([c.args[0] for c in capture.call_args_list],
                             [k for k in PARTS if not k.endswith('_hand')])
        self.assertFalse(w.zero_confirmed)
        self.assertFalse(w.control.connect_button.isEnabled())
        saved = {k: {m: value.copy() for m, value in w.engine.mode_neutral[
            w.engine.assignments[k]].items()} for k in PARTS if not k.endswith('_hand')}
        with patch.object(w.engine, 'zero', wraps=w.engine.zero) as capture:
            w.zero('hands')
            self.assertEqual([c.args[0] for c in capture.call_args_list],
                             [k for k in PARTS if k.endswith('_hand')])
        self.assertTrue(w.zero_confirmed)
        for k, modes in saved.items():
            for m, value in modes.items():
                self.assertTrue((value == w.engine.mode_neutral[w.engine.assignments[k]][m]).all())

    def test_countdown_captures_at_deadline_and_cancel_invalidates(self):
        w = self.window; w.review.setChecked(True); self.feed(PARTS)
        w.zero_delays['arms'].setValue(3)
        with patch.object(w.engine, 'zero', wraps=w.engine.zero) as capture:
            w.zero('arms')
            self.assertIsNotNone(w.zero_pending)
            capture.assert_not_called()
            group, keys, deadline = w.zero_pending
            w.zero_pending = (group, keys, time.monotonic()-1)
            w.tick_zero()
            self.assertEqual(capture.call_count, 2)
        w.zero('hands'); self.assertTrue(w.zero_confirmed)
        w.zero('arms'); self.assertFalse(w.zero_confirmed)
        w.cancel_zero()
        self.assertIsNone(w.zero_pending)
        self.assertFalse(w.zero_timer.isActive())
        self.assertFalse(w.control.connect_button.isEnabled())

    def test_stale_group_at_deadline_does_not_capture(self):
        w = self.window; w.review.setChecked(True); self.feed(PARTS)
        w.zero_delays['hands'].setValue(3); w.zero('hands')
        self.age_devices(PARTS, 3)
        group, keys, deadline = w.zero_pending
        w.zero_pending = (group, keys, time.monotonic()-1)
        with patch.object(w.engine, 'zero') as capture:
            w.tick_zero(); capture.assert_not_called()
        self.assertIn('0/1', w.message.text())
        self.assertIn('跳过', w.message.text())
        self.assertFalse(w.zero_confirmed)

    def test_mode_change_cancels_pending_and_uart_blocks_capture(self):
        w = self.window; w.review.setChecked(True); self.feed(PARTS)
        w.zero_delays['arms'].setValue(3); w.zero('arms')
        self.assertIsNotNone(w.zero_pending)
        w.control.mode_select.setCurrentIndex(3)
        self.assertIsNone(w.zero_pending)
        self.assertFalse(w.zeroed_parts)
        w.review.setChecked(True)
        w.control.worker = Mock()
        try:
            w.zero('hands'); self.assertIsNone(w.zero_pending)
        finally:
            w.control.worker = None

    def test_lightweight_imports_and_render(self):
        self.window.show(); self.window.poll(); APP.processEvents()
        image = self.window.grab()
        self.assertGreater(image.width(), 0)
        for name in ('vtk', 'pyvista', 'cv2', 'torch', 'mediapipe', 'pyqtgraph'):
            self.assertNotIn(name, sys.modules)

    def test_learning_stages_are_display_only_for_neutral_capture(self):
        w = self.window; w.review.setChecked(True)
        device = w.engine.assignments[PARTS[0]]
        self.assertFalse(w.engine.tilt_status(device)['gravity_ready'])
        self.feed(PARTS[:3], learn=False)
        with patch.object(w.engine, 'zero') as capture:
            w.zero('arms'); self.assertEqual(capture.call_count, 2)
        self.assertIsNone(w.zero_pending)
        self.assertIn('2/2', w.message.text())
        for _ in range(20): self.feed(PARTS[:3], learn=False)
        state = w.engine.tilt_status(device)
        self.assertTrue(state['gravity_ready'])
        self.assertFalse(state['bias_ready'])
        w.zero('arms'); self.assertIsNone(w.zero_pending)
        self.feed(PARTS[:3])
        self.assertTrue(w.engine.tilt_status(device)['bias_ready'])
        w.next_display = 0; w.poll()
        self.assertIn('重力：已建立', w.devices.text())
        self.assertIn('零偏：已完成', w.devices.text())
        for line in w.devices.text().splitlines():
            self.assertRegex(line, r' · 距末包 (?:未收到|\d+ ms)$')
        self.assertIn('就绪 3/3', w.learning_summary.text())
        self.assertTrue(w.zero_button.isEnabled())
        self.zero_both(); self.assertTrue(w.zero_confirmed)
        self.assertTrue(w.engine.tilt_status(device)['bias_ready'])  # zero is not reset.

    def test_capture_allows_one_second_age_but_skips_over_two_seconds(self):
        w = self.window; w.review.setChecked(True)
        self.feed(PARTS[:2])
        w.zero('arms')
        self.assertEqual(len(w.zeroed_parts), 2)
        w.zero('hands'); self.assertIsNone(w.zero_pending)
        self.assertFalse(w.zero_confirmed)
        self.feed(PARTS[:3]); self.zero_both()
        self.assertTrue(w.zero_confirmed)
        device = w.engine.assignments[PARTS[0]]
        self.age_devices(PARTS[:3], 1)
        self.assertTrue(w.engine.tilt_status(device)['bias_ready'])
        self.zero_both(); self.assertTrue(w.zero_confirmed)
        self.age_devices(PARTS[:1], 3)
        w.zero('arms')
        self.assertFalse(w.zero_confirmed)
        self.assertNotIn(PARTS[0], w.zeroed_parts)
        self.assertIn(PARTS[1], w.zeroed_parts)
        self.assertIn('1/2', w.message.text())
        self.assertIn('跳过', w.message.text())
        w.next_display = 0; w.poll()
        self.assertTrue(w.zero_button.isEnabled())
        self.assertIn('超时', w.devices.text())

    def test_online_device_without_pose_is_skipped(self):
        w = self.window; w.review.setChecked(True)
        self.feed(PARTS[:2], learn=False)
        del w.engine.current[w.engine.assignments[PARTS[0]]]
        with patch.object(w.engine, 'zero') as capture:
            w.zero('arms')
            capture.assert_called_once_with(PARTS[1])
        self.assertIn('1/2', w.message.text())
        self.assertFalse(w.zero_confirmed)

    def test_motion_at_deadline_does_not_block_pc_style_capture(self):
        w = self.window; w.review.setChecked(True); self.feed(PARTS)
        w.zero_delays['arms'].setValue(3); w.zero('arms')
        device = w.engine.assignments[PARTS[0]]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(packet(device, self.test_tick+20, gyro=1024), self.source.sock.getsockname())
            w.poll()
        finally:
            sender.close()
        self.assertTrue(w.engine.tilt_status(device)['bias_ready'])
        self.assertFalse(w.engine.tilt_status(device)['static'])
        w.next_display = 0; w.poll()
        device_line = next(line for line in w.devices.text().splitlines() if device in line)
        self.assertIn('重力：已建立', device_line)
        self.assertIn('零偏：已完成', device_line)
        self.assertNotIn('运动', device_line)
        self.assertNotIn('静止', device_line)
        group, keys, deadline = w.zero_pending
        w.zero_pending = (group, keys, time.monotonic()-1)
        with patch.object(w.engine, 'zero') as capture:
            w.tick_zero(); self.assertEqual(capture.call_count, 2)
        self.assertIn('2/2', w.message.text())
        self.assertFalse(w.zero_confirmed)

    def test_main_mode_selector_syncs_and_resets_neutral(self):
        w = self.window
        self.assertIn('右臂不参与', w.mode_summary.text())
        self.feed(PARTS); w.review.setChecked(True); self.zero_both()
        self.assertTrue(w.zero_confirmed)
        w.mode_choice.setCurrentIndex(1)
        self.assertTrue(w.control._bilateral_mode())
        self.assertFalse(w.zero_confirmed)
        self.assertFalse(w.review.isChecked())
        self.assertIn('4 台', w.zero_buttons['arms'].text())
        self.assertIn('2 台', w.zero_buttons['hands'].text())
        w.review.setChecked(True); self.zero_both()
        self.assertTrue(w.zero_confirmed)
        self.assertIn('臂段 4/4', w.mode_summary.text())
        w.control.mode_select.setCurrentIndex(2)
        self.assertEqual(w.mode_choice.currentIndex(), 0)
        self.assertIn('2 台', w.zero_buttons['arms'].text())
        self.assertFalse(w.zero_confirmed)

    def test_preview_front_left_right_and_mirror_are_display_only(self):
        # Model +Y is anatomical left: front view puts it on screen right.
        points = [[0, .2, 0], [0, -.2, 0]]
        front = Preview.project(points)
        mirror = Preview.project(points, True)
        self.assertGreater(front[0, 0], front[1, 0])
        self.assertLess(mirror[0, 0], mirror[1, 0])
        w = self.window
        self.feed(PARTS); w.review.setChecked(True); self.zero_both()
        assignments = w.engine.assignments.copy()
        before = w.engine.transforms()
        self.assertFalse(w.preview_mirror.isChecked())
        w.preview_mirror.setChecked(True)
        self.assertTrue(w.zero_confirmed)
        self.assertEqual(w.engine.assignments, assignments)
        self.assertTrue(all((a == b).all() for a, b in zip(before, w.engine.transforms())))

    def test_parent_close_waits_for_control_shutdown(self):
        # Keep input flowing while the original transport brakes asynchronously.
        self.window.timer.start()
        event = QCloseEvent()
        with patch.object(self.window.control, 'close', return_value=False), \
                patch('jetson_teleop.window.QtCore.QTimer.singleShot') as retry:
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(self.window.timer.isActive())
        retry.assert_called_once()
        event = QCloseEvent()
        with patch.object(self.window.control, 'close', return_value=True):
            self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertFalse(self.window.timer.isActive())


if __name__ == '__main__': unittest.main()
