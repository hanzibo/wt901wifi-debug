"""One-use startup authorization and real Qt/IK/virtual-UART acceptance. No hardware."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import pty
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtWidgets import QWidget, QMessageBox
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_start_diagnostics import snapshot, delta_text
from optusbproj_ubuntu.pose import PoseEngine
from test_real_robot_shared import APP, BASE, rest_sample, wait_for


class AutoStartTests(unittest.TestCase):
    def setUp(self):
        self.viewer = QWidget(); self.viewer.engine = None; self.viewer.replay_mode = False
        self.d = RealRobotModesDialog(self.viewer); self.d.timer.stop()
        self.d.mode_select.setCurrentIndex(2)
        self.side = rest_sample(PoseEngine().transforms(include_natural=False))
        self.d.worker = Mock(active=False, pending_start=None, quit_requested=False)
        self.d.worker.isRunning.return_value = True
        self.d.worker.return_snapshot.return_value = (None, None, False)
        self.d.connected = True; self.d.origin = self.d.last_frame = BASE
        self.d.safety.setChecked(True)
        self.d._poll = Mock()
        self.fresh()
        self.a = self.d.auto_start

    def fresh(self):
        now = time.monotonic()
        self.d.arm_sample = (deepcopy(self.side), now+100., ('stable',))
        self.d.signature = ('stable',)
        self.d.arm_history = deque((now-.6+i*.1, deepcopy(self.side)) for i in range(7))

    def arm(self):
        with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes) as box:
            self.a.arm()
        self.assertEqual(box.call_count, 1)
        self.assertTrue(self.a.authorized)
        self.assertEqual(self.a.phase, 'countdown')

    def expire(self):
        self.fresh()
        now = time.monotonic(); self.a.deadline = now; self.a.last_tick = now-.02
        self.a.tick(now)

    def tearDown(self):
        self.a.cancel('test cleanup')
        self.d.busy = False; self.d.worker = None; self.d.ik_worker = None
        self.d.close(); self.viewer.close()

    def test_default_cancel_and_no_authorization_bypass(self):
        self.assertEqual(self.a.values(), dict(tolerance=2., countdown=10., settle=2.))
        with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Cancel): self.a.arm()
        self.assertFalse(self.a.pending); self.assertFalse(self.a.authorized)
        self.d._start(automatic=True); self.d._confirm_follow(automatic=True)
        self.d.worker.begin_control.assert_not_called()
        self.d.worker.confirm_follow.assert_not_called()

    def test_nearby_safety_confirmation_is_synchronized_without_starting(self):
        self.assertTrue(self.a.safety_confirm.isChecked())
        self.a.safety_confirm.click()
        self.assertFalse(self.d.safety.isChecked())
        self.assertFalse(self.a.start.isEnabled())
        self.d.safety.setChecked(True)
        self.assertTrue(self.a.safety_confirm.isChecked())
        self.assertTrue(self.a.start.isEnabled())
        self.assertFalse(self.a.authorized)
        self.d.worker.begin_control.assert_not_called()
        self.d.worker.confirm_follow.assert_not_called()
        self.d.safety.setEnabled(False)
        self.a.controls()
        self.assertFalse(self.a.safety_confirm.isEnabled())

    def test_nearby_safety_uncheck_cancels_countdown_without_reauthorizing(self):
        self.arm()
        self.a.safety_confirm.click()
        self.assertFalse(self.d.safety.isChecked())
        self.assertFalse(self.a.pending)
        self.assertFalse(self.a.authorized)
        self.a.safety_confirm.click()
        self.assertTrue(self.d.safety.isChecked())
        self.assertFalse(self.a.pending)
        self.d.worker.begin_control.assert_not_called()

    def test_checkpoint_clears_nearby_safety_confirmation(self):
        data = self.d.checkpoints.snapshot()
        self.d.checkpoints.apply(data)
        self.assertFalse(self.d.safety.isChecked())
        self.assertFalse(self.a.safety_confirm.isChecked())
        self.assertFalse(self.a.start.isEnabled())
        self.d.worker.begin_control.assert_not_called()

    def test_locks_new_pose_once_then_waits_and_follows_once(self):
        self.arm()
        self.assertIsNone(self.d.arm_capture)
        self.assertFalse(self.d.mode_select.isEnabled())
        self.assertFalse(self.d.checkpoints.available())
        # A different pose after releasing the mouse, not the authorization pose.
        self.side['hand'] = self.side['hand'] @ Rotation.from_euler('z', 15, degrees=True).as_matrix()
        self.d._start = Mock(side_effect=lambda **kw: setattr(self.d, 'busy', True))
        self.expire()
        self.assertEqual(self.a.phase, 'transition')
        np.testing.assert_allclose(self.d.arm_capture[0]['hand'], self.side['hand'])
        self.d._start.assert_called_once_with(automatic=True)
        self.a.progress(dict(mode='active', phase='transition'))
        self.assertIsNone(self.a.wait_until)
        self.a.progress(dict(mode='active', phase='await_confirmation'))
        deadline = self.a.wait_until
        self.a.progress(dict(mode='active', phase='await_confirmation'))
        self.assertEqual(self.a.wait_until, deadline)
        self.d._confirm_follow = Mock()
        self.a.last_tick = deadline-.2
        self.a.tick(deadline-.1)
        self.d._confirm_follow.assert_not_called()
        self.a.tick(deadline)
        self.a.tick(deadline+.02)
        self.d._confirm_follow.assert_called_once_with(automatic=True)
        self.a.progress(dict(mode='active', phase='following'))
        self.assertFalse(self.a.pending); self.assertFalse(self.a.authorized)

    def test_stale_changed_safety_disconnected_and_stalled_cancel(self):
        for case in ('stale', 'signature', 'safety', 'disconnect', 'stall', 'worker'):
            with self.subTest(case=case):
                self.d.connected = True; self.d.safety.setChecked(True)
                self.d.worker.isRunning.return_value = True
                self.fresh(); self.arm()
                now = time.monotonic()
                if case == 'stale': self.d.arm_sample = (self.side, now-.1, ('stable',))
                if case == 'signature': self.d.arm_sample = (self.side, now+10., ('changed',))
                if case == 'safety': self.d.safety.setChecked(False)
                if case == 'disconnect': self.d.connected = False
                if case == 'stall': self.a.last_tick = now-1.
                if case == 'worker': self.d.worker.isRunning.return_value = False
                self.a.tick(now)
                self.assertFalse(self.a.pending); self.assertFalse(self.a.authorized)
                self.d.worker.begin_control.assert_not_called()

    def test_unstable_lock_config_change_and_pause_do_not_retry(self):
        self.arm(); self.d._capture = Mock(); self.d._start = Mock()
        self.expire()
        self.assertFalse(self.a.pending); self.d._start.assert_not_called()
        self.arm(); self.a.widgets['tolerance'].setValue(3.)
        self.assertFalse(self.a.pending)
        self.arm(); self.d.busy = True; self.a.phase = 'transition'
        self.a.progress(dict(mode='braking', reason='hard timeout'))
        self.d.worker.pause.assert_called_once()
        self.assertFalse(self.a.authorized)
        self.a.progress(dict(mode='active', phase='await_confirmation'))
        self.assertFalse(self.a.pending)

    def test_gap_then_fresh_active_continues_without_cancel_or_follow(self):
        for mode in (2, 3):
            with self.subTest(mode=mode):
                self.d.mode_select.setCurrentIndex(mode)
                self.d.safety.setChecked(True); self.fresh(); self.arm()
                self.d.busy = True; self.a.phase = 'transition'
                self.a.progress(dict(mode='gap', phase='transition'))
                self.assertTrue(self.a.pending); self.assertTrue(self.a.authorized)
                self.d.worker.pause.assert_not_called()
                self.d._confirm_follow = Mock()
                self.a.tick()
                self.d._confirm_follow.assert_not_called()
                # A queued gap notification can arrive after writer recovery.
                self.a.progress(dict(mode='active', phase='transition'))
                self.assertIsNone(self.a.gap_until)
                self.assertEqual(self.a.phase, 'transition')
                self.a.cancel('test'); self.d.busy = False

    def test_gap_restarts_settle_wait_and_does_not_extend_grace(self):
        self.arm(); self.d.busy = True; self.a.phase = 'settling'
        self.a.wait_until = time.monotonic()-1
        self.d._confirm_follow = Mock()
        self.a.progress(dict(mode='gap', phase='await_confirmation'))
        deadline = self.a.gap_until
        self.a.progress(dict(mode='gap', phase='await_confirmation'))
        self.assertEqual(self.a.gap_until, deadline)
        self.a.tick(); self.d._confirm_follow.assert_not_called()
        self.a.progress(dict(mode='active', phase='await_confirmation'))
        self.assertGreater(self.a.wait_until, time.monotonic())
        self.a.tick(); self.d._confirm_follow.assert_not_called()

    def test_motion_stale_input_waits_only_within_existing_grace(self):
        self.arm(); self.d.busy = True; self.a.phase = 'transition'
        now = time.monotonic()
        self.a.input_deadline = now-.1; self.d.arm_sample = None
        self.a.tick(now)
        self.assertTrue(self.a.pending)
        self.d.worker.pause.assert_not_called()
        self.a.tick(now+.16)
        self.assertFalse(self.a.pending); self.assertFalse(self.a.authorized)
        self.d.worker.pause.assert_called_once()
        self.a.progress(dict(mode='active', phase='following'))
        self.assertFalse(self.a.pending)

    def test_gap_hard_timeout_and_paused_do_not_resume(self):
        for mode in ('braking', 'paused', 'expired'):
            with self.subTest(mode=mode):
                self.d.busy = False; self.fresh(); self.arm()
                self.d.busy = True; self.a.phase = 'transition'
                self.a.progress(dict(mode='gap', phase='transition'))
                if mode == 'expired':
                    self.a.gap_until = time.monotonic()-.01
                    self.a.tick()
                else:
                    self.a.progress(dict(mode=mode, reason='hard timeout or user pause'))
                self.assertFalse(self.a.pending); self.assertFalse(self.a.authorized)
                self.a.progress(dict(mode='active', phase='following'))
                self.assertFalse(self.a.pending)

    def test_discontinuous_history_rejected_at_deadline(self):
        self.arm(); self.d._start = Mock()
        now = time.monotonic(); self.a.deadline = now; self.a.last_tick = now-.01
        self.d.arm_history = deque((now-t, self.side) for t in (.7, .65, .6, .1, 0.))
        self.a.tick(now)
        self.assertFalse(self.a.pending); self.d._start.assert_not_called()

    def test_bilateral_confirmation_and_recheck_after_modal(self):
        self.d.mode_select.setCurrentIndex(3)
        self.d.safety.setChecked(True)
        self.d.arm_sample = ({'left': deepcopy(self.side), 'right': deepcopy(self.side)},
                             time.monotonic()+10., ('stable',))
        self.arm()
        self.a.cancel('test cancel')
        def expire_input(*args):
            self.d.arm_sample = (self.d.arm_sample[0], time.monotonic()-.1, ('stable',))
            return QMessageBox.Yes
        with patch.object(QMessageBox, 'warning', side_effect=expire_input): self.a.arm()
        self.assertFalse(self.a.pending); self.assertFalse(self.a.authorized)
        self.d.worker.begin_control.assert_not_called()

    def test_poll_reports_segment_and_disables_follow_on_pose_drift(self):
        self.d.arm_capture = deepcopy(self.d.arm_sample)
        self.d.busy = True; self.d.ik_worker = Mock()
        self.d.shared_phase = 'await_confirmation'; self.d.shared_control_mode = 'active'
        self.d._controls(); self.assertTrue(self.d.follow_button.isEnabled())
        self.side['hand'] = self.side['hand'] @ Rotation.from_euler('z', 4., degrees=True).as_matrix()
        now = time.monotonic()
        with patch.object(self.d, '_read_arm_sample', return_value=(
                dict(at=now, device_ages_ms={'u': 1.}), ('stable',), self.side)):
            RealRobotModesDialog._poll(self.d)
        self.assertIn('手掌变化 4.00°', self.d.start_notice.text())
        self.assertIn('不能开始跟随', self.d.start_notice.text())
        self.assertFalse(self.d.follow_button.isEnabled())
        self.d.worker.pause.assert_called_once()

    def test_segment_diagnostics_and_configurable_threshold(self):
        self.d.mode_select.setCurrentIndex(3)
        original = {'left': deepcopy(self.side), 'right': deepcopy(self.side)}
        moved = deepcopy(original)
        moved['right']['rotations']['forearm'] = moved['right']['rotations']['forearm'] @ Rotation.from_euler('x', 4, degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError, '右臂 前臂变化 4.00° > 容差 2°'):
            self.d._pose_guard(original, moved, '过渡期间人体超差')
        self.a.widgets['tolerance'].setValue(5.)
        self.d._pose_guard(original, moved, 'test')
        self.d.arm_capture = (original, 0., ('stable',))
        self.d.arm_sample = (moved, time.monotonic()+.2, ('stable',))
        data = snapshot(self.d)
        self.assertEqual(data['orientation_threshold_deg'], 5.)
        self.assertIn('阈值 5°', delta_text(data))


class VirtualAutoStartTests(unittest.TestCase):
    def test_countdown_to_follow_over_virtual_uart(self):
        viewer = QWidget(); viewer.engine = None; viewer.replay_mode = False
        side = rest_sample(PoseEngine().transforms(include_natural=False))
        def sample(*args):
            return dict(at=time.monotonic(), sides={'left': deepcopy(side)},
                        device_ages_ms={'u': 1., 'f': 1., 'h': 1.}), ('stable',)
        master, slave = pty.openpty(); d = None
        with tempfile.TemporaryDirectory() as directory, patch(
                'optusbproj_ubuntu.real_robot_arm_gui.pose_sample', side_effect=sample), patch.object(
                    QMessageBox, 'warning', return_value=QMessageBox.Yes) as confirm:
            try:
                d = RealRobotModesDialog(viewer); d.mode_select.setCurrentIndex(2)
                d.shared_palm.setChecked(False)
                for row in d.axis_rows: row[0].setChecked(False)
                d.axis_rows[3][0].setChecked(True)
                d.auto_start.widgets['countdown'].setValue(1.)
                d.auto_start.widgets['settle'].setValue(.1)
                worker = ElbowSerialWorker(os.ttyname(slave), Recorder(Path(directory)/'control.jsonl'), d)
                d.worker = worker
                for signal, slot in ((worker.ready, d._ready), (worker.progress, d._progress),
                                     (worker.sent, d._sent), (worker.failed, d._failed),
                                     (worker.finished, d._finished)): signal.connect(slot)
                writes = []; worker.sent.connect(writes.append)
                worker.start(); wait_for(lambda: d.connected)
                d.origin = d.last_frame = BASE; d.safety.setChecked(True)
                wait_for(lambda: d.arm_sample is not None)
                d.auto_start.arm()
                self.assertEqual(writes, [])
                wait_for(lambda: d.shared_phase == 'following', timeout=8.)
                self.assertFalse(d.auto_start.authorized)
                self.assertEqual(confirm.call_count, 1)
                self.assertEqual(writes[0], BASE)
                d._pause('virtual auto acceptance')
                wait_for(lambda: not d.busy and d.ik_worker is None)
                d._disconnect(); wait_for(lambda: d.worker is None)
                events = [json.loads(s)['event'] for s in (Path(directory)/'control.jsonl').read_text().splitlines()]
                for name in ('auto_start_authorized', 'arm_reference', 'arm_start_accepted',
                             'auto_start_settling', 'auto_start_completed'):
                    self.assertEqual(events.count(name), 1)
            finally:
                if d is not None:
                    d._disconnect(); wait_for(lambda: d.worker is None and d.ik_worker is None)
                    d.close()
                viewer.close(); os.close(master); os.close(slave)
