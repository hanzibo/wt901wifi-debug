"""Bounded input loss tests with synthetic clocks; never open hardware ports."""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_arm import ArmControl, ArmLimits
from optusbproj_ubuntu.real_robot_control import ElbowControl, Limits
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_bilateral import BilateralLimits, right_defaults
from optusbproj_ubuntu.real_robot_protocol import encode, decode


BASE = encode([90.]*15, [0]*10)


class Events:
    def __init__(self): self.rows = []
    def record(self, event, **data): self.rows.append((event, data))


class GapTests(unittest.TestCase):
    def test_gui_stale_input_waits_but_other_faults_pause(self):
        from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
        from optusbproj_ubuntu.real_robot_gui import RealRobotDialog
        from optusbproj_ubuntu.scene_robot_state import StaleImuInput
        for exc in (StaleImuInput('left_forearm', 'test', .3),
                    StaleImuInput('left_forearm', 'test', 2.),
                    StaleImuInput('left_forearm', 'test', -.2), ValueError('映射变化')):
            for arm in (True, False):
                with self.subTest(exc=str(exc), arm=arm):
                    d = Mock(end_session=None, busy=True, arm_input_error=None)
                    d._read_arm_sample.side_effect = exc
                    if arm:
                        with patch('optusbproj_ubuntu.real_robot_arm_gui.observe_input_timing', return_value={}), \
                             patch('optusbproj_ubuntu.real_robot_arm_gui.start_snapshot', return_value={}):
                            RealRobotModesDialog._poll(d)
                    else:
                        with patch('optusbproj_ubuntu.real_robot_gui.pose_sample', side_effect=exc):
                            RealRobotDialog._poll(d)
                    self.assertEqual(d._pause.called,
                                     not (isinstance(exc, StaleImuInput) and exc.age > .25))

    def test_queued_ik_result_does_not_bypass_gap_on_stale_current_input(self):
        from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
        from optusbproj_ubuntu.scene_robot_state import StaleImuInput
        d = Mock(busy=True, arm_capture=(None, None, 'signature'))
        d.sender.return_value = d.ik_worker
        d._read_arm_sample.side_effect = StaleImuInput('left_hand', 'test', .3)
        RealRobotModesDialog._ik_result(d, {'deadline':time.monotonic()+1})
        d._pause.assert_not_called()
        d.worker.update_arm_target.assert_not_called()

    def setup_control(self, kind='arm'):
        limits = ArmLimits(enabled=(True,)*6)
        if kind == 'shared':
            c = SharedArmControl(BASE, limits, [95.]*6)
        elif kind == 'bilateral':
            c = ArmControl(BASE, BilateralLimits(
                ArmLimits(**{**vars(right_defaults()), 'enabled': (True,)*6}), limits))
        elif kind == 'elbow':
            c = ElbowControl(BASE, 90., Limits())
        else:
            c = ArmControl(BASE, limits)
        target = 70. if kind == 'elbow' else [95.]*c.size
        w = ElbowSerialWorker('NEVER_OPEN', Events())
        w._reserve(c, target, 10.25)
        w._input_guard(c, 10., 10.25, None)
        for _ in range(4): c.step()
        return w, c, target

    def test_transient_decelerates_and_fresh_target_recovers_all_modes(self):
        for kind in ('arm', 'elbow', 'shared', 'bilateral'):
            with self.subTest(kind=kind):
                w, c, target = self.setup_control(kind)
                speed = np.abs(c.v).copy()
                w._input_guard(c, 10.26, 10.25, None)
                self.assertEqual(c.mode, 'gap')
                for _ in range(4): c.step()
                self.assertTrue(np.all(np.abs(c.v) <= speed+1e-7))
                self.assertEqual(c.mode, 'gap')  # even fully stopped: waiting, not latched
                before = np.array(c.q).copy()
                w.latest = (target, 10.65)
                w._input_guard(c, 10.4, 10.65, None)
                self.assertEqual(c.mode, 'active')
                np.testing.assert_allclose(c.q, before)  # no position reset / jump
                frame = c.step()
                self.assertEqual(decode(frame)[1], [0]*10)
                self.assertEqual([e for e,_ in w.recorder.rows],
                                 ['input_gap_started', 'input_gap_recovered'])

    def test_timeout_latches_even_if_fresh_target_arrives_after_window(self):
        for fresh in (False, True):
            w, c, target = self.setup_control()
            w._input_guard(c, 10.26, 10.25, None)
            if fresh: w.latest = (target, 10.8)
            w._input_guard(c, 10.51, w.latest[1], None)
            self.assertEqual(c.mode, 'braking')
            w.latest = (target, 11.)
            w._input_guard(c, 10.52, 11., None)
            self.assertEqual(c.mode, 'braking')
            self.assertEqual(sum(e == 'input_timeout' for e,_ in w.recorder.rows), 1)

    def test_expired_updates_cannot_extend_window(self):
        w, c, target = self.setup_control()
        w._input_guard(c, 10.26, 10.25, None)
        w.latest = (target, 10.45)
        w._input_guard(c, 10.49, 10.45, None)
        self.assertEqual(w.gap_deadline, 10.5)
        w._input_guard(c, 10.51, 10.45, None)
        self.assertEqual(c.mode, 'braking')

    def test_stalled_writer_no_extra_window(self):
        w, c, _ = self.setup_control()
        w._input_guard(c, 10.7, 10.25, None)
        self.assertEqual(c.mode, 'braking')

    def test_explicit_pause_never_auto_resumes(self):
        w, c, target = self.setup_control()
        w._input_guard(c, 10.26, 10.25, None)
        w._input_guard(c, 10.27, 10.25, '用户暂停')
        self.assertEqual(c.reason, '用户暂停')
        w.latest = (target, 10.7)
        w._input_guard(c, 10.3, 10.7, None)
        self.assertEqual(c.mode, 'braking')

    def test_shared_transition_resumes_original_frozen_target(self):
        w, c, _ = self.setup_control('shared')
        w._input_guard(c, 10.26, 10.25, None)
        for _ in range(10): c.step()
        self.assertEqual(c.phase, 'transition')
        w.latest = ([100.]*6, 10.7)
        w._input_guard(c, 10.3, 10.7, None)
        np.testing.assert_allclose(c.target, c.initial_target)


if __name__ == '__main__': unittest.main()
