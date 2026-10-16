"""Pure models, fake simulator and /dev/pts only; never real hardware."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import pty
import tempfile
import json
from pathlib import Path
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from optusbproj_ubuntu.real_robot_shared import SharedArmControl, SharedArmIK
from optusbproj_ubuntu.real_robot_arm import ArmLimits, load_profile
from optusbproj_ubuntu.real_robot_protocol import encode, decode
from optusbproj_ubuntu.real_robot_mirror import mirror_targets, CommandMirror
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PART_BY_KEY
from optusbproj_ubuntu.scene_robot_model import SceneArm
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.scene_robot_model import palm_target

APP = QApplication.instance() or QApplication([])
BASE = encode([90.]*15, [0]*10)


def wait_for(predicate, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        APP.processEvents()
        if predicate():
            return
        time.sleep(.003)
    raise AssertionError('virtual chain timed out')


def rest_sample(rest):
    points = tuple(rest[PART_BY_KEY['left_'+part]][:3,3]
                   for part in ('upper_arm','forearm','hand'))
    s,e,w = points
    return dict(points=points, directions=[(e-s)/np.linalg.norm(e-s), (w-e)/np.linalg.norm(w-e)],
                rotations={p:rest[PART_BY_KEY['left_'+p]][:3,:3]
                           for p in ('upper_arm','forearm')},
                hand=rest[PART_BY_KEY['left_hand']][:3,:3])


class SharedTests(unittest.TestCase):
    def test_offset_controls_copy_disabled_z_and_invalidate_capture(self):
        viewer = QWidget()
        viewer.engine = None
        viewer.replay_mode = False
        offset = dict(enabled=False, axis='z', degrees=90.)
        viewer.scene_robot_dialog = SimpleNamespace(palm_offsets=lambda:{'left':dict(offset)})
        dialog = RealRobotModesDialog(viewer)
        dialog.timer.stop()
        try:
            dialog.mode_select.setCurrentIndex(2)
            self.assertEqual(dialog._palm_offset(), offset)
            offset.update(enabled=True, axis='y', degrees=-45.)
            dialog.arm_capture = ({},0.,('test',))
            dialog._copy_scene_offset()
            self.assertEqual(dialog._palm_offset(), offset)
            self.assertIsNone(dialog.arm_capture)
            for widget,change in (
                    (dialog.shared_offset_axis,lambda:dialog.shared_offset_axis.setCurrentText('z')),
                    (dialog.shared_offset_angle,lambda:dialog.shared_offset_angle.setValue(90.)),
                    (dialog.shared_offset,lambda:dialog.shared_offset.setChecked(False))):
                dialog.arm_capture = ({},0.,('test',))
                change()
                self.assertIsNone(dialog.arm_capture)
            dialog.busy = True
            dialog._controls()
            self.assertFalse(dialog.shared_offset_axis.isEnabled())
            self.assertFalse(dialog.shared_offset_angle.isEnabled())
            self.assertFalse(dialog.shared_offset_copy.isEnabled())
            before = dialog._palm_offset()
            offset.update(axis='x',degrees=10.)
            dialog._copy_scene_offset()
            self.assertEqual(dialog._palm_offset(),before)
        finally:
            dialog.busy = False
            dialog.close();viewer.close()

    def test_offset_defaults_and_z_reaches_shared_solver(self):
        viewer = QWidget()
        viewer.engine = None
        viewer.replay_mode = False
        dialog = RealRobotModesDialog(viewer)
        dialog.timer.stop()
        try:
            self.assertEqual(dialog._palm_offset(),dict(enabled=False,axis='z',degrees=90.))
            dialog.shared_offset.setChecked(True)
            rest = PoseEngine().transforms(include_natural=False)
            side = rest_sample(rest)
            solver = SharedArmIK(BASE, ArmLimits(enabled=(True,)*6), human_rest=rest,
                                 offset=dialog._palm_offset())
            captured = []
            original = solver.solver.solve
            def solve(*args,**kwargs):
                result = original(*args,**kwargs)
                captured.append(result.palm_targets['left'])
                return result
            with patch.object(solver.solver,'solve',side_effect=solve) as spy:
                solver.solve(side)
            self.assertEqual(spy.call_args.args[0]['palm_offsets']['left'],
                             dict(enabled=True,axis='z',degrees=90.))
            C = solver.solver.axes
            robot_rest = solver.model.tool_frame(solver.model.neutral)[:3,:3]
            expected = palm_target(C,side['hand'],side['hand'],robot_rest,
                                   dict(enabled=True,axis='z',degrees=90.))
            np.testing.assert_allclose(captured[0],expected)
            wrong_x = palm_target(C,side['hand'],side['hand'],robot_rest,
                                  dict(enabled=True,axis='x',degrees=90.))
            self.assertFalse(np.allclose(captured[0],wrong_x))
        finally:
            dialog.close();viewer.close()

    def test_defaults_and_transition_freezes_until_confirmation(self):
        limits = ArmLimits(enabled=(True,)*6)
        self.assertEqual(limits.upper, (226.,233.,180.,119.,180.,122.))
        self.assertEqual(limits.lower, (0.,90.,0.,0.,0.,51.))
        self.assertEqual(limits.speed, (100.,)*6)
        self.assertEqual(limits.acceleration, (150.,)*6)
        c = SharedArmControl(BASE, limits, [100.,95.,80.,0.,90.,80.])
        self.assertEqual(c.step(), BASE)
        with self.assertRaises(ValueError):
            c.request_follow()
        for _ in range(180):
            c.set_target([90.]*6)  # Refreshes must NOT chase new targets yet.
            c.step()
            self.assertTrue(np.all(np.abs(c.v) <= 100+1e-7))
            self.assertTrue(np.all(np.abs(c.a) <= 150+1e-7))
        self.assertEqual(c.phase, 'await_confirmation')
        np.testing.assert_allclose(c.q, c.initial_target)
        c.request_follow()
        c.step()
        self.assertEqual(c.phase, 'following')
        c.set_target([90.]*6)
        before = c.q.copy()
        c.step()
        self.assertLess(np.max(np.abs(c.q-before)), 100/limits.frequency)
        c.brake('test')
        with self.assertRaises(ValueError):
            c.request_follow()

    def test_mask_and_power_on_only(self):
        mask = (False,False,False,True,False,False)
        c = SharedArmControl(BASE, ArmLimits(enabled=mask), [0.,200.,180.,119.,0.,122.])
        for _ in range(130):
            angles,fingers = decode(c.step())
            self.assertEqual(angles[:9]+angles[10:], [90.]*14)
            self.assertEqual(fingers, [0]*10)
        self.assertAlmostEqual(c.q[3],119.)
        altered = [90.]*15
        altered[9] = 60.
        with self.assertRaises(ValueError):
            SharedArmControl(encode(altered,[0]*10), ArmLimits(enabled=mask), [90.]*6)

    def test_shared_rest_targets_are_hanging_not_power_on(self):
        rest = PoseEngine().transforms(include_natural=False)
        solver = SharedArmIK(BASE, ArmLimits(enabled=(True,)*6), human_rest=rest,
                             palm=False)
        result = solver.solve(rest_sample(rest))
        self.assertLess(result['target'][3], 1.)
        self.assertLess(result['position_error_mm'], 2.)
        self.assertEqual(solver.model.neutral[3], -np.pi/2)
        expected = solver.solver.solve(
            dict(sides={'left':{**rest_sample(rest), 'hand':None}}, task='positions',
                 scale='uniform', solver_mode='staged', segment_orientation=True,
                 segment_weight=.15), {'left':solver.model.neutral.copy()}, 100.,100.,1.,
            step_override=np.full(6,2*np.pi))
        np.testing.assert_allclose(result['target'],solver.limits.to_command(expected.targets['left']))

    def test_mirror_uses_quantized_written_frame_and_signs(self):
        angles = [90.]*15
        angles[6:12] = [100.,110.,80.,0.,100.,90.]
        targets = mirror_targets(encode(angles,[0]*10), (1,-1,1,1,1,1))
        np.testing.assert_allclose(np.rad2deg(targets['left']), [10.,-20.,-10.,-90.,10.,0.])
        np.testing.assert_allclose(targets['right'], np.zeros(6))

    def test_fake_mirror_lifecycle_without_initial_write(self):
        arm,_,_ = load_profile()
        bridge = SimpleNamespace(data={'arms':{'left':arm}}, exchange=Mock(), close=Mock())
        worker = CommandMirror(bridge_factory=lambda port:bridge)
        ready, errors = [],[]
        worker.ready.connect(lambda:ready.append(True))
        worker.failed.connect(errors.append)
        worker.start()
        try:
            end = time.monotonic()+2
            while not ready and time.monotonic()<end:
                APP.processEvents();time.sleep(.002)
            self.assertTrue(ready)
            bridge.exchange.assert_not_called()
            worker.submit(BASE,(1,)*6)
            end = time.monotonic()+1
            while not bridge.exchange.called and time.monotonic()<end:
                APP.processEvents();time.sleep(.002)
            self.assertTrue(bridge.exchange.called)
            self.assertFalse(errors)
        finally:
            worker.shutdown()
            self.assertTrue(worker.wait(2500))
            bridge.close.assert_called_once()

    def test_gui_defaults_and_changed_pose_refuses_follow(self):
        viewer = QWidget()
        viewer.engine = None
        viewer.replay_mode = False
        dialog = RealRobotModesDialog(viewer)
        dialog.timer.stop()
        try:
            dialog.mode_select.setCurrentIndex(2)
            self.assertEqual(dialog._limits().enabled, (True,)*6)
            self.assertEqual(dialog._limits().upper, ArmLimits().upper)
            rest = PoseEngine().transforms(include_natural=False)
            side = rest_sample(rest)
            signature = ('test',)
            dialog.worker = SimpleNamespace(confirm_follow=Mock(), pause=Mock(), recorder=Mock())
            dialog.busy = True
            dialog.shared_phase = 'await_confirmation'
            dialog.arm_capture = (deepcopy(side), time.monotonic()+.25, signature)
            dialog.arm_sample = (deepcopy(side), time.monotonic()+.25, signature)
            with patch.object(dialog,'_poll'), patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):
                dialog._confirm_follow()
            dialog.worker.confirm_follow.assert_called_once()
            dialog.worker.confirm_follow.reset_mock()
            dialog.arm_sample = (side, time.monotonic()-.01, signature)
            dialog._confirm_follow()
            dialog.worker.confirm_follow.assert_not_called()
            dialog.worker.pause.assert_called_once()
        finally:
            dialog.busy = False
            dialog.worker = None
            dialog.close();viewer.close()

    def test_virtual_uart_full_transition_confirmation_and_pause(self):
        viewer = QWidget()
        viewer.engine = None
        viewer.replay_mode = False
        rest = PoseEngine().transforms(include_natural=False)
        side = rest_sample(rest)
        def sample(*args):
            return dict(at=time.monotonic(), sides={'left':deepcopy(side)},
                        device_ages_ms={'u':1.,'f':1.,'h':1.}), ('stable',)
        master,slave = pty.openpty()
        dialog = None
        with tempfile.TemporaryDirectory() as directory, patch(
                'optusbproj_ubuntu.real_robot_arm_gui.pose_sample',side_effect=sample), patch.object(
                    QMessageBox,'warning',return_value=QMessageBox.Yes):
            try:
                dialog = RealRobotModesDialog(viewer)
                dialog.mode_select.setCurrentIndex(2)
                dialog.shared_palm.setChecked(False)
                dialog.shared_offset.setChecked(False)
                for row in dialog.axis_rows:
                    row[0].setChecked(False)
                dialog.axis_rows[3][0].setChecked(True)
                worker = ElbowSerialWorker(os.ttyname(slave), Recorder(Path(directory)/'control.jsonl'), dialog)
                dialog.worker = worker
                worker.ready.connect(dialog._ready)
                worker.progress.connect(dialog._progress)
                worker.sent.connect(dialog._sent)
                worker.failed.connect(dialog._failed)
                worker.finished.connect(dialog._finished)
                writes = []
                worker.sent.connect(writes.append)
                worker.start()
                wait_for(lambda:dialog.connected)
                dialog.origin = dialog.last_frame = BASE
                dialog.safety.setChecked(True)
                wait_for(lambda:len(dialog.arm_history)>28)
                dialog._capture()
                self.assertIsNotNone(dialog.arm_capture)
                dialog._start()
                self.assertTrue(dialog.busy)
                wait_for(lambda:dialog.shared_phase=='await_confirmation')
                self.assertEqual(writes[0], BASE)
                self.assertLess(decode(writes[-1])[0][9],1.)
                self.assertTrue(dialog.follow_button.isEnabled())
                for frame in writes:
                    angles,fingers = decode(frame)
                    self.assertEqual(angles[:9]+angles[10:], [90.]*14)
                    self.assertEqual(fingers,[0]*10)
                dialog._confirm_follow()
                wait_for(lambda:dialog.shared_phase=='following')
                dialog._pause('virtual shared acceptance')
                wait_for(lambda:not dialog.busy and dialog.ik_worker is None)
                dialog._disconnect()
                wait_for(lambda:dialog.worker is None)
                rows = [json.loads(line) for line in (Path(directory)/'control.jsonl').read_text().splitlines()]
                self.assertTrue(any(row['event']=='shared_transition' for row in rows))
                self.assertTrue(any(row['event']=='shared_follow_confirmed' for row in rows))
                self.assertTrue(any(row['event']=='write' and row.get('phase')=='following' for row in rows))
            finally:
                if dialog is not None:
                    dialog._disconnect()
                    wait_for(lambda:dialog.worker is None and dialog.ik_worker is None)
                    dialog.close()
                viewer.close()
                os.close(master);os.close(slave)
