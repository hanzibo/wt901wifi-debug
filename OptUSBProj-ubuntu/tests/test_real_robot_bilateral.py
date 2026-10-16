"""Bilateral acceptance uses exported geometry, fake RPC and virtual UART only."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import pty
import tempfile
import time
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from optusbproj_ubuntu.real_robot_arm import ArmLimits, load_profile
from optusbproj_ubuntu.real_robot_bilateral import BilateralLimits, BilateralIK, right_defaults
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_protocol import encode, decode
from optusbproj_ubuntu.real_robot_mirror import mirror_targets, CommandMirror
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog, pose_distance
from optusbproj_ubuntu.real_robot_start_diagnostics import reference_deltas
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PART_BY_KEY

APP = QApplication.instance() or QApplication([])
BASE = encode([90.]*15,[0]*10)
MASK = (False,False,False,True,False,False)


def limits(mask=MASK):
    return BilateralLimits(replace(right_defaults(), enabled=mask), ArmLimits(enabled=mask))


def human():
    rest = PoseEngine().transforms(include_natural=False)
    result = {}
    for side in ('left','right'):
        points = tuple(rest[PART_BY_KEY[f'{side}_{p}']][:3,3] for p in ('upper_arm','forearm','hand'))
        s,e,w = points
        result[side] = dict(points=points, directions=[(e-s)/np.linalg.norm(e-s),(w-e)/np.linalg.norm(w-e)],
                            rotations={p:rest[PART_BY_KEY[f'{side}_{p}']][:3,:3] for p in ('upper_arm','forearm')},
                            hand=rest[PART_BY_KEY[f'{side}_hand']][:3,:3])
    return result


def wait_for(predicate, timeout=7):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        APP.processEvents()
        if predicate(): return
        time.sleep(.003)
    raise AssertionError('virtual bilateral chain timed out')


class BilateralTests(unittest.TestCase):
    def test_actual_right_profile_and_task_reuse(self):
        left,C,_ = load_profile()
        right,D,source = load_profile('right')
        np.testing.assert_allclose(C,D)
        self.assertIn('20261011-213953',source)
        self.assertFalse(np.allclose(left['joints'],right['joints']))
        config = limits((True,)*6)
        solver = BilateralIK(BASE,config,palms={'left':False,'right':False})
        result = solver.solve(human())
        self.assertEqual(len(result['target']),12)
        self.assertEqual(set(result['errors']),{'left','right'})
        self.assertLess(result['target'][3],1.)
        self.assertLess(result['target'][9],1.)
        self.assertLess(result['position_error_mm'],2.)
        for side in ('right','left'):
            self.assertEqual(set(solver.solvers[side].solver.models),{side})

    def test_one_packet_order_permissions_and_shared_transition(self):
        c = SharedArmControl(BASE,limits(),[0.,240.,180.,60.,180.,121.,226.,233.,0.,30.,180.,122.])
        self.assertEqual(c.step(),BASE)
        for _ in range(160):
            c.set_target([90.]*12)
            frame = c.step()
            q,f = decode(frame)
            self.assertEqual([x for i,x in enumerate(q) if i not in (3,9)],[90.]*13)
            self.assertEqual(f,[0]*10)
            self.assertLessEqual(max(abs(c.v)),100.+1e-6)
            self.assertLessEqual(max(abs(c.a)),150.+1e-6)
        self.assertEqual(c.phase,'await_confirmation')
        self.assertEqual(q[3],60.)
        self.assertEqual(q[9],30.)
        c.request_follow();c.step();c.set_target([90.]*12)
        for _ in range(6): c.step()
        c.brake('one participating IMU stale')
        for _ in range(100): c.step()
        self.assertEqual(c.mode,'paused')
        self.assertEqual(c.status()['names'][0],'R1')
        self.assertEqual(c.status()['names'][6],'L1')

    def test_right_only_does_not_need_left_imu(self):
        config = BilateralLimits(replace(right_defaults(),enabled=MASK),ArmLimits())
        solver = BilateralIK(BASE,config,palms={'right':False})
        result = solver.solve({'right':human()['right']})
        self.assertEqual(result['target'][6:],[90.]*6)
        self.assertEqual(config.sides,('right',))
        with self.assertRaises(ValueError): solver.solve(human())
        with self.assertRaises(ValueError):
            BilateralLimits(replace(right_defaults(),frequency=10),ArmLimits())

    def test_both_direction_inverses_and_diagnostics(self):
        q = [90.]*15;q[5]=100.;q[11]=80.
        targets = mirror_targets(encode(q,[0]*10),{'left':(1,1,1,1,1,-1),'right':(1,1,1,1,1,-1)})
        self.assertAlmostEqual(np.rad2deg(targets['right'][5]),-10.)
        self.assertAlmostEqual(np.rad2deg(targets['left'][5]),10.)
        h = human()
        self.assertEqual(pose_distance(h,deepcopy(h)),0.)
        delta = reference_deltas((h,), (deepcopy(h),))
        self.assertEqual(len(delta),6)
        self.assertIn('right.hand',delta)

    def test_mirror_validates_right_before_first_write(self):
        arms = {s:load_profile(s)[0] for s in ('left','right')}
        arms['right'] = deepcopy(arms['right'])
        arms['right']['joints'][0][3] += .02
        bridge = SimpleNamespace(data={'arms':arms},exchange=Mock(),close=Mock())
        mirror = CommandMirror(bridge_factory=lambda _:bridge,sides=('left','right'))
        errors=[];mirror.failed.connect(errors.append)
        mirror.start()
        wait_for(lambda:not mirror.isRunning())
        APP.processEvents()
        self.assertTrue(errors)
        bridge.exchange.assert_not_called()
        bridge.close.assert_called_once()

    def test_gui_settings_and_selected_input(self):
        viewer = QWidget();viewer.engine=None;viewer.replay_mode=False
        dialog = RealRobotModesDialog(viewer);dialog.timer.stop()
        try:
            dialog.axis_rows[5][5].setCurrentIndex(1)
            dialog.mode_select.setCurrentIndex(3)
            self.assertEqual(dialog._limits().left.signs[-1],-1)
            self.assertFalse(any(dialog._limits().right.enabled))
            for row in dialog.axis_rows: row[0].setChecked(False)
            dialog.right_axis_rows[3][0].setChecked(True)
            self.assertEqual(dialog._selected_sides(),('right',))
            with patch('optusbproj_ubuntu.real_robot_arm_gui.pose_sample',return_value=(dict(sides={'right':human()['right']}),('sig',))) as read:
                dialog._read_arm_sample()
            self.assertEqual(read.call_args.args[1:3],(('right',),('right',)))
            viewer.scene_robot_dialog=SimpleNamespace(palm_offsets=lambda:{'right':dict(enabled=True,axis='z',degrees=-90.)})
            dialog._copy_right_offset()
            self.assertEqual(dialog._right_offset(),dict(enabled=True,axis='z',degrees=-90.))
        finally: dialog.close();viewer.close()

    def test_virtual_uart_bilateral_follow_then_one_side_loss(self):
        viewer = QWidget();viewer.engine=None;viewer.replay_mode=False
        current = human();failed=[False]
        def sample(*args):
            if failed[0]: raise ValueError('right_forearm 数据不新鲜')
            return dict(at=time.monotonic(),sides=deepcopy(current),device_ages_ms={f'd{i}':1. for i in range(6)}),('stable',)
        master,slave=pty.openpty();dialog=None
        with tempfile.TemporaryDirectory() as directory, patch('optusbproj_ubuntu.real_robot_arm_gui.pose_sample',side_effect=sample), patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):
            try:
                dialog=RealRobotModesDialog(viewer)
                dialog.mode_select.setCurrentIndex(3)
                dialog.shared_palm.setChecked(False);dialog.right_options[0].setChecked(False)
                for rows in (dialog.axis_rows,dialog.right_axis_rows):
                    for row in rows: row[0].setChecked(False)
                    rows[3][0].setChecked(True)
                worker=ElbowSerialWorker(os.ttyname(slave),Recorder(Path(directory)/'control.jsonl'),dialog)
                dialog.worker=worker
                worker.ready.connect(dialog._ready);worker.progress.connect(dialog._progress)
                worker.sent.connect(dialog._sent);worker.failed.connect(dialog._failed);worker.finished.connect(dialog._finished)
                writes=[];worker.sent.connect(writes.append)
                worker.start();wait_for(lambda:dialog.connected)
                dialog.origin=dialog.last_frame=BASE;dialog.safety.setChecked(True)
                wait_for(lambda:len(dialog.arm_history)>28)
                dialog._capture();self.assertIsNotNone(dialog.arm_capture)
                dialog._start();self.assertTrue(dialog.busy)
                wait_for(lambda:dialog.shared_phase=='await_confirmation')
                self.assertEqual(writes[0],BASE)
                self.assertLess(decode(writes[-1])[0][3],1.)
                self.assertLess(decode(writes[-1])[0][9],1.)
                dialog._confirm_follow();wait_for(lambda:dialog.shared_phase=='following')
                failed[0]=True
                wait_for(lambda:not dialog.busy and dialog.ik_worker is None)
                self.assertFalse(dialog.safety.isChecked())
                for frame in writes:
                    q,f=decode(frame)
                    self.assertEqual([x for i,x in enumerate(q) if i not in (3,9)],[90.]*13)
                    self.assertEqual(f,[0]*10)
                dialog._disconnect();wait_for(lambda:dialog.worker is None)
                logs=[json.loads(l) for l in (Path(directory)/'control.jsonl').read_text().splitlines()]
                self.assertTrue(any(l['event']=='arm_ik' and 'right' in l['errors'] for l in logs))
                self.assertTrue(any(l['event']=='paused' and 'right_forearm' in l['reason'] for l in logs))
            finally:
                if dialog is not None:
                    dialog._disconnect();wait_for(lambda:dialog.worker is None and dialog.ik_worker is None)
                    dialog.close()
                viewer.close();os.close(master);os.close(slave)
