"""Finger homing is verified only with pure trajectories and virtual UARTs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
import json
from pathlib import Path
import pty
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QWidget,QMessageBox
from optusbproj_ubuntu.real_robot_finger_return import FingerReturnControl
from optusbproj_ubuntu.real_robot_hand import HandConfig
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from optusbproj_ubuntu.real_robot_bilateral import BilateralLimits,right_defaults
from optusbproj_ubuntu.real_robot_protocol import encode,decode
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.real_robot_return import ReturnConfig
from test_real_robot_shared import APP,wait_for


def config():return HandConfig(enabled=(True,False,False,False,False))
LIMITS=ArmLimits(enabled=(True,)*6,frequency=50.)
BASE=encode([90.]*15,[0]*5+[100,0,0,0,0])


class FingerReturnTests(unittest.TestCase):
    def test_first_frame_fixed_other_channels_speed_acceleration_and_zero(self):
        c=FingerReturnControl(BASE,LIMITS,{'left':config()})
        self.assertEqual(c.step(),BASE)
        previous=0.
        for _ in range(400):
            f=c.step();angles,values=decode(f);m=c.motions['left']
            self.assertEqual(angles,[90.]*15);self.assertEqual(values[:5]+values[6:],[0]*9)
            self.assertTrue(0<=values[5]<=100)
            self.assertLessEqual(abs(m.v[0]),200+1e-6)
            self.assertLessEqual(abs(m.v[0]-previous),400/50+1e-6);previous=m.v[0]
            if c.mode=='paused':break
        self.assertEqual(c.phase,'complete');self.assertEqual(decode(f)[1],[0]*10)
        self.assertEqual(c.status()['return_kind'],'fingers')

    def test_requires_home_angles_permissions_side_and_zero_envelope(self):
        q=[90.]*15;q[6]=91.
        for frame,configs in ((encode(q,[0]*5+[100,0,0,0,0]),{'left':config()}),
                              (BASE,{}),(BASE,{'left':replace(config(),enabled=(False,)*5)}),
                              (BASE,{'left':replace(config(),lower=(1,)*5,opened=(1,)*5)}),
                              (encode([90.]*15,[100]+[0]*9),{'right':config()})):
            with self.assertRaises(ValueError):FingerReturnControl(frame,LIMITS,configs)

    def test_pause_brakes_latches_and_does_not_falsely_complete(self):
        c=FingerReturnControl(BASE,LIMITS,{'left':config()})
        for _ in range(10):c.step()
        c.brake('manual pause')
        for _ in range(100):
            frame=c.step()
            if c.mode=='paused':break
        self.assertEqual(c.mode,'paused');self.assertEqual(c.phase,'aborted')
        self.assertGreater(decode(frame)[1][5],0)
        c.set_target([0.]*6);self.assertEqual(c.step(),frame)

    def test_bilateral_returns_only_nonzero_previous_fingers(self):
        limits=BilateralLimits(replace(right_defaults(),enabled=(True,)*6),ArmLimits(enabled=(True,)*6))
        frame=encode([90.]*15,[80,0,0,0,0,100,0,0,0,0])
        c=FingerReturnControl(frame,limits,{'left':config(),'right':config()})
        self.assertEqual(c.step(),frame)
        for _ in range(300):
            frame=c.step()
            if c.mode=='paused':break
        self.assertEqual(c.phase,'complete');self.assertEqual(decode(frame)[1],[0]*10)

    def test_worker_requires_completed_arm_return_before_any_new_task(self):
        worker=ElbowSerialWorker('not-opened',Mock())
        worker.last_written=BASE;worker.last_state={'mode':'paused','phase':'following'}
        worker.authorized_limits=LIMITS;worker.authorized_hands={'left':config()}
        with self.assertRaises(ValueError):worker.begin_finger_return()
        self.assertIsNone(worker.pending_start)
        worker.last_state={'mode':'paused','phase':'complete'};worker.home_pending=True
        worker.begin_finger_return()
        self.assertTrue(worker.active);self.assertFalse(worker.home_pending)
        self.assertTrue(worker.pending_start[0].controls_fingers)

    def test_virtual_uart_arm_then_finger_return_then_confirm_without_repower(self):
        master,slave=pty.openpty()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'control.jsonl';worker=ElbowSerialWorker(os.ttyname(slave),Recorder(path))
            frames=[];worker.sent.connect(frames.append)
            worker.last_written=BASE;worker.last_state={'mode':'paused','phase':'following'}
            worker.authorized_limits=LIMITS;worker.authorized_hands={'left':config()}
            try:
                worker.start();wait_for(lambda:worker.isRunning())
                worker.begin_return(LIMITS,ReturnConfig(sides=('left',),manual=False,dwell=0.))
                wait_for(lambda:not worker.active)
                self.assertTrue(worker.home_pending)
                with self.assertRaises(ValueError):worker.confirm_home()
                self.assertIn('left',worker.authorized_hands)
                worker.begin_finger_return()
                deadline=time.monotonic()+5.
                while worker.active and time.monotonic()<deadline:
                    worker.heartbeat();APP.processEvents();time.sleep(.01)
                self.assertFalse(worker.active,'手指回零任务未在限定时间完成')
                APP.processEvents();self.assertTrue(worker.home_pending)
                self.assertEqual(worker.confirm_home(),encode([90.]*15,[0]*10))
                worker.shutdown();self.assertTrue(worker.wait(2500));APP.processEvents()
                self.assertEqual(os.read(master,65536),b''.join(frames))
                for frame in frames:self.assertEqual(decode(frame)[0],[90.]*15)
                self.assertEqual(frames[0],BASE);self.assertEqual(decode(frames[-1])[1],[0]*10)
                rows=[json.loads(l) for l in path.read_text().splitlines()]
                self.assertTrue(any(r['event']=='finger_return_requested' for r in rows))
                self.assertTrue(any(r['event']=='home_manually_confirmed' for r in rows))
            finally:worker.shutdown();worker.wait(2500);os.close(master);os.close(slave)

    def test_gui_explicit_button_cancel_and_confirmation_gate(self):
        viewer=QWidget();viewer.engine=None;viewer.replay_mode=False
        d=RealRobotModesDialog(viewer);d.timer.stop();d.hand_session.timer.stop();d.mode_select.setCurrentIndex(2)
        worker=Mock();worker.hand_snapshot.return_value={}
        worker.return_snapshot.return_value=(BASE,{'mode':'paused','phase':'complete'},True)
        d.worker=worker;d.connected=True;d.busy=False;end=d.end_session;end.await_home=True
        try:
            d._controls();self.assertTrue(end.finger_return_button.isEnabled())
            self.assertFalse(end.confirm_button.isEnabled())
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Cancel):end.start_finger_return()
            worker.begin_finger_return.assert_not_called();self.assertFalse(d.busy)
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):end.start_finger_return()
            worker.begin_finger_return.assert_called_once();self.assertTrue(d.busy)
            d.busy=False;end.await_home=True
            worker.return_snapshot.return_value=(encode([90.]*15,[0]*10),{'mode':'paused','phase':'complete'},True)
            d._controls();self.assertFalse(end.finger_return_button.isEnabled());self.assertTrue(end.confirm_button.isEnabled())
        finally:d.worker=None;d.busy=False;d.close();viewer.close()
