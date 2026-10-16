"""End/return regressions use pure clocks, /dev/pts and fake GUI inputs only."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import pty
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QWidget, QMessageBox
from optusbproj_ubuntu.real_robot_stop_gesture import EndGesture, GestureConfig, GESTURE_INDICES
from optusbproj_ubuntu.real_robot_return import ReturnControl, ReturnConfig, ORDER
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from optusbproj_ubuntu.real_robot_bilateral import BilateralLimits, right_defaults
from optusbproj_ubuntu.real_robot_protocol import decode, encode
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from test_real_robot_bilateral import APP, BASE, human, wait_for


def limits(mask=(True,)*6, speed=30.):
    return BilateralLimits(replace(right_defaults(),enabled=mask,speed=(speed,)*6),
                           ArmLimits(enabled=mask,speed=(speed,)*6))


def frame(q=None):
    values = [95.]*12+[90.]*3 if q is None else q
    return encode(values,[0]*10)


class GestureTests(unittest.TestCase):
    def test_custom_hold_duration(self):
        for seconds in (.5, 2., 8.):
            g = EndGesture(GestureConfig((90.,)*4, seconds=seconds))
            for t in np.arange(0., seconds-.05, .05):
                self.assertFalse(self.update(g, float(t)))
            self.assertFalse(self.update(g, seconds-.01))
            self.assertTrue(self.update(g, seconds))

    def test_left_only_ignores_right_and_requires_left_permissions(self):
        g=EndGesture(GestureConfig((90.,90.),(5.,5.),indices=(7,9)))
        q=encode([130.]*6+[90.]*9,[0]*10)
        for t in np.arange(0.,7.,.1):
            self.assertFalse(self.update(g,float(t),q,enabled=(False,)*6+(True,)*6))
        self.assertTrue(self.update(g,7.,q,enabled=(False,)*6+(True,)*6))
        g=EndGesture(GestureConfig((90.,90.),(5.,5.),indices=(7,9)))
        self.update(g,0.,enabled=(True,)*12)
        self.update(g,.1,enabled=(False,)*12)
        self.assertIsNone(g.started)

    def update(self, detector,t,q=BASE,**kw):
        return detector.update(q,t,following=kw.get('following',True),fresh=kw.get('fresh',True),
                               enabled=kw.get('enabled',(True,)*12),max_gap=.3)

    def test_exact_seven_seconds_and_one_shot(self):
        g=EndGesture(GestureConfig((90.,)*4))
        for t in np.arange(0.,6.91,.1): self.assertFalse(self.update(g,float(t)))
        self.assertFalse(self.update(g,6.99))
        self.assertTrue(self.update(g,7.))
        self.assertFalse(self.update(g,7.1))
        self.assertTrue(g.triggered)

    def test_four_axes_tolerance_exit_staleness_permission_phase_and_gap(self):
        for kw in ({'fresh':False},{'following':False},{'enabled':(False,)*12}):
            g=EndGesture(GestureConfig((90.,)*4))
            self.update(g,0.);self.update(g,.1)
            self.update(g,.2,**kw)
            self.assertIsNone(g.started)
        g=EndGesture(GestureConfig((90.,)*4))
        self.update(g,0.);self.update(g,.1)
        q=[90.]*15;q[9]=95.01
        self.update(g,.2,encode(q,[0]*10))
        self.assertIsNone(g.started)
        q[9]=95.
        self.update(g,.3,encode(q,[0]*10))
        self.assertEqual(g.started,.3)
        self.update(g,1.)
        self.assertEqual(g.started,1.)
        self.update(g,.5)
        self.assertEqual(g.started,.5)

    def test_non_monitored_axes_do_not_constrain_gesture(self):
        g=EndGesture(GestureConfig((90.,)*4))
        q=[90.]*15
        for i in (0,2,4,5,6,8,10,11): q[i]=100.
        self.update(g,0.,encode(q,[0]*10))
        self.assertEqual(g.started,0.)
        with self.assertRaises(ValueError): GestureConfig((90.,)*3)
        with self.assertRaises(ValueError): GestureConfig((90.,)*4,tolerances=(0.,)*4)


class ReturnTests(unittest.TestCase):
    def test_left_six_axes_order_and_no_right_motion(self):
        config=ArmLimits(enabled=(True,)*6,speed=(30.,)*6)
        initial=encode([90.]*6+[95.]*6+[90.]*3,[0]*10)
        c=ReturnControl(initial,config,ReturnConfig(sides=('left',),dwell=0.,manual=False))
        previous=decode(initial)[0];seen=[]
        for _ in range(2000):
            result=c.step();q,fingers=decode(result)
            changed=[i for i in range(15) if q[i]!=previous[i]]
            self.assertLessEqual(len(changed),1)
            if changed and (not seen or seen[-1]!=changed[0]): seen.append(changed[0])
            self.assertEqual(q[:6],[90.]*6);self.assertEqual(q[12:],[90.]*3)
            self.assertEqual(fingers,[0]*10)
            previous=q
            if c.mode=='paused': break
        self.assertEqual(seen,[6+j-1 for j in ORDER])
        self.assertEqual(result,BASE);self.assertEqual(c.phase,'complete')
        with self.assertRaises(ValueError): ReturnControl(initial,config,ReturnConfig())

    def test_full_order_one_joint_changes_and_final_frame(self):
        c=ReturnControl(frame(),limits(),ReturnConfig(dwell=0.,manual=False))
        self.assertEqual(c.step(),frame())
        seen=[];last=decode(frame())[0]
        for _ in range(2000):
            out=c.step();q,f=decode(out)
            changed=[i for i,(a,b) in enumerate(zip(last,q)) if a!=b]
            self.assertLessEqual(len(changed),1)
            if changed and (not seen or changed[0]!=seen[-1]): seen.append(changed[0])
            self.assertEqual(q[12:],[90.]*3);self.assertEqual(f,[0]*10)
            self.assertLessEqual(max(abs(c.v)),30.+1e-6)
            last=q
            if c.mode=='paused': break
        expected=[6+j-1 for j in ORDER]+[j-1 for j in ORDER]
        self.assertEqual(seen,expected)
        self.assertEqual(c.phase,'complete');self.assertEqual(out,BASE)

    def test_manual_gate_freezes_and_ignores_ik(self):
        mask=(False,False,False,True,False,False)
        q=[90.]*15;q[3]=85.;q[9]=80.
        c=ReturnControl(frame(q),limits(mask),ReturnConfig(dwell=0.))
        for _ in range(10): self.assertEqual(c.step(),frame(q))
        c.set_target([0.]*12)
        self.assertEqual(c.phase,'await_step')
        c.advance()
        with self.assertRaises(ValueError): c.advance()
        for _ in range(200):
            out=c.step()
            if c.phase=='await_step': break
        self.assertEqual(decode(out)[0][9],90.)
        self.assertEqual(decode(out)[0][3],85.)
        c.advance()
        for _ in range(200):
            out=c.step()
            if c.mode=='paused': break
        self.assertEqual(out,BASE)

    def test_permissions_ranges_order_and_brake(self):
        mask=(False,False,False,True,False,False)
        with self.assertRaises(ValueError): ReturnControl(frame(),limits(mask))
        config=limits()
        with self.assertRaises(ValueError):
            ReturnControl(frame(),BilateralLimits(replace(config.right,lower=(0.,91.,0.,0.,0.,50.)),config.left))
        with self.assertRaises(ValueError): ReturnConfig(order=(1,3,5,6,4,1))
        c=ReturnControl(frame(),config,ReturnConfig(manual=False))
        for _ in range(4): c.step()
        c.brake('user stop')
        for _ in range(200):
            c.step()
            if c.mode=='paused': break
        self.assertEqual(c.phase,'aborted')
        with self.assertRaises(ValueError): c.advance()


class WorkerReturnTests(unittest.TestCase):
    def setup_worker(self,directory,parent=None):
        master,slave=pty.openpty()
        worker=ElbowSerialWorker(os.ttyname(slave),Recorder(Path(directory)/'control.jsonl'),parent)
        ready=[];errors=[];writes=[]
        worker.ready.connect(lambda:ready.append(True));worker.failed.connect(errors.append);worker.sent.connect(writes.append)
        worker.start();wait_for(lambda:bool(ready))
        return master,slave,worker,errors,writes

    def seed_paused(self,worker):
        mask=(False,False,False,True,False,False)
        config=limits(mask)
        c=SharedArmControl(BASE,config,[90.]*12)
        c.brake('test clean pause')
        worker.begin_control(c,[90.]*12,time.monotonic()+.25)
        wait_for(lambda:not worker.active)
        # Establish a paused, successfully written state by a real planner,
        # then use a small move through the real virtual transport.
        c=SharedArmControl(BASE,config,[90.]*3+[85.]+[90.]*5+[85.]+[90.]*2)
        worker.begin_control(c,[90.]*12,time.monotonic()+.25)
        end=time.monotonic()+3.
        while time.monotonic()<end:
            worker.update_arm_target([90.]*12,time.monotonic()+.25)
            APP.processEvents();time.sleep(.01)
            if worker.last_state and worker.last_state.get('phase')=='await_confirmation': break
        worker.pause('operator pause');wait_for(lambda:not worker.active)
        return config

    def test_return_no_imu_rejects_old_ik_and_requires_manual_home(self):
        with tempfile.TemporaryDirectory() as directory:
            master,slave,worker,errors,writes=self.setup_worker(directory)
            try:
                config=self.seed_paused(worker)
                self.assertEqual(decode(worker.return_snapshot()[0])[0][3],85.)
                worker.begin_return(config,ReturnConfig(dwell=0.,manual=False))
                end=time.monotonic()+5.
                while worker.active and time.monotonic()<end:
                    worker.update_arm_target([0.]*12,time.monotonic()+.25)
                    worker.heartbeat();APP.processEvents();time.sleep(.01)
                self.assertFalse(worker.active);self.assertFalse(errors)
                self.assertEqual(worker.return_snapshot()[0],BASE)
                self.assertTrue(worker.home_pending)
                with self.assertRaises(ValueError):
                    worker.begin_control(SharedArmControl(BASE,config,[90.]*12),[90.]*12,time.monotonic()+.25)
                self.assertEqual(worker.confirm_home(),BASE)
                self.assertFalse(worker.home_pending)
            finally:
                worker.shutdown();wait_for(lambda:not worker.isRunning());os.close(master);os.close(slave)
            logs=[json.loads(l) for l in (Path(directory)/'control.jsonl').read_text().splitlines()]
            self.assertTrue(any(l['event']=='return_commands_complete' for l in logs))
            self.assertTrue(any(l['event']=='home_manually_confirmed' for l in logs))

    def test_heartbeat_timeout_brakes_without_completion_or_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            master,slave,worker,errors,writes=self.setup_worker(directory)
            try:
                config=self.seed_paused(worker)
                with self.assertRaises(ValueError):
                    worker.begin_return(limits(),ReturnConfig())
                worker.begin_return(config,ReturnConfig())
                wait_for(lambda:not worker.active,timeout=3.)
                self.assertFalse(errors)
                self.assertEqual(worker.return_snapshot()[1]['phase'],'aborted')
                self.assertFalse(worker.home_pending)
                with self.assertRaises(ValueError): worker.confirm_home()
            finally:
                worker.shutdown();wait_for(lambda:not worker.isRunning());os.close(master);os.close(slave)

    def test_gesture_worker_triggers_only_after_written_following_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            master,slave,worker,errors,writes=self.setup_worker(directory)
            try:
                c=SharedArmControl(BASE,limits(),[90.]*12)
                c.gesture=EndGesture(GestureConfig((90.,)*4,seconds=.15))
                worker.begin_control(c,[90.]*12,time.monotonic()+.25)
                wait_for(lambda:c.phase=='await_confirmation')
                self.assertFalse(c.gesture.triggered)
                worker.confirm_follow()
                end=time.monotonic()+2.
                while worker.active and time.monotonic()<end:
                    worker.update_arm_target([90.]*12,time.monotonic()+.25)
                    APP.processEvents();time.sleep(.01)
                self.assertFalse(worker.active);self.assertTrue(c.gesture.triggered);self.assertFalse(errors)
            finally:
                worker.shutdown();wait_for(lambda:not worker.isRunning());os.close(master);os.close(slave)


class GuiEndTests(unittest.TestCase):
    def test_left_gui_gesture_pause_ordered_return_and_manual_home(self):
        viewer=QWidget();viewer.engine=None;viewer.replay_mode=False
        dialog=RealRobotModesDialog(viewer);dialog.mode_select.setCurrentIndex(2)
        for row in dialog.axis_rows: row[0].setChecked(False)
        for i in (1,3): dialog.axis_rows[i][0].setChecked(True)
        with tempfile.TemporaryDirectory() as directory, patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):
            master,slave,worker,errors,writes=WorkerReturnTests().setup_worker(directory,dialog)
            try:
                dialog.worker=worker;dialog.connected=True
                worker.progress.connect(dialog._progress);worker.sent.connect(dialog._sent)
                worker.failed.connect(dialog._failed);worker.finished.connect(dialog._finished)
                config=ArmLimits(enabled=(False,True,False,True,False,False),speed=(30.,)*6)
                target=[90.,95.,90.,85.,90.,90.]
                end=dialog.end_session
                self.assertFalse(end.panel.isHidden());self.assertFalse(end.side_order.isEnabled())
                end.gesture_rows[7][0].setValue(95.);end.gesture_rows[9][0].setValue(85.)
                end.configure(False);end.enable.setChecked(True)
                detector=end.gesture(config)
                self.assertEqual(detector.config.indices,(7,9))
                control=SharedArmControl(BASE,config,target)
                control.gesture=EndGesture(replace(detector.config,seconds=.15))
                # Timer heartbeat remains live, but no IMU access may drive this test.
                with patch.object(dialog,'_read_arm_sample',side_effect=ValueError('offline')):
                    worker.begin_control(control,target,time.monotonic()+.25)
                    deadline=time.monotonic()+3.
                    while time.monotonic()<deadline and control.phase!='await_confirmation':
                        worker.update_arm_target(target,time.monotonic()+.25)
                        APP.processEvents();time.sleep(.01)
                    self.assertEqual(control.phase,'await_confirmation')
                    worker.confirm_follow()
                    deadline=time.monotonic()+2.
                    while worker.active and time.monotonic()<deadline:
                        worker.update_arm_target(target,time.monotonic()+.25)
                        APP.processEvents();time.sleep(.01)
                    wait_for(lambda:not worker.active)
                    self.assertTrue(control.gesture.triggered)
                    end.manual.setChecked(True);end.dwell.setValue(0.);end.speed.setValue(30.)
                    dialog._controls()
                    self.assertTrue(end.continuous_button.isEnabled())
                    with patch.object(QMessageBox,'warning',return_value=QMessageBox.Cancel):
                        end.continuous_button.click()
                    self.assertFalse(end.returning);self.assertFalse(worker.active)
                    end.continuous_button.click();self.assertTrue(end.returning)
                    self.assertTrue(end.manual.isChecked())
                    self.assertFalse(end.continuous_button.isEnabled())
                    wait_for(lambda:end.await_home and not dialog.busy)
                    self.assertEqual(worker.last_state['return_order'],['L4','L2'])
                    self.assertEqual(worker.last_written,BASE)
                    self.assertFalse(dialog.capture_button.isEnabled())
                    end.confirm_home();self.assertFalse(end.await_home)
                self.assertFalse(errors)
                self.assertTrue(all(decode(f)[0][:6]==[90.]*6 for f in writes))
            finally:
                dialog._disconnect();wait_for(lambda:dialog.worker is None)
                dialog.timer.stop();dialog.close();viewer.close();os.close(master);os.close(slave)

    def test_virtual_gui_return_ignores_pose_and_manual_confirm_rearms(self):
        viewer=QWidget();viewer.engine=None;viewer.replay_mode=False
        dialog=RealRobotModesDialog(viewer);dialog.mode_select.setCurrentIndex(3)
        for rows in (dialog.axis_rows,dialog.right_axis_rows):
            for row in rows: row[0].setChecked(False)
            rows[3][0].setChecked(True)
        with tempfile.TemporaryDirectory() as directory, patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):
            master,slave,worker,errors,writes=WorkerReturnTests().setup_worker(directory,dialog)
            try:
                dialog.worker=worker;dialog.connected=True
                worker.progress.connect(dialog._progress);worker.sent.connect(dialog._sent)
                worker.failed.connect(dialog._failed);worker.finished.connect(dialog._finished)
                config=WorkerReturnTests().seed_paused(worker)
                dialog.arm_configuration=config
                end=dialog.end_session
                end.manual.setChecked(True);end.speed.setValue(30.);end.dwell.setValue(0.)
                reads_during_return=[]
                def offline_sample():
                    if end.returning:
                        reads_during_return.append(True)
                    raise ValueError('IMU offline')
                # Scope to this dialog: other regression dialogs may still own timers.
                with patch.object(dialog,'_read_arm_sample',side_effect=offline_sample):
                    end.start_return();self.assertTrue(end.returning)
                    wait_for(lambda:worker.last_state.get('phase')=='await_step')
                    end.next_step()
                    wait_for(lambda:worker.last_state.get('return_step')==2 and worker.last_state.get('phase')=='await_step')
                    end.next_step()
                    wait_for(lambda:end.await_home and not dialog.busy)
                    self.assertEqual(reads_during_return,[])
                self.assertFalse(dialog.capture_button.isEnabled())
                self.assertTrue(end.confirm_button.isEnabled())
                end.confirm_home()
                self.assertFalse(end.await_home);self.assertEqual(dialog.origin,BASE)
                self.assertIsNone(dialog.arm_capture)
                self.assertFalse(errors)
            finally:
                dialog._disconnect();wait_for(lambda:dialog.worker is None)
                dialog.close();viewer.close();os.close(master);os.close(slave)

    def test_reference_confirmation_permission_gate_and_defaults(self):
        viewer=QWidget();viewer.engine=None;viewer.replay_mode=False
        dialog=RealRobotModesDialog(viewer);dialog.timer.stop();dialog.mode_select.setCurrentIndex(3)
        try:
            end=dialog.end_session
            self.assertFalse(end.enable.isChecked());self.assertTrue(end.manual.isChecked())
            self.assertEqual(end.order.text(),'1,3,5,6,4,2')
            end.enable.setChecked(True)
            with self.assertRaises(ValueError): end.gesture(limits())
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes): end.configure(False)
            self.assertIsNotNone(end.gesture(limits()))
            with self.assertRaises(ValueError): end.gesture(limits((False,)*6))
            end.gesture_rows[1][0].setValue(100.)
            with self.assertRaises(ValueError): end.gesture(limits())
        finally: dialog.close();viewer.close()
