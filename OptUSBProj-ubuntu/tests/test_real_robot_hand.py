"""Pure/Qt/virtual-UART tests only; never open a physical robot port."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
import json
from pathlib import Path
import pty
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PySide6.QtWidgets import QWidget, QMessageBox
from optusbproj_ubuntu.real_robot_hand import HandConfig, HandMotion, HandBank, observation_valid
from optusbproj_ubuntu.real_robot_hand_profiles import save_profile, read_profile
from optusbproj_ubuntu.real_robot_mirror import mirror_hand_targets, CommandMirror
from optusbproj_ubuntu.real_robot_protocol import encode, decode
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_arm import ArmLimits, load_profile
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_return import ReturnControl, ReturnConfig
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from test_real_robot_shared import APP, BASE, wait_for


def obs(now, state=True, valid=True, signature=('back','finger',1)):
    return dict(valid=valid,state=state,stamp=now,signature=signature,residual_deg=0.)


def config(**kwargs):
    return replace(HandConfig(enabled=(True,False,False,False,False)),**kwargs)


class HandTests(unittest.TestCase):
    def test_residual_bypass_is_explicit_and_cannot_bypass_freshness_or_reference(self):
        data={**obs(1.),'residual_deg':45.,'residual_bypass':True}
        self.assertFalse(observation_valid(data,1.1))
        self.assertTrue(observation_valid(data,1.1,allow_residual=True))
        self.assertFalse(observation_valid(data,1.3,allow_residual=True))
        self.assertFalse(observation_valid({**data,'residual_deg':float('inf')},1.1,allow_residual=True))
        with self.assertRaises(ValueError):HandMotion(config(),[0]*5,data,20.,1.)
        m=HandMotion(config(),[0]*5,data,20.,1.,allow_residual=True)
        m.observe({**data,'stamp':1.1},1.1);self.assertEqual(m.mode,'active')
        m.observe({**data,'stamp':1.2,'signature':('changed',)},1.2);self.assertEqual(m.mode,'braking')
        m.observe({**data,'stamp':1.3},1.3);self.assertEqual(m.mode,'braking')

    def test_strict_configuration_and_freshness(self):
        for kwargs in (dict(opened=(True,)*5),dict(closed=(4096,)*5),dict(speed=(float('nan'),)*5),
                       dict(lower=(200,)*5),dict(enabled=(1,)*5),dict(speed=(0.,)*5)):
            with self.assertRaises(ValueError):HandConfig(**kwargs)
        self.assertTrue(observation_valid(obs(2.),2.1))
        for data in (obs(1.),obs(3.),obs(2.,state=None),obs(2.,valid=False),
                     {**obs(2.),'residual_deg':float('nan')},{**obs(2.),'residual_deg':21.}):
            self.assertFalse(observation_valid(data,2.1))

    def test_first_frame_and_mask_preserve_all_other_channels(self):
        bank=HandBank(); angles=list(range(80,95)); fingers=[9,8,7,6,5,0,4,3,2,1]
        base=encode(angles,fingers); bank.reset(base); bank.observe('left',obs(1.),1.)
        bank.enable('left',config(),20.,1.)
        self.assertEqual(bank.merge(base,1.,arm_mode='active',following=True),base)
        for i in range(1,101):
            now=1.+i*.05; bank.observe('left',obs(now),now)
            a,f=decode(bank.merge(base,now,arm_mode='active',following=True))
            self.assertEqual(a,angles); self.assertEqual(f[:5],fingers[:5]); self.assertEqual(f[6:],fingers[6:])
        self.assertEqual(f[5],200)

    def test_target_reversal_random_limits_and_analytic_stop(self):
        rng=random.Random(16); m=HandMotion(config(),[0]*5,obs(1.),20.,1.)
        for i in range(1,1001):
            now=1.+i*.05; m.observe(obs(now,state=bool(rng.randrange(2))),now)
            before=m.v[0]; m.step(now)
            self.assertTrue(0<=m.q[0]<=200)
            self.assertLessEqual(abs(m.v[0]),200+1e-6)
            self.assertLessEqual(abs(m.v[0]-before),400*.05+1e-6)
        m.brake('stop')
        for i in range(100):m.step(now+i*.05)
        self.assertEqual(m.mode,'paused')
        self.assertAlmostEqual(m.v[0],0.)

    def test_gap_recovery_and_latched_timeout(self):
        m=HandMotion(config(),[0]*5,obs(1.),20.,1.)
        m.step(1.); m.step(1.05); m.step(1.26)
        self.assertEqual(m.mode,'gap')
        m.observe(obs(1.30),1.30); self.assertEqual(m.mode,'active')
        m.step(1.81); self.assertIn(m.mode,('braking','paused'))
        m.observe(obs(1.82),1.82); self.assertNotEqual(m.mode,'active')

    def test_invalid_quality_and_reference_changes(self):
        m=HandMotion(config(),[0]*5,obs(1.),20.,1.)
        m.observe(obs(1.1,valid=False),1.1); self.assertEqual(m.mode,'gap')
        m.observe(obs(1.2),1.2); self.assertEqual(m.mode,'active')
        m.observe(obs(1.3,signature=('different',)),1.3); self.assertEqual(m.mode,'braking')
        m.observe(obs(1.4),1.4); self.assertEqual(m.mode,'braking')

    def test_global_pause_stops_hands_no_automatic_reactivation(self):
        bank=HandBank(); bank.reset(BASE); bank.observe('left',obs(1.),1.)
        bank.enable('left',config(),20.,1.)
        for i in range(5):
            now=1.+i*.05; bank.observe('left',obs(now),now); bank.merge(BASE,now,arm_mode='active',following=True)
        for i in range(40):bank.merge(BASE,1.25+i*.05,arm_mode='paused',following=True)
        self.assertTrue(bank.stopped); held=bank.values.copy()
        bank.observe('left',obs(4.),4.); bank.merge(BASE,4.,arm_mode='active',following=True)
        self.assertEqual(bank.values,held)

    def test_both_hands_independent_states(self):
        bank=HandBank(); bank.reset(BASE)
        for side,state in (('left',True),('right',False)):
            bank.observe(side,obs(1.,state),1.);bank.enable(side,config(),20.,1.)
        for i in range(70):
            now=1.+i*.05
            for side,state in (('left',True),('right',False)):bank.observe(side,obs(now,state),now)
            a,f=decode(bank.merge(BASE,now,arm_mode='active',following=True))
        self.assertEqual(f[0],0); self.assertEqual(f[5],200); self.assertEqual(a,[90.]*15)

    def test_profiles_roundtrip_no_permissions_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            path=save_profile('握拳',dict(left=config(),right=config()),root)
            name,rows=read_profile(path)
            self.assertEqual(name,'握拳'); self.assertFalse(any(rows['left'].enabled))
            self.assertEqual(rows['left'].closed,(200,)*5)
            path.write_text('{"version":1,"version":1}')
            with self.assertRaises(ValueError):read_profile(path)

    def test_mirror_scales_each_protocol_finger_to_three_nominal_joints(self):
        fingers=[4095,0,0,0,0,0,4095,0,0,0]
        targets=mirror_hand_targets(encode([90.]*15,fingers))
        np.testing.assert_allclose(np.rad2deg(targets['right'][:3]),[18,65,55])
        np.testing.assert_allclose(np.rad2deg(targets['left'][3:6]),[60,50,40])
        self.assertEqual(np.count_nonzero(targets['right']),3)

    def test_return_preserves_nonzero_fingers_and_rejects_false_power_on_confirmation(self):
        frame=encode([90.]*15,[20]*10)
        limits=ArmLimits(enabled=(True,)*6)
        c=ReturnControl(frame,limits,ReturnConfig(sides=('left',)))
        self.assertEqual(c.step(),frame)
        worker=ElbowSerialWorker('not-opened',Mock())
        worker.last_written=frame;worker.home_pending=True;worker.last_state=dict(phase='complete')
        with self.assertRaisesRegex(ValueError,'手指非零'):worker.confirm_home()


class HandGuiTests(unittest.TestCase):
    def setUp(self):
        self.viewer=QWidget();self.viewer.engine=None;self.viewer.replay_mode=False
        self.d=RealRobotModesDialog(self.viewer);self.d.timer.stop();self.d.hand_session.timer.stop()
        self.d.mode_select.setCurrentIndex(2);self.s=self.d.hand_session

    def tearDown(self):
        self.d.worker=None;self.d.busy=False;self.d.ik_worker=None;self.d.close();self.viewer.close()

    def test_default_off_and_checkpoint_load_is_parameter_only(self):
        self.assertFalse(any(self.s.config('left').enabled))
        self.assertFalse(self.s.starts['left'].isEnabled())
        self.s.rows['left'][0][0].setChecked(True)
        worker=Mock();worker.hand_snapshot.return_value={};worker.return_snapshot.return_value=(None,None,False);self.d.worker=worker
        with tempfile.TemporaryDirectory() as root:
            path=save_profile('small',dict(left=config(),right=config()),root);self.s.apply(path)
        self.assertFalse(any(self.s.config('left').enabled)); worker.enable_hand.assert_not_called()
        worker.observe_hand.assert_not_called();worker.stop_hand.assert_not_called()

    def test_status_reports_live_input_before_permission_and_startup_blockers(self):
        self.s.receive('left',obs(time.monotonic()));self.s.controls()
        text=self.s.labels['left'].text()
        self.assertIn('闭合（有效）',text);self.assertIn('输出：未授权',text)
        self.assertIn('串口未连接',text);self.assertIn('手臂尚未进入同源跟随',text)
        worker=Mock();worker.hand_snapshot.return_value={};worker.return_snapshot.return_value=(None,None,False)
        self.d.worker=worker;self.d.busy=True;self.d.safety.setChecked(True)
        self.d.shared_phase='await_confirmation';self.d.shared_control_mode='active'
        self.s.rows['left'][0][0].setChecked(True);self.s.controls()
        self.assertFalse(self.s.starts['left'].isEnabled())
        self.d.shared_phase='following';self.s.controls()
        self.assertTrue(self.s.starts['left'].isEnabled())
        self.assertIn('等待明确启用',self.s.labels['left'].text())

    def test_policy_toggle_stops_clears_old_observation_and_syncs_source(self):
        worker=Mock();worker.hand_snapshot.return_value={};worker.return_snapshot.return_value=(None,None,False)
        self.d.worker=worker;self.d.busy=True;self.d.safety.setChecked(True)
        self.d.shared_phase='following';self.d.shared_control_mode='active'
        panel=Mock();self.viewer.gripper_gesture_dialog=SimpleNamespace(panels={'left':panel,'right':Mock()})
        self.assertFalse(self.s.residual_checks['left'].isChecked())
        self.s.receive('left',obs(time.monotonic()));self.s.residual_checks['left'].setChecked(True)
        worker.stop_hand.assert_called_once_with('left');panel.set_real_residual_policy.assert_called_once_with(True)
        self.assertNotIn('left',self.s.observations);worker.enable_hand.assert_not_called()
        self.s.receive('left',{**obs(time.monotonic()),'residual_deg':45.,'residual_bypass':False})
        self.assertFalse(self.s.ready('left'))
        self.s.receive('left',{**obs(time.monotonic()),'residual_deg':45.,'residual_bypass':True})
        self.assertTrue(self.s.ready('left'));self.s.rows['left'][0][0].setChecked(True)
        with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):self.s.enable('left')
        self.assertTrue(worker.enable_hand.call_args.kwargs['allow_residual'])
        self.s.residual_checks['left'].setChecked(False)
        self.assertEqual(worker.stop_hand.call_count,2);self.assertFalse(self.s.ready('left'))

    def test_fresh_modal_required_and_no_right_hand_in_left_mode(self):
        worker=Mock();worker.hand_snapshot.return_value={};worker.return_snapshot.return_value=(None,None,False);self.d.worker=worker
        self.d.busy=True;self.d.safety.setChecked(True);self.d.shared_phase='following';self.d.shared_control_mode='active'
        self.s.receive('left',obs(time.monotonic()));self.s.rows['left'][0][0].setChecked(True)
        self.assertTrue(self.s.ready('left'));self.assertFalse(self.s.ready('right'))
        with patch.object(QMessageBox,'warning',return_value=QMessageBox.Cancel): self.s.enable('left')
        worker.enable_hand.assert_not_called()
        with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes): self.s.enable('left')
        worker.enable_hand.assert_called_once()

    def test_invalid_profile_is_atomic_and_running_hand_blocks_loading(self):
        before=self.s.config('left')
        with tempfile.TemporaryDirectory() as root:
            path=save_profile('bad',dict(left=config(),right=config()),root)
            row=json.loads(path.read_text());row['sides']['left']['closed']=[100]*5
            row['sides']['right']['upper']=[-1]*5;path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):self.s.apply(path)
            self.assertEqual(self.s.config('left'),before)
            worker=Mock();worker.hand_snapshot.return_value={'left':dict(mode='active')};self.d.worker=worker
            with self.assertRaisesRegex(ValueError,'停用'):self.s.apply(path)

    def test_real_classifier_never_inherits_preview_residual_bypass(self):
        from optusbproj_ubuntu.gripper_gesture_gui import GestureWindow, SharedGestureSource
        with tempfile.TemporaryDirectory() as root:
            panel=GestureWindow(SharedGestureSource(),root);panel.timer.stop()
            try:
                panel.dwell.setValue(0);panel.warn_only.setChecked(True)
                self.assertTrue(panel.update_real_state(0.,1.,0.,True))
                self.assertFalse(panel.real_discrete.state)
                panel.discrete.update(1.,2.)
                self.assertFalse(panel.update_real_state(1.,2.,40.,True))
                self.assertFalse(panel.real_discrete.state)
                self.assertEqual(panel.real_filtered,0.)
                self.assertTrue(panel.discrete.state)
                panel.set_real_residual_policy(True)
                self.assertIsNone(panel.real_discrete.state)
                self.assertTrue(panel.update_real_state(1.,3.,40.,True))
                self.assertTrue(panel.real_discrete.state)
                self.assertTrue(panel.discrete.state)
                panel.set_real_residual_policy(False)
                self.assertIsNone(panel.real_discrete.state)
                self.assertFalse(panel.update_real_state(1.,4.,40.,True))
                previous=panel.real_revision;panel.reset_discrete()
                self.assertGreater(panel.real_revision,previous);self.assertIsNone(panel.real_discrete.state)
            finally:panel.shutdown();panel.close()

    def test_real_bypass_emits_fresh_gesture_even_when_preview_rejects_residual(self):
        from optusbproj_ubuntu.gripper_gesture_gui import GestureWindow, SharedGestureSource
        with tempfile.TemporaryDirectory() as root:
            panel=GestureWindow(SharedGestureSource(),root);panel.timer.stop();rows=[]
            panel.hand_observation.connect(rows.append)
            try:
                panel.dwell.setValue(0);panel.set_real_residual_policy(True)
                panel.calibration=Mock()
                panel.calibration.evaluate.return_value=SimpleNamespace(angle_deg=73.,residual_deg=35.,closure=1.)
                panel.selected_pair=Mock(return_value=(time.monotonic(),np.eye(3),np.eye(3)))
                self.assertFalse(panel.warn_only.isChecked());panel.tick()
                self.assertTrue(rows[-1]['valid']);self.assertTrue(rows[-1]['state'])
                self.assertEqual(rows[-1]['residual_deg'],35.);self.assertTrue(rows[-1]['residual_bypass'])
                panel.set_real_residual_policy(False);panel.tick()
                self.assertFalse(rows[-1]['valid']);self.assertFalse(rows[-1]['residual_bypass'])
            finally:panel.shutdown();panel.close()


class HandVirtualTests(unittest.TestCase):
    def test_mirror_sends_arm_and_both_hand_targets_from_same_written_frame(self):
        bridge=SimpleNamespace(data=dict(arms={s:load_profile(s)[0] for s in ('left','right')},
                                        hands={'left':{},'right':{}}),exchange=Mock(),close=Mock())
        mirror=CommandMirror(bridge_factory=lambda _:bridge,sides=('left','right'))
        ready=[];errors=[];mirror.ready.connect(lambda:ready.append(True));mirror.failed.connect(errors.append)
        try:
            mirror.start();wait_for(lambda:bool(ready) or bool(errors));self.assertFalse(errors)
            bridge.exchange.assert_not_called()
            frame=encode([90.]*15,[100]*10);mirror.submit(frame,{'left':(1,)*6,'right':(1,)*6})
            wait_for(lambda:bridge.exchange.called)
            actual=bridge.exchange.call_args.kwargs['hand_targets'];expected=mirror_hand_targets(frame)
            for side in ('left','right'):np.testing.assert_allclose(actual[side],expected[side])
        finally:mirror.shutdown();self.assertTrue(mirror.wait(2500))
        bridge.close.assert_called_once()

    def test_finger_timeout_latches_without_stopping_healthy_arms(self):
        master,slave=pty.openpty();limits=ArmLimits(enabled=(True,)*6,frequency=50.)
        c=SharedArmControl(BASE,limits,[90.]*6);c.phase='following'
        with tempfile.TemporaryDirectory() as root:
            worker=ElbowSerialWorker(os.ttyname(slave),Recorder(Path(root)/'control.jsonl'))
            frames=[];worker.sent.connect(frames.append)
            try:
                worker.start();wait_for(lambda:worker.isRunning())
                worker.begin_control(c,[90.]*6,time.monotonic()+30);wait_for(lambda:len(frames)>2)
                worker.observe_hand('left',obs(time.monotonic()));worker.enable_hand('left',config())
                wait_for(lambda:worker.hand_snapshot()['left']['mode']=='paused')
                self.assertTrue(worker.active)
                held=worker.hand_snapshot()['left']['command']
                worker.observe_hand('left',obs(time.monotonic()));time.sleep(.08);APP.processEvents()
                self.assertEqual(worker.hand_snapshot()['left']['command'],held)
                self.assertEqual(worker.hand_snapshot()['left']['mode'],'paused')
            finally:worker.shutdown();worker.wait(3000);os.close(master);os.close(slave)

    def test_shared_uart_merges_then_global_stop_drains_both_planners(self):
        master,slave=pty.openpty()
        limits=ArmLimits(enabled=(True,)*6,frequency=50.)
        c=SharedArmControl(BASE,limits,[90.]*6); c.phase='following'
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'control.jsonl';worker=ElbowSerialWorker(os.ttyname(slave),Recorder(path))
            frames=[];states=[];worker.sent.connect(frames.append);worker.progress.connect(states.append)
            try:
                worker.start();wait_for(lambda:worker.isRunning())
                worker.begin_control(c,[90.]*6,time.monotonic()+30)
                wait_for(lambda:len(frames)>2)
                now=time.monotonic();worker.observe_hand('left',obs(now));worker.enable_hand('left',config())
                for _ in range(15):
                    now=time.monotonic();worker.observe_hand('left',obs(now));APP.processEvents();time.sleep(.02)
                wait_for(lambda:any(decode(f)[1][5]>0 for f in frames))
                worker.pause('test global pause');wait_for(lambda:not worker.active)
                count=len(frames);worker.observe_hand('left',obs(time.monotonic()));time.sleep(.08);APP.processEvents()
                self.assertEqual(len(frames),count)
                self.assertTrue(worker.hands.stopped)
                worker.shutdown();self.assertTrue(worker.wait(3000));APP.processEvents()
                raw=os.read(master,65536);self.assertEqual(raw,b''.join(frames))
                self.assertEqual(frames[0],BASE)
                for f in frames:
                    a,values=decode(f);self.assertEqual(a,[90.]*15)
                    self.assertEqual(values[:5]+values[6:],[0]*9)
                rows=[json.loads(l) for l in path.read_text().splitlines()]
                self.assertTrue(any(r['event']=='hand_enabled' for r in rows))
                self.assertTrue(any(r.get('hands',{}).get('left',{}).get('command',[0])[0]>0 for r in rows))
            finally:
                worker.shutdown();worker.wait(3000);os.close(master);os.close(slave)
