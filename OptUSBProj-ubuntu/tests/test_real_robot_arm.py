"""Nominal IK and virtual UART only: never communicate with physical hardware."""
import os
import json
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
import pty
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from optusbproj_ubuntu.real_robot_arm import ArmLimits, ArmControl, RelativeArmIK, load_profile
from optusbproj_ubuntu.real_robot_protocol import encode, decode
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_arm_worker import ArmIKWorker
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.scene_robot_model import SceneArm
from scipy.spatial.transform import Rotation

APP = QApplication.instance() or QApplication([])
BASE = encode([90.]*15, [0]*10)


def wait_for(predicate, timeout=4):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        APP.processEvents()
        if predicate():
            return
        time.sleep(.003)
    raise AssertionError('virtual test timed out')


def sample_at(model, q):
    s,e,w = model.points(q)
    return dict(points=(s,e,w), directions=[(e-s)/np.linalg.norm(e-s), (w-e)/np.linalg.norm(w-e)],
                hand=model.tool_frame(q)[:3,:3],
                rotations={k:model.segment_frame(q,k)[:3,:3] for k in ('upper_arm','forearm')})


class ArmTests(unittest.TestCase):
    def test_limit_validation_and_mapping(self):
        for kwargs in ({'frequency':0}, {'speed':(251.,)*6}, {'acceleration':(0.,)*6},
                       {'lower':(80.,)*6}, {'signs':(0,)*6}, {'enabled':(1,)*6},
                       {'upper':(226.,233.,180.,120.,180.,122.)}):
            with self.assertRaises(ValueError):
                ArmLimits(**kwargs)
        limits = ArmLimits(signs=(1,-1,1,-1,1,-1))
        q = np.array([.1,-.1,.2,-.2,.3,-.3])
        np.testing.assert_allclose(limits.to_sim(limits.to_command(q)), q)
        with self.assertRaises(ValueError):
            ArmControl(BASE, ArmLimits())

    def test_first_frame_mask_and_bounds(self):
        for mask in ((True,)*6, (False,False,True,True,True,False), (False,False,False,True,False,False)):
            limits = ArmLimits(enabled=mask, speed=(60.,)*6, acceleration=(60.,)*6,
                               lower=(80.,90.,80.,0.,80.,80.), upper=(100.,)*6)
            c = ArmControl(BASE, limits)
            c.set_target([200.,200.,200.,-50.,-10.,0.])
            self.assertEqual(c.step(), BASE)
            for _ in range(160):
                frame = c.step()
                angles, fingers = decode(frame)
                self.assertEqual(fingers, [0]*10)
                for i, value in enumerate(angles):
                    if not (6 <= i < 12 and mask[i-6]):
                        self.assertEqual(value, 90.)
                    else:
                        self.assertTrue(limits.lower[i-6] <= value <= limits.upper[i-6])
                self.assertLessEqual(np.max(np.abs(c.v)), 60.+1e-7)
                self.assertLessEqual(np.max(np.abs(c.a)), 60.+1e-7)
            self.assertEqual(c.mode, 'active')
            np.testing.assert_allclose(c.q, np.where(mask, [100.,100.,100.,0.,80.,80.], 90.))

    def test_reverse_and_brake_all_axes_at_high_rates(self):
        rng = np.random.default_rng(19)
        for hz in (5.,20.,100.):
            limits = ArmLimits(frequency=hz, enabled=(True,)*6, speed=(250.,)*6, acceleration=(250.,)*6)
            c = ArmControl(BASE, limits)
            previous = np.zeros(6)
            for _ in range(300):
                c.set_target(rng.uniform(limits.lower, limits.upper))
                c.step()
                self.assertTrue(np.all(c.q >= np.asarray(limits.lower)-1e-7))
                self.assertTrue(np.all(c.q <= np.asarray(limits.upper)+1e-7))
                self.assertTrue(np.all(np.abs(c.v-previous) <= 250/hz+1e-7))
                previous = c.v.copy()
            c.brake('test')
            for _ in range(200):
                c.step()
            self.assertEqual(c.mode, 'paused')
            frozen = c.q.copy()
            c.set_target([90.]*6)
            c.step()
            np.testing.assert_allclose(c.q, frozen)

    def test_nominal_clutch_and_palm_cannot_move_shoulders(self):
        arm, _, provenance = load_profile()
        model = SceneArm(arm, [-np.pi]*6, [np.pi]*6)
        reference = sample_at(model, np.zeros(6))
        limits = ArmLimits(enabled=(True,)*6)
        solver = RelativeArmIK(reference, BASE, limits, (arm, np.eye(3), provenance))
        result = solver.solve(reference)
        np.testing.assert_allclose(result['target'], [90.]*6, atol=1e-4)
        self.assertLess(result['position_error_mm'], 1e-4)
        # Keep all segment points/rotations exactly fixed and change only wrist FK.
        current = sample_at(model, np.array([0.,0.,0.,0.,.12,-.1]))
        for _ in range(5):
            result = solver.solve(current)
        np.testing.assert_allclose(result['target'][:4], [90.]*4, atol=.001)
        self.assertGreater(abs(result['target'][4]-90), 2)
        self.assertGreater(abs(result['target'][5]-90), 2)
        self.assertLess(result['orientation_error_deg'], 1.)

    def test_nonzero_retake_anchor_and_invalid_input(self):
        arm, _, provenance = load_profile()
        model = SceneArm(arm, [-np.pi]*6, [np.pi]*6)
        commands = [90.]*15
        commands[6:12] = [92.,94.,93.,60.,91.,88.]
        limits = ArmLimits(enabled=(True,)*6)
        reference = sample_at(model, np.zeros(6))  # Human need not match robot!
        solver = RelativeArmIK(reference, encode(commands,[0]*10), limits, (arm,np.eye(3),provenance))
        result = solver.solve(reference)
        np.testing.assert_allclose(result['target'], commands[6:12], atol=.001)
        bad = deepcopy(reference)
        bad['hand'] = np.full((3,3), np.nan)
        with self.assertRaises(ValueError):
            solver.solve(bad)


class WorkerTests(unittest.TestCase):
    def test_virtual_uart_gap_recovers_then_explicit_pause_latches(self):
        master, slave = pty.openpty()
        states, errors, ready = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            worker = ElbowSerialWorker(os.ttyname(slave), Recorder(Path(directory)/'control.jsonl'))
            worker.ready.connect(lambda:ready.append(True))
            worker.progress.connect(states.append)
            worker.failed.connect(errors.append)
            try:
                worker.start()
                wait_for(lambda:ready)
                worker.begin_control(ArmControl(BASE, ArmLimits(enabled=(True,)*6)),
                                     [92.]*6, time.monotonic()+.1)
                wait_for(lambda:states and states[-1]['mode']=='gap')
                worker.update_arm_target([93.]*6, time.monotonic()+.25)
                wait_for(lambda:states and states[-1]['mode']=='active')
                worker.pause('测试人工暂停')
                wait_for(lambda:states and states[-1]['mode']=='paused')
                worker.update_arm_target([94.]*6, time.monotonic()+.25)
                self.assertFalse(worker.active)
                self.assertFalse(errors)
            finally:
                worker.shutdown()
                self.assertTrue(worker.wait(3000))
                os.close(master);os.close(slave)
            events=[json.loads(line)['event'] for line in
                    (Path(directory)/'control.jsonl').read_text().splitlines()]
            self.assertIn('input_gap_started',events)
            self.assertIn('input_gap_recovered',events)
            self.assertNotIn('input_timeout',events)

    def test_virtual_uart_mask_and_stale_braking(self):
        master, slave = pty.openpty()
        writes, states, errors = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            worker = ElbowSerialWorker(os.ttyname(slave), Recorder(Path(directory)/'control.jsonl'))
            worker.sent.connect(writes.append)
            worker.progress.connect(states.append)
            worker.failed.connect(errors.append)
            ready = []
            worker.ready.connect(lambda:ready.append(True))
            try:
                worker.start()
                wait_for(lambda:ready)
                self.assertFalse(writes)
                limits = ArmLimits(enabled=(True,False,True,True,False,True))
                worker.begin_control(ArmControl(BASE,limits), [90.]*6, time.monotonic()+.25)
                wait_for(lambda:writes)
                self.assertEqual(writes[0], BASE)
                for _ in range(6):
                    worker.update_arm_target([98.,99.,95.,60.,99.,85.], time.monotonic()+.25,
                                             {'sequence':42,'input_timing':{'devices':{'left_hand':{'age_ms':10.}}}})
                    until = time.monotonic()+.07
                    wait_for(lambda:time.monotonic() >= until)
                wait_for(lambda:states and states[-1]['mode']=='paused')
                for frame in writes:
                    q,f = decode(frame)
                    self.assertEqual(q[:6]+q[12:], [90.]*9)
                    self.assertEqual(q[7],90.)
                    self.assertEqual(q[10],90.)
                    self.assertEqual(f,[0]*10)
                self.assertGreater(decode(writes[-1])[0][6],90.)
                count = len(writes)
                worker.update_arm_target([90.]*6,time.monotonic()+1)
                until=time.monotonic()+.15
                wait_for(lambda:time.monotonic()>=until)
                self.assertEqual(len(writes),count)
                self.assertFalse(errors)
            finally:
                worker.shutdown()
                self.assertTrue(worker.wait(3000))
                os.close(master)
                os.close(slave)

            rows=[json.loads(line) for line in (Path(directory)/'control.jsonl').read_text().splitlines()]
            timeout=[row for row in rows if row['event']=='input_timeout']
            self.assertEqual(len(timeout),1)
            self.assertEqual(timeout[0]['timing']['sequence'],42)
            self.assertGreaterEqual(timeout[0]['expired_ms'],0.)
            self.assertGreater(timeout[0]['target_age_ms'],200.)
            self.assertTrue(any(row['event']=='arm_target_received' for row in rows))

    def test_ik_worker_discards_expired_result(self):
        class Slow:
            def solve(self, side):
                time.sleep(.07)
                return {'target':[90.]*6}
        worker=ArmIKWorker(Slow())
        results=[]
        worker.result.connect(results.append)
        worker.start()
        try:
            worker.submit({},time.monotonic()+.02)
            until=time.monotonic()+.15
            wait_for(lambda:time.monotonic()>=until)
            self.assertFalse(results)
            worker.submit({},time.monotonic()+.25)
            wait_for(lambda:results)
            self.assertLess(results[-1]['deadline'],time.monotonic()+.25)
        finally:
            worker.shutdown()
            self.assertTrue(worker.wait(1000))

    def test_gui_six_mode_default_permissions_and_live_chain(self):
        viewer=QWidget()
        viewer.replay_mode=False
        viewer.engine=SimpleNamespace()
        arm,C,provenance=load_profile()
        model=SceneArm(arm,[-np.pi]*6,[np.pi]*6)
        def human_at(q):
            raw=sample_at(model,q)
            raw['points']=tuple(C.T@p for p in raw['points'])
            raw['directions']=[C.T@p for p in raw['directions']]
            raw['hand']=C.T@raw['hand']
            raw['rotations']={k:C.T@R for k,R in raw['rotations'].items()}
            return raw
        reference=human_at(np.zeros(6))
        human=deepcopy(reference)
        def pose(*args):
            return dict(at=time.monotonic(),sides={'left':deepcopy(human)},device_ages_ms={'u':1,'f':1,'h':1}), ('stable',)
        master,slave=pty.openpty()
        dialog=None
        with patch('optusbproj_ubuntu.real_robot_arm_gui.pose_sample',side_effect=pose), tempfile.TemporaryDirectory() as directory:
            try:
                dialog=RealRobotModesDialog(viewer)
                dialog.mode_select.setCurrentIndex(1)
                dialog.show()
                self.assertEqual(dialog._limits().enabled,(True,)*6)
                for row in dialog.axis_rows:
                    row[0].setChecked(False)
                worker=ElbowSerialWorker(os.ttyname(slave),Recorder(Path(directory)/'control.jsonl'),dialog)
                dialog.worker=worker
                worker.ready.connect(dialog._ready)
                worker.progress.connect(dialog._progress)
                worker.sent.connect(dialog._sent)
                worker.finished.connect(dialog._finished)
                worker.start()
                wait_for(lambda:dialog.connected)
                dialog.origin=dialog.last_frame=BASE
                dialog.safety.setChecked(True)
                wait_for(lambda:len(dialog.arm_history)>28)
                dialog._capture()
                self.assertIsNone(dialog.arm_capture) # No output consent.
                dialog.axis_rows[3][0].setChecked(True)
                dialog._capture()
                self.assertIsNotNone(dialog.arm_capture)
                QTest.mousePress(dialog.start_button, Qt.LeftButton)
                QTest.qWait(100) # Real polling refreshes the GUI while pressed.
                self.assertTrue(dialog.start_button.isDown())
                self.assertFalse(dialog.busy)
                QTest.mouseRelease(dialog.start_button, Qt.LeftButton)
                self.assertTrue(dialog.busy)
                self.assertEqual(dialog.last_frame,BASE)
                wait_for(lambda:'全六轴 IK' in dialog.ik_label.text())
                human=human_at(np.array([0.,0.,0.,-.08,0.,0.]))
                wait_for(lambda:abs(decode(dialog.last_frame)[0][9]-90.)>.1)
                moved=decode(dialog.last_frame)[0]
                self.assertEqual(moved[:9]+moved[10:], [90.]*14)
                dialog._pause('virtual acceptance')
                wait_for(lambda:not dialog.busy and dialog.ik_worker is None)
                self.assertFalse(dialog.safety.isChecked())
                self.assertIsNone(dialog.arm_capture)
                # Re-take over at the last command, then close WHILE actively armed.
                until=time.monotonic()+.8
                wait_for(lambda:time.monotonic()>=until)
                dialog.safety.setChecked(True)
                dialog._capture()
                self.assertIsNotNone(dialog.arm_capture)
                origin=dialog.origin
                dialog._start()
                self.assertTrue(dialog.busy)
                self.assertEqual(dialog.last_frame,origin)
                dialog.close()
                wait_for(lambda:dialog.worker is None and dialog.ik_worker is None)
                rows=list(map(json.loads,(Path(directory)/'control.jsonl').read_text().splitlines()))
                attempts=[r for r in rows if r['event']=='arm_start_attempt']
                accepted=[r for r in rows if r['event']=='arm_start_accepted']
                self.assertEqual(len(attempts),2)
                self.assertEqual(len(accepted),2)
                self.assertEqual(attempts[0]['enabled'],[False,False,False,True,False,False])
                self.assertLess(max(attempts[0]['orientation_delta_deg'].values()),2.)
            finally:
                if dialog is not None:
                    dialog._disconnect()
                    wait_for(lambda:dialog.worker is None and dialog.ik_worker is None)
                    dialog.close()
                viewer.close()
                os.close(master)
                os.close(slave)


class ButtonRefreshTests(unittest.TestCase):
    """Exercise mouse press/release, not just atomic click() or direct _start()."""

    def test_start_press_survives_periodic_refresh_in_both_modes(self):
        viewer = QWidget()
        viewer.replay_mode = False
        viewer.engine = None
        with patch.object(RealRobotModesDialog, '_start') as start:
            dialog = RealRobotModesDialog(viewer)
            dialog.timer.stop()
            refresh = QTimer(dialog)
            refresh.setInterval(20)
            refresh.timeout.connect(dialog._controls)
            try:
                dialog.worker = SimpleNamespace(isRunning=lambda: True, recorder=Mock())
                dialog.connected = True
                for mode in (0, 1):
                    with self.subTest(mode=mode):
                        dialog.mode_select.setCurrentIndex(mode)
                        dialog.origin = BASE
                        dialog.safety.setChecked(True)
                        dialog.capture = (0., 0., ('test',)) if mode == 0 else None
                        dialog.arm_capture = ({}, 0., ('test',)) if mode == 1 else None
                        dialog._controls()
                        start.reset_mock()
                        self.assertTrue(dialog.start_button.isEnabled())
                        QTest.mousePress(dialog.start_button, Qt.LeftButton)
                        self.assertTrue(dialog.start_button.isDown())
                        refresh.start()
                        QTest.qWait(100)
                        refresh.stop()
                        self.assertTrue(dialog.start_button.isDown())
                        start.assert_not_called()
                        QTest.mouseRelease(dialog.start_button, Qt.LeftButton)
                        start.assert_called_once()
            finally:
                refresh.stop()
                dialog.worker = None
                dialog.close()
                viewer.close()

    def test_six_axis_controls_still_require_permission_and_live_worker(self):
        viewer = QWidget()
        viewer.replay_mode = False
        viewer.engine = None
        dialog = RealRobotModesDialog(viewer)
        dialog.timer.stop()
        running = [True]
        try:
            dialog.mode_select.setCurrentIndex(1)
            dialog.worker = SimpleNamespace(isRunning=lambda: running[0], recorder=Mock())
            dialog.connected = True
            dialog.origin = BASE
            dialog.safety.setChecked(True)
            for blocker in ('reference', 'safety', 'busy', 'ik_worker', 'worker_stopped', 'disconnected'):
                with self.subTest(blocker=blocker):
                    dialog.busy = False
                    dialog.ik_worker = None
                    running[0] = True
                    dialog.connected = True
                    dialog.safety.setChecked(True)
                    dialog.arm_capture = ({}, 0., ('test',))
                    dialog._controls()
                    self.assertTrue(dialog.start_button.isEnabled())
                    if blocker == 'reference':
                        dialog.arm_capture = None
                    elif blocker == 'safety':
                        dialog.safety.setChecked(False)
                    elif blocker == 'busy':
                        dialog.busy = True
                    elif blocker == 'ik_worker':
                        dialog.ik_worker = object()
                    elif blocker == 'worker_stopped':
                        running[0] = False
                    else:
                        dialog.connected = False
                    dialog._controls()
                    self.assertFalse(dialog.start_button.isEnabled())
                    if blocker != 'reference':
                        self.assertFalse(dialog.capture_button.isEnabled())
        finally:
            dialog.worker = None
            dialog.ik_worker = None
            dialog.busy = False
            dialog.close()
            viewer.close()


class StartDiagnosticsTests(unittest.TestCase):
    def exercise(self, change, expected_code):
        viewer=QWidget()
        viewer.replay_mode=False
        viewer.engine=None
        dialog=RealRobotModesDialog(viewer)
        dialog.timer.stop()
        dialog.mode_select.setCurrentIndex(1)
        arm,_,_=load_profile()
        model=SceneArm(arm,[-np.pi]*6,[np.pi]*6)
        reference=sample_at(model,np.zeros(6))
        with tempfile.TemporaryDirectory() as directory:
            recorder=Recorder(Path(directory)/'diagnostics.jsonl')
            serial=SimpleNamespace(recorder=recorder,isRunning=lambda:True,begin_control=Mock())
            try:
                dialog.worker=serial
                dialog.connected=True
                dialog.origin=dialog.last_frame=BASE
                for row in dialog.axis_rows:
                    row[0].setChecked(False)
                dialog.axis_rows[3][0].setChecked(True)
                dialog.safety.setChecked(True)
                dialog.arm_capture=(deepcopy(reference),time.monotonic()+.25,('same',))
                dialog.arm_sample=(deepcopy(reference),time.monotonic()+.25,('same',))
                dialog.arm_ages={'upper':10.,'forearm':12.,'hand':15.}
                change(dialog)
                dialog._start()
                self.assertFalse(dialog.busy)
                if expected_code!='initialization_failed':
                    serial.begin_control.assert_not_called()
                self.assertIn('六轴未启动',dialog.start_notice.text())
                recorder.close()
                rows=list(map(json.loads,recorder.path.read_text().splitlines()))
                attempt=next(r for r in rows if r['event']=='arm_start_attempt')
                rejected=next(r for r in rows if r['event']=='arm_start_rejected')
                self.assertEqual(rejected['reason_code'],expected_code)
                self.assertEqual(attempt['attempt'],rejected['attempt'])
                self.assertEqual(attempt['enabled'],[False,False,False,True,False,False])
                return rejected
            finally:
                recorder.close()
                dialog.worker=None
                dialog.busy=False
                dialog.close()
                viewer.close()

    def test_each_segment_movement_is_logged_with_exact_deltas(self):
        for segment in ('upper_arm','forearm','hand'):
            def change(d, segment=segment):
                side=d.arm_sample[0]
                if segment=='hand':
                    side['hand']=side['hand']@Rotation.from_euler('x',5,degrees=True).as_matrix()
                else:
                    side['rotations'][segment]=side['rotations'][segment]@Rotation.from_euler('x',5,degrees=True).as_matrix()
            row=self.exercise(change,'reference_moved')
            self.assertAlmostEqual(row['orientation_delta_deg'][segment],5.,places=6)
            for key,value in row['orientation_delta_deg'].items():
                if key!=segment:
                    self.assertAlmostEqual(value,0.,places=6)

    def test_expired_missing_signature_and_safety_are_distinct(self):
        def expired(d):
            d.arm_sample=(d.arm_sample[0],time.monotonic()-.02,d.arm_sample[2])
        row=self.exercise(expired,'input_expired')
        self.assertGreater(row['input_age_ms'],250.)
        def missing(d):
            d.arm_sample=None
            d.arm_input_error='left_hand 数据不新鲜'
        self.assertIn('left_hand',self.exercise(missing,'missing_input')['reason'])
        def signature(d):
            d.arm_sample=(d.arm_sample[0],d.arm_sample[1],('changed',))
        self.assertFalse(self.exercise(signature,'signature_changed')['signature_matches'])
        # Suppress checkbox's normal invalidation to isolate the startup guard.
        def safety(d):
            d.safety.blockSignals(True)
            d.safety.setChecked(False)
            d.safety.blockSignals(False)
        self.exercise(safety,'safety_not_confirmed')

    def test_constructor_failure_is_logged(self):
        def failure(d):
            d.worker.begin_control.side_effect=RuntimeError('test initialization error')
        row=self.exercise(failure,'initialization_failed')
        self.assertIn('test initialization error',row['reason'])
