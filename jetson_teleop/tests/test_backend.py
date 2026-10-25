"""Production background pipeline: synthetic UDP + PTY, never real robot ports."""
import copy
import json
import os
from pathlib import Path
import pty
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from test_jetson import APP, Window, GuiUdpSource, raw_config, checkpoint, MODES, PARTS, packet, load
from jetson_teleop.backend import LiveIK
from jetson_teleop.imu_manager import ImuManager
from optusbproj_ubuntu.real_robot_protocol import encode
from optusbproj_ubuntu.real_robot_recording import Recorder
from optusbproj_ubuntu.real_robot_transport import ElbowSerialWorker
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from optusbproj_ubuntu.real_robot_stop_gesture import EndGesture, GestureConfig
from PySide6.QtWidgets import QMessageBox


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for mode in MODES:
            (self.root/(mode+'.json')).write_text(json.dumps(checkpoint(mode)))
        (self.root/'runtime.json').write_text(json.dumps(raw_config()))
        self.source = GuiUdpSource(0,'127.0.0.1')
        self.window = Window(self.source, load(self.root/'runtime.json'))
        self.service = self.window.pose_service
        self.feed_stop = threading.Event(); self.feed_pause = threading.Event()
        self.sender = threading.Thread(target=self.feed, daemon=True); self.sender.start()
        self.master = self.slave = None
        self.jobs = []
        self.writers = []
        self.wait(lambda:len(self.window.source.seen_devices()) == 6)
        self.service.zero(list(PARTS))

    def feed(self):
        tick = 0
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
            while not self.feed_stop.is_set():
                if not self.feed_pause.is_set():
                    tick += 20
                    for device in self.window.config['assignments'].values():
                        sender.sendto(packet(device,tick),self.source.sock.getsockname())
                self.feed_stop.wait(.02)

    def wait(self, predicate, timeout=5., gui=True):
        end = time.monotonic()+timeout
        while time.monotonic()<end:
            if gui: APP.processEvents()
            if predicate(): return
            time.sleep(.003)
        self.fail('background chain timed out')

    def tearDown(self):
        d = self.window.control
        d._disconnect()
        for job in self.jobs:
            job.shutdown(); job.wait(3000)
        for writer in self.writers:
            writer.shutdown(); writer.wait(3000)
        if d.worker is not None:
            self.wait(lambda:d.worker is None and d.ik_worker is None)
        self.feed_stop.set(); self.sender.join(2)
        self.window.close(); self.window.stop_backend(); APP.processEvents()
        self.source.close()
        if self.master is not None: os.close(self.master); os.close(self.slave)
        self.temp.cleanup()

    def test_acquisition_continues_without_gui_and_snapshots_are_detached(self):
        sample, signature, version = self.service.sample(('left',))
        rotation = sample['sides']['left']['hand'].copy()
        sample['sides']['left']['hand'][:] = 99.
        time.sleep(.2)  # Intentionally no processEvents.
        after, _, newer = self.service.sample(('left',))
        self.assertGreater(newer,version)
        self.assertTrue((after['sides']['left']['hand'] == rotation).all())
        self.assertLess(max(after['device_ages_ms'].values()),100.)
        self.assertEqual(self.source._diagnostic_sink_errors,0)
        completed = self.service.zero(['left_hand'])
        self.assertEqual(completed,['left_hand'])
        self.assertNotEqual(signature,self.service.sample(('left',))[1])

    def test_read_only_diagnostic_button_records_live_snapshot_without_uart(self):
        from jetson_teleop.diagnostics import DiagnosticLog
        from jetson_teleop.replay_ik import replay
        w=self.window
        def temporary_log(root,settings,seconds):
            return DiagnosticLog(self.root/'diagnostics',settings,seconds)
        with patch('jetson_teleop.diagnostics.DiagnosticLog',side_effect=temporary_log),patch(
                'serial.Serial',side_effect=AssertionError('diagnostic must not open serial')):
            w.diagnostic_button.click()
            self.assertIsNotNone(w.diagnostic_log)
            path=w.last_diagnostic
            self.wait(lambda:w.diagnostic_log.queue.qsize()==0)
            time.sleep(.15);APP.processEvents()
            w.diagnostic_button.click()
            self.assertIsNone(w.diagnostic_log)
            self.assertIn('已保存',w.diagnostic_status.text())
            self.assertIsNone(w.control.worker)
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        for event in ('metadata','imu_packet','imu_parsed','pose_snapshot','recording_end'):
            self.assertIn(event,[r['event'] for r in rows])
        self.assertTrue(rows[-1]['complete'])
        self.assertGreater(replay(path,max_frames=2)['frames'],0)

    def test_ap_and_zero_commands_execute_only_on_owner_thread(self):
        ids = []; key = self.window.config['assignments'][PARTS[0]]
        with patch.object(self.source,'switch_to_ap_udp',side_effect=lambda k:ids.append((k,threading.get_ident()))):
            self.window.source.switch_to_ap_udp(key)
        self.assertEqual(ids,[(key,self.service.thread.ident)])
        with patch.object(self.service.engine,'zero',wraps=self.service.engine.zero) as zero:
            self.service.zero(['left_hand']); zero.assert_called_once_with('left_hand')

    def test_manager_reads_all_six_devices_from_background_proxy(self):
        with patch.object(ImuManager,'refresh_network'):
            manager = ImuManager(self.window)
        try:
            manager.render()
            self.assertEqual(manager.table.rowCount(),6)
            self.assertTrue(all(manager.table.item(i,7).text()=='在线' for i in range(6)))
            self.assertEqual(manager.source.sock.getsockname(),self.source.sock.getsockname())
        finally:
            manager.timer.stop(); manager.network_timer.stop(); manager.close()

    def virtual_writer(self):
        self.master,self.slave = pty.openpty()
        writer = ElbowSerialWorker(os.ttyname(self.slave),Recorder(self.root/'control.jsonl'))
        writer.start(); self.wait(lambda:writer.isRunning())
        self.writers.append(writer)
        return writer

    def start_mock(self, solver=None, gesture=False):
        writer = self.virtual_writer(); captured = self.capture()
        limits = ArmLimits(enabled=(False,True,False,True,False,False))
        control = SharedArmControl(encode([90.]*15,[0]*10),limits,[90.]*6)
        if gesture:
            control.gesture = EndGesture(GestureConfig((90.,90.),(5.,5.),.15,(7,9)))
        writer.begin_control(control,[90.]*6,captured[1])
        if solver is None:
            solver = Mock()
            solver.solve.return_value = dict(target=[90.]*6,
                position_error_mm=0.,orientation_error_deg=None)
        job = LiveIK(solver,self.service,writer,('left',),captured,2.)
        self.jobs.append(job); job.start()
        return writer, job, control

    def test_short_gap_recovers_without_gui_and_gesture_still_stops(self):
        writer, job, control = self.start_mock(gesture=True)
        self.wait(lambda:writer.input_snapshot()['phase']=='await_confirmation',gui=False)
        self.feed_pause.set()
        try:
            self.wait(lambda:writer.input_snapshot()['mode']=='gap',timeout=1.,gui=False)
        finally:
            self.feed_pause.clear()
        self.wait(lambda:writer.input_snapshot()['mode']=='active',gui=False)
        self.assertTrue(writer.active)
        writer.confirm_follow()
        self.wait(lambda:control.gesture.triggered,gui=False)
        self.wait(lambda:job.cancelled.is_set(),gui=False)
        self.wait(lambda:not writer.active,gui=False)

    def test_manual_pause_rejects_inflight_solve_without_gui(self):
        entered = threading.Event(); release = threading.Event()
        def solve(payload):
            entered.set(); release.wait(2.)
            return dict(target=[90.]*6,position_error_mm=0.,orientation_error_deg=None)
        writer, job, _ = self.start_mock(Mock(solve=solve))
        self.assertTrue(entered.wait(1.))
        writer.pause('manual test')
        release.set(); self.assertTrue(job.wait(2000))
        self.assertIsNone(writer.target_received_at)
        self.assertFalse(writer.update_arm_target([90.]*6,time.monotonic()+.25,
                                                task_id=writer.task_id+1))

    def test_owner_failure_pauses_even_with_blocked_solver(self):
        entered = threading.Event(); release = threading.Event()
        def solve(payload):
            entered.set(); release.wait(2.)
            return dict(target=[90.]*6,position_error_mm=0.,orientation_error_deg=None)
        writer, job, _ = self.start_mock(Mock(solve=solve))
        self.assertTrue(entered.wait(1.))
        with patch.object(self.source,'poll_timed',side_effect=RuntimeError('synthetic acquisition fault')):
            self.wait(lambda:job.cancelled.is_set(),gui=False)
        release.set(); self.assertTrue(job.wait(2000))
        self.assertIn('synthetic acquisition fault',writer.pause_reason)
        self.assertIsNone(writer.target_received_at)
        self.window.source.remove_diagnostic_sink(lambda row:None)

    def capture(self, sides=('left',)):
        sample, signature, _ = self.service.sample(sides)
        payload = sample['sides']['left'] if sides == ('left',) else sample['sides']
        return (payload, sample['at']+.25-max(sample['device_ages_ms'].values())/1000, signature)

    def test_hard_timeout_latches_and_late_result_does_not_recover(self):
        writer = self.virtual_writer(); captured = self.capture()
        baseline = encode([90.]*15,[0]*10)
        limits = ArmLimits(enabled=(False,False,False,True,False,False))
        writer.begin_control(SharedArmControl(baseline,limits,[90.]*6),[90.]*6,captured[1])
        solver = Mock(); solver.solve.return_value = dict(target=[90.]*6,
                    position_error_mm=0.,orientation_error_deg=None)
        job = LiveIK(solver,self.service,writer,('left',),captured,2.)
        self.jobs.append(job); job.start()
        self.wait(lambda:writer.target_received_at is not None)
        self.feed_pause.set()
        self.wait(lambda:not writer.active,timeout=2.,gui=False)
        self.assertIn('500 ms',writer.last_state['reason'])
        received = writer.target_received_at
        self.feed_pause.clear(); time.sleep(.25)
        self.assertFalse(writer.active)
        self.assertEqual(received,writer.target_received_at)
        self.assertFalse(writer.update_arm_target([90.]*6,time.monotonic()+.25,task_id=writer.task_id))

    def test_cancel_inflight_solve_and_signature_change_cannot_publish(self):
        writer = self.virtual_writer(); captured = self.capture()
        baseline = encode([90.]*15,[0]*10)
        limits = ArmLimits(enabled=(False,False,False,True,False,False))
        writer.begin_control(SharedArmControl(baseline,limits,[90.]*6),[90.]*6,captured[1])
        entered = threading.Event(); release = threading.Event()
        def solve(side):
            entered.set(); release.wait(2.)
            return dict(target=[90.]*6,position_error_mm=0.,orientation_error_deg=None)
        job = LiveIK(Mock(solve=solve),self.service,writer,('left',),captured,2.)
        self.jobs.append(job); job.start(); self.assertTrue(entered.wait(1.))
        self.service.zero(['left_hand'])  # Same angles, but a NEW reference generation.
        self.wait(lambda:job.cancelled.is_set(),gui=False)
        release.set(); self.assertTrue(job.wait(2000))
        self.assertIsNone(writer.target_received_at)
        self.assertIn('变化',writer.pause_reason)

    def test_production_gui_auto_takeover_both_modes_and_750ms_gui_stall(self):
        d = self.window.control
        for mode in (2,3):
            with self.subTest(mode=mode):
                d.mode_select.setCurrentIndex(mode)
                self.window.review.setChecked(True)
                for delay in self.window.zero_delays.values(): delay.setValue(0)
                self.window.zero('arms'); self.window.zero('hands')
                self.assertTrue(self.window.zero_confirmed)
                d.shared_palm.setChecked(True); d.right_options[0].setChecked(True)
                for rows in (d.axis_rows,d.right_axis_rows):
                    for i,row in enumerate(rows): row[0].setChecked(i == 3)
                d.auto_start.widgets['countdown'].setValue(1.)
                d.auto_start.widgets['settle'].setValue(.1)
                master,slave = pty.openpty()
                try:
                    writer = ElbowSerialWorker(os.ttyname(slave),Recorder(self.root/f'gui-{mode}.jsonl'),d)
                    d.worker = writer
                    for signal,slot in ((writer.ready,d._ready),(writer.progress,d._progress),
                                        (writer.sent,d._sent),(writer.failed,d._failed),(writer.finished,d._finished)):
                        signal.connect(slot)
                    writer.start(); self.wait(lambda:d.connected)
                    d.origin = d.last_frame = encode([90.]*15,[0]*10)
                    d.safety.setChecked(True)
                    self.wait(lambda:d.arm_sample is not None)
                    with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):
                        d.auto_start.arm()
                        self.wait(lambda:d.shared_phase=='following',timeout=12.)
                    before = writer.target_received_at
                    time.sleep(.75)  # Longer than hard timeout; Qt delivery is not required.
                    self.assertGreater(writer.target_received_at,before)
                    self.assertEqual(writer.last_state['mode'],'active')
                    d._pause('test'); self.wait(lambda:not d.busy and d.ik_worker is None)
                    d._disconnect(); self.wait(lambda:d.worker is None)
                    events = [json.loads(r) for r in (self.root/f'gui-{mode}.jsonl').read_text().splitlines()]
                    self.assertFalse(any(r['event']=='input_timeout' for r in events))
                    self.assertTrue(any(r['event']=='arm_ik' and r['timing'].get('backend') for r in events))
                finally:
                    d._disconnect(); self.wait(lambda:d.worker is None and d.ik_worker is None)
                    os.close(master); os.close(slave)
