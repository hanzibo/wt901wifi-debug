"""Offline Qt + real pyserial/PTY tests; never touch a physical port."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OptUSBProj-ubuntu'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import math
import json
import pty
import random
import select
import struct
import time
import tempfile
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QMessageBox
from real_robot_manual.protocol import encode, decode
from real_robot_manual.transport import SerialWorker
from real_robot_manual.gui import Window
from real_robot_manual.trajectory import Trajectory
from real_robot_manual.recording import Recorder
from real_robot_manual.live import LiveTrajectory

APP = QApplication.instance() or QApplication([])


def wait_until(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if predicate():return
        time.sleep(.005)
    raise AssertionError('timed out')


class ProtocolTests(unittest.TestCase):
    def test_golden_zero_fingers(self):
        frame = encode([90.]*15, [0]*10)
        self.assertEqual(len(frame),54)
        self.assertEqual(frame,b'\xaa\x55'+b'\x28\x23'*15+bytes(20)+b'\x64\x05')
        self.assertEqual(decode(frame),([90.]*15,[0]*10))

    def test_random_roundtrip_and_independent_packing(self):
        rng = random.Random(13)
        for _ in range(1000):
            centidegrees = [rng.randrange(0,32768) for _ in range(15)]
            fingers = [rng.randrange(4096) for _ in range(10)]
            frame = encode([v/100 for v in centidegrees],fingers)
            self.assertEqual(frame[2:32],struct.pack('<15h',*centidegrees))
            for i, group in enumerate((fingers[:5],fingers[5:])):
                bits = ''.join(f'{value:012b}' for value in group)+'0000'
                self.assertEqual(frame[32+8*i:40+8*i],int(bits,2).to_bytes(8,'big'))
            self.assertEqual(decode(frame),([v/100 for v in centidegrees],fingers))

    def test_invalid_values(self):
        for value in (math.nan,math.inf,-.1,327.68,360.,True):
            with self.assertRaises(ValueError):encode([value]+[90.]*14,[0]*10)
        for value in (-1,4096,1.5,True):
            with self.assertRaises(ValueError):encode([90.]*15,[value]+[0]*9)
        with self.assertRaises(ValueError):encode([90.]*12,[0]*10)
        frame = bytearray(encode([90.]*15,[0]*10));frame[4]^=1
        with self.assertRaises(ValueError):decode(frame)


class WorkerTests(unittest.TestCase):
    def test_live_latest_mailbox_requires_permission_and_logs_virtual_writes(self):
        master, slave = pty.openpty()
        worker = SerialWorker(os.ttyname(slave))
        ready, sent = [], []
        worker.ready.connect(lambda:ready.append(True)); worker.sent.connect(sent.append)
        baseline = encode([90.]*15, [0]*10)
        motion = LiveTrajectory(baseline, 9, 85, 95, 10, 20, 25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'live.jsonl'
            worker.recorder = Recorder(path)
            try:
                with self.assertRaises(ValueError):worker.retarget_live(91)
                worker.enable(True); worker.submit_motion(motion)
                for _ in range(10000):worker.retarget_live(95)
                worker.retarget_live(89)
                self.assertEqual(worker.live_target, 89)
                worker.start(); wait_until(lambda:ready and len(sent)>4)
                worker.enable(False)
                with self.assertRaises(ValueError):worker.retarget_live(90)
                worker.stop(); self.assertTrue(worker.wait(2000))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertTrue(any(row['event']=='live_start' for row in rows))
                targets = [row['target_deg'] for row in rows if row['event']=='live_target']
                self.assertEqual(targets, [89])
                writes = [row for row in rows if row['event']=='write']
                self.assertEqual(bytes.fromhex(writes[0]['frame_hex']), baseline)
                self.assertTrue(all(row['os_write_complete'] for row in writes))
                self.assertTrue(all(row['interval_ms']>=39 for row in writes[1:]))
                self.assertEqual(os.read(master,65536), b''.join(bytes.fromhex(row['frame_hex']) for row in writes))
            finally:
                worker.stop(); worker.wait(2000); os.close(master); os.close(slave)

    def test_recorded_trajectory_matches_virtual_serial(self):
        master,slave=pty.openpty();worker=SerialWorker(os.ttyname(slave))
        ready,done=[],[]
        worker.ready.connect(lambda:ready.append(True));worker.motion_done.connect(lambda:done.append(True))
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'commands.jsonl');worker.recorder=Recorder(path)
            motion=Trajectory(encode([90.]*15,[0]*10),encode([91.]+[90.]*14,[0]*10),10,20,200,400,25)
            try:
                worker.start();wait_until(lambda:ready)
                self.assertFalse(select.select([master],[],[],.03)[0])
                worker.enable(True);worker.submit_motion(motion);wait_until(lambda:done)
                worker.stop();self.assertTrue(worker.wait(2000))
                with open(path,encoding='utf-8') as source:events=[json.loads(line) for line in source]
                writes=[e for e in events if e['event']=='write']
                self.assertEqual(len(writes),motion.steps)
                self.assertEqual(os.read(master,4096),b''.join(bytes.fromhex(e['frame_hex']) for e in writes))
                for i,item in enumerate(writes):
                    self.assertTrue(item['os_write_complete']);self.assertEqual(item['bytes_written'],54)
                    self.assertEqual(item['step'],i+1);self.assertGreaterEqual(item['write_duration_ms'],0)
                    if i:self.assertGreaterEqual(item['interval_ms'],39)
                self.assertEqual(next(e for e in events if e['event']=='motion_start')['frequency_hz'],25)
                self.assertIn('motion_complete',[e['event'] for e in events])
                self.assertEqual(events[-1]['event'],'serial_close')
            finally:
                worker.stop();worker.wait(2000);os.close(master);os.close(slave)

    def test_partial_write_is_recorded_as_uncertain(self):
        class Fake:
            in_waiting=0
            def open(self):pass
            def close(self):pass
            def write(self,frame):return 3
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'commands.jsonl')
            worker=SerialWorker('/dev/fake',factory=lambda **kwargs:Fake())
            worker.recorder=Recorder(path)
            worker.enable(True);worker.submit(encode([90.]*15,[0]*10))
            worker.start();self.assertTrue(worker.wait(2000))
            with open(path,encoding='utf-8') as source:events=[json.loads(line) for line in source]
            writes=[e for e in events if e['event']=='write']
            self.assertEqual(len(writes),1);self.assertEqual(writes[0]['bytes_written'],3)
            self.assertFalse(writes[0]['os_write_complete'])
            self.assertTrue(next(e for e in events if e['event']=='serial_error')['write_result_uncertain'])

    def test_smooth_pty_and_revoke(self):
        master, slave = pty.openpty(); worker = SerialWorker(os.ttyname(slave))
        ready, sent, done = [], [], []
        worker.ready.connect(lambda:ready.append(True));worker.sent.connect(sent.append)
        worker.motion_done.connect(lambda:done.append(True))
        start = encode([90.]*15,[0]*10)
        target = encode([91.]+[90.]*14,[0]*10)
        motion = Trajectory(start,target,10,20,200,400)
        try:
            worker.start();wait_until(lambda:ready)
            worker.enable(True);worker.submit_motion(motion)
            wait_until(lambda:done)
            self.assertEqual(sent,[motion.frame(i) for i in range(1,motion.steps+1)])
            self.assertEqual(sent[-1],target)
            worker.enable(True);worker.submit_motion(Trajectory(target,start,1,2,200,400))
            count=len(sent);wait_until(lambda:len(sent)>count)
            worker.enable(False);APP.processEvents();count=len(sent)
            time.sleep(.08);APP.processEvents()
            self.assertEqual(len(sent),count)
        finally:
            worker.stop();worker.wait(1000);os.close(master);os.close(slave)

    def test_mailbox_requires_unlock_and_revokes(self):
        worker = SerialWorker('/dev/NOT_A_DEVICE')
        frame = encode([90.]*15,[0]*10)
        with self.assertRaises(ValueError):worker.submit(frame)
        worker.enable(True)
        for _ in range(10000):worker.submit(frame)
        self.assertEqual(worker.pending[0],frame)
        worker.enable(False);self.assertIsNone(worker.pending)
        worker.enable(True);worker.submit(frame);worker.stop()
        self.assertIsNone(worker.pending)
        with self.assertRaises(ValueError):worker.submit(frame)

    def test_real_pyserial_pty_no_automatic_write_and_exact_frame(self):
        master, slave = pty.openpty();worker = SerialWorker(os.ttyname(slave))
        ready, sent, failures = [], [], []
        worker.ready.connect(lambda:ready.append(True))
        worker.sent.connect(sent.append);worker.failed.connect(failures.append)
        try:
            worker.start();wait_until(lambda:ready or failures)
            self.assertEqual(failures,[])
            self.assertFalse(select.select([master],[],[],.05)[0])
            frame = encode([90.]*15,list(range(10)))
            worker.enable(True);worker.submit(frame)
            wait_until(lambda:sent or failures)
            self.assertEqual(failures,[]);self.assertEqual(sent,[frame])
            self.assertEqual(os.read(master,4096),frame)
            with self.assertRaises(ValueError):worker.submit(frame)
            self.assertFalse(select.select([master],[],[],.05)[0])
            worker.stop();self.assertTrue(worker.wait(1000))
            self.assertFalse(select.select([master],[],[],.05)[0])
        finally:
            worker.stop();worker.wait(1000);os.close(master);os.close(slave)

    def test_partial_write_fails_without_retry(self):
        class Fake:
            in_waiting = 0
            calls = 0
            closed = False
            def __init__(self,**kwargs):pass
            def open(self):pass
            def close(self):self.closed=True
            def write(self,frame):self.calls+=1;return 3
        fake = Fake();worker = SerialWorker('/dev/fake',factory=lambda **kw:fake)
        failures=[];worker.failed.connect(failures.append)
        worker.enable(True);worker.submit(encode([90.]*15,[0]*10))
        worker.start();wait_until(lambda:failures)
        self.assertTrue(worker.wait(1000));self.assertEqual(fake.calls,1)
        self.assertTrue(fake.closed);self.assertIn('不重试',failures[0])


class GuiTests(unittest.TestCase):
    def test_live_slider_pty_latest_target_and_channel_freezing(self):
        master, slave = pty.openpty()
        window = Window(); window.ports.setEditText(os.ttyname(slave))
        try:
            with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):
                window.connect_serial()
            wait_until(lambda:window.connected)
            with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):
                window.confirm_origin()
            window.joints[0].setValue(110) # Unsent unrelated target must not move R1.
            window.live_mode.setChecked(True)
            window.confirm.setChecked(True); window.arm.setChecked(True)
            with patch.object(QMessageBox, 'warning', return_value=QMessageBox.No):window.start_live()
            self.assertFalse(select.select([master], [], [], .03)[0])
            with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes):window.start_live()
            self.assertTrue(window.live_active)
            self.assertEqual(window.joints[0].value(), 90)
            self.assertFalse(window.joints[0].isEnabled())
            self.assertFalse(window.fingers[0].isEnabled())
            self.assertTrue(window.sliders[9].isEnabled())
            window.sliders[9].setValue(10000)
            self.assertEqual(window.joints[9].value(), 95)
            window.sliders[9].setValue(9200)
            wait_until(lambda:decode(window.origin)[0][9] > 90.1)
            self.assertTrue(window.arm.isChecked())
            for value in range(9300, 9000, -1):window.sliders[9].setValue(value)
            window.sliders[9].setValue(8900)
            wait_until(lambda:abs(decode(window.origin)[0][9]-89) < .02)
            data = os.read(master, 65536)
            self.assertEqual(len(data)%54, 0)
            self.assertEqual(data[:54], encode([90.]*15, [0]*10))
            for i in range(0, len(data), 54):
                angles, fingers = decode(data[i:i+54])
                self.assertEqual(angles[:9]+angles[10:], [90.]*14)
                self.assertEqual(fingers, [0]*10)
                self.assertTrue(85 <= angles[9] <= 95)
            window.arm.setChecked(False)
            self.assertFalse(window.live_active)
            self.assertIsNone(window.origin)
            time.sleep(.08); APP.processEvents()
            while select.select([master], [], [], 0)[0]:os.read(master, 65536)
            self.assertFalse(select.select([master], [], [], .08)[0])
        finally:
            if window.worker:window.disconnect_serial();wait_until(lambda:window.worker is None)
            window.close();os.close(master);os.close(slave)

    def test_live_requires_valid_envelope_and_explicit_unlock(self):
        window = Window()
        try:
            window.live_mode.setChecked(True)
            self.assertFalse(window.live_button.isEnabled())
            window.sliders[9].setValue(9200)
            self.assertIsNone(window.worker)
            self.assertFalse(window.live_active)
            window.live_axis.setCurrentIndex(1)
            self.assertEqual(window.live_lower.value(), 90)
            self.assertEqual(window.live_upper.value(), 95)
        finally:window.close()

    def test_interruption_and_error_invalidate_origin(self):
        window=Window()
        try:
            window.connected=True
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):window.confirm_origin()
            self.assertIsNotNone(window.origin)
            window.confirm.setChecked(True);window.arm.setChecked(True)
            window.executing=True
            window.joints[0].setValue(91.)
            self.assertFalse(window.executing);self.assertIsNone(window.origin)
            window.origin=encode([90.]*15,[0]*10)
            window.on_failed('test error')
            self.assertIsNone(window.origin);self.assertFalse(window.connected)
        finally:window.close()

    def test_no_startup_io_slider_is_preview_only(self):
        with patch('real_robot_manual.gui.SerialWorker') as factory:
            window=Window(worker_factory=factory)
            self.assertEqual(len(window.joints),15);self.assertEqual(len(window.fingers),10)
            self.assertFalse(window.send_button.isEnabled());self.assertIsNone(window.worker)
            window.joints[0].setValue(100.)
            self.assertEqual(decode(window.frame())[0][0],100.)
            self.assertFalse(window.confirm.isChecked());factory.assert_not_called()
            window.close()

    def test_gui_pty_cancel_send_edit_revokes_disconnect(self):
        master,slave=pty.openpty();window=Window();window.ports.setEditText(os.ttyname(slave))
        try:
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):window.connect_serial()
            wait_until(lambda:window.connected)
            self.assertFalse(window.send_button.isEnabled())
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):window.confirm_origin()
            window.confirm.setChecked(True);window.arm.setChecked(True)
            self.assertTrue(window.send_button.isEnabled())
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.No):window.send_once()
            self.assertFalse(select.select([master],[],[],.05)[0])
            window.joints[0].setValue(91.)
            self.assertFalse(window.send_button.isEnabled());self.assertFalse(window.arm.isChecked())
            window.confirm.setChecked(True);window.arm.setChecked(True)
            expected=window.frame()
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):window.send_once()
            wait_until(lambda:not window.arm.isChecked())
            data=os.read(master,4096)
            self.assertGreater(len(data),54)
            self.assertEqual(len(data)%54,0)
            self.assertEqual(data[-54:],expected)
            window.disconnect_serial();wait_until(lambda:window.worker is None)
            self.assertFalse(select.select([master],[],[],.05)[0])
        finally:
            if window.worker:window.disconnect_serial();wait_until(lambda:window.worker is None)
            window.close();os.close(master);os.close(slave)


class TrajectoryTests(unittest.TestCase):
    def test_live_random_retargets_envelope_rates_and_frozen_channels(self):
        baseline = encode([90.]*15, [0]*10)
        rng = random.Random(14)
        for axis in (0, 9, 14):
            motion = LiveTrajectory(baseline, axis, 85, 95, 10, 20, 20)
            motion.set_target(95)
            self.assertEqual(motion.frame(0), baseline)
            self.assertEqual(motion.frame(1), baseline)
            previous_v = 0.
            for step in range(2, 500):
                if step%6 == 0:motion.set_target(rng.uniform(85, 95))
                angles, fingers = decode(motion.frame(step))
                self.assertTrue(85 <= angles[axis] <= 95)
                self.assertEqual(angles[:axis]+angles[axis+1:], [90.]*14)
                self.assertEqual(fingers, [0]*10)
                self.assertLessEqual(abs(motion.v), 10+1e-7)
                self.assertLessEqual(abs(motion.v-previous_v), 20*motion.period+1e-7)
                previous_v = motion.v
            for value in (84, 96, float('nan')):
                with self.assertRaises(ValueError):motion.set_target(value)
        with self.assertRaises(ValueError):LiveTrajectory(baseline, 0, 91, 95, 10, 20, 20)

    def test_frequency_changes_sampling_not_motion_speed(self):
        start=encode([90.]*15,[0]*10);target=encode([120.]+[90.]*14,[0]*10)
        durations=[];steps=[]
        for frequency in (50,25,10):
            motion=Trajectory(start,target,10,20,200,400,frequency)
            self.assertEqual(motion.period,1/frequency)
            self.assertEqual(motion.frame(0),start)
            self.assertEqual(motion.frame(motion.steps),target)
            self.assertGreaterEqual(motion.duration,5.625)
            self.assertLess(motion.duration-5.625,motion.period+1e-9)
            durations.append(motion.duration);steps.append(motion.steps)
        self.assertLess(max(durations)-min(durations),.1)
        self.assertGreater(steps[0],steps[1]);self.assertGreater(steps[1],steps[2])
        for value in (0,101,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):Trajectory(start,target,10,20,200,400,value)

    def test_endpoints_limits_and_monotonicity(self):
        start=encode([90.]*15,[0]*10);target=encode([100.]+[90.]*14,[1000]+[0]*9)
        motion=Trajectory(start,target,10,20,200,400)
        self.assertGreaterEqual(motion.duration,1.875*1000/200)
        self.assertGreaterEqual(motion.duration,math.sqrt((10/math.sqrt(3))*1000/400))
        self.assertEqual(motion.frame(0),start);self.assertEqual(motion.frame(motion.steps),target)
        values=[decode(motion.frame(i)) for i in range(motion.steps+1)]
        for group,limit in ((0,10),(1,200)):
            sequence=[v[group][0] for v in values]
            self.assertEqual(sequence,sorted(sequence))
            quantum=.01 if group==0 else 1
            self.assertLessEqual(max((b-a)/.02 for a,b in zip(sequence,sequence[1:])),limit+quantum/.02)
        for args in ((0,20,200,400),(10,float('nan'),200,400)):
            with self.assertRaises(ValueError):Trajectory(start,target,*args)


if __name__=='__main__':unittest.main()
