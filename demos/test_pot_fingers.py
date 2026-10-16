"""POT5 integration tests: only fresh PTYs, never physical UARTs."""
import os
import sys
import time
import pty
import select
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'OptUSBProj-ubuntu'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication, QMessageBox
from real_robot_manual.gui import Window
from real_robot_manual.protocol import encode, decode
from real_robot_manual.potentiometers import FingerMap, parse_line, parse_a4, targets
from real_robot_manual.finger_live import FingerTrajectory
from real_robot_manual.transport import SerialWorker

APP=QApplication.instance() or QApplication([])


def wait(predicate, seconds=3):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        APP.processEvents()
        if predicate():return
        time.sleep(.005)
    raise AssertionError('timed out')


def mappings():
    return [FingerMap(i,100,900,0,200,i==0) for i in range(5)]


class PureTests(unittest.TestCase):
    def test_a4_ignores_disconnected_inputs(self):
        self.assertEqual(parse_a4(b'POT5,3,60,noise,-1,9999,nan,322'),(3,60,(0,0,0,0,322)))
        self.assertIsNone(parse_a4(b'POT5,3,60,0,0,0,0,1024'))
    def test_parse_strict(self):
        self.assertEqual(parse_line(b'POT5,42,840,775,650,537,422,322\r'),(42,840,(775,650,537,422,322)))
        for line in (b'BOOT,x',b'POT5,0,20,1,2,3,4',b'POT5,0,20,1024,2,3,4,5',b'POT5,-1,20,1,2,3,4,5'):
            self.assertIsNone(parse_line(line))

    def test_mapping_reverse_permutation_clamp(self):
        maps=[FingerMap(4-i,900,100,0,200) for i in range(5)]
        self.assertEqual(targets(maps,[0,100,500,900,1023]),[0,0,100,200,200])
        with self.assertRaises(ValueError):FingerMap(0,200,210)
        with self.assertRaises(ValueError):targets([maps[0]]*5,[0]*5)

    def test_random_trajectory_freezes_other_channels_and_limits(self):
        import random
        baseline=encode([90+i for i in range(15)],[0,11,22,33,44,55,66,77,88,99])
        motion=FingerTrajectory(baseline,'right',mappings(),100,200,50)
        rng=random.Random(12)
        motion.set_target({'values':[200]*5,'deadline':time.monotonic()+30})
        self.assertEqual(motion.frame(1),baseline)
        for step in range(2,1002):
            if step%37==0:motion.set_target({'values':[rng.randrange(201)]*5,'deadline':time.monotonic()+30})
            frame=motion.frame(step);angles,fingers=decode(frame)
            self.assertEqual(angles,decode(baseline)[0]);self.assertEqual(fingers[1:],decode(baseline)[1][1:])
            self.assertTrue(0<=fingers[0]<=200)
            self.assertLessEqual(abs(motion.v[0]),100+1e-5);self.assertLessEqual(abs(motion.a[0]),200+1e-5)
        motion.brake('stop')
        for step in range(1002,1102):
            motion.frame(step)
            if motion.done:break
        self.assertTrue(motion.done)
        q=motion.q.copy();motion.set_target({'values':[0]*5,'deadline':time.monotonic()+30})
        self.assertEqual(motion.q,q);self.assertTrue(motion.stopping)

    def test_watchdog_independent_of_qt_and_latched(self):
        m=FingerTrajectory(encode([90.]*15,[0]*10),'left',mappings(),200,400,50)
        m.set_target({'values':[200]*5,'deadline':0})
        m.frame(1);self.assertTrue(m.done);self.assertTrue(m.stopping)
        m.set_target({'values':[200]*5,'deadline':time.monotonic()+30})
        self.assertEqual(m.frame(2),m.baseline)


class IntegrationTests(unittest.TestCase):
    def test_a4_selects_each_of_ten_fingers(self):
        window=Window();panel=window.pot_panel
        try:
            panel.rows[0][2].setValue(100);panel.rows[0][3].setValue(900)
            panel.rows[0][0].setChecked(True)
            baseline=encode([90.]*15,[0]*10)
            for side in range(2):
                panel.side.setCurrentIndex(side)
                for finger in range(5):
                    panel.finger.setCurrentIndex(finger)
                    maps=panel.maps();self.assertEqual([m.enabled for m in maps],[i==finger for i in range(5)])
                    motion=FingerTrajectory(baseline,'left' if side==0 else 'right',maps,200,400,50)
                    panel.latest=(0,0,0,0,900)
                    motion.set_target({'values':panel.target_values(maps),'deadline':time.monotonic()+10})
                    for step in range(1,101):frame=motion.frame(step)
                    values=decode(frame)[1];expected=[0]*10;expected[(5 if side==0 else 0)+finger]=200
                    self.assertEqual(values,expected)
        finally:window.close();APP.processEvents()

    def test_worker_watchdog_stops_without_gui_events(self):
        master,slave=pty.openpty();worker=SerialWorker(os.ttyname(slave));frames=[];done=[]
        motion=FingerTrajectory(encode([90.]*15,[0]*10),'right',mappings(),100,400,50)
        motion.set_target({'values':[200]*5,'deadline':time.monotonic()+.15})
        worker.sent.connect(frames.append);worker.motion_done.connect(lambda:done.append(True))
        try:
            worker.enable(True);worker.submit_motion(motion);worker.start()
            time.sleep(.6) # Deliberately no Qt event loop; trajectory watchdog lives in writer.
            wait(lambda:done)
            self.assertTrue(motion.stopping);self.assertTrue(motion.done)
            count=len(frames);time.sleep(.08);APP.processEvents();self.assertEqual(len(frames),count)
            with self.assertRaises(ValueError):worker.retarget_live({'values':[200]*5,'deadline':time.monotonic()+1})
        finally:worker.stop();worker.wait(2000);os.close(master);os.close(slave)

    def test_buffered_old_frames_do_not_refresh_freshness(self):
        master,slave=pty.openpty();window=Window();panel=window.pot_panel
        try:
            panel.port.setEditText(os.ttyname(slave));panel.open_input()
            os.write(master,b'POT5,0,20,1,2,3,4,5\n');wait(lambda:panel.latest is not None)
            self.assertTrue(panel.fresh())
            time.sleep(.55)
            os.write(master,b'POT5,1,40,1,2,3,4,5\n');wait(lambda:panel.previous[0]==1)
            self.assertFalse(panel.fresh())
        finally:window.close();APP.processEvents();os.close(master);os.close(slave)

    def test_nano_read_only_and_capture(self):
        master,slave=pty.openpty();window=Window();panel=window.pot_panel
        try:
            self.assertIsNone(panel.connection);self.assertIsNone(window.worker)
            panel.port.setEditText(os.ttyname(slave));panel.open_input()
            os.write(master,b'BOOT,FivePotentiometers,1\nPOT5,0,20,775,650,537,422,322\n')
            wait(lambda:panel.latest is not None)
            self.assertEqual(panel.latest,(0,0,0,0,322))
            panel.capture(2);self.assertEqual(panel.rows[0][2].value(),322)
            self.assertFalse(panel.start.isEnabled());self.assertIsNone(window.worker)
            self.assertFalse(select.select([master],[],[],.05)[0]) # No Nano TX, not even commands.
        finally:window.close();APP.processEvents();os.close(master);os.close(slave)

    def test_full_nano_to_robot_virtual_serial_and_manual_stop(self):
        nm,ns=pty.openpty();rm,rs=pty.openpty();window=Window();panel=window.pot_panel
        worker=SerialWorker(os.ttyname(rs),window);frames=[];done=[]
        window.worker=worker
        worker.ready.connect(window.on_ready);worker.sent.connect(window.on_sent);worker.sent.connect(frames.append)
        worker.motion_done.connect(window.on_motion_done);worker.motion_done.connect(lambda:done.append(True))
        worker.failed.connect(lambda msg:self.fail(msg))
        try:
            worker.start();wait(lambda:window.connected)
            panel.port.setEditText(os.ttyname(ns));panel.open_input()
            os.write(nm,b'POT5,0,20,100,100,100,100,900\n');wait(lambda:panel.latest is not None)
            panel.side.setCurrentIndex(1);panel.finger.setCurrentIndex(3)
            for r in panel.rows:r[2].setValue(100);r[3].setValue(900)
            panel.rows[0][0].setChecked(True)
            window.origin=encode([90.]*15,[0]*10)
            window.confirm.setChecked(True);window.arm.setChecked(True);panel.update_controls()
            self.assertTrue(panel.start.isEnabled())
            with patch.object(QMessageBox,'warning',return_value=QMessageBox.Yes):panel.start_output()
            wait(lambda:len(frames)>=5)
            panel.stop.click();wait(lambda:done)
            self.assertIsInstance(window.motion.reason,str)
            self.assertFalse(window.pot_active);self.assertFalse(window.arm.isChecked())
            self.assertTrue(window.motion.stopping);self.assertEqual(frames[0],encode([90.]*15,[0]*10))
            self.assertTrue(any(decode(f)[1][3]>0 for f in frames))
            for frame in frames:
                angles,fingers=decode(frame);self.assertEqual(angles,[90.]*15)
                self.assertEqual(fingers[:3]+fingers[4:],[0]*9)
            self.assertEqual(os.read(rm,65536),b''.join(frames))
            self.assertFalse(select.select([nm],[],[],.02)[0])
        finally:
            worker.stop();worker.wait(2000);window.worker=None;window.close();APP.processEvents()
            for fd in (nm,ns,rm,rs):os.close(fd)


if __name__=='__main__':unittest.main()
