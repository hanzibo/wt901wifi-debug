from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from optusbproj_ubuntu.gripper_gesture import PairBuffer, GestureCalibration, static_mean


def rot(v):
    return Rotation.from_rotvec(np.deg2rad(v)).as_matrix()


class GestureTests(unittest.TestCase):
    def test_independent_worlds_mount_and_common_motion_cancel(self):
        h0=rot([12,30,-20]);f0=rot([-8,17,42]);k=rot([0,0,90])
        cal=GestureCalibration(h0,f0,k)
        # The fixed reference conversion constructed from the open mount.
        c=h0@k@f0.T
        h=rot([15,-20,50])@h0
        f=c.T@h@rot([0,70,0])@k
        cal.set_closed(h,f)
        self.assertAlmostEqual(cal.closed_deg,70)
        for common in (np.eye(3),rot([25,40,-50])):
            h1=common@h
            f1=c.T@common@c@f
            result=cal.evaluate(h1,f1)
            self.assertAlmostEqual(result.angle_deg,70)
            self.assertAlmostEqual(result.residual_deg,0,places=6)
            self.assertAlmostEqual(result.closure,1)
        r=cal.evaluate(h0,f0)
        self.assertAlmostEqual(r.closure,0)

    def test_signed_angle_offaxis_and_endpoint_rejection(self):
        cal=GestureCalibration(np.eye(3),np.eye(3),np.eye(3))
        with self.assertRaises(ValueError):cal.set_closed(np.eye(3),rot([0,5,0]))
        cal.set_closed(np.eye(3),rot([0,80,0]))
        r=cal.evaluate(np.eye(3),rot([0,40,0]))
        self.assertAlmostEqual(r.closure,.5)
        r=cal.evaluate(np.eye(3),rot([0,-20,0]))
        self.assertAlmostEqual(r.angle_deg,-20);self.assertEqual(r.closure,0)
        self.assertAlmostEqual(cal.evaluate(np.eye(3),rot([20,0,0])).residual_deg,20)
        with self.assertRaises(ValueError):cal.evaluate(np.eye(3),rot([0,175,0]))
        with self.assertRaises(ValueError):GestureCalibration(np.eye(3),np.eye(3),np.diag([1,1,-1]))

    def test_pair_bracketing_stale_gap_duplicate(self):
        b=PairBuffer();b.append('h',1.,rot([0,0,0]));b.append('h',1.04,rot([0,0,40]))
        b.append('f',1.02,np.eye(3));b.append('f',1.02,np.eye(3))
        stamp,h,f=b.pair('h','f',1.05)
        self.assertEqual(stamp,1.02)
        np.testing.assert_allclose(h,rot([0,0,20]),atol=1e-10)
        self.assertEqual(len(b.rows['f']),1)
        for args in (('h','h',1.05),('h','f',2),('h','missing',1.05)):
            with self.assertRaises(ValueError):b.pair(*args)
        b.append('h',1.3,np.eye(3));b.append('f',1.2,np.eye(3))
        with self.assertRaisesRegex(ValueError,'间隔过大'):b.pair('h','f',1.3)

    def test_static_samples_reject_motion_and_short_windows(self):
        with self.assertRaises(ValueError):static_mean([np.eye(3)]*9)
        with self.assertRaises(ValueError):static_mean([rot([0,0,i]) for i in range(20)])
        np.testing.assert_allclose(static_mean([np.eye(3)]*15),np.eye(3))


class GestureGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_discovery_selection_calibration_record_and_invalidation(self):
        from optusbproj_ubuntu.gripper_gesture_gui import GestureWindow
        from optusbproj_ubuntu.pose import VENDOR_RPY
        import time
        import json
        class Source:
            packets=[]
            def poll_timed(self):rows=self.packets;self.packets=[];return rows
            def close(self):pass
        def packet(device,ms,pitch):
            return SimpleNamespace(device_id=device,roll=0.,pitch=pitch,yaw=0.,hour=0,minute=0,
                second=ms//1000,millisecond=ms%1000,gyro_x=0.,gyro_y=0.,gyro_z=0.,acc_x=0.,acc_y=0.,acc_z=1.,battery_percent=90)
        with TemporaryDirectory() as tmp:
            source=Source();w=GestureWindow(source,tmp);w.timer.stop()
            try:
                w.mode.setCurrentIndex(w.mode.findData(VENDOR_RPY))
                def feed(ms,pitch):
                    ns=time.monotonic_ns();source.packets=[(ns,packet('arbitrary-H',ms,0)),(ns+1,packet('arbitrary-F',ms,pitch))];w.tick()
                feed(0,0)
                self.assertEqual(w.hand.currentData(),'')
                self.assertEqual(w.hand.count(),3)
                w.hand.setCurrentIndex(w.hand.findData('arbitrary-H'));w.finger.setCurrentIndex(w.finger.findData('arbitrary-F'))
                feed(20,0);feed(40,0)
                w.start_capture('open');self.assertIsNotNone(w.capture)
                w.capture['state']='hold';w.capture['deadline']=time.monotonic()+10
                for i in range(10):feed(60+i*20,0)
                w.capture['deadline']=0;feed(250,0)
                self.assertIsNotNone(w.calibration)
                w.start_capture('closed');w.capture['state']='hold';w.capture['deadline']=time.monotonic()+10
                for i in range(10):feed(300+i*20,70)
                w.capture['deadline']=0;feed(550,70)
                self.assertAlmostEqual(w.calibration.closed_deg,70,delta=1e-3)
                w.toggle_record();feed(570,35);w.stop_record()
                rows=[json.loads(line) for line in (w.record_path/'diagnostics.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['hand'],'arbitrary-H')
                self.assertTrue(any(r['kind']=='imu' for r in rows));self.assertTrue(any(r['kind']=='gesture' for r in rows))
                from unittest.mock import patch
                from optusbproj_ubuntu.gripper_gesture import GestureResult
                self.assertFalse(w.warn_only.isChecked())
                w.toggle_record()
                with patch.object(w.calibration,'evaluate',return_value=GestureResult(70.,25.,1.)):
                    previous=w.filtered
                    feed(590,35);self.assertEqual(w.filtered,previous)
                    w.warn_only.setChecked(True)
                    time.sleep(.005);feed(610,35)
                    self.assertGreater(w.filtered,previous)
                    self.assertIn('继续更新',w.status.text())
                    w.stop_record()
                    rows=[json.loads(line) for line in (w.record_path/'diagnostics.jsonl').read_text().splitlines()]
                    gesture=next(r for r in rows if r['kind']=='gesture' and r['residual_warn_only'])
                    self.assertFalse(gesture['valid']);self.assertTrue(gesture['update_allowed'])
                    self.assertTrue(gesture['residual_warn_only'])
                    self.assertTrue(any(r['kind']=='residual_policy' and r['warn_only'] for r in rows))
                # Reboot clock discontinuity retires paired installation refs.
                feed(0,0);self.assertIsNone(w.calibration)
                w.mode.setCurrentIndex(0);self.assertIsNone(w.calibration);self.assertFalse(w.buffer.rows)
                w.show();self.app.processEvents()
                self.assertFalse(w.grab().isNull())
            finally:w.close()
