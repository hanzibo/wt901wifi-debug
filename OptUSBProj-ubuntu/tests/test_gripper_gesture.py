from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from optusbproj_ubuntu.gripper_gesture import PairBuffer, GestureCalibration, static_mean, DiscreteClosure


def rot(v):
    return Rotation.from_rotvec(np.deg2rad(v)).as_matrix()


class GestureTests(unittest.TestCase):
    def test_discrete_hysteresis_dwell_and_invalid_pause(self):
        d=DiscreteClosure(.3,.7,.1)
        self.assertIsNone(d.update(.5,0))
        self.assertIsNone(d.update(.8,1))
        self.assertTrue(d.update(.8,1.11))
        self.assertTrue(d.update(.4,1.2))
        self.assertTrue(d.update(.2,1.3))
        d.pause()  # A gap / residual rejection cannot count toward confirmation.
        self.assertTrue(d.update(.2,2))
        self.assertFalse(d.update(.2,2.11))
        self.assertFalse(d.update(.6,3))
        d.update(.8,4);d.update(.5,4.1)
        self.assertFalse(d.update(.8,4.2))
        for args in ((.8,.3,0),(.3,.3,0),(-.1,.7,0),(.3,.7,-1)):
            with self.assertRaises(ValueError):DiscreteClosure(*args)
        with self.assertRaises(ValueError):d.update(float('nan'),4)

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

    def test_host_pose_reuses_learning_per_packet_without_mutating_host(self):
        import time
        from unittest.mock import patch
        from PySide6 import QtWidgets
        from optusbproj_ubuntu.gripper_gesture_gui import BilateralGestureWindow, SharedGestureSource
        from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT
        host=QtWidgets.QWidget();host.engine=PoseEngine();host._show_x1_preview=lambda:None
        def packet(device,ms):
            return SimpleNamespace(device_id=device,roll=0.,pitch=0.,yaw=0.,hour=0,minute=0,second=ms//1000,millisecond=ms%1000,
                gyro_x=0.,gyro_y=0.,gyro_z=0.,acc_x=0.,acc_y=0.,acc_z=1.,battery_percent=90)
        with TemporaryDirectory() as tmp:
            # Learn in the host BEFORE opening the gripper popup.
            for ms in range(0,1100,20):
                for device in ('H','F'):host.engine.feed(packet(device,ms),received_at=time.monotonic())
            self.assertTrue(host.engine._tilt_filters['H'].bias_ready)
            w=BilateralGestureWindow(SharedGestureSource(),tmp,host);w.timer.stop()
            left=w.panels['left'];right=w.panels['right']
            try:
                for device in ('H','F'):
                    w.feed_host_packet(round(host.engine.received_at[device]*1e9),host.engine.packets[device],host.engine)
                with (patch.object(left.engine,'feed',side_effect=AssertionError('must not reintegrate')),
                      patch.object(right.engine,'feed',side_effect=AssertionError('must not reintegrate'))):
                    w.tick()
                self.assertIn('请先选择',left.status.text())
                left.hand.setCurrentIndex(left.hand.findData('H'));left.finger.setCurrentIndex(left.finger.findData('F'))
                # Host remains ready even when actively moving; no popup still window.
                first=time.monotonic();before=host.engine._tilt_filters['F'].rotation.copy()
                for index,angle in enumerate((10,20,30)):
                    stamp=round((first+index*.0001)*1e9)
                    for device in ('H','F'):
                        host.engine.mode_current[device][GYRO_TILT]=rot([0,angle if device=='F' else 0,0])
                        w.feed_host_packet(stamp,host.engine.packets[device],host.engine)
                # Mutating host's latest matrix must not overwrite old queued samples.
                host.engine.mode_current['F'][GYRO_TILT][:]=rot([0,90,0])
                with patch.object(left.engine,'feed',side_effect=AssertionError('must not reintegrate')):w.tick()
                left.selected_pair(time.monotonic()+.001)
                self.assertIn('复用主窗口：零偏就绪',left.devices.text())
                angles=[Rotation.from_matrix(r).as_euler('xyz',degrees=True)[1] for _,r in left.buffer.rows['F']]
                np.testing.assert_allclose(angles, [10,20,30],atol=1e-8)
                np.testing.assert_allclose(host.engine._tilt_filters['F'].rotation,before)
                self.assertFalse(left.engine._tilt_filters)
                left.calibration=GestureCalibration(np.eye(3),np.eye(3),np.eye(3));left.calibration.set_closed(np.eye(3),rot([0,80,0]))
                # Display zero / popup reset cannot change the host filter.
                epoch=host.engine.epochs['F'];left.invalidate()
                self.assertEqual(host.engine.epochs['F'],epoch)
                self.assertTrue(host.engine._tilt_filters['F'].bias_ready)
                left.calibration=GestureCalibration(np.eye(3),np.eye(3),np.eye(3))
                host.engine.invalidate_neutral('F')
                with self.assertRaisesRegex(ValueError,'参考系已改变'):left.selected_pair(time.monotonic())
                right.role_validator()  # A different side's epoch change cannot block this panel.
                host.engine.feed(packet('F',1200),received_at=time.monotonic())
                w.feed_host_packet(round(host.engine.received_at['F']*1e9),host.engine.packets['F'],host.engine);w.tick()
                self.assertIsNone(left.calibration)
                self.assertFalse(left.host_bias_ready['F'])
                with self.assertRaisesRegex(ValueError,'静止零偏学习'):left.selected_pair(time.monotonic())
            finally:w.shutdown();w.hide();host.deleteLater()

    def test_bilateral_four_devices_independent_states_stale_and_conflicts(self):
        import time
        from unittest.mock import Mock
        from optusbproj_ubuntu.gripper_gesture_gui import BilateralGestureWindow, SharedGestureSource
        from optusbproj_ubuntu.pose import VENDOR_RPY
        with TemporaryDirectory() as tmp:
            source=SharedGestureSource();w=BilateralGestureWindow(source,tmp);w.timer.stop()
            originals=[]
            try:
                for panel in w.panels.values():
                    panel.mode.setCurrentIndex(panel.mode.findData(VENDOR_RPY))
                    originals.append(panel.preview_socket)
                    panel.preview_socket=Mock();panel.preview_socket.recvfrom.side_effect=BlockingIOError
                def feed(pitches):
                    ns=time.monotonic_ns()
                    for device,pitch in pitches.items():
                        source.feed(ns,SimpleNamespace(device_id=device,roll=0.,pitch=pitch,yaw=0.,hour=0,minute=0,second=0,millisecond=20,
                            gyro_x=0.,gyro_y=0.,gyro_z=0.,acc_x=0.,acc_y=0.,acc_z=1.,battery_percent=90))
                    w.tick()
                feed(dict(LH=0,LF=0,RH=0,RF=0))
                left,right=w.panels['left'],w.panels['right']
                self.assertEqual(left.hand.count(),5);self.assertEqual(right.hand.count(),5)
                for panel,h,f,joint in ((left,'LH','LF','7'),(right,'RH','RF','57')):
                    panel.hand.setCurrentIndex(panel.hand.findData(h));panel.finger.setCurrentIndex(panel.finger.findData(f))
                    self.assertEqual(panel.target_joint.currentData(),joint);self.assertFalse(panel.target_joint.isEnabled())
                    panel.calibration=GestureCalibration(np.eye(3),np.eye(3),np.eye(3));panel.calibration.set_closed(np.eye(3),rot([0,80,0]))
                    panel.dwell.setValue(0);panel.send_preview.setChecked(True)
                feed(dict(LH=0,LF=80,RH=0,RF=0))
                self.assertTrue(left.discrete.state);self.assertFalse(right.discrete.state)
                self.assertAlmostEqual(left.filtered,1);self.assertAlmostEqual(right.filtered,0)
                import json
                for panel,joint,value in ((left,'7',0.),(right,'57',1.)):
                    frame=json.loads(panel.preview_socket.sendto.call_args.args[0]);self.assertEqual((frame['joint'],frame['position']),(joint,value))
                left.start_capture('open')
                self.assertIsNotNone(left.capture)
                self.assertFalse(left.send_preview.isChecked());self.assertTrue(right.send_preview.isChecked())
                left.cancel_capture();left.send_preview.setChecked(True)
                # Left source expires; the fresh right side continues to send.
                for device in ('LH','LF'):
                    left.buffer.rows[device]=type(left.buffer.rows[device])([(time.monotonic()-1,np.eye(3))],maxlen=150)
                left.preview_socket.sendto.reset_mock();right.preview_socket.sendto.reset_mock()
                right.last_send=0
                feed(dict(RH=0,RF=0))
                left.preview_socket.sendto.assert_not_called();right.preview_socket.sendto.assert_called()
                self.assertTrue(left.discrete.state)
                right.hand.setCurrentIndex(right.hand.findData('LH'))
                self.assertIn('冲突',w.assignment_status.text())
                self.assertFalse(left.send_preview.isChecked());self.assertFalse(right.send_preview.isChecked())
                with self.assertRaisesRegex(ValueError,'冲突'):w.validate_roles()
                right.hand.setCurrentIndex(right.hand.findData('RH'));w.validate_roles()
                self.assertIsNotNone(left.calibration);self.assertIsNone(right.calibration)
                w.show();self.app.processEvents();self.assertFalse(w.grab().isNull())
            finally:
                w.shutdown();w.close()
                for sock in originals:sock.close()

    def test_shared_source_discrete_udp_and_popup_lifecycle(self):
        import socket, time, json
        from unittest.mock import patch
        from PySide6 import QtWidgets
        from optusbproj_ubuntu.gripper_gesture_gui import SharedGestureSource, GestureWindow
        s=SharedGestureSource();p=object();s.feed(123,p)
        self.assertEqual(s.poll_timed(),[(123,p)]);self.assertEqual(s.poll_timed(),[])
        host=QtWidgets.QWidget();host._show_x1_preview=lambda:None
        with TemporaryDirectory() as tmp:
            w=GestureWindow(s,tmp,host);w.timer.stop()
            original=w.preview_socket
            fake=unittest.mock.Mock();fake.recvfrom.side_effect=BlockingIOError
            w.preview_socket=fake
            try:
                self.assertFalse(w.send_preview.isChecked())
                w.discrete=DiscreteClosure(dwell=0)
                w.discrete.update(1.,1.)
                w.update_preview(time.monotonic(),True);fake.sendto.assert_not_called()
                w.send_preview.setChecked(True)
                w.update_preview(time.monotonic(),True)
                frame=json.loads(fake.sendto.call_args.args[0]);self.assertEqual(frame['position'],0.);self.assertEqual(frame['joint'],'57')
                fake.sendto.reset_mock();w.last_send=0
                w.update_preview(time.monotonic(),False);fake.sendto.assert_not_called()
                w.discrete.update(0.,2.);w.last_send=0
                w.update_preview(time.monotonic(),True)
                self.assertEqual(json.loads(fake.sendto.call_args.args[0])['position'],1.)
                w.show();self.app.processEvents();w.close()
                self.assertFalse(w.isVisible());self.assertFalse(w.send_preview.isChecked())
                w.open_threshold.setValue(90)
                self.assertIsNone(w.discrete)
                w.closed_threshold.setValue(95)
                self.assertIsNone(w.discrete.state)
                w.shutdown();fake.close.assert_called_once()
            finally:
                original.close();host.deleteLater()

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
