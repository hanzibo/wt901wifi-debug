import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6 import QtWidgets
from optusbproj_ubuntu.camera_calibration_gui import CameraCalibrationDialog,calibration_steps


class Owner(QtWidgets.QWidget):
    def __init__(self,path,clock):
        super().__init__();self.path=path;self.clock=clock;self.live=True;self.replay_mode=False;self.camera_source='orbbec'
        self.session=None;self._record_start_deadline=None
        self.engine=SimpleNamespace(assignments={'right_upper_arm':'A','right_forearm':'B'},is_live=lambda *a,**k:self.live)
        self.rgbd_pose_enabled=QtWidgets.QCheckBox();self.rgbd_pose_enabled.setChecked(True)
        for key in ['session_button','timed_session_button','cancel_session_button']:setattr(self,key,QtWidgets.QPushButton())
        self.refresh();self.starts=self.stops=0
        self.gravity_calls=[]
        self.live_fusion=SimpleNamespace(at=lambda *_:None)

    def prepare_camera_gravity(self,devices,start_ns,end_ns):
        self.gravity_calls.append((devices,start_ns,end_ns))
        return {device:{'up_world':[0,0,1]} for device in devices}

    def refresh(self):
        self.last_rgbd_observation=SimpleNamespace(captured_mono_ns=int(self.clock[0]*1e9))
        self.last_pose_estimate=SimpleNamespace(captured_mono_ns=int(self.clock[0]*1e9),
            to_record=lambda:{'valid':[False]*8})

    def _start_session(self):
        self.starts+=1;self.path.mkdir(exist_ok=True)
        self.session=SimpleNamespace(path=self.path,started_mono_ns=int(self.clock[0]*1e9));return True

    def _stop_session(self,**kwargs):
        self.stops+=1;self.session=None
        (self.path/'manifest.json').write_text(json.dumps({'error':None}))


class GuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_delay_then_actual_phase_times_and_complete(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock);dialog=CameraCalibrationDialog(owner)
            dialog.auto_analyze.setChecked(False)
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic',side_effect=lambda:clock[0]),patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',side_effect=lambda:int(clock[0]*1e9)):
                dialog.begin();dialog.timer.stop()
                self.assertEqual(owner.starts,0)
                clock[0]+=11;owner.refresh();dialog.tick()
                self.assertEqual(owner.starts,1);self.assertEqual(dialog.state,'prepare')
                while dialog.active:
                    clock[0]=dialog.deadline+.1;owner.refresh();dialog.tick()
                self.assertEqual(dialog.state,'complete');self.assertEqual(owner.stops,1)
                self.assertTrue(dialog.analyze.isEnabled())
                saved=json.loads((owner.path/'camera_guide.json').read_text())
                self.assertEqual(saved['status'],'complete')
                self.assertEqual(len(owner.gravity_calls),1)
                self.assertEqual(set(saved['gravity_reference']),{'A','B'})
                self.assertEqual(len(saved['events']),20)
                self.assertEqual(saved['camera_pose_protocol'],'visible-oblique-v1')
                neutral,train=dialog.analysis_windows()
                self.assertLess(train[0],neutral[0]);self.assertLess(neutral[1],train[1])
                self.assertTrue(all(e['end_s']>e['start_s'] for e in saved['events']))
            dialog.close();owner.close()

    def test_cancel_before_recording_creates_no_files(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock);dialog=CameraCalibrationDialog(owner)
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic',return_value=100.),patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',return_value=100_000_000_000):
                dialog.begin();dialog.abort()
                self.assertEqual(owner.starts,0);self.assertFalse(owner.path.exists());self.assertFalse(dialog.active)
            dialog.close();owner.close()

    def test_gravity_failure_stops_before_arm_actions(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock);dialog=CameraCalibrationDialog(owner)
            dialog.delay.setValue(0)
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic',side_effect=lambda:clock[0]),patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',side_effect=lambda:int(clock[0]*1e9)),patch.object(owner,'prepare_camera_gravity',side_effect=ValueError('未充分静止')):
                dialog.begin();dialog.timer.stop()
                for _ in range(2):clock[0]=dialog.deadline+.1;owner.refresh();dialog.tick()
                self.assertEqual(dialog.state,'cancelled')
                self.assertEqual(dialog.index,0)
                self.assertIn('重力初始化',dialog.status.text())
                self.assertEqual(owner.stops,1)
                self.assertFalse(dialog.analyze.isEnabled())
            dialog.close();owner.close()

    def test_failed_application_has_explicit_warning(self):
        with TemporaryDirectory() as folder:
            owner=Owner(Path(folder),[100.]);dialog=CameraCalibrationDialog(owner)
            dialog.analysis_path=Path(folder);(Path(folder)/'report.json').write_text('{}')
            owner.apply_camera_report=lambda report:(_ for _ in ()).throw(ValueError('实时重力参考与标定不一致'))
            with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.StandardButton.Yes),patch.object(QtWidgets.QMessageBox,'warning') as warning,patch.object(QtWidgets.QMessageBox,'information') as success:
                dialog.apply_live();warning.assert_called_once();success.assert_not_called()
                self.assertIn('未应用',dialog.status.text())
            dialog.close();owner.close()

    def test_device_disconnect_aborts_record_without_analysis(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock);dialog=CameraCalibrationDialog(owner);dialog.delay.setValue(0)
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic',return_value=100.),patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',return_value=100_000_000_000):
                dialog.begin();self.assertEqual(owner.starts,1)
                owner.live=False;dialog.tick()
                self.assertEqual(dialog.state,'cancelled');self.assertEqual(owner.stops,1)
                self.assertIsNone(dialog.process);self.assertFalse(dialog.analyze.isEnabled())
                self.assertEqual(json.loads((owner.path/'camera_guide.json').read_text())['status'],'cancelled')
            dialog.close();owner.close()

    def test_conflicting_record_and_missing_vision_are_rejected(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock);dialog=CameraCalibrationDialog(owner)
            owner.session=SimpleNamespace(path=owner.path)
            self.assertIn('已有',dialog.readiness_error());owner.session=None
            owner.rgbd_pose_enabled.setChecked(False)
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',return_value=100_000_000_000):
                self.assertIn('三维',dialog.readiness_error())
            dialog.close();owner.close()

    def test_protocol_has_tail_beyond_cutoff_and_repeated_actions(self):
        steps=calibration_steps(4,4,8,90)
        self.assertEqual(steps[-1][0],'tail');self.assertGreater(steps[-1][4],3)
        self.assertEqual(steps[-2][0],'validation')
        self.assertIn('forward_repeat',[s[0] for s in steps])

    def test_zero_validation_omits_window_but_retains_checks_and_tail(self):
        steps=calibration_steps(4,4,5,0,'bilateral')
        keys=[s[0] for s in steps]
        self.assertNotIn('validation',keys)
        for key in ('forward_repeat','lateral_repeat','neutral_return','tail'):
            self.assertIn(key,keys)
        with TemporaryDirectory() as folder:
            owner=Owner(Path(folder)/'record',[100.]);dialog=CameraCalibrationDialog(owner)
            self.assertEqual(dialog.validation.minimum(),0)
            dialog.validation.setValue(0)
            self.assertNotIn('validation',[s[0] for s in dialog.capture_steps()])
            dialog.close();owner.close()

    def test_live_feedback_counts_only_unique_synced_hold_frames(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock);dialog=CameraCalibrationDialog(owner)
            dialog.devices={'right_upper_arm':'A','right_forearm':'B'}
            dialog.started_ns=99_000_000_000;dialog.events=[{'start_s':0}]
            dialog.deadline=105.;dialog.state='hold';dialog._observed_counts={'right_upper_arm':0,'right_forearm':0};dialog._observed_stamp=None
            owner.live_fusion.at=lambda *_:object()
            row={'valid':[True]*8,'confidence':[1.]*133,'support':[1.]*8,
                 'points_camera_m':[[0,0,1],[0,0,1],[0,0,1],[.3,0,1],[0,0,1],[.6,0,1],[0,0,1],[0,0,1]]}
            owner.last_pose_estimate.to_record=lambda:row
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',return_value=100_000_000_000):
                dialog.observation_feedback('oblique');dialog.observation_feedback('oblique')
                self.assertEqual(dialog._observed_counts['right_upper_arm'],1)
                owner.last_pose_estimate.captured_mono_ns+=10_000_000
                row['valid'][5]=False;dialog.observation_feedback('oblique')
                self.assertEqual(dialog._observed_counts['right_upper_arm'],2)
                self.assertEqual(dialog._observed_counts['right_forearm'],1)
            dialog.state='idle';dialog.close();owner.close()
