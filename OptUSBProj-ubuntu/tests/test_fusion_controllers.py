"""Coordination tests with fake hosts: no camera, desktop, ROS or simulator."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock
import subprocess
import sys
import numpy as np

from optusbproj_ubuntu.fusion_camera_controller import FusionCameraController
from optusbproj_ubuntu.fusion_recording_controller import FusionRecordingController
from optusbproj_ubuntu.fusion_calibration_controller import FusionCalibrationController
from optusbproj_ubuntu.fusion_vision_controller import FusionVisionController


class Worker:
    def __init__(self, *args, **kwargs):
        self.error = NS(connect=Mock())
        self.finished = NS(connect=Mock())
        self.status = NS(connect=Mock())
        self.start = Mock()
        self.stop = Mock()
        self.deleteLater = Mock()
        self.calibration = 'calibration'
        self.consumer_skipped = 3


class ControllerTests(TestCase):
    def test_imports_do_not_load_qt_or_fusion_window(self):
        code = ('import sys; from optusbproj_ubuntu import '
                'fusion_camera_controller, fusion_recording_controller, '
                'fusion_calibration_controller, fusion_vision_controller; '
                'assert not any(x.startswith("PySide6") for x in sys.modules); '
                'assert "optusbproj_ubuntu.fusion_gui" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], check=True)

    def camera_host(self):
        return NS(camera_worker=None, replay_mode=False, camera_source='orbbec',
                  orbbec_python=Path('python'), rgbd_rate=Mock(value=lambda:10),
                  rgbd_warmup=2., last_rgbd_observation=object(), rgbd_timer=Mock(),
                  _on_camera_error=Mock(), _on_camera_finished=Mock(),
                  camera_source_combo=Mock(), start_camera_button=Mock(),
                  stop_camera_button=Mock(), camera_status=Mock(), session=None,
                  rgbd_pose_enabled=Mock(), live_fusion=Mock(),
                  _stop_session=Mock(), pose_worker=None, pose_dialog=None,
                  vision_config='config', vision_epoch='epoch', person_target='person',
                  _on_pose_error=Mock(), _on_pose_finished=Mock(), pose_status=Mock())

    def camera_controller(self, host):
        return FusionCameraController(host, camera_factory=Mock(),
                                      orbbec_type=Worker, pose_factory=Worker)

    def test_camera_start_idempotent_and_finish_invalidates(self):
        w=self.camera_host(); c=self.camera_controller(w)
        c.start_camera(); worker=w.camera_worker
        self.assertIsNone(w.last_rgbd_observation)
        worker.start.assert_called_once(); w.rgbd_timer.start.assert_called_once()
        c.start_camera(); self.assertIs(w.camera_worker,worker)
        c._on_camera_finished()
        self.assertIsNone(w.camera_worker)
        worker.deleteLater.assert_called_once()
        w.live_fusion.invalidate.assert_called_once_with('相机停止/重启，需重新标定')
        w.rgbd_timer.stop.assert_called_once()

    def test_old_pose_callbacks_cannot_overwrite_new_worker(self):
        w=self.camera_host(); c=self.camera_controller(w)
        c._set_rgbd_pose(True); old=w.pose_worker
        error=old.error.connect.call_args.args[0]
        status=old.status.connect.call_args.args[0]
        w.pose_worker=Worker()
        error('late failure'); status('late status')
        w._on_pose_error.assert_not_called(); w.pose_status.setText.assert_not_called()
        w.pose_worker=old; error('current failure'); status('current status')
        w._on_pose_error.assert_called_once_with('current failure')
        w.pose_status.setText.assert_called_once_with('current status')

    def test_quick_pose_restart_waits_for_finished_callback(self):
        w=self.camera_host(); c=self.camera_controller(w)
        c._set_rgbd_pose(True); old=w.pose_worker
        c._set_rgbd_pose(False); c._set_rgbd_pose(True)
        self.assertIs(w.pose_worker,old); old.start.assert_called_once()
        old.stop.assert_called_once()
        w.camera_worker=Worker(); w.rgbd_pose_enabled.isChecked.return_value=True
        w._set_rgbd_pose=Mock()
        c._on_pose_finished()
        self.assertIsNone(w.pose_worker); old.deleteLater.assert_called_once()
        w._set_rgbd_pose.assert_called_once_with(True)

    def recording_host(self):
        widgets={name:Mock() for name in ('record_schedule_status','session_button',
            'camera_source_combo','rgbd_rate','timed_session_button','cancel_session_button',
            'record_delay','record_duration','record_schedule_timer')}
        return NS(**widgets, camera_calibration_dialog=None, replay_mode=False,
            session=None, _record_start_deadline=None, _record_stop_deadline=None,
            palm_visual_dialog=None, natural_hand_dialog=None, camera_source='orbbec',
            last_rgbd_observation=NS(captured_mono_ns=10_000_000_000),
            camera_worker=Worker(), _fusion_events_file=None,
            engine=NS(natural_hand_previews={}, assignments={}, axis_mappings={},
                algorithm_modes={}, continuous_basis={}, natural_hand_references={'x':{'v':1}},
                calibration_references={}), vision_metadata={'signature':'s'},
            vision_epoch='v',last_pose_estimate=None,
            live_fusion=NS(max_offset_deg=30,bindings={'arm':({'device_id':'a'},NS(offset_rad=.2))}),
            statusBar=Mock(return_value=Mock()), _start_session=Mock(return_value=True),
            _stop_session=Mock(), _update_record_schedule=Mock())

    def recording_controller(self,w,factory,clock=lambda:10.):
        return FusionRecordingController(w,recorder_factory=factory,orbbec_type=Worker,
            warn=Mock(),clock=clock,clock_ns=lambda:int(clock()*1e9))

    def test_recording_stale_camera_and_natural_preview_block_storage(self):
        w=self.recording_host();factory=Mock(); c=self.recording_controller(w,factory)
        w.last_rgbd_observation.captured_mono_ns=0
        self.assertFalse(c._start_session());factory.assert_not_called()
        w.engine.natural_hand_previews={'hand':{}}
        self.assertFalse(c._start_session());factory.assert_not_called()

    def test_recording_snapshots_nonzero_offset_and_closes_once(self):
        with TemporaryDirectory() as folder:
            w=self.recording_host();recorder=NS(path=Path(folder),close=Mock(),
                trimmed_imu_packets=0,trimmed_camera_frames=0,imu_count=1,
                frame_count=1,dropped_frames=0)
            factory=Mock(return_value=recorder);c=self.recording_controller(w,factory)
            self.assertTrue(c._start_session())
            self.assertAlmostEqual(recorder.fusion_at_start['arm']['offset_deg'],np.degrees(.2))
            w.engine.natural_hand_references['x']['v']=2
            self.assertEqual(recorder.natural_hand_references_at_start['x']['v'],1)
            self.assertEqual(factory.call_args.kwargs['max_pending_frames'],16)
            stream=w._fusion_events_file
            c._stop_session();c._stop_session()
            recorder.close.assert_called_once();self.assertTrue(stream.closed)
            self.assertIsNone(w.session);self.assertIsNone(w._fusion_events_file)

    def test_timed_recording_clock_and_cancel(self):
        w=self.recording_host();now=[10.];c=self.recording_controller(w,Mock(),lambda:now[0])
        w.record_delay.value.return_value=3;w.record_duration.value.return_value=5
        c._start_timed_session();self.assertEqual(w._record_start_deadline,13.)
        c._update_record_schedule();w._start_session.assert_not_called()
        now[0]=13.;c._update_record_schedule();w._start_session.assert_called_once()
        self.assertEqual(w._record_stop_deadline,18.)
        now[0]=18.;c._update_record_schedule();w._stop_session.assert_called_once_with(auto=True)
        w._record_start_deadline=20.;c._cancel_record_schedule()
        self.assertIsNone(w._record_start_deadline)

    def test_failed_scheduled_start_releases_controls(self):
        w=self.recording_host();c=self.recording_controller(w,Mock())
        w._record_start_deadline=9.;w._start_session.return_value=False
        c._update_record_schedule()
        self.assertIsNone(w._record_start_deadline);self.assertIsNone(w._record_stop_deadline)
        w.record_schedule_timer.stop.assert_called_once()
        w.record_delay.setEnabled.assert_called_once_with(True)

    def test_calibration_report_epoch_and_recording_gates(self):
        w=NS(replay_mode=False,session=None,x1_preview_dialog=None,
            camera_worker=NS(capture_epoch='camera'),_check_vision_report=Mock(),
            live_fusion=Mock(),vision_metadata={'signature':'s'},_update_live_fusion_status=Mock())
        c=FusionCalibrationController(w)
        with self.assertRaises(ValueError):c.apply_camera_report({'camera_epoch':'old'})
        w.live_fusion.install.assert_not_called()
        report={'camera_epoch':'camera'};c.apply_camera_report(report)
        w.live_fusion.install.assert_called_once_with(report)
        self.assertEqual(w._live_camera_epoch,'camera')
        self.assertEqual(w._person_restore_vision_signature,'s')
        w.session=object()
        with self.assertRaises(ValueError):c.apply_camera_report(report)
        self.assertEqual(w.live_fusion.install.call_count,1)

    def test_person_restore_rejects_old_frame_and_changed_camera(self):
        w=NS(_pose_change_error=lambda:None,camera_source='orbbec',
             camera_worker=NS(capture_epoch='camera'),_live_camera_epoch='camera',
             rgbd_pose_enabled=Mock(isChecked=lambda:True),pose_worker=object(),
             person_combo=Mock(currentData=lambda:{'captured_mono_ns':0}),live_fusion=Mock())
        c=FusionVisionController(w,clock_ns=lambda:2_000_000_000)
        self.assertIn('原标定者',c._person_restore_error())
        w.live_fusion.validate_person_restore.assert_not_called()
        w.camera_worker.capture_epoch='other'
        self.assertIn('相机采集会话',c._person_restore_error())
