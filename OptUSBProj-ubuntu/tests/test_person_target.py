import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path[:0]=[str(Path(__file__).resolve().parents[1])]
from optusbproj_ubuntu.person_target import PersonTarget, torso_depth
from optusbproj_ubuntu.pose_2d import YOLOBackend, Pose2DConfig
from optusbproj_ubuntu.rgbd_pose import DepthPoseEstimator
from optusbproj_ubuntu.rgbd_pose_worker import infer_observation, RGBDPoseWorker
from optusbproj_ubuntu.camera_preview import CameraPreview
from test_rgbd_pose import fixture

BOXES=np.array([[10,10,65,110],[90,10,150,110]],float)


def selected_target(obs,index=0):
    target=PersonTarget()
    _,info=target.choose(BOXES,obs)
    target.reset(info['person_candidates'][index],mode='manual',selection_id='chosen')
    return target


class TargetTests(unittest.TestCase):
    def test_pose_change_low_reid_score_does_not_interrupt_continuous_tracking(self):
        obs,target,identity=self.reid_fixture()
        # Clear the simulated overlap as an explicit fresh selection.
        target.reset();target.encoder.feature=np.array([1.,0.])
        for i in range(3):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
            self.assertEqual(target.choose(BOXES[:1],obs)[0],0)
        target.encoder.feature=np.array([.51,np.sqrt(1-.51**2)])
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        index,info=target.choose(BOXES[:1],obs)
        self.assertEqual(index,0);self.assertFalse(target.latched)
        self.assertAlmostEqual(info['reid_tracking_similarity'],.51)
        # Abrupt dissimilar appearance must not be learned into the gallery.
        self.assertEqual(len(target.gallery),3)

    def test_gradual_pose_changes_collect_diverse_reference(self):
        obs,_,_=fixture();target=PersonTarget(lambda rgb,boxes:np.tile(self.feature,(len(boxes),1)))
        self.feature=np.array([1.,0.])
        for i in range(3):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
            target.choose(BOXES[:1],obs)
        for angle in np.linspace(0,1.05,12):
            self.feature=np.array([np.cos(angle),np.sin(angle)])
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
            self.assertEqual(target.choose(BOXES[:1],obs)[0],0)
        self.assertGreater(len(target.gallery),3)
        self.assertGreater(max(np.asarray(target.gallery)@self.feature),.9)
        identity=target.selection_id
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        self.assertIsNone(target.choose(np.array([BOXES[0],BOXES[0]+[5,0,5,0]]),obs)[0])
        # The new pose, poorly matched to the initial neutral reference, can
        # still recover against the diverse clean reference collection.
        for i in range(5):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
            index,info=target.choose(BOXES[:1],obs)
        self.assertEqual(index,0);self.assertTrue(info['reid_recovered'])
        self.assertEqual(target.selection_id,identity)

    def test_arm_raise_bbox_width_change_preserves_tracking(self):
        obs,_,_=fixture();target=PersonTarget()
        target.choose([[50,10,90,110]],obs)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        index,_=target.choose([[15,10,125,110]],obs)
        self.assertEqual(index,0);self.assertFalse(target.latched)

    def reid_fixture(self):
        obs,_,_=fixture()
        class Encoder:
            metadata={'model':'synthetic'}
            feature=np.array([1.,0.])
            def __call__(self,rgb,boxes):
                return np.stack([self.feature if b[0]<70 else np.array([0.,1.]) for b in boxes])
        target=PersonTarget(Encoder())
        for i in range(4):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
            self.assertEqual(target.choose(BOXES[:1],obs)[0],0)
        original_id=target.selection_id
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        close=BOXES[0]+[5,0,5,0]
        self.assertIsNone(target.choose(np.array([BOXES[0],close]),obs)[0])
        return obs,target,original_id

    def test_reid_overlap_recovery_keeps_identity_and_gallery(self):
        obs,target,identity=self.reid_fixture();gallery=np.asarray(target.gallery).copy()
        for i in range(5):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
            index,info=target.choose(BOXES[::-1],obs)
            self.assertEqual(index,1 if i==4 else None)
        self.assertTrue(info['reid_recovered']);self.assertEqual(target.selection_id,identity)
        np.testing.assert_equal(target.gallery,gallery)
        json.dumps(info,allow_nan=False)

    def test_reid_impostor_alone_does_not_restore(self):
        obs,target,_=self.reid_fixture();target.encoder.feature=np.array([0.,1.])
        for i in range(8):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
            self.assertIsNone(target.choose(BOXES[:1],obs)[0])

    def test_reid_similar_people_and_missing_depth_do_not_restore(self):
        obs,target,_=self.reid_fixture()
        target.encoder=lambda rgb,boxes:np.tile([1.,0.],(len(boxes),1))
        for i in range(6):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
            self.assertIsNone(target.choose(BOXES,obs)[0])
        obs.rgbd.aligned_depth_mm[:]=0
        for i in range(6):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
            self.assertIsNone(target.choose(BOXES[:1],obs)[0])

    def test_reid_long_loss_requires_manual_confirmation(self):
        obs,target,_=self.reid_fixture()
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+31_000_000_000)
        index,info=target.choose(BOXES[:1],obs)
        self.assertIsNone(index);self.assertIn('30秒',info['reject_reason'])

    def test_reid_pending_interrupted_by_missing_frame(self):
        obs,target,_=self.reid_fixture()
        for i in range(4):
            obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
            self.assertIsNone(target.choose(BOXES[:1],obs)[0])
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
        target.choose([],obs)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+110_000_000)
        self.assertIsNone(target.choose(BOXES[:1],obs)[0]);self.assertEqual(target.pending_count,1)

    def test_multi_waits_for_choice_and_reordered_detections_preserve_target(self):
        obs,_,_=fixture();target=PersonTarget()
        index,info=target.choose(BOXES,obs)
        self.assertIsNone(index);self.assertIn('多人',info['reject_reason'])
        self.assertEqual(info['person_candidates'][0]['depth_m'],1.)
        target.reset(info['person_candidates'][0],mode='manual')
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        index,info=target.choose(BOXES[::-1],obs)
        self.assertEqual(index,1);self.assertEqual(info['selected_index'],1)
        json.dumps(info,allow_nan=False)

    def test_auto_only_acquires_single_once_never_switches_after_loss(self):
        obs,_,_=fixture();target=PersonTarget()
        self.assertEqual(target.choose(BOXES[:1],obs)[0],0)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        self.assertIsNone(target.choose(BOXES[1:],obs)[0])
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        self.assertIsNone(target.choose(BOXES[:1],obs)[0])
        target.reset()
        self.assertEqual(target.choose(BOXES[:1],obs)[0],0)

    def test_crossing_and_long_gap_latch_even_when_only_one_person_remains(self):
        obs,_,_=fixture();target=selected_target(obs)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        close=BOXES[0]+[5,0,5,0]
        _,info=target.choose(np.array([BOXES[0],close]),obs)
        self.assertIn('歧义',info['reject_reason'])
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        self.assertIsNone(target.choose(BOXES[:1],obs)[0])
        target.reset(info['person_candidates'][0],mode='manual')
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+2_000_000_000)
        self.assertIn('间隔',target.choose(BOXES[:1],obs)[1]['reject_reason'])

    def test_depth_jump_and_missing_depth_and_cancel(self):
        obs,_,_=fixture();target=selected_target(obs)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000);obs.rgbd.aligned_depth_mm[:]=2000
        self.assertIsNone(target.choose(BOXES[:1],obs)[0])
        obs.rgbd.aligned_depth_mm[:]=0
        self.assertIsNone(torso_depth(obs,BOXES[0]))
        target.reset(mode='paused')
        self.assertIsNone(target.choose(BOXES[:1],obs)[0])

    def test_depth_grid_uses_distortion_mapping(self):
        obs,_,_=fixture(distortion=.4)
        self.assertAlmostEqual(torso_depth(obs,BOXES[0]),1.)

    def test_rtmw_only_runs_pose_for_selected_box(self):
        obs,xy,scores=fixture();target=selected_target(obs,1)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        pose=Mock(return_value=(None,scores[None],None,xy[None]))
        model=SimpleNamespace(det_model=lambda bgr:BOXES,pose_model=pose)
        estimate=infer_observation(model,DepthPoseEstimator(),obs,target=target)
        self.assertEqual(estimate.diagnostics['selected_index'],1)
        np.testing.assert_equal(pose.call_args.kwargs['bboxes'],BOXES[1:])
        self.assertTrue(estimate.valid.all())
        # No detection must not retain old keypoints as valid observations.
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000);model.det_model=lambda bgr:np.empty((0,4))
        rejected=infer_observation(model,DepthPoseEstimator(),obs,target=target)
        self.assertFalse(rejected.valid.any());self.assertTrue(np.isnan(rejected.keypoints_xy).all())

    def test_yolo_selects_correct_keypoint_batch(self):
        obs,xy,scores=fixture();target=selected_target(obs,1)
        obs=replace(obs,captured_mono_ns=obs.captured_mono_ns+100_000_000)
        model=YOLOBackend.__new__(YOLOBackend);model.config=Pose2DConfig('yolo26');model.metadata={}
        # Ultralytics Boxes supports len; a small fake avoids loading torch/GPU.
        class Boxes:
            xyxy=BOXES
            def __len__(self):return 2
        model.model=Mock()
        model.model.predict.return_value=[SimpleNamespace(boxes=Boxes(),keypoints=SimpleNamespace(
            xy=np.stack([np.zeros((17,2)),xy[:17]]),conf=np.ones((2,17))))]
        estimate=infer_observation(model,DepthPoseEstimator(),obs,target=target)
        np.testing.assert_equal(estimate.keypoints_xy[:17],xy[:17])
        self.assertEqual(estimate.diagnostics['selected_index'],1)
        self.assertTrue(estimate.valid.all())

    def test_worker_selection_resets_pending_result_without_gpu_reload(self):
        worker=RGBDPoseWorker(vision_epoch='old')
        worker._result=object()
        worker.select_target(None,'paused','new')
        self.assertIsNone(worker.take_latest())
        self.assertEqual(worker.vision_epoch,'new')
        self.assertEqual(worker._target_command,(None,'paused','new'))
        worker.stop()


class GuiTests(unittest.TestCase):
    def test_original_person_confirmation_restores_without_reinstall(self):
        from optusbproj_ubuntu.fusion_gui import FusionWindow
        from test_pose_backend_gui import Source
        from test_live_rgbd_fusion import LiveTests
        from optusbproj_ubuntu.live_rgbd_fusion import PERSON_SELECTION_INVALIDATION
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                engine,service,device,q,_=LiveTests().setup_system()
                w.engine=engine;w.live_fusion=service
                w.camera_worker=SimpleNamespace(capture_epoch='camera')
                w._live_camera_epoch='camera'
                w._person_restore_vision_signature=w.vision_metadata['signature']
                engine.received_at[device]=time.monotonic()
                worker=Mock();w.pose_worker=worker
                with patch.object(w,'_set_rgbd_pose'):w.rgbd_pose_enabled.setChecked(True)
                obs,_,_=fixture();_,info=PersonTarget().choose(BOXES,obs)
                item=info['person_candidates'][0];item['captured_mono_ns']=time.monotonic_ns()
                w.person_combo.addItem('原人物',item)
                service.bindings['right_upper_arm'][1].offset_rad=.3
                service.invalidate(PERSON_SELECTION_INVALIDATION)
                old_epoch=w.vision_epoch
                w._restore_original_person()  # No human acknowledgement.
                self.assertNotIn(device,engine.fusion_ready)
                w.person_restore_ack.setChecked(True)
                self.assertTrue(w.person_restore.isEnabled())
                w._restore_original_person()
                self.assertIn(device,engine.fusion_ready)
                self.assertEqual(service.bindings['right_upper_arm'][1].offset_rad,.3)
                self.assertNotEqual(w.vision_epoch,old_epoch)
                worker.select_target.assert_called_once_with(item,'manual',w.vision_epoch)
                self.assertFalse(w.person_restore_ack.isChecked())
                # A model change is recoverable, but requires a new direction check.
                from optusbproj_ubuntu.live_rgbd_fusion import MODEL_SELECTION_INVALIDATION
                service.invalidate(MODEL_SELECTION_INVALIDATION)
                w.vision_metadata=dict(w.vision_metadata,signature='another-model')
                item['captured_mono_ns']=time.monotonic_ns()
                w.person_combo.addItem('原人物',item)
                w.person_restore_ack.setChecked(True)
                self.assertTrue(w.person_restore.isEnabled())
                w._restore_original_person()
                self.assertNotIn(device,engine.fusion_ready)
                self.assertIn('right_upper_arm',service.model_rechecks)
                self.assertEqual(service.bindings['right_upper_arm'][1].offset_rad,.3)
                # A camera restart blocks this shortcut even after acknowledgement.
                w.camera_worker.capture_epoch='different'
                self.assertIn('相机',w._person_restore_error())
            finally:w.pose_worker=None;w.camera_worker=None;w.close()

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_preview_click_maps_letterboxed_pixels_and_ignores_margin(self):
        from PySide6 import QtCore,QtGui,QtTest
        preview=CameraPreview();preview.resize(400,400)
        image=QtGui.QImage(640,320,QtGui.QImage.Format.Format_RGB888);image.fill(0)
        preview.display_image(image);preview.show();self.app.processEvents()
        clicked=[];preview.image_clicked.connect(lambda x,y:clicked.append((x,y)))
        QtTest.QTest.mouseClick(preview,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(200,200))
        np.testing.assert_allclose(clicked[0],[320,160],atol=1)
        QtTest.QTest.mouseClick(preview,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(200,10))
        self.assertEqual(len(clicked),1);preview.close()

    def test_choice_invalidates_camera_not_wearing_and_blocks_during_calibration(self):
        from optusbproj_ubuntu.fusion_gui import FusionWindow
        from test_pose_backend_gui import Source
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w.engine.continuous_basis['left_upper_arm']=np.eye(3)
                w.engine.fusion_ready.add('A')
                w.live_fusion.bindings['left_upper_arm']=({'device_id':'A'},SimpleNamespace(offset_rad=.2))
                worker=Mock();w.pose_worker=worker;epoch=w.vision_epoch
                w._apply_person_target(None,'paused')
                self.assertNotEqual(w.vision_epoch,epoch)
                self.assertNotIn('A',w.engine.fusion_ready)
                np.testing.assert_equal(w.engine.continuous_basis['left_upper_arm'],np.eye(3))
                worker.select_target.assert_called_once_with(None,'paused',w.vision_epoch)
                epoch=w.vision_epoch
                w._combined_calibration_dialog=SimpleNamespace(active=False,state='analyzing')
                w._apply_person_target(None,'auto')
                self.assertEqual(w.vision_epoch,epoch)
                self.assertIn('未切换',w.person_status.text())
            finally:w.pose_worker=None;w._combined_calibration_dialog=None;w.close()

    def test_selection_snapshot_not_future_detection_index_and_stale_rejected(self):
        from optusbproj_ubuntu.fusion_gui import FusionWindow
        from test_pose_backend_gui import Source
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                with patch.object(w,'_set_rgbd_pose'):w.rgbd_pose_enabled.setChecked(True)
                obs,_,_=fixture();_,info=PersonTarget().choose(BOXES,obs)
                selected=info['person_candidates'][0]
                w._person_candidates=list(reversed(info['person_candidates']))
                with patch.object(w,'_apply_person_target') as apply:
                    w._select_person(selected)
                    apply.assert_called_once_with(selected,'manual')
                    selected['captured_mono_ns']-=2_000_000_000
                    w._select_person(selected)
                    self.assertEqual(apply.call_count,1)
                with patch.object(w,'_select_person') as select:
                    w._person_frame_ns=obs.captured_mono_ns;w._person_preview_frame_ns=None
                    w._click_person(30,30);select.assert_not_called()
            finally:w.close()


if __name__=='__main__':unittest.main()
