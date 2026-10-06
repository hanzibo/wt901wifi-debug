from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,Mock
import sys
import time
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtCore,QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.palm_geometry import frame_from_points,lift_palm,estimate_c0,distance
from optusbproj_ubuntu.palm_visual_gui import PalmVisualDialog
from optusbproj_ubuntu.palm_worker import PalmDetector,draw_palm_diagnostics
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.pose import GYRO_TILT
from test_pose import imu_packet
from test_pose_backend_gui import Source
from test_rgbd_pose import fixture


def palm():
    p=np.array([[0,.04,1],[-.035,-.04,1],[-.012,-.05,1],[.012,-.05,1],[.035,-.04,1]])
    x,y=np.meshgrid(np.linspace(-.03,.03,12),np.linspace(-.035,.025,12))
    q=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
    return p,q


class PalmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_detection_failure_keeps_crop_and_distinguishes_zero_multiple(self):
        obs,_,_=fixture();xy=np.zeros((133,2));xy[7]=[80,100];xy[9]=[80,60];xy[10]=[10,100]
        body={'captured_mono_ns':obs.captured_mono_ns,'keypoints_xy':xy,'confidence':np.ones(133)}
        detector=PalmDetector.__new__(PalmDetector);detector.metadata={}
        detector.mp=SimpleNamespace(Image=lambda **kw:kw,ImageFormat=SimpleNamespace(SRGB=1))
        detector.detector=Mock()
        for count in (0,2):
            points=[SimpleNamespace(x=.5,y=.5) for _ in range(21)]
            detector.detector.detect.return_value=SimpleNamespace(hand_landmarks=[points]*count)
            row=detector.infer(obs,body)
            self.assertFalse(row['accepted']);self.assertEqual(row['detected_hand_count'],count)
            self.assertEqual(len(row['detected_hands_xy']),count);self.assertGreater(row['frame_rgb'].size,0)
            self.assertIn(f'检出{count}个',row['reason'])
        detector.detector.detect.return_value=SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=.1,y=.1) for _ in range(21)]])
        row=detector.infer(obs,body)
        self.assertTrue(row['detection_2d']);self.assertEqual(row['stage'],'association')
        self.assertFalse(row['accepted']);self.assertGreater(row['wrist_distance_px'],row['wrist_gate_px'])
        self.assertEqual(len(row['keypoints_xy']),21);self.assertGreater(row['frame_rgb'].size,0)
        np.testing.assert_array_equal(draw_palm_diagnostics(obs.frame_rgb,row).shape,obs.frame_rgb.shape)

    def test_depth_failure_keeps_2d_and_never_accepts(self):
        obs,_,_=fixture();xy=np.zeros((133,2));xy[7]=[80,100];xy[9]=[80,60];xy[10]=[10,100]
        body={'captured_mono_ns':obs.captured_mono_ns,'keypoints_xy':xy,'confidence':np.ones(133)}
        detector=PalmDetector.__new__(PalmDetector);detector.metadata={}
        detector.mp=SimpleNamespace(Image=lambda **kw:kw,ImageFormat=SimpleNamespace(SRGB=1))
        detector.detector=Mock()
        # ROI: x 16..144, y 0..110. Match body wrist exactly.
        detector.detector.detect.return_value=SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=.5,y=60/110) for _ in range(21)]])
        with patch('optusbproj_ubuntu.palm_worker.lift_palm',side_effect=ValueError('模拟深度空洞')):
            row=detector.infer(obs,body)
        self.assertTrue(row['detection_2d']);self.assertTrue(row['association_valid']);self.assertFalse(row['accepted'])
        self.assertEqual(row['stage'],'depth_geometry');self.assertEqual(row['reason'],'模拟深度空洞')
        self.assertGreater(row['frame_rgb'].size,0)

    def test_gui_failed_observation_visible_bounded_snapshots(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec')
            for timer in w.findChildren(QtCore.QTimer):timer.stop()
            d=PalmVisualDialog(w)
            for i in range(20):
                d.consume({'captured_mono_ns':i,'accepted':False,'stage':'association','reason':'腕点不一致',
                    'detected_hand_count':1,'detection_2d':True,'wrist_distance_px':50.,'wrist_gate_px':30.,'frame_rgb':np.zeros((100,100,3),np.uint8)})
            self.assertFalse(d.image.pixmap().isNull());self.assertIn('50.0',d.detection_status.text())
            self.assertEqual(len(d.diagnostic_images),3);self.assertFalse(d.fit_rows)
            saved=d.save_diagnostics(Path(tmp));self.assertEqual(len(saved),3)
            self.assertTrue(all((Path(tmp)/s['file']).is_file() for s in saved))
            d.close();w.close()

    def test_left_surface_frame_so3_and_semantic_axes(self):
        p,q=palm();r,metrics=frame_from_points(p,q)
        np.testing.assert_allclose(r[:,0],[0,0,1],atol=1e-12)
        np.testing.assert_allclose(r[:,2],[0,1,0],atol=1e-12)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12);self.assertAlmostEqual(np.linalg.det(r),1)
        self.assertLess(metrics['plane_rms_mm'],1e-10)
        rotated=Rotation.from_euler('z',30,degrees=True).as_matrix()
        rr,_=frame_from_points((rotated@p.T).T,(rotated@q.T).T)
        np.testing.assert_allclose(rr,rotated@r,atol=1e-12)

    def test_degenerate_side_back_or_nonplanar_rejected(self):
        p,q=palm()
        with self.assertRaises(ValueError):frame_from_points(p,q[:20])
        with self.assertRaises(ValueError):frame_from_points(np.tile(p[0],(5,1)),q)
        # Swap index/little finger association: normal becomes dorsal view.
        p[[1,4]]=p[[4,1]]
        with self.assertRaisesRegex(ValueError,'掌心'):frame_from_points(p,q)

    def test_true_depth_lifting_and_missing_depth_reject(self):
        obs,_,_=fixture();xy=np.tile([80.,55.],(21,1))
        for name in ('color_intrinsic','aligned_depth_intrinsic'):obs.rgbd.calibration[name].update(fx=400,fy=400)
        xy[[0,5,9,13,17]]=[[80,78],[60,43],[73,38],[87,38],[100,43]]
        r,quality,p=lift_palm(obs,xy)
        np.testing.assert_allclose(p[:,2],1.)
        np.testing.assert_allclose(r[:,0],[0,0,1],atol=1e-12)
        obs.rgbd.aligned_depth_mm[:]=0
        with self.assertRaises(ValueError):lift_palm(obs,xy)

    def test_c0_svd_fit_manual_comparison_and_independent_failure(self):
        c0=Rotation.from_euler('xyz',[3,-4,-35],degrees=True).as_matrix()
        manual=Rotation.from_euler('z',-25,degrees=True).as_matrix()
        rows=[]
        for i in range(20):
            d=Rotation.from_euler('xy',[30+i*.1,-40],degrees=True).as_matrix()
            rows.append({'D':d.tolist(),'C':(d@c0).tolist()})
        checks=deepcopy(rows);result=estimate_c0(rows,checks,manual)
        self.assertTrue(result['accepted']);self.assertLess(distance(result['C0_visual'],c0),1e-8)
        self.assertGreater(result['manual_visual_so3_deg'],9)
        wrong=Rotation.from_euler('z',40,degrees=True).as_matrix()
        for row in checks:row['C']=(wrong@np.asarray(row['C'])).tolist()
        failed=estimate_c0(rows,checks,manual)
        self.assertFalse(failed['accepted'])
        self.assertLess(distance(failed['C0_visual'],c0),1e-8)
        with self.assertRaises(ValueError):estimate_c0(rows[:5],checks,manual)

    def test_gui_entry_session_guard_and_explicit_apply(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{'left_hand':'A'},{},camera_source='orbbec')
            for timer in w.findChildren(QtCore.QTimer):timer.stop()
            e=w.engine;e.feed(imu_packet('A',0));e.set_algorithm_mode('left_hand',GYRO_TILT)
            e.capture_manual_standard_hand('left_hand',np.eye(3))
            d=PalmVisualDialog(w);w.palm_visual_dialog=d
            w.camera_worker=SimpleNamespace(capture_epoch='camera');w.rgbd_pose_enabled.blockSignals(True);w.rgbd_pose_enabled.setChecked(True);w.rgbd_pose_enabled.blockSignals(False)
            d.seed=e.make_natural_hand_reference('left_hand',np.eye(3),np.eye(3));d.camera_epoch='camera';d.vision_epoch=w.vision_epoch
            d.before=None
            c0=Rotation.from_euler('z',-35,degrees=True).as_matrix();d.manual_angle=-25;d.path=Path(tmp)
            d.report={'accepted':True,'C0_visual':c0.tolist(),'C0_manual':np.eye(3).tolist()}
            d.phase='train';self.assertFalse(w._start_session());self.assertIn('视觉C0',w._pose_change_error())
            d.phase='complete';d.preview(True)
            self.assertFalse(e.natural_hand_references);self.assertIn('left_hand',e.natural_hand_previews)
            with patch('optusbproj_ubuntu.palm_visual_gui.QtWidgets.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.No):d.apply_visual()
            self.assertFalse(e.natural_hand_references)
            with patch('optusbproj_ubuntu.palm_visual_gui.QtWidgets.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes):d.apply_visual()
            np.testing.assert_allclose(e.natural_hand_references['left_hand']['neutral_pose'],c0)
            self.assertFalse(e.natural_hand_previews)
            d.undo_visual();self.assertFalse(e.natural_hand_references)
            d.close();w.palm_visual_dialog=None;w.camera_worker=None;w.close()

    def test_guided_capture_fixed_body_frame_separate_validation_and_report(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{'left_hand':'A'},{},camera_source='orbbec')
            for timer in w.findChildren(QtCore.QTimer):timer.stop()
            e=w.engine;e.feed(imu_packet('A',0));e.set_algorithm_mode('left_hand',GYRO_TILT)
            e.set_axis_mapping('left_hand',('+X','+Y','+Z'));e.capture_manual_standard_hand('left_hand',np.eye(3))
            w.camera_worker=SimpleNamespace(capture_epoch='camera');w.rgbd_pose_enabled.blockSignals(True);w.rgbd_pose_enabled.setChecked(True);w.rgbd_pose_enabled.blockSignals(False)
            d=PalmVisualDialog(w);w.palm_visual_dialog=d;d.confirm.setChecked(True)
            clock=[time.monotonic_ns()];c0=Rotation.from_euler('z',-35,degrees=True).as_matrix()
            root=Path(tmp)/'reports';model=Path(tmp)/'hand.task';model.write_bytes(b'test')
            worker=Mock();worker.take_latest.return_value=None;worker.wait.return_value=True
            body={'valid':[True]*8,'support':[1.]*8,'confidence':[1.]*133,
                'points_camera_m':[[0,.2,1.5],[0,-.2,1.5],[0,.2,1.2],[0,-.2,1.2],[0,.2,1],[0,-.2,1],[0,.15,1],[0,-.15,1]]}
            with patch('optusbproj_ubuntu.palm_visual_gui.REPORT_ROOT',root),patch('optusbproj_ubuntu.palm_visual_gui.DEFAULT_HAND_MODEL',model),patch('optusbproj_ubuntu.palm_visual_gui.PalmWorker',return_value=worker),patch('optusbproj_ubuntu.palm_visual_gui.time.monotonic_ns',side_effect=lambda:clock[0]),patch('optusbproj_ubuntu.pose.time.monotonic',side_effect=lambda:clock[0]/1e9):
                d.begin();d.timer.stop()
                for _ in range(600):
                    if not d.active:break
                    clock[0]+=100_000_000;e.received_at['A']=clock[0]/1e9
                    r=(Rotation.from_euler('x',40,degrees=True).as_matrix() if d.phase in ('train','check') else np.eye(3))
                    e.mode_current['A'][GYRO_TILT]=r;e._sync_active('A')
                    w.live_fusion.at=lambda *_:r.copy()
                    if d.phase=='neutral':
                        row=deepcopy(body);row['captured_mono_ns']=clock[0]
                        d.on_body(SimpleNamespace(to_record=lambda:row,captured_mono_ns=clock[0]))
                    if d.phase in ('train','check'):
                        d.consume({'captured_mono_ns':clock[0],'accepted':True,'R_camera_hand':(r@c0).tolist(),'reason':'synthetic'})
                    d.tick()
                self.assertEqual(d.phase,'complete',d.status.text());self.assertTrue(d.report['accepted'],d.report)
                self.assertTrue((d.path/'report.json').is_file());self.assertFalse(e.natural_hand_references)
                np.testing.assert_allclose(d.K,np.eye(3),atol=1e-12)
                np.testing.assert_allclose(d.report['C0_visual'],c0,atol=1e-10)
                train_stamps={r['stamp_ns'] for r in d.report['training']};check_stamps={r['stamp_ns'] for r in d.report['validation']}
                self.assertFalse(train_stamps&check_stamps)
                d.manual.setValue(-20)
                self.assertEqual(d.report['manual_estimate_offset_deg'],-20)
                np.testing.assert_allclose(d.report['C0_visual'],c0,atol=1e-10)
                self.assertAlmostEqual(d.report['manual_visual_so3_deg'],15)
            d.close();w.palm_visual_dialog=None;w.camera_worker=None;w.close()


if __name__=='__main__':unittest.main()
