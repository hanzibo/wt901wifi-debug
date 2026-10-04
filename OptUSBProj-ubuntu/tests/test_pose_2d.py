import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.pose_2d import Pose2DConfig,YOLOBackend
from optusbproj_ubuntu.rgbd_pose import DepthPoseEstimator,INDICES
from optusbproj_ubuntu.rgbd_pose_worker import infer_observation
from optusbproj_ubuntu.pose_quality import PoseQualityStats
from test_rgbd_pose import fixture
from rgbd_compare import compare


class BackendTests(unittest.TestCase):
    def test_missing_torch_names_interpreter_and_correct_launcher(self):
        with patch.dict(sys.modules,{'torch':None}):
            with self.assertRaisesRegex(RuntimeError,'OptUSBProj-yolo') as error:
                YOLOBackend(Pose2DConfig('yolo26'))
            self.assertIn(sys.executable,str(error.exception))

    def yolo(self,xy,scores,count=1):
        backend=YOLOBackend.__new__(YOLOBackend)
        backend.config=Pose2DConfig('yolo26');backend.metadata=backend.config.metadata()
        backend.model=Mock()
        backend.model.predict.return_value=[SimpleNamespace(boxes=[None]*count,keypoints=SimpleNamespace(xy=xy[None,:17],conf=scores[None,:17]))]
        return backend

    def test_yolo_adapter_original_pixels_rgb_conversion_and_real_depth(self):
        obs,xy,scores=fixture();obs.frame_rgb[:]=[10,20,30]
        detector=self.yolo(xy,scores)
        result=infer_observation(detector,DepthPoseEstimator(),obs)
        self.assertTrue(result.valid.all());self.assertTrue(result.diagnostics['valid_2d']==[True]*8)
        np.testing.assert_allclose(result.keypoints_xy[:17],xy[:17])
        self.assertTrue(np.isnan(result.keypoints_xy[17:]).all())
        self.assertTrue((result.confidence[17:]==0).all())
        np.testing.assert_allclose(result.points_camera_m[0],[-.2,-.3,1])
        image=detector.model.predict.call_args.args[0]
        np.testing.assert_equal(image[0,0],[30,20,10])
        self.assertEqual(detector.model.predict.call_args.kwargs['device'],0)
        self.assertEqual(result.diagnostics['vision_config']['native_keypoints'],17)
        json.dumps(result.to_record(),allow_nan=False)

    def test_yolo_rejects_no_person_and_multiple_people(self):
        obs,xy,scores=fixture();estimator=DepthPoseEstimator()
        infer_observation(self.yolo(xy,scores),estimator,obs)
        for count in (0,2):
            result=infer_observation(self.yolo(xy,scores,count),estimator,obs)
            self.assertFalse(result.valid.any());self.assertFalse(any(result.diagnostics['valid_2d']))
            self.assertIn('reject_reason',result.diagnostics)

    def test_2d_usable_but_depth_invalid_are_distinguished(self):
        obs,xy,scores=fixture();obs.rgbd.aligned_depth_mm[:]=0
        result=infer_observation(self.yolo(xy,scores),DepthPoseEstimator(),obs)
        self.assertTrue(all(result.diagnostics['valid_2d']));self.assertFalse(result.valid.any())
        xy[7]=[-1,20];scores[9]=.1
        result=infer_observation(self.yolo(xy,scores),DepthPoseEstimator(),obs)
        self.assertFalse(result.diagnostics['valid_2d'][2]);self.assertFalse(result.diagnostics['valid_2d'][4])

    def test_config_identity_changes_for_weights_and_imgsz(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'old')
            a=Pose2DConfig('yolo26',path).metadata()['signature']
            b=Pose2DConfig('yolo26',path,960).metadata()['signature']
            self.assertNotEqual(a,b)
            path.write_bytes(b'changed')
            self.assertNotEqual(a,Pose2DConfig('yolo26',path).metadata()['signature'])

    def test_quality_statistics_and_disjoint_windows(self):
        obs,xy,scores=fixture();stats=PoseQualityStats()
        good=infer_observation(self.yolo(xy,scores),DepthPoseEstimator(),obs)
        bad=infer_observation(self.yolo(xy,scores,0),DepthPoseEstimator(),obs)
        for t,r in enumerate((good,bad,bad,good)):stats.update(r,time_s=t,include_timing=t>0)
        report=stats.report();self.assertEqual(report['frames'],4);self.assertEqual(report['timed_frames'],3)
        elbow=report['joints']['left_elbow']
        self.assertEqual(elbow['available_2d_rate'],.5);self.assertEqual(elbow['longest_failure_frames_2d'],2)
        self.assertEqual(elbow['longest_failure_span_s_2d'],1)
        stats.break_sequence();stats.update(bad,time_s=100)
        self.assertEqual(stats.report()['joints']['left_elbow']['longest_failure_span_s_2d'],1)

    def test_comparison_rejects_mismatched_source_frames_and_gates(self):
        with TemporaryDirectory() as directory:
            paths=[Path(directory)/n for n in ('a','b')]
            for p in paths:
                p.mkdir();(p/'report.json').write_text(json.dumps({'source_session':'source','quality':{},'side_raise':{}}))
                (p/'results.jsonl').write_text(json.dumps({'captured_mono_ns':1,'frame':'a.jpg','diagnostics':{'gates':{'score':.6}}}))
            self.assertEqual(compare(*paths)['paired_frames'],1)
            (paths[1]/'results.jsonl').write_text(json.dumps({'captured_mono_ns':2,'frame':'a.jpg'}))
            with self.assertRaisesRegex(ValueError,'源帧'):compare(*paths)
            (paths[1]/'results.jsonl').write_text(json.dumps({'captured_mono_ns':1,'frame':'a.jpg','diagnostics':{'gates':{'score':.7}}}))
            with self.assertRaisesRegex(ValueError,'门限'):compare(*paths)


if __name__=='__main__':unittest.main()
