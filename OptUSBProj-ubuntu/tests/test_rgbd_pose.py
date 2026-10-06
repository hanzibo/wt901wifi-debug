import json
from pathlib import Path
import tempfile
import time
import unittest
import sys
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.fusion_capture import CameraObservation, RGBDData
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay
from optusbproj_ubuntu.rgbd_pose import DepthPoseEstimator, INDICES, RGBDPoseEstimate, camera_matrix, distortion_coefficients
from optusbproj_ubuntu.rgbd_pose_worker import infer_observation


def fixture(stamp=None, distortion=0.):
    intrinsic = dict(width=160, height=120, fx=100., fy=100., cx=80., cy=60.)
    calibration = dict(color_intrinsic=intrinsic, aligned_depth_intrinsic=intrinsic,
                       color_distortion=dict(model="BROWN_CONRADY", k1=distortion),
                       aligned_depth_distortion=dict(model="BROWN_CONRADY"))
    obs = CameraObservation(stamp or time.monotonic_ns(), "orbbec", np.zeros((120, 160, 3), np.uint8), None, None,
                            RGBDData(np.full((120, 160), 1000, np.uint16),
                                     np.full((120, 160), 1000., np.float32), calibration, {}))
    xy = np.zeros((133, 2))
    xy[INDICES] = [[60, 30], [100, 30], [45, 50], [115, 50], [30, 70], [130, 70], [65, 85], [95, 85]]
    return obs, xy, np.ones(133)


class DepthPoseTests(unittest.TestCase):
    def test_metric_lifting_and_json_roundtrip(self):
        obs, xy, scores = fixture()
        result = DepthPoseEstimator().estimate(obs, xy, scores)
        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(result.points_camera_m[0], [-.2, -.3, 1])
        self.assertAlmostEqual(result.bone_lengths_m["left_upper_arm"], .25)
        restored = RGBDPoseEstimate.from_record(json.loads(json.dumps(result.to_record(), allow_nan=False)))
        np.testing.assert_allclose(restored.points_camera_m, result.points_camera_m)

    def test_distorted_rgb_maps_to_rectified_depth_pixel(self):
        obs, xy, scores = fixture(distortion=.4)
        k = camera_matrix(obs.rgbd.calibration["color_intrinsic"])
        d = distortion_coefficients(obs.rgbd.calibration["color_distortion"])
        ray = np.array([[.3, .2, 1.]])
        point, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), k, d)
        xy[5] = point[0, 0]
        result = DepthPoseEstimator().estimate(obs, xy, scores)
        np.testing.assert_allclose(result.depth_pixels[0], [110, 80], atol=1e-4)
        np.testing.assert_allclose(result.points_camera_m[0], [.3, .2, 1.], atol=1e-5)

    def test_depth_holes_edges_and_low_confidence(self):
        obs, xy, scores = fixture()
        obs.rgbd.aligned_depth_mm[27:34, 57:64] = 0
        obs.rgbd.aligned_depth_mm[27:34, 100:104] = 3000
        scores[7] = .1
        result = DepthPoseEstimator().estimate(obs, xy, scores)
        self.assertFalse(result.valid[:3].any())
        self.assertIn("支持不足", result.reasons[0])
        self.assertIn("混合", result.reasons[1])
        self.assertIn("置信度", result.reasons[2])
        self.assertTrue(np.isnan(result.points_camera_m[:3]).all())
        json.dumps(result.to_record(), allow_nan=False)

    def test_temporal_jump_not_held_as_valid(self):
        obs, xy, scores = fixture()
        estimator = DepthPoseEstimator()
        estimator.estimate(obs, xy, scores)
        new, xy, scores = fixture(obs.captured_mono_ns+10_000_000)
        new.rgbd.aligned_depth_mm[:] = 1500
        result = estimator.estimate(new, xy, scores)
        self.assertFalse(result.valid.any())
        self.assertTrue(all("跳变" in reason for reason in result.reasons))

    def test_unknown_distortion_rejected(self):
        obs, xy, scores = fixture()
        obs.rgbd.calibration["color_distortion"] = dict(model="UNKNOWN", k1=.1)
        with self.assertRaises(ValueError):
            DepthPoseEstimator().estimate(obs, xy, scores)

    def test_smooth_occluder_rejected_by_arm_ratio(self):
        obs, xy, scores = fixture()
        obs.rgbd.aligned_depth_mm[67:74, 27:34] = 400
        result = DepthPoseEstimator().estimate(obs, xy, scores)
        self.assertFalse(result.valid[4])
        self.assertIn("比例异常", result.reasons[4])

    def test_inference_discards_predicted_xyz_and_rejects_multi_person(self):
        obs, xy, scores = fixture()
        model = SimpleNamespace(det_model=lambda image: np.ones((1, 4)),
                                pose_model=lambda image, bboxes: (np.full((1, 133, 3), 999999), scores[None], None, xy[None]))
        result = infer_observation(model, DepthPoseEstimator(), obs)
        self.assertTrue(result.valid.all())
        self.assertAlmostEqual(result.points_camera_m[0, 2], 1.)
        model.det_model = lambda image: np.ones((2, 4))
        result = infer_observation(model, DepthPoseEstimator(), obs)
        self.assertFalse(result.valid.any())
        self.assertIn("多人", result.diagnostics["reject_reason"])

    def test_recorded_vision_associated_with_source_frame_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = FusionSessionRecorder(Path(folder), tail_trim_seconds=0)
            obs, xy, scores = fixture()
            recorder.write_camera(obs)
            recorder.write_vision(DepthPoseEstimator().estimate(obs, xy, scores))
            recorder.close()
            row = json.loads((recorder.path/"vision.jsonl").read_text())
            self.assertEqual(row["captured_mono_ns"], obs.captured_mono_ns)
            self.assertTrue((recorder.path/row["frame"]).is_file())
            replay = FusionSessionReplay(recorder.path)
            try:
                self.assertTrue(replay.has_vision)
                replay.poll_timed(replay._started_at+1)
                self.assertEqual(len(replay.take_vision()), 1)
            finally:
                replay.close()

    def test_unmatched_and_tail_vision_not_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = FusionSessionRecorder(Path(folder), tail_trim_seconds=3)
            recorder.started_mono_ns -= 10_000_000_000
            obs, xy, scores = fixture(time.monotonic_ns()-4_000_000_000)
            recorder.write_vision(DepthPoseEstimator().estimate(obs, xy, scores))
            obs, xy, scores = fixture()
            recorder.write_camera(obs)
            recorder.write_vision(DepthPoseEstimator().estimate(obs, xy, scores))
            recorder.close()
            self.assertEqual(recorder.vision_count, 0)
            self.assertEqual(recorder.unmatched_vision, 1)
            self.assertEqual(recorder.trimmed_vision, 1)


if __name__ == "__main__":
    unittest.main()
