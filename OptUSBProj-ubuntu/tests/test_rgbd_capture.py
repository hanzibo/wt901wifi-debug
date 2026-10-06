"""SDK-free shape, lossless depth, replay and tail-trim tests."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from test_fusion_capture import packet
from optusbproj_ubuntu.fusion_capture import CameraObservation, RGBDData
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


def observation(stamp):
    depth = np.arange(24, dtype=np.uint16).reshape(4, 6)
    metric = np.arange(48, dtype=np.float32).reshape(6, 8) * .25
    return CameraObservation(stamp, "orbbec", np.zeros((6, 8, 3), dtype=np.uint8), None, None,
                             RGBDData(depth, metric, {"profiles": "test", "alignment": "D2C"},
                                      {"color_device_us": 100, "depth_device_us": 120,
                                       "raw_depth_scale_mm": .25}))


class RGBDTests(unittest.TestCase):
    def test_shape_and_depth_validation(self):
        with self.assertRaises(ValueError):
            CameraObservation(1, "orbbec", np.zeros((6, 8, 3)), None, None)
        d = observation(1).rgbd
        with self.assertRaises(ValueError):
            CameraObservation(1, "orbbec", np.zeros((3, 8, 3)), None, None, d)
        with self.assertRaises(ValueError):
            RGBDData(d.raw_depth.astype(float), d.aligned_depth_mm, {}, {})

    def test_record_replay_lossless_metric_and_raw_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = FusionSessionRecorder(Path(folder), tail_trim_seconds=0)
            o = observation(time.monotonic_ns())
            recorder.write_camera(o)
            recorder.write_imu(o.captured_mono_ns, packet())
            recorder.close()
            manifest = json.loads((recorder.path/"manifest.json").read_text())
            self.assertEqual(manifest["format"], "wt901-fusion-session-v2")
            self.assertEqual(manifest["rgbd_frames"], 1)
            replay = FusionSessionReplay(recorder.path)
            try:
                replay.poll_timed(now=replay._started_at+1)
                result = replay.take_camera()[0]
                np.testing.assert_array_equal(result.rgbd.raw_depth, o.rgbd.raw_depth)
                np.testing.assert_array_equal(result.rgbd.aligned_depth_mm, o.rgbd.aligned_depth_mm)
                self.assertEqual(result.rgbd.calibration, o.rgbd.calibration)
                self.assertEqual(result.rgbd.timing, o.rgbd.timing)
            finally:
                replay.close()

    def test_tail_trim_removes_depth_and_metadata_consistently(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = FusionSessionRecorder(Path(folder))
            recorder.started_mono_ns -= 10_000_000_000
            recorder.write_camera(observation(time.monotonic_ns()-4_000_000_000))
            recorder.write_camera(observation(time.monotonic_ns()-1_000_000_000))
            recorder.close()
            self.assertEqual(recorder.rgbd_count, 1)
            self.assertEqual(recorder.trimmed_camera_frames, 1)
            self.assertEqual(len(list((recorder.path/"depth").glob("*.npy"))), 2)

    def test_changed_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = FusionSessionRecorder(Path(folder), tail_trim_seconds=0)
            o = observation(time.monotonic_ns())
            recorder.write_camera(o)
            bad = CameraObservation(time.monotonic_ns(), "orbbec", o.frame_rgb, None, None,
                                    RGBDData(o.rgbd.raw_depth, o.rgbd.aligned_depth_mm, {}, {}))
            try:
                with self.assertRaises(ValueError):
                    recorder.write_camera(bad)
            finally:
                recorder.close()

    def test_writer_failure_close_does_not_deadlock(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = FusionSessionRecorder(Path(folder), tail_trim_seconds=0, max_pending_frames=1)
            with patch("numpy.save", side_effect=OSError("disk full")):
                recorder.write_camera(observation(time.monotonic_ns()))
                recorder._worker.join(timeout=2)
                self.assertFalse(recorder._worker.is_alive())
                with self.assertRaises(OSError):
                    recorder.close()
