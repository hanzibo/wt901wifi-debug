from __future__ import annotations

import csv
import json
import sys
import time
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from wt901.protocol import AttitudePacket
from wt901.gui_udp import GuiUdpSource
from optusbproj_ubuntu.fusion_capture import CameraObservation
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


def packet() -> AttitudePacket:
    return AttitudePacket(
        device_id="WT5500011381", year=2026, month=9, day=26,
        hour=10, minute=0, second=0, millisecond=0,
        acc_x=0, acc_y=0, acc_z=1, gyro_x=0, gyro_y=0, gyro_z=0,
        mag_x=20, mag_y=0, mag_z=40, roll=12, pitch=0, yaw=0,
        temperature=25, battery_v=3.9, rssi=-50, version=1)


class FusionCaptureTests(unittest.TestCase):
    def test_imu_only_session_does_not_require_camera_backend(self):
        with TemporaryDirectory() as directory:
            recorder = FusionSessionRecorder(Path(directory), tail_trim_seconds=0)
            recorder.write_imu(time.monotonic_ns(), packet())
            recorder.close()
            self.assertEqual(recorder.imu_count, 1)
            self.assertEqual((recorder.path / "camera.jsonl").read_text(), "")

    def test_udp_source_timestamp_keeps_legacy_poll_compatible(self):
        raw = bytearray(54)
        raw[:12] = b"WT5500011381"
        raw[52:] = b"\r\n"

        class FakeSocket:
            def __init__(self):
                self.pending = [(bytes(raw), ("10.42.0.174", 9250))]

            def recvfrom(self, _size):
                if not self.pending:
                    raise BlockingIOError
                return self.pending.pop(0)

        source = object.__new__(GuiUdpSource)
        source.sock = FakeSocket()
        source._buffers = {}
        source._devices = {}
        source._endpoints = {}
        source._latest_packets = {}
        source._replies = deque()
        timed = source.poll_timed()
        self.assertEqual(len(timed), 1)
        self.assertGreater(timed[0][0], 0)
        self.assertEqual(timed[0][1].device_id, "WT5500011381")
        self.assertIs(source.latest_packet("WT5500011381"), timed[0][1])
        self.assertIsNone(source.latest_packet("WT5500011381", max_age=-1))
        source.sock.pending.append((bytes(raw), ("10.42.0.174", 9250)))
        self.assertEqual(source.poll()[0].device_id, "WT5500011381")

    def test_camera_observation_shapes(self):
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        points = np.zeros((133, 2))
        scores = np.ones(133)
        observation = CameraObservation(10, "rtmw3d", frame, points, scores)
        self.assertEqual(observation.image_size, (12, 8))
        with self.assertRaises(ValueError):
            CameraObservation(10, "rtmw3d", frame, points[:33], scores[:33])

    def test_record_and_replay_preserves_raw_2d_and_monotonic_times(self):
        with TemporaryDirectory() as directory:
            recorder = FusionSessionRecorder(
                Path(directory), tail_trim_seconds=0,
                assignments={"right_upper_arm": "WT5500011381"},
                axis_mappings={"right_upper_arm": ("+Y", "-X", "+Z")})
            captured_ns = time.monotonic_ns()
            received_ns = captured_ns + 10_000_000
            points = np.full((133, 2), np.nan)
            points[5] = [120.5, 42.25]
            scores = np.zeros(133)
            scores[5] = 0.95
            frame = np.full((8, 12, 3), 128, dtype=np.uint8)
            recorder.write_camera(CameraObservation(
                captured_ns, "rtmw3d", frame, points, scores))
            recorder.write_imu(received_ns, packet())
            time.sleep(0.02)
            recorder.close()
            with (recorder.path / "imu.csv").open(newline="", encoding="utf-8") as source:
                imu_row = next(csv.DictReader(source))
            self.assertEqual(int(imu_row["host_recv_mono_ns"]), received_ns)
            camera_row = json.loads((recorder.path / "camera.jsonl").read_text())
            manifest = json.loads((recorder.path / "manifest.json").read_text())
            self.assertEqual(manifest["assignments_at_start"],
                             {"right_upper_arm": "WT5500011381"})
            self.assertEqual(manifest["axis_mappings_at_start"],
                             {"right_upper_arm": ["+Y", "-X", "+Z"]})
            self.assertEqual(camera_row["captured_mono_ns"], captured_ns)
            self.assertIsNone(camera_row["keypoints_xy"][0][0])
            self.assertEqual(camera_row["keypoints_xy"][5], [120.5, 42.25])
            replay = FusionSessionReplay(recorder.path)
            try:
                self.assertEqual(replay.poll_timed(now=replay._started_at), [])
                frames = replay.take_camera()
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].keypoints_xy[5].tolist(), [120.5, 42.25])
                self.assertTrue(np.isnan(frames[0].keypoints_xy[0]).all())
                due = replay.poll_timed(now=replay._started_at + 0.03)
                self.assertEqual([(stamp, item.device_id) for stamp, item in due],
                                 [(received_ns, "WT5500011381")])
            finally:
                replay.close()

    def test_last_three_seconds_are_excluded_from_both_streams(self):
        with TemporaryDirectory() as directory:
            recorder = FusionSessionRecorder(Path(directory))
            recorder.started_mono_ns -= 10_000_000_000  # synthetic session clock
            now = time.monotonic_ns()
            old_ns = now - 4_000_000_000
            tail_ns = now - 1_000_000_000
            frame = np.zeros((8, 12, 3), dtype=np.uint8)
            recorder.write_camera(CameraObservation(
                old_ns, "rtmw3d", frame, None, None))
            recorder.write_imu(old_ns, packet())
            recorder.write_camera(CameraObservation(
                tail_ns, "rtmw3d", frame, None, None))
            recorder.write_imu(tail_ns, packet())
            recorder.close()
            with (recorder.path / "imu.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            camera_rows = (recorder.path / "camera.jsonl").read_text().splitlines()
            manifest = json.loads((recorder.path / "manifest.json").read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["host_recv_mono_ns"]), old_ns)
            self.assertEqual(len(camera_rows), 1)
            self.assertEqual(len(list((recorder.path / "frames").glob("*.jpg"))), 1)
            self.assertEqual(manifest["tail_trim_seconds"], 3.0)
            self.assertEqual(manifest["trimmed_imu_packets"], 1)
            self.assertEqual(manifest["trimmed_camera_frames"], 1)
            replay = FusionSessionReplay(recorder.path)
            try:
                self.assertEqual(len(replay.poll_timed(now=replay._started_at)), 1)
                self.assertEqual(len(replay.take_camera()), 1)
                self.assertEqual(replay.poll_timed(now=replay._started_at + 10), [])
                self.assertEqual(replay.take_camera(), [])
            finally:
                replay.close()


if __name__ == "__main__":
    unittest.main()
