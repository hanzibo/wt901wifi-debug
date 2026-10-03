"""Time-paced replay of a recorded WT901 + camera session."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from wt901.protocol import AttitudePacket

from .fusion_capture import CameraObservation, RGBDData
from .fusion_recording import IMU_FIELDS
from .rgbd_pose import RGBDPoseEstimate
from .recording import INTEGER_FIELDS, PACKET_FIELDS


class FusionSessionReplay:
    is_replay = True

    def __init__(self, path: Path, *, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("回放速度必须大于 0")
        self.path = path
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") not in ("wt901-fusion-session-v1", "wt901-fusion-session-v2"):
            raise ValueError("不是可识别的融合录制目录")
        self.camera_calibration = (json.loads((path / "camera_calibration.json").read_text())
                                   if manifest.get("format") == "wt901-fusion-session-v2" else None)
        self.speed = speed
        self.imu_file = (path / "imu.csv").open(newline="", encoding="utf-8")
        self.runtime_file=((path/'runtime_pose.jsonl').open() if (path/'runtime_pose.jsonl').is_file() else None)
        self._next_runtime=self._read_runtime()
        self._runtime_due={}
        self.camera_file = (path / "camera.jsonl").open(encoding="utf-8")
        self.vision_file = ((path / "vision.jsonl").open(encoding="utf-8")
                            if (path / "vision.jsonl").is_file() else None)
        self._next_vision = self._read_vision()
        self._vision_due = []
        self.has_vision = self._next_vision is not None
        self.imu_reader = csv.DictReader(self.imu_file)
        if self.imu_reader.fieldnames != list(IMU_FIELDS):
            self.close()
            raise ValueError("IMU CSV 字段不匹配")
        self._next_imu = self._read_imu()
        self._next_camera = self._read_camera()
        first_times = [value for value in (
            self._next_imu[0] if self._next_imu else None,
            self._next_camera["captured_mono_ns"] if self._next_camera else None)
            if value is not None]
        self._first_ns = min(first_times) if first_times else 0
        self._started_at = time.monotonic()
        self._camera_due: list[CameraObservation] = []
        self._seen: dict[str, float] = {}

    def _read_imu(self) -> tuple[int, AttitudePacket] | None:
        row = next(self.imu_reader, None)
        if row is None:
            return None
        values = {name: (row[name] if name == "device_id" else
                         int(row[name]) if name in INTEGER_FIELDS else float(row[name]))
                  for name in PACKET_FIELDS}
        return int(row["host_recv_mono_ns"]), AttitudePacket(**values)

    def _read_camera(self) -> dict | None:
        line = self.camera_file.readline()
        return json.loads(line) if line else None

    def _read_vision(self):
        line = self.vision_file.readline() if self.vision_file else ""
        return json.loads(line) if line else None

    def _read_runtime(self):
        line=self.runtime_file.readline() if self.runtime_file else ''
        return json.loads(line) if line else None

    def take_runtime(self):
        rows,self._runtime_due=self._runtime_due,{}
        return rows

    def poll_timed(self, now: float | None = None) -> list[tuple[int, AttitudePacket]]:
        moment = time.monotonic() if now is None else now
        due_ns = self._first_ns + int((moment - self._started_at) * self.speed * 1e9)
        packets = []
        while (self._next_imu is not None and self._next_imu[0] <= due_ns
               and len(packets) < 512):
            received_ns, packet = self._next_imu
            packets.append((received_ns, packet))
            while self._next_runtime is not None and self._next_runtime['host_recv_mono_ns']<=received_ns:
                row=self._next_runtime
                self._runtime_due[(row['host_recv_mono_ns'],row['device_id'])]=row
                self._next_runtime=self._read_runtime()
            self._seen[packet.device_id] = moment
            self._next_imu = self._read_imu()
        while (self._next_camera is not None
               and self._next_camera["captured_mono_ns"] <= due_ns
               and len(self._camera_due) < 32):
            row = self._next_camera
            import cv2
            bgr = cv2.imread(str(self.path / row["frame"]))
            if bgr is None:
                raise OSError(f"无法读取录制帧：{row['frame']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            xy = (None if row["keypoints_xy"] is None else
                  np.asarray(row["keypoints_xy"], dtype=float))
            confidence = (None if row["confidence"] is None else
                          np.asarray(row["confidence"], dtype=float))
            rgbd = None
            if row.get("rgbd") is not None:
                depth = row["rgbd"]
                rgbd = RGBDData(np.load(self.path / depth["raw_depth"], allow_pickle=False),
                                np.load(self.path / depth["aligned_depth_mm"], allow_pickle=False),
                                self.camera_calibration, depth["timing"])
            self._camera_due.append(CameraObservation(
                int(row["captured_mono_ns"]), row["backend"], rgb,
                xy, confidence, rgbd))
            self._next_camera = self._read_camera()
        while self._next_vision is not None and self._next_vision["captured_mono_ns"] <= due_ns and len(self._vision_due) < 32:
            import cv2
            row = self._next_vision
            bgr = cv2.imread(str(self.path / row["frame"]))
            if bgr is None:
                raise OSError(f"无法读取视觉源帧：{row['frame']}")
            self._vision_due.append(RGBDPoseEstimate.from_record(row, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            self._next_vision = self._read_vision()
        return packets

    def take_vision(self):
        due, self._vision_due = self._vision_due, []
        return due

    def take_camera(self) -> list[CameraObservation]:
        due, self._camera_due = self._camera_due, []
        return due

    def poll(self) -> list[AttitudePacket]:
        return [packet for _, packet in self.poll_timed()]

    def take_replies(self) -> list:
        return []

    def seen_devices(self, max_age: float = 3.0) -> dict[str, tuple[str, float]]:
        now = time.monotonic()
        return {device_id: ("回放", seen) for device_id, seen in self._seen.items()
                if now - seen <= max_age}

    def seen_endpoints(self, max_age: float = 3.0) -> dict[str, tuple[str, int, float]]:
        return {device_id: ("回放", 0, seen)
                for device_id, (_, seen) in self.seen_devices(max_age).items()}

    def close(self) -> None:
        self.imu_file.close()
        self.camera_file.close()
        if self.runtime_file:self.runtime_file.close()
        if self.vision_file:
            self.vision_file.close()
