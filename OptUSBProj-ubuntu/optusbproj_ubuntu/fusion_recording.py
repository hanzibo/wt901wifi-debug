"""Record one camera/WT901 session without blocking image capture on disk I/O."""

from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from queue import Full, Queue
from threading import Thread
import time
from uuid import uuid4

import numpy as np

from wt901.protocol import AttitudePacket

from .fusion_capture import CameraObservation
from .recording import PACKET_FIELDS


IMU_FIELDS = ("host_recv_mono_ns", "host_record_mono_ns") + PACKET_FIELDS


def _json_points(values: np.ndarray | None) -> list | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in array]
    return [[float(value) if np.isfinite(value) else None for value in row]
            for row in array]


class FusionSessionRecorder:
    """One session directory: IMU CSV, camera JSONL, and unannotated JPEGs.

    Recent samples stay in memory until they are older than the configured
    tail window. On close, that window is discarded from both streams. Older
    camera frames use a bounded background queue; disk overload is counted.
    """

    def __init__(self, parent: Path, *, jpeg_quality: int = 85,
                 max_pending_frames: int = 64,
                 tail_trim_seconds: float = 3.0,
                 assignments: dict[str, str] | None = None,
                 axis_mappings: dict[str, tuple[str, str, str]] | None = None,
                 algorithm_modes: dict[str, str] | None = None) -> None:
        if (not 1 <= jpeg_quality <= 100 or max_pending_frames < 1
                or tail_trim_seconds < 0):
            raise ValueError("录制参数无效")
        self.path = parent / (datetime.now().strftime("%Y%m%d-%H%M%S")
                              + "-" + uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir()
        self.started_mono_ns = time.monotonic_ns()
        self.started_wall_ns = time.time_ns()
        self.jpeg_quality = jpeg_quality
        self.tail_trim_seconds = tail_trim_seconds
        self.assignments = dict(assignments or {})
        self.axis_mappings = {key: list(value) for key, value in
                              (axis_mappings or {}).items()}
        self.algorithm_modes = dict(algorithm_modes or {})
        self.tail_trim_ns = round(tail_trim_seconds * 1e9)
        self.imu_file = (self.path / "imu.csv").open("w", newline="", encoding="utf-8")
        self.imu_writer = csv.DictWriter(self.imu_file, fieldnames=IMU_FIELDS)
        self.imu_writer.writeheader()
        self.imu_count = 0
        self.frame_count = 0
        self.dropped_frames = 0
        self.trimmed_imu_packets = 0
        self.trimmed_camera_frames = 0
        self._submitted_frames = 0
        self._imu_pending: deque[tuple[int, int, AttitudePacket]] = deque()
        self._camera_pending: deque[CameraObservation] = deque()
        self.error: str | None = None
        self._closed = False
        self._queue: Queue[tuple[int, CameraObservation] | None] = Queue(
            maxsize=max_pending_frames)
        self._worker = Thread(target=self._write_frames, name="fusion-recorder")
        self._worker.start()

    def write_imu(self, received_ns: int, packet: AttitudePacket) -> None:
        if self._closed:
            raise RuntimeError("录制已结束")
        if received_ns < self.started_mono_ns:
            return
        recorded_ns = time.monotonic_ns()
        self._imu_pending.append((received_ns, recorded_ns, packet))
        self._flush_ready(recorded_ns - self.tail_trim_ns)

    def write_camera(self, observation: CameraObservation) -> bool:
        if self._closed:
            raise RuntimeError("录制已结束")
        if self.error is not None:
            raise OSError(self.error)
        if observation.captured_mono_ns < self.started_mono_ns:
            return False
        self._camera_pending.append(observation)
        self._flush_ready(time.monotonic_ns() - self.tail_trim_ns)
        return True

    def _flush_ready(self, cutoff_ns: int) -> None:
        """Commit only samples outside the trailing exclusion window."""
        while self._imu_pending and self._imu_pending[0][0] <= cutoff_ns:
            received_ns, recorded_ns, packet = self._imu_pending.popleft()
            row = {name: getattr(packet, name) for name in PACKET_FIELDS}
            row["host_recv_mono_ns"] = received_ns
            row["host_record_mono_ns"] = recorded_ns
            self.imu_writer.writerow(row)
            self.imu_count += 1
            if self.imu_count % 100 == 0:
                self.imu_file.flush()
        while (self._camera_pending
               and self._camera_pending[0].captured_mono_ns <= cutoff_ns):
            observation = self._camera_pending.popleft()
            self._submitted_frames += 1
            try:
                self._queue.put_nowait((self._submitted_frames, observation))
            except Full:
                self.dropped_frames += 1

    def _write_frames(self) -> None:
        try:
            with (self.path / "camera.jsonl").open("w", encoding="utf-8") as output:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    import cv2

                    number, observation = item
                    filename = f"frames/{number:08d}.jpg"
                    bgr = cv2.cvtColor(observation.frame_rgb, cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(
                        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if not ok:
                        raise OSError("相机帧 JPEG 编码失败")
                    (self.path / filename).write_bytes(encoded.tobytes())
                    width, height = observation.image_size
                    row = {
                        "frame": filename,
                        "captured_mono_ns": observation.captured_mono_ns,
                        "backend": observation.backend,
                        "width": width,
                        "height": height,
                        "keypoints_xy": _json_points(observation.keypoints_xy),
                        "confidence": _json_points(observation.confidence),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        output.flush()
        except Exception as exc:
            self.error = f"写入相机帧失败：{exc}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stopped_mono_ns = time.monotonic_ns()
        retained_until_ns = stopped_mono_ns - self.tail_trim_ns
        self._flush_ready(retained_until_ns)
        self.trimmed_imu_packets = len(self._imu_pending)
        self.trimmed_camera_frames = len(self._camera_pending)
        self._imu_pending.clear()
        self._camera_pending.clear()
        if self._worker.is_alive():
            self._queue.put(None)
        self._worker.join()
        self.imu_file.close()
        manifest = {
            "format": "wt901-fusion-session-v1",
            "started_wall_ns": self.started_wall_ns,
            "started_mono_ns": self.started_mono_ns,
            "ended_mono_ns": stopped_mono_ns,
            "retained_until_mono_ns": retained_until_ns,
            "tail_trim_seconds": self.tail_trim_seconds,
            "imu_packets": self.imu_count,
            "camera_frames": self.frame_count,
            "dropped_camera_frames": self.dropped_frames,
            "trimmed_imu_packets": self.trimmed_imu_packets,
            "trimmed_camera_frames": self.trimmed_camera_frames,
            "error": self.error,
            "note": "host_recv_mono_ns is UDP arrival, not device sample time",
            "assignments_at_start": self.assignments,
            "axis_mappings_at_start": self.axis_mappings,
            "algorithm_modes_at_start": self.algorithm_modes,
        }
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        if self.error is not None:
            raise OSError(self.error)

    def __enter__(self) -> "FusionSessionRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
