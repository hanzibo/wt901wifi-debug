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
        self._vision_pending: deque[dict] = deque()
        self._camera_numbers: dict[int, int] = {}
        self.vision_count = self.dropped_vision = self.unmatched_vision = self.trimmed_vision = 0
        self.error: str | None = None
        self.camera_calibration: dict | None = None
        self.camera_last_timing: dict = {}
        self.camera_bridge_skipped = 0
        self.rgbd_count = 0
        self.runtime_count = 0
        self._runtime_pending = {}
        self.runtime_file = (self.path/'runtime_pose.jsonl').open('w',encoding='utf-8')
        self.continuous_basis_at_start = {}
        self.camera_epoch=None
        self.fusion_at_start={}
        self._closed = False
        self._queue: Queue[tuple[int, CameraObservation] | tuple[str, dict] | None] = Queue(
            maxsize=max_pending_frames)
        self._worker = Thread(target=self._write_frames, name="fusion-recorder")
        self._worker.start()

    def write_imu(self, received_ns: int, packet: AttitudePacket, runtime=None) -> None:
        if self._closed:
            raise RuntimeError("录制已结束")
        if received_ns < self.started_mono_ns:
            return
        recorded_ns = time.monotonic_ns()
        self._imu_pending.append((received_ns, recorded_ns, packet))
        if runtime is not None:self._runtime_pending[(received_ns,packet.device_id)]=runtime
        self._flush_ready(recorded_ns - self.tail_trim_ns)

    def write_camera(self, observation: CameraObservation) -> bool:
        if self._closed:
            raise RuntimeError("录制已结束")
        if self.error is not None:
            raise OSError(self.error)
        if observation.captured_mono_ns < self.started_mono_ns:
            return False
        if observation.rgbd is not None:
            if self.camera_calibration is not None and self.camera_calibration != observation.rgbd.calibration:
                raise ValueError("录制中相机标定/profile 改变，请结束并重新录制")
            self.camera_calibration = observation.rgbd.calibration
            self.camera_last_timing = dict(observation.rgbd.timing)
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
            runtime=self._runtime_pending.pop((received_ns,packet.device_id),None)
            if runtime is not None:
                self.runtime_file.write(json.dumps(runtime,ensure_ascii=False,allow_nan=False)+'\n')
                self.runtime_count+=1
            self.imu_count += 1
            if self.imu_count % 100 == 0:
                self.imu_file.flush()
        while (self._camera_pending
               and self._camera_pending[0].captured_mono_ns <= cutoff_ns):
            observation = self._camera_pending.popleft()
            self._submitted_frames += 1
            try:
                self._queue.put_nowait((self._submitted_frames, observation))
                self._camera_numbers[observation.captured_mono_ns] = self._submitted_frames
            except Full:
                self.dropped_frames += 1
        while self._vision_pending and self._vision_pending[0]["captured_mono_ns"] <= cutoff_ns:
            row = self._vision_pending.popleft()
            number = self._camera_numbers.get(row["captured_mono_ns"])
            if number is None:
                self.unmatched_vision += 1
                continue
            row["frame"] = f"frames/{number:08d}.jpg"
            try:
                self._queue.put_nowait(("vision", row))
            except Full:
                self.dropped_vision += 1

    def write_vision(self, estimate):
        """Only small numeric results; bind to the exact saved source frame."""
        if self._closed:
            raise RuntimeError("录制已结束")
        if self.error is not None:
            raise OSError(self.error)
        if estimate.captured_mono_ns < self.started_mono_ns:
            return
        self._vision_pending.append(estimate.to_record())
        self._flush_ready(time.monotonic_ns()-self.tail_trim_ns)

    def _write_frames(self) -> None:
        try:
            with (self.path / "camera.jsonl").open("w", encoding="utf-8") as output, \
                    (self.path / "vision.jsonl").open("w", encoding="utf-8") as vision:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    import cv2

                    number, observation = item
                    if number == "vision":
                        vision.write(json.dumps(observation, ensure_ascii=False, allow_nan=False)+"\n")
                        self.vision_count += 1
                        continue
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
                    if observation.rgbd is not None:
                        (self.path / "depth").mkdir(exist_ok=True)
                        raw_path = f"depth/{number:08d}_raw.npy"
                        metric_path = f"depth/{number:08d}_aligned_mm.npy"
                        np.save(self.path / raw_path, observation.rgbd.raw_depth, allow_pickle=False)
                        np.save(self.path / metric_path, observation.rgbd.aligned_depth_mm, allow_pickle=False)
                        row["rgbd"] = {
                            "raw_depth": raw_path, "aligned_depth_mm": metric_path,
                            "calibration": "camera_calibration.json",
                            "timing": observation.rgbd.timing,
                        }
                        self.rgbd_count += 1
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
        self.trimmed_vision = len(self._vision_pending)
        self._imu_pending.clear()
        self._runtime_pending.clear()
        self._camera_pending.clear()
        self._vision_pending.clear()
        while self._worker.is_alive():
            try:
                self._queue.put(None, timeout=.1)
                break
            except Full:
                continue  # Writer failures must not leave close blocked on a full queue.
        self._worker.join()
        self.imu_file.close()
        self.runtime_file.close()
        manifest = {
            "format": "wt901-fusion-session-v2" if self.camera_calibration else "wt901-fusion-session-v1",
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
            "continuous_basis_at_start":self.continuous_basis_at_start,
            "runtime_pose_records":self.runtime_count,
            "camera_epoch":self.camera_epoch,
            "fusion_at_start":self.fusion_at_start,
            "rgbd_frames": self.rgbd_count,
            "camera_bridge_skipped": self.camera_bridge_skipped,
            "camera_last_timing": self.camera_last_timing,
            "vision_records": self.vision_count,
            "dropped_vision_records": self.dropped_vision,
            "unmatched_vision_records": self.unmatched_vision,
            "trimmed_vision_records": self.trimmed_vision,
            "vision_note": "independent RGB-D surface-joint observations; no IMU correction; each record references its actual source frame",
            "camera_timestamp_note": "captured_mono_ns for Orbbec is host frameset receipt, not exposure; device clocks are not yet aligned to WT901",
        }
        if self.camera_calibration is not None:
            (self.path / "camera_calibration.json").write_text(
                json.dumps(self.camera_calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        if self.error is not None:
            raise OSError(self.error)

    def __enter__(self) -> "FusionSessionRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
