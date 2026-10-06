"""Qt-side SDK bridge with one owned mmap slot and one latest-frame mailbox."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
import subprocess
import tempfile
from threading import Lock, Timer
import time

import numpy as np
from PySide6 import QtCore

from .fusion_capture import CameraObservation, RGBDData

DEFAULT_PYTHON = Path.home() / "miniconda3/envs/orbbec336l/bin/python"


class OrbbecWorker(QtCore.QThread):
    error = QtCore.Signal(str)

    def __init__(self, python: Path = DEFAULT_PYTHON, *, fps=10., warmup=2., parent=None):
        super().__init__(parent)
        self.capture_epoch=uuid.uuid4().hex
        self.python = Path(python)
        self.fps, self.warmup = float(fps), float(warmup)
        self.calibration = None
        self.last_timing = {}
        self.consumer_skipped = 0
        self._latest = None
        self._lock = Lock()
        self._stop_requested = False
        self.process = None
        self._killer = None

    def take_latest(self):
        with self._lock:
            frame, self._latest = self._latest, None
            return frame

    def stop(self):
        self._stop_requested = True
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.stdin.write("STOP\n")
                process.stdin.flush()
            except (OSError, ValueError):
                pass
            # Do not block the GUI if a USB/SDK call hangs.
            self._killer = Timer(2., self._terminate_owned)
            self._killer.daemon = True
            self._killer.start()

    def _terminate_owned(self):
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    def run(self):
        mapping = None
        if not self.python.is_file():
            self.error.emit(f"Orbbec 环境解释器不存在：{self.python}")
            return
        # Private directory prevents another user/process replacing the slot.
        shared_root = "/dev/shm" if Path("/dev/shm").is_dir() else None
        with tempfile.TemporaryDirectory(prefix="wt901-rgbd-", dir=shared_root) as directory:
            folder = Path(directory)
            with (folder / "stderr.log").open("w+") as log:
                try:
                    script = Path(__file__).with_name("orbbec_process.py")
                    self.process = subprocess.Popen(
                        [str(self.python), "-u", str(script), "--buffer", str(folder / "slot.bin"),
                         "--fps", str(self.fps), "--warmup", str(self.warmup)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                        text=True, bufsize=1, cwd=folder)
                    if self._stop_requested:
                        self.stop()
                    layouts = None
                    reported = False
                    for line in self.process.stdout:
                        if not line.startswith("@RGBD@"):
                            continue  # Native SDK startup logging is not protocol data.
                        event = json.loads(line[len("@RGBD@"):])
                        if event["type"] == "ready":
                            self.calibration = event["calibration"]
                            layouts = event["layouts"]
                            if len(layouts) != 3 or event["nbytes"] > 128 * 1024 * 1024:
                                raise ValueError("RGB-D 共享缓冲区布局无效")
                            mapping = np.memmap(folder / "slot.bin", mode="r", dtype=np.uint8,
                                                shape=(event["nbytes"],))
                        elif event["type"] == "frame":
                            if mapping is None:
                                raise ValueError("RGB-D 数据先于布局到达")
                            arrays = [np.ndarray(tuple(x["shape"]), dtype=x["dtype"],
                                                 buffer=mapping, offset=x["offset"]).copy()
                                      for x in layouts]
                            timing = event["timing"]
                            timing["host_bridge_mono_ns"] = time.monotonic_ns()
                            observation = CameraObservation(
                                timing["host_received_mono_ns"], "orbbec", arrays[0], None, None,
                                RGBDData(arrays[1], arrays[2], self.calibration, timing))
                            with self._lock:
                                if self._latest is not None:
                                    self.consumer_skipped += 1
                                self._latest = observation
                                self.last_timing = timing
                            self.process.stdin.write("STOP\n" if self._stop_requested else "ACK\n")
                            self.process.stdin.flush()
                        elif event["type"] == "error":
                            reported = True
                            self.error.emit(event["message"])
                        elif event["type"] == "stopped":
                            self.last_timing = {**self.last_timing, "final_capture_stats": event}
                    code = self.process.wait(timeout=3)
                    if code and not reported and not self._stop_requested:
                        log.seek(0)
                        self.error.emit(f"Orbbec 采集进程退出 {code}：{log.read()[-1500:]}")
                except Exception as exc:
                    if not self._stop_requested:
                        self.error.emit(f"RGB-D 接口失败：{exc}")
                finally:
                    if self._killer is not None:
                        self._killer.cancel()
                    self._terminate_owned()
                    if self.process is not None:
                        self.process.wait(timeout=3)
                        self.process.stdin.close()
                        self.process.stdout.close()
                    if mapping is not None:
                        mapping._mmap.close()
