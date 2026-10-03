"""Bounded latest-frame RTMW 2D inference in the GUI environment."""
from pathlib import Path
from threading import Condition
import time

import numpy as np
from PySide6 import QtCore

from .rgbd_pose import DepthPoseEstimator


def load_detector():
    import onnxruntime as ort
    from rtmlib import Wholebody3d
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    cache = Path.home()/".cache/rtmlib/hub/checkpoints"
    det = cache/"yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
    pose = cache/"rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
    if not det.is_file() or not pose.is_file():
        raise RuntimeError("缺少已下载的 RTMW3D/YOLOX ONNX 模型，请先运行原 rtmw3d_probe.py 下载；本窗口不会自动下载")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("未发现 CUDA 执行后端；不静默回退到 CPU")
    model = Wholebody3d(det=str(det), pose=str(pose), backend="onnxruntime", device="cuda")
    for component in (model.det_model, model.pose_model):
        if "CUDAExecutionProvider" not in component.session.get_providers():
            raise RuntimeError("RTMW 模型未使用 CUDA；请检查 ONNX Runtime 运行库")
    return model


def infer_observation(detector, estimator, observation):
    """Shared live/offline path; RTMW predicted 3D coordinates are discarded."""
    import cv2
    started = time.perf_counter()
    bgr = cv2.cvtColor(observation.frame_rgb, cv2.COLOR_RGB2BGR)
    boxes = detector.det_model(bgr)
    xy = np.full((133, 2), np.nan)
    scores = np.zeros(133)
    diagnostics = {"people_detected": len(boxes), "inference_backend": "RTMW3D 2D outputs / CUDA"}
    if len(boxes) == 1:
        # rtmlib has inverse-affine transformed these to original RGB pixels.
        _predicted_xyz, scores_batch, _simcc, image_points = detector.pose_model(bgr, bboxes=boxes)
        xy, scores = np.asarray(image_points[0]), np.asarray(scores_batch[0])
    else:
        estimator.reset()
        diagnostics["reject_reason"] = "未检测到人" if len(boxes) == 0 else "多人画面：暂不自动选择身份"
    return estimator.estimate(observation, xy, scores,
                              inference_ms=(time.perf_counter()-started)*1000,
                              diagnostics=diagnostics)


class RGBDPoseWorker(QtCore.QThread):
    error = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = Condition()
        self._input = self._result = None
        self._running = True
        self.input_skipped = self.result_skipped = 0

    def submit(self, observation):
        with self._condition:
            if not self._running:
                return
            if self._input is not None:
                self.input_skipped += 1
            self._input = observation
            self._condition.notify()

    def take_latest(self):
        with self._condition:
            result, self._result = self._result, None
            return result

    def stop(self):
        with self._condition:
            self._running = False
            self._input = None
            self._condition.notify()

    def run(self):
        try:
            self.status.emit("正在加载本地 RTMW 模型…RGB-D 原始采集仍继续")
            detector = load_detector()
            estimator = DepthPoseEstimator()
            self.status.emit("模型就绪；只取二维关键点，三维距离来自相机")
            while True:
                with self._condition:
                    while self._running and self._input is None:
                        self._condition.wait(.5)
                    if not self._running:
                        break
                    observation, self._input = self._input, None
                estimate = infer_observation(detector, estimator, observation)
                with self._condition:
                    if self._result is not None:
                        self.result_skipped += 1
                    self._result = estimate
        except Exception as exc:
            self.error.emit(f"RGB-D 三维观测失败：{exc}")
