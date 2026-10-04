"""Bounded latest-frame RTMW 2D inference in the GUI environment."""
from threading import Condition
import time

from PySide6 import QtCore

from .rgbd_pose import DepthPoseEstimator
from .pose_2d import Pose2DConfig,load_backend,infer_rtmw


def load_detector(config=None):
    return load_backend(config or Pose2DConfig())


def infer_observation(detector, estimator, observation):
    """Shared live/offline path; RTMW predicted 3D coordinates are discarded."""
    import cv2
    started = time.perf_counter()
    bgr = cv2.cvtColor(observation.frame_rgb, cv2.COLOR_RGB2BGR)
    xy,scores,diagnostics=(detector.infer(bgr) if hasattr(detector,'infer') else infer_rtmw(detector,bgr))
    if 'reject_reason' in diagnostics:
        estimator.reset()
    if hasattr(detector,'metadata'):diagnostics['vision_config']=detector.metadata
    return estimator.estimate(observation, xy, scores,
                              inference_ms=(time.perf_counter()-started)*1000,
                              diagnostics=diagnostics)


class RGBDPoseWorker(QtCore.QThread):
    error = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, parent=None, *, config=None, vision_epoch=None):
        super().__init__(parent)
        self._condition = Condition()
        self._input = self._result = None
        self._running = True
        self.input_skipped = self.result_skipped = 0
        self.config=config or Pose2DConfig();self.vision_epoch=vision_epoch

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
            self.status.emit(f"正在加载本地 {self.config.backend} 模型…RGB-D 原始采集仍继续")
            detector = load_detector(self.config)
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
                estimate.diagnostics['vision_epoch']=self.vision_epoch
                with self._condition:
                    if self._result is not None:
                        self.result_skipped += 1
                    self._result = estimate
        except Exception as exc:
            self.error.emit(f"RGB-D 三维观测失败：{exc}")
        finally:
            # Drop sessions before the GUI starts the replacement worker.
            if 'detector' in locals():del detector
            import gc,sys
            gc.collect()
            if self.config.backend=='yolo26' and 'torch' in sys.modules:
                torch=sys.modules['torch']
                if torch.cuda.is_available():torch.cuda.empty_cache()
