"""Timestamped camera observations shared by the future fusion stages.

Pixel coordinates are deliberately kept in the original image frame. The
camera-only avatar's approximate 3-D retargeting is not a metric observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDData:
    """Raw sensor depth + metric aligned depth; never a colored depth preview."""
    raw_depth: np.ndarray
    aligned_depth_mm: np.ndarray
    calibration: dict
    timing: dict

    def __post_init__(self):
        if self.raw_depth.ndim != 2 or self.raw_depth.dtype != np.uint16:
            raise ValueError("原始深度必须为 uint16 二维数组")
        if self.aligned_depth_mm.ndim != 2 or self.aligned_depth_mm.dtype != np.float32:
            raise ValueError("配准深度必须为 float32 毫米数组")
        if not np.isfinite(self.aligned_depth_mm).all() or np.any(self.aligned_depth_mm < 0):
            raise ValueError("配准深度包含无效数值")


@dataclass(frozen=True)
class CameraObservation:
    captured_mono_ns: int
    backend: str
    frame_rgb: np.ndarray
    keypoints_xy: np.ndarray | None
    confidence: np.ndarray | None
    rgbd: RGBDData | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("mediapipe", "rtmw3d", "orbbec"):
            raise ValueError("未知相机识别后端")
        if self.frame_rgb.ndim != 3 or self.frame_rgb.shape[2] != 3:
            raise ValueError("相机帧必须为 RGB 图像")
        if self.rgbd is not None and self.rgbd.aligned_depth_mm.shape != self.frame_rgb.shape[:2]:
            raise ValueError("配准深度和彩色尺寸不一致")
        if self.backend == "orbbec" and (self.rgbd is None or self.keypoints_xy is not None):
            raise ValueError("阶段 A Orbbec 观测必须包含 RGB-D 且不包含估计关键点")
        if (self.keypoints_xy is None) != (self.confidence is None):
            raise ValueError("关键点坐标和置信度必须同时提供")
        if self.keypoints_xy is not None:
            expected = 33 if self.backend == "mediapipe" else 133
            if self.keypoints_xy.shape != (expected, 2):
                raise ValueError(f"{self.backend} 需要 {expected} 个 2D 关键点")
            if self.confidence.shape != (expected,):
                raise ValueError("置信度数量与关键点不一致")

    @property
    def image_size(self) -> tuple[int, int]:
        height, width = self.frame_rgb.shape[:2]
        return width, height
