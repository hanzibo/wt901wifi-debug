"""Metric surface-joint observations; no monocular predicted Z and no IMU writes."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

NAMES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip")
INDICES = np.array([5, 6, 7, 8, 9, 10, 11, 12])
EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7))
BONES = {"left_upper_arm": (0, 2), "left_forearm": (2, 4),
         "right_upper_arm": (1, 3), "right_forearm": (3, 5), "shoulder_width": (0, 1)}


def json_array(array):
    a = np.asarray(array)
    if a.dtype == bool:
        return a.tolist()
    if a.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in a]
    return [json_array(row) for row in a]


@dataclass
class RGBDPoseEstimate:
    captured_mono_ns: int
    keypoints_xy: np.ndarray
    confidence: np.ndarray
    points_camera_m: np.ndarray
    valid: np.ndarray
    reasons: list[str]
    depth_pixels: np.ndarray
    support: np.ndarray
    spread_mm: np.ndarray
    bone_lengths_m: dict
    inference_ms: float = 0.
    completed_mono_ns: int = 0
    frame_rgb: np.ndarray | None = field(default=None, repr=False)
    diagnostics: dict = field(default_factory=dict)

    def to_record(self):
        return {"format": "rgbd-surface-joints-v1", "captured_mono_ns": self.captured_mono_ns,
                "keypoints_xy": json_array(self.keypoints_xy), "confidence": json_array(self.confidence),
                "joint_names": list(NAMES), "points_camera_m": json_array(self.points_camera_m),
                "valid": self.valid.tolist(), "reasons": list(self.reasons),
                "depth_pixels": json_array(self.depth_pixels), "support": json_array(self.support),
                "spread_mm": json_array(self.spread_mm), "bone_lengths_m": self.bone_lengths_m,
                "inference_ms": self.inference_ms, "completed_mono_ns": self.completed_mono_ns,
                "diagnostics": self.diagnostics,
                "coordinate_frame": "color_optical: X right, Y down, Z forward; metres",
                "measurement_note": "surface depth at estimated pixels, NOT ground-truth skeletal joint centres"}

    @classmethod
    def from_record(cls, row, frame_rgb=None):
        if row.get("format") != "rgbd-surface-joints-v1" or row.get("joint_names") != list(NAMES):
            raise ValueError("无法识别 RGB-D 三维观测")
        return cls(int(row["captured_mono_ns"]), np.asarray(row["keypoints_xy"], float),
                   np.asarray(row["confidence"], float), np.asarray(row["points_camera_m"], float),
                   np.asarray(row["valid"], bool), row["reasons"], np.asarray(row["depth_pixels"], float),
                   np.asarray(row["support"], float), np.asarray(row["spread_mm"], float),
                   row["bone_lengths_m"], float(row["inference_ms"]), int(row["completed_mono_ns"]),
                   frame_rgb, row.get("diagnostics", {}))


def camera_matrix(intrinsic):
    w, h = int(intrinsic["width"]), int(intrinsic["height"])
    fx, fy, cx, cy = (float(intrinsic[k]) for k in ("fx", "fy", "cx", "cy"))
    if w < 1 or h < 1 or not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("相机内参无效")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], float)


def distortion_coefficients(value):
    if value is None:
        raise ValueError("缺少畸变标定，不能假设图像无畸变")
    coeff = np.array([value.get(k, 0.) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], float)
    if not np.isfinite(coeff).all():
        raise ValueError("畸变参数无效")
    if np.any(coeff != 0) and "BROWN_CONRADY" not in str(value.get("model", "")):
        raise ValueError("当前只支持 Brown-Conrady 畸变，拒绝猜测其他模型")
    return coeff


class DepthPoseEstimator:
    def __init__(self, *, min_confidence=.5, radius=3, min_support=.5,
                 min_depth_mm=250., max_depth_mm=6000.):
        self.min_confidence = min_confidence
        self.radius = radius
        self.min_support = min_support
        self.min_depth_mm, self.max_depth_mm = min_depth_mm, max_depth_mm
        self._previous = None

    def reset(self):
        self._previous = None

    def estimate(self, observation, keypoints, confidence, *, inference_ms=0., diagnostics=None):
        import cv2
        if observation.rgbd is None:
            raise ValueError("缺少真实 RGB-D 观测")
        xy, scores = np.asarray(keypoints, float), np.asarray(confidence, float)
        if xy.shape != (133, 2) or scores.shape != (133,):
            raise ValueError("需要还原到原始彩色像素系的 133 个二维关键点")
        calibration = observation.rgbd.calibration
        color_k = camera_matrix(calibration["color_intrinsic"])
        aligned_k = camera_matrix(calibration["aligned_depth_intrinsic"])
        color_d = distortion_coefficients(calibration["color_distortion"])
        aligned_d = distortion_coefficients(calibration["aligned_depth_distortion"])
        rgb_h, rgb_w = observation.frame_rgb.shape[:2]
        body_xy=xy[INDICES];body_scores=scores[INDICES]
        valid_2d=(np.isfinite(body_xy).all(axis=1)&np.isfinite(body_scores)&(body_scores>=self.min_confidence)
                  &(body_xy[:,0]>=0)&(body_xy[:,0]<rgb_w)&(body_xy[:,1]>=0)&(body_xy[:,1]<rgb_h))
        if (calibration["color_intrinsic"]["width"], calibration["color_intrinsic"]["height"]) != (rgb_w, rgb_h):
            raise ValueError("彩色内参与活动图像尺寸不匹配")
        depth = observation.rgbd.aligned_depth_mm
        if (calibration["aligned_depth_intrinsic"]["width"], calibration["aligned_depth_intrinsic"]["height"]) != (depth.shape[1], depth.shape[0]):
            raise ValueError("配准深度内参与活动图像尺寸不匹配")
        xyz = np.full((8, 3), np.nan)
        pixels = np.full((8, 2), np.nan)
        support, spread = np.zeros(8), np.full(8, np.nan)
        reasons = ["低二维置信度"] * 8
        for i, k in enumerate(INDICES):
            if not np.isfinite(scores[k]) or scores[k] < self.min_confidence:
                continue
            u, v = xy[k]
            if not np.isfinite([u, v]).all() or not (0 <= u < rgb_w and 0 <= v < rgb_h):
                reasons[i] = "彩色关键点越界"
                continue
            # Original RGB pixels -> undistorted colour ray -> aligned depth pixels.
            ray = cv2.undistortPoints(np.array([[[u, v]]], float), color_k, color_d)[0, 0]
            target, _ = cv2.projectPoints(np.array([[ray[0], ray[1], 1.]]), np.zeros(3), np.zeros(3), aligned_k, aligned_d)
            pixels[i] = target[0, 0]
            x, y = np.rint(pixels[i]).astype(int)
            dh, dw = depth.shape
            r = self.radius
            if x-r < 0 or y-r < 0 or x+r >= dw or y+r >= dh:
                reasons[i] = "配准深度关键点越界"
                continue
            patch = depth[y-r:y+r+1, x-r:x+r+1]
            usable = (patch >= self.min_depth_mm) & (patch <= self.max_depth_mm)
            support[i] = usable.mean()
            if support[i] < self.min_support:
                reasons[i] = "深度有效支持不足"
                continue
            central = depth[y-1:y+2, x-1:x+2]
            central = central[(central >= self.min_depth_mm) & (central <= self.max_depth_mm)]
            if len(central) < 3:
                reasons[i] = "关键点中心深度空洞"
                continue
            values = patch[usable]
            z = float(np.median(values))
            spread[i] = float(np.percentile(values, 90)-np.percentile(values, 10))
            threshold = 100.+.02*z
            if spread[i] > threshold or abs(float(np.median(central))-z) > threshold/2:
                reasons[i] = "深度边缘/前景背景混合"
                continue
            xyz[i] = np.r_[ray, 1.] * (z/1000.)
            reasons[i] = "有效表面观测"
        valid = np.isfinite(xyz).all(axis=1)
        # Broad sanity checks, not fixed adult bone lengths or a calibration fit.
        raw_lengths = {}
        for name, (a, b) in BONES.items():
            if valid[a] and valid[b]:
                length = float(np.linalg.norm(xyz[a]-xyz[b]))
                raw_lengths[name] = length
                if not .08 <= length <= .85:
                    valid[a] = valid[b] = False
                    reasons[a] = reasons[b] = "骨段长度异常（宽松范围）"
        # A smooth nearby occluder may pass patch tests but create an absurd
        # forearm/upper-arm ratio. Reject the distal observation conservatively.
        for side, (shoulder, elbow, wrist) in (("left", (0, 2, 4)), ("right", (1, 3, 5))):
            if valid[[shoulder, elbow, wrist]].all():
                upper = raw_lengths.get(side+"_upper_arm")
                fore = raw_lengths.get(side+"_forearm")
                if upper and fore and not .35 <= fore/upper <= 1.8:
                    valid[wrist] = False
                    reasons[wrist] = "前臂/上臂比例异常（疑似遮挡取错深度）"
        if self._previous is not None:
            old_stamp, old_xyz, old_valid = self._previous
            dt = (observation.captured_mono_ns-old_stamp)/1e9
            if 0 < dt < .6:
                for i in range(8):
                    if valid[i] and old_valid[i] and np.linalg.norm(xyz[i]-old_xyz[i]) > .18+3.*dt:
                        valid[i] = False
                        reasons[i] = "三维观测孤立跳变"
        # Never hold a stale point and label it valid. Gate reacquisition afresh.
        self._previous = observation.captured_mono_ns, xyz.copy(), valid.copy()
        xyz[~valid] = np.nan
        lengths = {name: (float(np.linalg.norm(xyz[a]-xyz[b])) if valid[a] and valid[b] else None)
                   for name, (a, b) in BONES.items()}
        details = {**(diagnostics or {}), "raw_bone_lengths_m": raw_lengths,
                   "valid_2d": valid_2d.tolist(),
                   "depth_source": "aligned sensor depth; predicted Z ignored",
                   "pixel_mapping": "RGB undistort -> aligned depth project",
                   "gates": {"min_2d_score": self.min_confidence, "radius_px": self.radius,
                             "min_patch_support": self.min_support,
                             "depth_range_mm": [self.min_depth_mm, self.max_depth_mm]}}
        return RGBDPoseEstimate(observation.captured_mono_ns, xy.copy(), scores.copy(), xyz,
                                valid, reasons, pixels, support, spread, lengths, inference_ms,
                                time.monotonic_ns(), observation.frame_rgb, details)
