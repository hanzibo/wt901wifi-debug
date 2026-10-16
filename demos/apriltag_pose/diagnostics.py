"""Display-only pose diagnostics; never a calibrated accuracy/safety certificate."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Variation:
    count: int
    span_s: float
    position_mm: float
    rotation_deg: float

    @property
    def ready(self):
        return self.count >= 10 and self.span_s >= 1.0


class PoseWindow:
    """Two-second per-tag window; discontinuities start a new measurement."""
    def __init__(self):
        self.samples = {}

    def clear(self):
        self.samples.clear()

    def retain(self, ids):
        self.samples = {key: value for key, value in self.samples.items() if key in ids}

    def update(self, tag_id, timestamp, size_mm, pose):
        samples = self.samples.setdefault(tag_id, deque())
        if samples and (timestamp <= samples[-1][0] or timestamp-samples[-1][0] > .5
                        or size_mm != samples[-1][1]):
            samples.clear()
        samples.append((timestamp, size_mm, pose.tvec.reshape(3).copy(),
                        cv2.Rodrigues(pose.rvec)[0]))
        while samples and (timestamp-samples[0][0] > 2.0 or len(samples) > 120):
            samples.popleft()
        positions = np.array([s[2] for s in samples])
        rotations = np.array([s[3] for s in samples])
        u, _, vt = np.linalg.svd(rotations.mean(axis=0))
        mean_rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        relative = rotations @ mean_rotation.T
        angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1))
        return Variation(len(samples), timestamp-samples[0][0],
                         float(np.sqrt(np.mean(np.sum((positions-positions.mean(axis=0))**2,
                                                       axis=1)))*1000),
                         float(np.rad2deg(np.sqrt(np.mean(angles**2)))))


def quality_reasons(edge_px, pose, variation, min_edge=32., max_error=1.,
                    max_position=2., max_rotation=2.):
    reasons = []
    if edge_px < min_edge:
        reasons.append('像素不足')
    if pose.error_px > max_error:
        reasons.append('重投影偏大')
    if pose.ambiguous:
        reasons.append('平面双解：方向不可靠')
    if variation is not None and variation.ready:
        if variation.position_mm > max_position:
            reasons.append('位置波动大/目标在运动')
        if variation.rotation_deg > max_rotation:
            reasons.append('朝向波动大/目标在运动')
    return reasons
