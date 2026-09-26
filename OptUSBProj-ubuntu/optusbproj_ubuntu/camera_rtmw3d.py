"""Bridge RTMW3D's COCO-WholeBody points to the camera pose engine.

RTMW3D's x/y values are in a person crop, while z has a separate model
scale. This bridge normalizes x/y by apparent shoulder width. It is an
orientation approximation, not a metric 3-D reconstruction.
"""

from __future__ import annotations

import math
import time

import numpy as np


# MediaPipe 33 index -> COCO-WholeBody 133 index. Hand MCP points approximate
# MediaPipe's index/pinky knuckle landmarks used by the existing hand frame.
RTMW_TO_MEDIAPIPE = {
    11: 5, 12: 6,      # shoulders
    13: 7, 14: 8,      # elbows
    15: 9, 16: 10,     # wrists
    17: 108, 18: 129,  # pinky MCPs
    19: 96, 20: 117,   # index MCPs
    23: 11, 24: 12,    # hips
}


def best_person_index(scores: np.ndarray) -> int:
    """Pick the person with the strongest shoulder-to-wrist evidence."""
    body_scores = np.nan_to_num(scores[:, [5, 6, 7, 8, 9, 10]], nan=0.0)
    return int(np.argmax(body_scores.mean(axis=1)))


def adapt_rtmw3d(keypoints: np.ndarray, scores: np.ndarray,
                 *, shoulder_width_override: float | None = None
                 ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return pseudo-camera xyz/visibility arrays accepted by CameraPoseEngine.

    Use the most confidently detected person's body. The resulting x/y are
    shoulder-width-normalized crop coordinates and z is RTMW3D's native depth.
    They are suitable for relative segment directions, not absolute position.
    """
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (keypoints.ndim != 3 or keypoints.shape[1:] != (133, 3)
            or scores.shape != keypoints.shape[:2] or len(keypoints) == 0):
        return None, None
    person_idx = best_person_index(scores)
    person = keypoints[person_idx]
    confidence = scores[person_idx]
    if not np.isfinite(person[[5, 6]]).all():
        return None, None
    shoulder_width = float(np.linalg.norm(person[5, :2] - person[6, :2]))
    if (shoulder_width < 10.0 or not np.isfinite(confidence[[5, 6]]).all()
            or min(confidence[5], confidence[6]) < 0.25):
        return None, None

    # Approximate a 0.4 m shoulder width. Clamp to avoid large jumps when
    # the shoulders are partially occluded or the person turns side-on.
    reference_width = (shoulder_width if shoulder_width_override is None
                       else shoulder_width_override)
    xy_scale = float(np.clip(0.4 / reference_width, 0.001, 0.012))
    origin = (person[5, :2] + person[6, :2]) * 0.5
    points = np.full((33, 3), np.nan, dtype=float)
    visibility = np.zeros(33, dtype=float)
    for mp_index, rtmw_index in RTMW_TO_MEDIAPIPE.items():
        point = person[rtmw_index]
        score = confidence[rtmw_index]
        if np.isfinite(point).all() and np.isfinite(score) and score > 0:
            points[mp_index, :2] = (point[:2] - origin) * xy_scale
            points[mp_index, 2] = point[2]
            visibility[mp_index] = float(np.clip(score, 0.0, 1.0))
    return points, visibility


class _OneEuroPoint:
    """Timestamp-aware adaptive low-pass filter for one 3-D landmark."""

    def __init__(self, value: np.ndarray, timestamp: float) -> None:
        self.raw = value.copy()
        self.filtered = value.copy()
        self.derivative = np.zeros(3, dtype=float)
        self.timestamp = timestamp
        self.last_seen = timestamp

    def update(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        dt = max(timestamp - self.timestamp, 1e-3)
        raw_derivative = (value - self.raw) / dt
        derivative_alpha = 1.0 - math.exp(-2.0 * math.pi * 1.0 * dt)
        self.derivative += derivative_alpha * (raw_derivative - self.derivative)
        # Depth estimates fluctuate more than image-plane landmarks.
        min_cutoff = np.array((1.2, 1.2, 0.7))
        cutoff = min_cutoff + 2.0 * np.abs(self.derivative)
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff * dt)
        self.filtered += alpha * (value - self.filtered)
        self.raw = value.copy()
        self.timestamp = timestamp
        self.last_seen = timestamp
        return self.filtered.copy()


class RTMWPoseStabilizer:
    """Stabilize RTMW3D scale and landmarks before computing joint frames."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.shoulder_width: float | None = None
        self.last_frame_at: float | None = None
        self.last_person_at: float | None = None
        self.points: dict[int, _OneEuroPoint] = {}
        self.outlier_counts: dict[int, int] = {}

    def update(self, keypoints: np.ndarray, scores: np.ndarray,
               timestamp: float | None = None
               ) -> tuple[np.ndarray | None, np.ndarray | None]:
        keypoints = np.asarray(keypoints, dtype=float)
        scores = np.asarray(scores, dtype=float)
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.last_frame_at is not None and now - self.last_frame_at > 0.8:
            self.reset()
        dt = (1.0 / 18.0 if self.last_frame_at is None
              else max(now - self.last_frame_at, 1e-3))
        self.last_frame_at = now

        raw_points, visibility = adapt_rtmw3d(keypoints, scores)
        if raw_points is None:
            if self.last_person_at is not None and now - self.last_person_at > 0.8:
                self.reset()
            return None, None
        self.last_person_at = now

        person = np.asarray(keypoints, dtype=float)[best_person_index(scores)]
        measured_width = float(np.linalg.norm(person[5, :2] - person[6, :2]))
        if self.shoulder_width is None:
            self.shoulder_width = measured_width
        else:
            # Limit the influence of a single detector/crop jump. Slow scale
            # changes are still allowed when a person approaches the camera.
            target_width = float(np.clip(measured_width,
                                         self.shoulder_width * 0.8,
                                         self.shoulder_width * 1.2))
            alpha = 1.0 - math.exp(-dt / 0.8)
            self.shoulder_width += alpha * (target_width - self.shoulder_width)
        raw_points, visibility = adapt_rtmw3d(
            keypoints, scores, shoulder_width_override=self.shoulder_width)

        filtered = np.full((33, 3), np.nan, dtype=float)
        stable_visibility = np.zeros(33, dtype=float)
        for index in RTMW_TO_MEDIAPIPE:
            value = raw_points[index]
            score = visibility[index]
            previous = self.points.get(index)
            usable = score >= 0.35 and np.isfinite(value).all()
            if usable and previous is not None:
                delta = value - previous.raw
                xy_jump = np.linalg.norm(delta[:2]) > 0.12 + 2.8 * dt
                z_jump = abs(delta[2]) > 0.12 + 1.8 * dt
                if xy_jump or z_jump:
                    count = self.outlier_counts.get(index, 0) + 1
                    self.outlier_counts[index] = count
                    # A single isolated spike is noise; two consecutive large
                    # changes may be real movement, so let the filter catch up.
                    usable = count >= 2
                else:
                    self.outlier_counts.pop(index, None)
            if usable:
                self.outlier_counts.pop(index, None)
                if previous is None:
                    previous = _OneEuroPoint(value, now)
                    self.points[index] = previous
                else:
                    previous.update(value, now)
                filtered[index] = previous.filtered
                stable_visibility[index] = score
            elif previous is not None and now - previous.last_seen <= 0.18:
                filtered[index] = previous.filtered
                stable_visibility[index] = 0.36
            elif previous is not None:
                self.points.pop(index, None)
                self.outlier_counts.pop(index, None)
        return filtered, stable_visibility
