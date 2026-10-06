"""Retarget MediaPipe 3-D body landmarks onto the existing fixed-length model.

This is a camera-only approximation. A segment direction does not determine
rotation about its own long axis; hand orientation also depends on small,
often-occluded landmarks. No absolute translation is reconstructed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .skeleton import PARTS, transform


UPPER_BODY_KEYS = (
    "torso", "left_upper_arm", "left_forearm", "left_hand",
    "right_upper_arm", "right_forearm", "right_hand",
)

# MediaPipe: image/camera right, down, away. Model: front, anatomical left, up.
CAMERA_TO_MODEL = np.array(((0.0, 0.0, -1.0),
                            (1.0, 0.0, 0.0),
                            (0.0, -1.0, 0.0)))

LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_hip": 23, "right_hip": 24,
}


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-5:
        return None
    return vector / length


def _frame(up_vector: np.ndarray, forward_hint: np.ndarray) -> np.ndarray | None:
    """Right-handed frame: columns are model forward, left, up."""
    up = _unit(up_vector)
    if up is None:
        return None
    forward = _unit(forward_hint - np.dot(forward_hint, up) * up)
    if forward is None:
        return None
    left = _unit(np.cross(up, forward))
    if left is None:
        return None
    forward = np.cross(left, up)
    return np.column_stack((forward, left, up))


def _swing_frame(parent_frame: np.ndarray, up_vector: np.ndarray) -> np.ndarray | None:
    """Align a segment's long axis with its measured direction, without twist.

    Projecting a fixed forward axis onto the segment-normal plane fails when
    the arm points forward. The shortest rotation from the parent's up axis
    remains defined there; axial twist is intentionally left unestimated.
    """
    target_up = _unit(up_vector)
    if target_up is None:
        return None
    parent_up = parent_frame[:, 2]
    cross = np.cross(parent_up, target_up)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(parent_up, target_up), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0:
            return parent_frame.copy()
        # Exactly opposite axes have no unique twist. Choose the parent's
        # forward axis rather than dropping the segment altogether.
        axis = parent_frame[:, 0]
        rotation = Rotation.from_rotvec(np.pi * axis).as_matrix()
    else:
        angle = np.arctan2(sine, cosine)
        rotation = Rotation.from_rotvec(cross / sine * angle).as_matrix()
    return rotation @ parent_frame


def frames_from_landmarks(points: np.ndarray, visibility: np.ndarray,
                          min_visibility: float = 0.5) -> dict[str, np.ndarray]:
    """Compute observable segment frames from MediaPipe's 33 world landmarks."""
    points = np.asarray(points, dtype=float)
    visibility = np.asarray(visibility, dtype=float)
    if points.shape != (33, 3) or visibility.shape != (33,):
        raise ValueError("需要 33 个三维关键点及其可见度")
    model_points = points @ CAMERA_TO_MODEL.T

    def point(name: str) -> np.ndarray | None:
        index = LANDMARKS[name]
        if visibility[index] < min_visibility or not np.isfinite(model_points[index]).all():
            return None
        return model_points[index]

    result: dict[str, np.ndarray] = {}
    ls, rs, lh, rh = (point(name) for name in
                      ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if ls is None or rs is None:
        return result
    if lh is not None and rh is not None:
        shoulder_center = (ls + rs) * 0.5
        hip_center = (lh + rh) * 0.5
        torso_up = _unit(shoulder_center - hip_center)
    else:
        # A laptop webcam may frame shoulders and arms but not the hips.
        # Retain approximate torso heading; pitch/lean then cannot be measured.
        torso_up = np.array([0.0, 0.0, 1.0])
    torso_left = _unit(ls - rs)
    if torso_up is None or torso_left is None:
        return result
    torso_forward = _unit(np.cross(torso_left, torso_up))
    if torso_forward is None:
        return result
    torso = _frame(torso_up, torso_forward)
    if torso is None:
        return result
    result["torso"] = torso

    for side in ("left", "right"):
        shoulder = point(f"{side}_shoulder")
        elbow = point(f"{side}_elbow")
        wrist = point(f"{side}_wrist")
        if shoulder is None or elbow is None:
            continue
        upper = _swing_frame(torso, shoulder - elbow)
        if upper is None:
            continue
        result[f"{side}_upper_arm"] = upper
        if wrist is None:
            continue
        forearm = _swing_frame(upper, elbow - wrist)
        if forearm is None:
            continue
        result[f"{side}_forearm"] = forearm

        pinky, index = point(f"{side}_pinky"), point(f"{side}_index")
        if pinky is None or index is None:
            continue
        palm_center = (pinky + index) * 0.5
        hand_up = _unit(wrist - palm_center)
        hand_left = _unit(pinky - index if side == "left" else index - pinky)
        if hand_up is None or hand_left is None:
            continue
        hand_forward = _unit(np.cross(hand_left, hand_up))
        if hand_forward is None:
            continue
        hand = _frame(hand_up, hand_forward)
        if hand is not None:
            result[f"{side}_hand"] = hand
    return result


class CameraPoseEngine:
    """Calibrate each observed segment, smooth rotations, and place STL parts."""

    def __init__(self, *, smoothing: float = 0.45,
                 min_visibility: float = 0.5) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing 必须在 (0, 1] 内")
        self.smoothing = smoothing
        self.min_visibility = min_visibility
        self.neutral: dict[str, np.ndarray] = {}
        self.latest: dict[str, np.ndarray] = {}
        self.rotations: dict[str, np.ndarray] = {}
        self.tracked: set[str] = set()
        self.has_person = False

    def update(self, points: np.ndarray | None,
               visibility: np.ndarray | None) -> set[str]:
        if points is None or visibility is None:
            # No detection: hold the last pose while the GUI marks tracking lost.
            self.has_person = False
            return set()
        frames = frames_from_landmarks(points, visibility, self.min_visibility)
        if not frames:
            self.has_person = False
            return set()
        self.has_person = True
        self.tracked = set(frames)
        self.latest = frames
        for key, frame in frames.items():
            if key not in self.neutral:
                self.neutral[key] = frame.copy()
            target = frame @ self.neutral[key].T
            previous = self.rotations.get(key)
            if previous is None:
                self.rotations[key] = target
            else:
                step = Rotation.from_matrix(previous.T @ target).as_rotvec()
                self.rotations[key] = (previous @
                    Rotation.from_rotvec(self.smoothing * step).as_matrix())
        return self.tracked.copy()

    def zero(self) -> bool:
        if not self.has_person or not self.latest:
            return False
        for key, frame in self.latest.items():
            self.neutral[key] = frame.copy()
            self.rotations[key] = np.eye(3)
        return True

    def transforms(self) -> tuple[np.ndarray, ...]:
        rotations: list[np.ndarray] = []
        worlds: list[np.ndarray] = []
        identity = np.eye(3)
        for part in PARTS:
            parent = part.parent
            parent_rotation = identity if parent is None else rotations[parent]
            # During partial tracking, an unavailable child follows its parent.
            absolute = (self.rotations[part.key] if part.key in self.tracked
                        and part.key in self.rotations else parent_rotation)
            local = transform(parent_rotation.T @ absolute, part.offset)
            world = local if parent is None else worlds[parent] @ local
            rotations.append(absolute)
            worlds.append(world)
        return tuple(worlds)
