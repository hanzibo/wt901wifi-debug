"""15-part hierarchy and mesh placement reconstructed from CfgDataShow.xml.

The seven live slots are torso, left/right upper arm, forearm and hand. Other
parts remain in the hierarchy so a missing sensor inherits its parent's pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Part:
    key: str
    label: str
    parent: int | None
    mesh: str
    color: str
    offset: tuple[float, float, float]  # metres, in parent/model coordinates
    sensor_slot: bool = False


PARTS: tuple[Part, ...] = (
    Part("pelvis", "骨盆", None, "basin.STL", "#8B0000", (0, 0, 0)),
    Part("torso", "躯干", 0, "body01.STL", "#DC143C", (-0.00978, 0, 0.13049), True),
    Part("head", "头部", 1, "head.STL", "#FFB6C1", (-0.02829, 0, 0.57115)),
    Part("left_upper_arm", "左上臂", 1, "lUpperArm01.STL", "#4682B4", (-0.05481, 0.18436, 0.36973), True),
    Part("left_forearm", "左前臂", 3, "lForeArm01.STL", "#87CEEB", (0, 0, -0.26349), True),
    Part("left_hand", "左手", 4, "lHand01.STL", "#E6E6FA", (0, 0, -0.2724), True),
    Part("right_upper_arm", "右上臂", 1, "rUpperArm01.STL", "#4682B4", (-0.05481, -0.18436, 0.36973), True),
    Part("right_forearm", "右前臂", 6, "rForeArm01.STL", "#87CEEB", (0, 0, -0.26349), True),
    Part("right_hand", "右手", 7, "rHand01.STL", "#E6E6FA", (0, 0.012, -0.2724), True),
    Part("left_thigh", "左大腿", 0, "lThigh.STL", "#32CD32", (0, 0.09974, 0)),
    Part("left_calf", "左小腿", 9, "lCalf.STL", "#228B22", (0, 0.00568, -0.4326)),
    Part("left_foot", "左脚", 10, "lFoot.STL", "#FFD700", (0.00503, 0.01356, -0.35294)),
    Part("right_thigh", "右大腿", 0, "rThigh.STL", "#32CD32", (0, -0.09974, 0)),
    Part("right_calf", "右小腿", 12, "rCalf.STL", "#228B22", (0, -0.00568, -0.4326)),
    Part("right_foot", "右脚", 13, "rFoot.STL", "#FFD700", (0.00503, -0.01356, -0.35294)),
)

SENSOR_KEYS = tuple(part.key for part in PARTS if part.sensor_slot)
PART_BY_KEY = {part.key: index for index, part in enumerate(PARTS)}

# The original UI remaps WT sensor XYZ to model XYZ with this orthogonal basis.
SENSOR_TO_MODEL = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=float)


def transform(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = identity_transform()
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix
