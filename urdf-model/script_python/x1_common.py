"""Shared joint metadata for the X1 RViz-only Python controllers."""

from typing import NamedTuple


class JointSpec(NamedTuple):
    name: str
    label: str
    minimum_deg: float
    maximum_deg: float


JOINT_SPECS = (
    JointSpec("101", "头部偏航", -90.0, 90.0),
    JointSpec("102", "头部俯仰", -45.0, 45.0),
    JointSpec("51", "右肩前后", -180.0, 180.0),
    JointSpec("52", "右肩左右", -180.0, 180.0),
    JointSpec("53", "右大臂旋转", -180.0, 180.0),
    JointSpec("54", "右肘", -180.0, 180.0),
    JointSpec("55", "右小臂旋转", -180.0, 180.0),
    JointSpec("56", "右手腕", -180.0, 180.0),
    JointSpec("57", "右夹爪", 0.0, 57.3),
    JointSpec("1", "左肩前后", -180.0, 180.0),
    JointSpec("2", "左肩左右", -180.0, 180.0),
    JointSpec("3", "左大臂旋转", -180.0, 180.0),
    JointSpec("4", "左肘", -180.0, 180.0),
    JointSpec("5", "左小臂旋转", -180.0, 180.0),
    JointSpec("6", "左手腕", -180.0, 180.0),
    JointSpec("7", "左夹爪", 0.0, 57.3),
)

JOINT_NAMES = tuple(spec.name for spec in JOINT_SPECS)

# Values are degrees. These poses are only for RViz visualization.
POSES_DEG = {
    "zero": {name: 0.0 for name in JOINT_NAMES},
    "arms_up": {
        "51": -75.0, "52": -20.0, "54": 35.0,
        "1": 75.0, "2": 20.0, "4": -35.0,
    },
    "greeting": {
        "101": 15.0,
        "51": -35.0, "52": -55.0, "54": 85.0, "56": 30.0, "57": 30.0,
        "1": 15.0, "4": -20.0,
    },
    "grippers_open": {"57": 50.0, "7": 50.0},
}


def clamp_degrees(joint_name: str, value: float) -> float:
    """Clamp a visualization command to this tool's conservative GUI range."""
    spec = next(spec for spec in JOINT_SPECS if spec.name == joint_name)
    return max(spec.minimum_deg, min(spec.maximum_deg, value))
