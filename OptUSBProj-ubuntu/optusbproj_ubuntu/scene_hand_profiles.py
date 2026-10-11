"""Versioned hand endpoint file storage and scene validation; no Qt dependency."""
import json
from pathlib import Path
import numpy as np

from .scene_hand import hand_pose


def write_hand_profiles(path, sides):
    if not sides:
        raise ValueError("至少确认一侧的两组端点后才能保存")
    data = {"version": 1, "sides": sides}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_hand_profiles(path, hands):
    """Validate every side before returning values for the GUI to apply.

Loading never grants output authorization or confirms endpoint poses.
Angles are degrees in files and radians in the returned arrays.
"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["version"] != 1:
        raise ValueError("不支持的手型格式")
    if (not isinstance(data["sides"], dict) or not data["sides"]
            or not set(data["sides"]) <= {"left", "right"}):
        raise ValueError("无效手侧")
    checked = {}
    for side, row in data["sides"].items():
        hand = hands[side]
        if row["paths"] != [obj["path"] for obj in hand["objects"]]:
            raise ValueError("手指对象路径不匹配")
        if not 5 <= row["speed_deg_s"] <= 720:
            raise ValueError("无效手指速度")
        checked[side] = (
            hand_pose(np.deg2rad(row["open_deg"]), hand["limits"]),
            hand_pose(np.deg2rad(row["closed_deg"]), hand["limits"]),
            row["speed_deg_s"],
        )
    return checked
