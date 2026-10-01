"""Right-handed mappings from neutral sensor axes to GUI model axes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_AXIS_MAPPING = ("+X", "+Y", "+Z")
AXES = {"X": 0, "Y": 1, "Z": 2}


def mapping_matrix(mapping: Sequence[str]) -> np.ndarray:
    """Return a rotation whose columns map sensor X/Y/Z into model coordinates."""
    if not isinstance(mapping, (list, tuple)) or len(mapping) != 3:
        raise ValueError("请为设备 X/Y/Z 三轴各选择一个模型方向")
    result = np.zeros((3, 3), dtype=float)
    used = set()
    for column, choice in enumerate(mapping):
        if not isinstance(choice, str) or len(choice) != 2 or choice[0] not in "+-" \
                or choice[1] not in AXES:
            raise ValueError(f"无效的模型轴方向：{choice!r}")
        axis = AXES[choice[1]]
        if axis in used:
            raise ValueError("模型 X/Y/Z 每根轴只能分配给一根设备轴")
        used.add(axis)
        result[axis, column] = 1.0 if choice[0] == "+" else -1.0
    if round(float(np.linalg.det(result))) != 1:
        raise ValueError("该组合会变成镜像坐标系；请调整一根轴的正负方向")
    return result


def normalized_mapping(mapping: Sequence[str]) -> tuple[str, str, str]:
    mapping_matrix(mapping)
    return mapping[0], mapping[1], mapping[2]
