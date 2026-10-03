"""Local-only device assignment and per-body-part axis mapping store."""

from __future__ import annotations

import json
from pathlib import Path

from .axis_mapping import normalized_mapping
from .skeleton import SENSOR_KEYS
from .pose import ORIENTATION_MODES, VENDOR_RPY


def load_config(path: Path) -> tuple[dict[str, str], dict[str, tuple[str, str, str]]]:
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("assignments"), dict):
        raise ValueError(f"配置格式错误：{path}")
    assignments = raw["assignments"]
    result = {key: value for key, value in assignments.items()
              if key in SENSOR_KEYS and isinstance(value, str)}
    used = [value for value in result.values() if value]
    if len(used) != len(set(used)):
        raise ValueError(f"配置中同一设备被分配给多个部位：{path}")
    mappings = raw.get("axis_mappings", {})
    if not isinstance(mappings, dict):
        raise ValueError(f"轴映射配置格式错误：{path}")
    valid_mappings = {}
    for key, mapping in mappings.items():
        if key in SENSOR_KEYS:
            try:
                valid_mappings[key] = normalized_mapping(mapping)
            except ValueError as exc:
                raise ValueError(f"{key} 的轴映射配置无效：{exc}") from exc
    return result, valid_mappings


def load_assignments(path: Path) -> dict[str, str]:
    return load_config(path)[0]


def load_algorithm_modes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    modes = raw.get("algorithm_modes", {})
    if not isinstance(modes, dict):
        raise ValueError(f"姿态算法配置格式错误：{path}")
    for key, mode in modes.items():
        if key not in SENSOR_KEYS or mode not in ORIENTATION_MODES:
            raise ValueError(f"姿态算法配置无效：{key} / {mode}")
    return modes


def save_config(path: Path, assignments: dict[str, str],
                axis_mappings: dict[str, tuple[str, str, str]],
                algorithm_modes: dict[str, str] | None = None) -> None:
    used = [value for value in assignments.values() if value]
    if len(used) != len(set(used)):
        raise ValueError("同一设备不能分配给多个部位")
    mappings = {key: list(normalized_mapping(value))
                for key, value in axis_mappings.items() if key in SENSOR_KEYS}
    modes = (load_algorithm_modes(path) if algorithm_modes is None else algorithm_modes)
    if any(key not in SENSOR_KEYS or mode not in ORIENTATION_MODES
           for key, mode in modes.items()):
        raise ValueError("姿态算法配置无效")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 3,
               "assignments": {key: assignments.get(key, "") for key in SENSOR_KEYS},
               "axis_mappings": mappings,
               "algorithm_modes": {key: ('gyro_tilt' if modes.get(key)=='gyro_rgbd' else modes.get(key, VENDOR_RPY))
                                   for key in SENSOR_KEYS}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_assignments(path: Path, assignments: dict[str, str]) -> None:
    # Compatibility helper: changing bindings must never erase saved mappings.
    _, mappings = load_config(path)
    save_config(path, assignments, mappings)
