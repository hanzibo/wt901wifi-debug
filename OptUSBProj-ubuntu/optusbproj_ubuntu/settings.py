"""Small, local-only device-to-body assignment store."""

from __future__ import annotations

import json
from pathlib import Path

from .skeleton import SENSOR_KEYS


def load_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("assignments"), dict):
        raise ValueError(f"配置格式错误：{path}")
    assignments = raw["assignments"]
    result = {key: value for key, value in assignments.items()
              if key in SENSOR_KEYS and isinstance(value, str)}
    used = [value for value in result.values() if value]
    if len(used) != len(set(used)):
        raise ValueError(f"配置中同一设备被分配给多个部位：{path}")
    return result


def save_assignments(path: Path, assignments: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "assignments": {key: assignments.get(key, "")
                                          for key in SENSOR_KEYS}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)

