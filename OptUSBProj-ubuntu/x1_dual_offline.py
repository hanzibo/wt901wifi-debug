#!/usr/bin/env python3
"""Generate synchronized X1 left/right arm RViz-only trajectory from one WT session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT))

from optusbproj_ubuntu.x1_dual_retarget import generate_dual_trajectory


DEFAULT_SESSION = PROJECT / "calibrations" / "20260926-174220-4304e9"
DEFAULT_URDF = ROOT / "urdf-model" / "pkg_robot_model" / "urdf" / "pkg_robot_model.urdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output or args.session / "x1_dual_trajectory.csv"
    summary = generate_dual_trajectory(args.session, args.urdf, output, rate_hz=args.rate)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("仅供 RViz 双臂预览；不是实体机器人控制轨迹。")


if __name__ == "__main__":
    main()
