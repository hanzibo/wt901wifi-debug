#!/usr/bin/env python3
"""Generate a visualization-only X1 right-arm trajectory from one WT session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parent
sys.path.insert(0, str(REPOSITORY))

from optusbproj_ubuntu.x1_retarget import generate_trajectory


DEFAULT_SESSION = PROJECT / "calibrations" / "20260926-174220-4304e9"
DEFAULT_URDF = REPOSITORY / "urdf-model" / "pkg_robot_model" / "urdf" / "pkg_robot_model.urdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help="同一目录须含 imu.csv、manifest.json 和右臂两份通过的标定结果")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path,
                        help="输出 CSV；默认写到录制目录内 x1_right_trajectory.csv")
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output or args.session / "x1_right_trajectory.csv"
    summary = generate_trajectory(args.session, args.urdf, output, rate_hz=args.rate)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("仅供 RViz 可视化；此文件不是真机控制轨迹。")


if __name__ == "__main__":
    main()
