#!/usr/bin/env python3
"""Create a read-only functional mounting-calibration candidate from one session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optusbproj_ubuntu.functional_calibration import (
    HoldWindow, fit_arm_basis, fit_arm_bone_basis)


def parse_window(text: str) -> HoldWindow:
    try:
        start, end = map(float, text.split(":"))
        return HoldWindow(start, end)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("时间窗格式应为 起始秒:结束秒，例如 2:4") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="原始联合录制目录")
    parser.add_argument("--part", required=True, choices=(
        "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm"))
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--method", choices=("axis", "bone"), default="bone",
                        help="bone 依据肢段指向，更适合普通抬臂；axis 要求纯正交轴动作")
    parser.add_argument("--neutral", type=parse_window, required=True,
                        help="中立位静止窗口，相对录制开始的秒数，如 2:4")
    parser.add_argument("--forward", type=parse_window, required=True,
                        help="直臂前举终点静止窗口")
    parser.add_argument("--lateral", type=parse_window, required=True,
                        help="直臂侧举终点静止窗口")
    parser.add_argument("--output", type=Path, help="可选：另存候选标定 JSON，不写入配置")
    args = parser.parse_args()
    if not args.session.is_dir():
        parser.error("录制目录不存在")
    if args.output is not None:
        if args.output.resolve().is_relative_to(args.session.resolve()):
            parser.error("候选标定请保存到录制目录之外")
        if args.output.exists():
            parser.error("输出文件已存在；请更换文件名以免覆盖")
    try:
        fit = fit_arm_bone_basis if args.method == "bone" else fit_arm_basis
        result = fit(args.session, args.part, args.device_id,
                     args.neutral, args.forward, args.lateral)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"无法标定：{exc}\n")
    print(f"{result.part} / {result.device_id}: {result.status}")
    if args.method == "axis":
        print(f"前举 {result.forward_angle_deg:.1f}°，侧举 {result.lateral_angle_deg:.1f}°；"
              f"两轴夹角 {result.axis_separation_deg:.1f}°")
        print(f"动作轴拟合误差：前举 {result.forward_axis_error_deg:.1f}°，"
              f"侧举 {result.lateral_axis_error_deg:.1f}°")
    else:
        print(f"骨段指向误差：前举 {result.forward_bone_error_deg:.1f}°，"
              f"侧举 {result.lateral_bone_error_deg:.1f}°")
        print(f"骨段仰角：前举 {result.forward_elevation_deg:+.1f}°，"
              f"侧举 {result.lateral_elevation_deg:+.1f}°")
    for name in ("neutral", "forward", "lateral"):
        print(f"{name}: 静止姿态 P90 波动 {result.hold_spread_deg[name]:.1f}°，"
              f"角速度 P90 {result.gyro_p90_deg_s[name]:.1f}°/s")
    for warning in result.warnings:
        print(f"警告：{warning}")
    print("B（中立传感器坐标 → 模型坐标）=\n" +
          "\n".join("  " + " ".join(f"{value:+.4f}" for value in row)
                    for row in result.basis_matrix))
    print("这是候选安装标定，不会自动改写现有轴映射或模型姿态。")
    if args.output is not None:
        payload = result.as_dict()
        payload["method"] = args.method
        payload["protocol"] = "neutral + straight-arm forward raise + lateral raise"
        payload["session"] = str(args.session.resolve())
        payload["windows_s"] = {
            name: [getattr(args, name).start_s, getattr(args, name).end_s]
            for name in ("neutral", "forward", "lateral")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"候选已另存：{args.output}")


if __name__ == "__main__":
    main()
