"""Read-only timing check for a recorded camera + WT901 session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optusbproj_ubuntu.fusion_sync import analyze_session, aligned_orientations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="录制会话目录")
    for side in ("left", "right"):
        for segment in ("upper_arm", "forearm"):
            parser.add_argument(f"--{side}-{segment.replace('_', '-')}-id", default="")
    parser.add_argument("--torso-id", default="", help="躯干只做旋转插值，不估计动作延迟")
    parser.add_argument("--max-lag-ms", type=int, default=400)
    parser.add_argument("--aligned-output", type=Path,
                        help="可选：另存每帧插值后的旋转矩阵 JSONL，不修改原始录制")
    parser.add_argument("--apply-estimated-lags", action="store_true",
                        help="输出对齐数据时采用可参考的估计延迟；默认全部采用 0 ms")
    args = parser.parse_args()
    assignments = {
        part: getattr(args, part + "_id")
        for part in ("left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm")
    }
    if not any(assignments.values()) and not args.torso_id:
        parser.error("请至少提供一个设备 ID；录制尚未保存绑定关系")
    if not args.session.is_dir():
        parser.error("会话目录不存在")
    if args.max_lag_ms < 0:
        parser.error("--max-lag-ms 不能为负")
    print("注意：正延迟表示录制时间轴上 IMU 动作晚于相机动作；不是硬件采样延迟。")
    results = analyze_session(args.session, assignments,
                              max_lag_ms=args.max_lag_ms)
    for result in results:
        lag = "不可估计" if result.lag_ms is None else f"{result.lag_ms:+d} ms"
        corr = "—" if result.correlation is None else f"{result.correlation:.2f}"
        print(f"{result.part} / {result.device_id}: {lag}; 相关系数 {corr}; "
              f"{result.visual_samples} 对视觉运动样本; {result.quality}")
        print(f"  {result.reason}")
    if args.aligned_output is not None:
        if args.aligned_output.resolve().is_relative_to(args.session.resolve()):
            parser.error("对齐结果请保存到录制目录之外，避免改动原始会话")
        if args.aligned_output.exists():
            parser.error("对齐输出已存在；请换一个文件名，避免覆盖")
        selected = {item.device_id: (item.lag_ms if args.apply_estimated_lags
                    and item.quality == "可参考" else 0) for item in results}
        if args.torso_id:
            selected[args.torso_id] = 0
        rows = aligned_orientations(args.session, selected)
        args.aligned_output.parent.mkdir(parents=True, exist_ok=True)
        with args.aligned_output.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        for device_id in selected:
            covered = sum(row["orientation_matrix"][device_id] is not None
                          for row in rows)
            print(f"{device_id}: {covered}/{len(rows)} 帧有可插值姿态；"
                  f"采用 {selected[device_id]:+d} ms")
        print(f"对齐结果已保存：{args.aligned_output}")


if __name__ == "__main__":
    main()
