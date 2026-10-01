#!/usr/bin/env python3
"""Run the joint WT901/camera capture baseline or replay a saved session."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))
DEFAULT_MODEL = PROJECT_DIR / "assets" / "mediapipe" / "pose_landmarker_lite.task"


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901 + 单目相机同步采集基线")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", choices=("rtmw3d", "mediapipe"), default="rtmw3d")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--settings", type=Path, default=PROJECT_DIR / "local_config.json")
    parser.add_argument("--replay", type=Path, help="回放联合录制目录，不占用摄像头或 UDP 端口")
    parser.add_argument("--calibration", type=Path, action="append", default=[],
                        help="仅回放：加载候选安装标定 JSON 作临时预览，可重复指定")
    parser.add_argument("--speed", type=float, default=1.0, help="回放倍速，默认 1")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("回放倍速必须大于 0")
    if args.calibration and args.replay is None:
        parser.error("候选标定只能在 --replay 模式预览，不能直接用于实时控制")
    if args.replay is None:
        required = ["cv2", "rtmlib", "onnxruntime"] if args.backend == "rtmw3d" else [
            "cv2", "mediapipe"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            parser.error(f"当前环境缺少依赖：{', '.join(missing)}")
        if args.backend == "mediapipe" and (
                not args.model.is_file() or args.model.stat().st_size == 0):
            parser.error(f"MediaPipe 模型不存在或为空：{args.model}")
    if (os.environ.get("WAYLAND_DISPLAY")
            and os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb")):
        os.environ["QT_QPA_PLATFORM"] = "wayland"

    from PySide6 import QtWidgets
    from wt901.gui_udp import GuiUdpSource
    from optusbproj_ubuntu.fusion_gui import FusionWindow
    from optusbproj_ubuntu.fusion_replay import FusionSessionReplay
    from optusbproj_ubuntu.settings import load_config, load_algorithm_modes

    try:
        assignments, axis_mappings = load_config(args.settings)
        algorithm_modes = load_algorithm_modes(args.settings)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"无法读取设备分配配置：{exc}\n")
    candidates = []
    manual_mapping_source = "当前配置"
    if args.replay:
        try:
            replay_manifest = json.loads(
                (args.replay / "manifest.json").read_text(encoding="utf-8"))
            algorithm_modes = replay_manifest.get("algorithm_modes_at_start", {})
            if not isinstance(algorithm_modes, dict):
                raise ValueError("姿态算法快照格式错误")
        except (OSError, ValueError) as exc:
            parser.exit(1, f"无法读取回放配置：{exc}\n")
    if args.calibration:
        try:
            manifest = json.loads((args.replay / "manifest.json").read_text(encoding="utf-8"))
            snapshot = manifest.get("assignments_at_start")
            if not isinstance(snapshot, dict):
                raise ValueError("该录制没有设备分配快照，无法安全预览候选标定")
            assignments = {key: value for key, value in snapshot.items()
                           if isinstance(value, str)}
            mapping_snapshot = manifest.get("axis_mappings_at_start")
            if isinstance(mapping_snapshot, dict) and mapping_snapshot:
                axis_mappings = {key: tuple(value) for key, value in
                                 mapping_snapshot.items()}
                manual_mapping_source = "录制时快照"
            used_parts = set()
            for path in args.calibration:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if Path(candidate["session"]).resolve() != args.replay.resolve():
                    raise ValueError(f"{path} 不属于当前回放会话")
                if ((candidate.get("method") == "bone" and
                     candidate.get("part") in ("left_upper_arm", "left_forearm",
                                               "right_upper_arm", "right_forearm")) or
                    (candidate.get("method") == "hand_bone_v1" and
                     candidate.get("part") in ("left_hand", "right_hand")) or
                    (candidate.get("method") == "torso_axis" and
                     candidate.get("part") == "torso")):
                    pass
                else:
                    raise ValueError(f"{path} 不是受支持的穿戴标定")
                if assignments.get(candidate["part"]) != candidate["device_id"]:
                    raise ValueError(f"{path} 的设备 ID 与录制时的部位分配不一致")
                if candidate["part"] in used_parts:
                    raise ValueError(f"{candidate['part']} 被重复指定候选标定")
                used_parts.add(candidate["part"])
                candidates.append(candidate)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            parser.exit(1, f"无法加载候选标定：{exc}\n")
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        source = (FusionSessionReplay(args.replay, speed=args.speed)
                  if args.replay else GuiUdpSource(args.listen_port, args.listen_ip))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"无法打开融合数据源：{exc}\n"
                      "实时模式请关闭其他占用 UDP 1399 的程序。\n")
    try:
        window = FusionWindow(source, args.settings, assignments, axis_mappings,
                              algorithm_modes=algorithm_modes,
                              camera_index=args.camera_index, backend=args.backend,
                              model_path=args.model,
                              calibration_candidates=candidates,
                              manual_mapping_source=manual_mapping_source)
    except Exception:
        source.close()
        raise
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
