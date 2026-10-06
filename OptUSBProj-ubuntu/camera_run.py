#!/usr/bin/env python3
"""Standalone camera-only upper-body model viewer (no WT901 UDP port)."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))
DEFAULT_MODEL = PROJECT_DIR / "assets" / "mediapipe" / "pose_landmarker_lite.task"


def main() -> int:
    parser = argparse.ArgumentParser(description="单目摄像头上身 7 部位随动实验")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV 摄像头编号，默认 0")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="MediaPipe Pose Landmarker .task 文件")
    parser.add_argument("--backend", choices=("mediapipe", "rtmw3d"),
                        default="mediapipe", help="初始姿态后端，GUI 中仍可切换")
    args = parser.parse_args()
    if args.backend == "mediapipe" and (
            not args.model.is_file() or args.model.stat().st_size == 0):
        parser.error(f"模型文件不存在或为空：{args.model}\n请先按 docs/CAMERA_SETUP.md 下载")
    if importlib.util.find_spec("cv2") is None:
        parser.error("缺少 OpenCV (cv2)；请按 docs/CAMERA_SETUP.md 准备组合环境")
    try:
        import PySide6  # noqa: F401
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        parser.error(f"缺少相机依赖：{exc}\n请按 docs/CAMERA_SETUP.md 准备组合环境")
    required = (("mediapipe",) if args.backend == "mediapipe"
                else ("rtmlib", "onnxruntime"))
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error(f"缺少 {', '.join(missing)}；请按 docs/CAMERA_SETUP.md 准备组合环境")

    from PySide6 import QtWidgets
    from optusbproj_ubuntu.camera_gui import CameraWindow

    app = QtWidgets.QApplication(sys.argv[:1])
    window = CameraWindow(args.model, args.camera_index, backend=args.backend)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
