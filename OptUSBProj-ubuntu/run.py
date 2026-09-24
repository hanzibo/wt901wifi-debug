#!/usr/bin/env python3
"""Run the Ubuntu WT901 upper-body demonstrator from any working directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901 上身姿态映射（Ubuntu 版）")
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--settings", type=Path,
                        default=PROJECT_DIR / "local_config.json")
    parser.add_argument("--replay", type=Path, help="回放本程序录制的 CSV，不占用 UDP 端口")
    args = parser.parse_args()

    # Match the existing wrist viewer's Wayland/OpenGL workaround.
    if (os.environ.get("WAYLAND_DISPLAY")
            and os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb")):
        os.environ["QT_QPA_PLATFORM"] = "wayland"

    from PySide6 import QtWidgets
    from wt901.gui_udp import GuiUdpSource
    from optusbproj_ubuntu.gui import BodyWindow
    from optusbproj_ubuntu.recording import ReplaySource
    from optusbproj_ubuntu.settings import load_assignments

    try:
        assignments = load_assignments(args.settings)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"无法读取设备分配配置：{exc}\n")
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        source = (ReplaySource(args.replay) if args.replay else
                  GuiUdpSource(port=args.listen_port, host=args.listen_ip))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"无法打开数据源：{exc}\n"
                      "实时模式请先关闭手腕控制台或其他占用 UDP 端口的程序。\n")
    try:
        window = BodyWindow(source, args.settings, assignments)
    except Exception:
        source.close()
        raise
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
