#!/usr/bin/env python3
"""Verify that the active Python can import the ROS modules used by the demos."""

import os
import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"ROS_DISTRO: {os.environ.get('ROS_DISTRO', '<not sourced>')}")

    failed = False
    for module_name in ("em", "yaml", "lark", "rclpy", "sensor_msgs"):
        try:
            module = __import__(module_name)
            print(f"[OK] {module_name}: {getattr(module, '__file__', '<built-in>')}")
        except Exception as error:
            failed = True
            print(f"[FAIL] {module_name}: {error}")

    try:
        import tkinter
        print(f"[OK] tkinter: Tk {tkinter.TkVersion}")
    except Exception as error:
        failed = True
        print(f"[FAIL] tkinter: {error}")

    try:
        from PySide6.QtWidgets import QApplication
        print(f"[OK] PySide6: {QApplication}")
    except Exception as error:
        failed = True
        print(f"[FAIL] PySide6: {error}")

    if failed:
        print("\nEnvironment is incomplete. Follow script_python/README.md.")
        return 1
    print("\nEnvironment is ready for the X1 Python visualization scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
