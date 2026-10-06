#!/usr/bin/env python3
"""Independent UDP1399 gesture observer; never publishes robot commands."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == '__main__':
    from optusbproj_ubuntu.gripper_gesture_gui import main
    raise SystemExit(main())
