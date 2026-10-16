#!/usr/bin/env python3
"""Launch independent explicit-send Qt demo; no IO until user connects."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OptUSBProj-ubuntu'))
from real_robot_manual.gui import main

if __name__ == '__main__':
    raise SystemExit(main())
