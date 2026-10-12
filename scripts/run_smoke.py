#!/usr/bin/env python3
"""Offline smoke suite: no Qt, camera, GPU, ROS or simulator required."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "tests": ("test_battery.py", "test_so3_filter.py", "test_test_runner.py",
              "test_repo_tools.py"),
    "OptUSBProj-ubuntu/tests": ("test_shared_core.py", "test_scene_baseline.py",
                              "test_x1_ik.py"),
}


def main() -> int:
    sys.path[:0] = [str(ROOT), str(ROOT / "OptUSBProj-ubuntu")]
    for directory, patterns in PATTERNS.items():
        for pattern in patterns:
            if not list((ROOT / directory).glob(pattern)):
                print(f"Missing smoke test: {directory}/{pattern}", file=sys.stderr)
                return 1
    suite = unittest.TestSuite()
    for directory, patterns in PATTERNS.items():
        for pattern in patterns:
            # Fresh loaders avoid retaining the previous suite's top-level dir.
            suite.addTests(unittest.TestLoader().discover(str(ROOT / directory), pattern))
    if suite.countTestCases() == 0:
        print("No smoke tests found; use an editable source checkout", file=sys.stderr)
        return 1
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
