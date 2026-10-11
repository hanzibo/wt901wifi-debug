#!/usr/bin/env python3
"""Run existing unittest suites with consistent paths and offscreen Qt.

Uses the current Python environment unless a suite's conda environment is
specified. Does not install dependencies or launch live simulator demos.
Separate processes isolate suites with duplicate names.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "core": (ROOT / "tests", "test_*.py"),
    "app": (ROOT / "OptUSBProj-ubuntu/tests", "test_*.py"),
    "scene": (ROOT / "OptUSBProj-ubuntu/tests", "test_scene*.py"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--pattern", help="Override unittest's file pattern")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--core-env", help="Conda environment for core tests")
    parser.add_argument("--app-env", help="Conda environment for app/scene tests")
    args = parser.parse_args()
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "OptUSBProj-ubuntu")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    selected = ("core", "app") if args.suite == "all" else (args.suite,)
    results = []
    for name in selected:
        directory, pattern = SUITES[name]
        conda_env = args.core_env if name == "core" else args.app_env
        interpreter = (["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
                       if conda_env else [sys.executable])
        command = [*interpreter, "-m", "unittest", "discover", "-s",
                   str(directory), "-p", args.pattern or pattern]
        if args.verbose:
            command.append("-v")
        print(f"\n[{name}] Python: {conda_env or sys.executable}", flush=True)
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            code = result.returncode
        except OSError as exc:
            print(f"[{name}] Cannot start tests: {exc}", file=sys.stderr)
            code = 1
        results.append((name, code, time.monotonic() - started))
    for name, code, seconds in results:
        print(f"[{name}] {'PASS' if code == 0 else 'FAIL'} ({seconds:.1f}s)")
    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
