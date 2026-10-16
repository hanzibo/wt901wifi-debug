#!/usr/bin/env python3
"""Launch an existing entry point without depending on the caller's cwd.

Only an explicit launch opens a GUI/device. --list and --dry-run are read-only.
This tool does not activate environments, install packages or start simulators.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "body": "OptUSBProj-ubuntu/run.py",
    "fusion": "OptUSBProj-ubuntu/fusion_run.py",
    "camera": "OptUSBProj-ubuntu/camera_run.py",
    "wrist": "wrist_demo.py",
    "imu-gripper": "OptUSBProj-ubuntu/gripper_gesture_demo.py",
    "lan-gripper": "OptUSBProj-ubuntu/lan_gripper_demo.py",
    "real-robot-manual": "demos/real_robot_manual_demo.py",
}


def launch_command(root, entry, arguments, conda_env=None):
    path = root / ENTRIES[entry]
    if not path.is_file():
        raise ValueError(f"Entry not found: {path}; use a source checkout")
    python = (["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
              if conda_env else [sys.executable])
    return [*python, "-u", str(path), *arguments]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--env", help="Use this existing Conda environment")
    parser.add_argument("entry", nargs="?", choices=tuple(ENTRIES))
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="Arguments for the original entry (after --)")
    args = parser.parse_args(argv)
    if args.list:
        for name, path in ENTRIES.items():
            print(f"{name:12} {path}")
        return 0
    if args.entry is None:
        parser.error("choose an entry or use --list")
    root = args.repo.resolve()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    try:
        command = launch_command(root, args.entry, arguments, args.env)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(shlex.join(command))
        return 0
    env = os.environ.copy()
    paths = [str(root), str(root / "OptUSBProj-ubuntu")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # Preserve cwd for user-supplied relative replay/settings paths. Built-in
    # entry and module/resource paths are absolute or source-relative.
    try:
        return subprocess.run(command, env=env, check=False).returncode
    except OSError as exc:
        print(f"Cannot launch: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
