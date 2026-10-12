#!/usr/bin/env python3
"""Read-only repository checks; never import applications or delete files."""
import argparse
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    from scripts.run_project import ENTRIES
else:
    from .run_project import ENTRIES


def inspect(root):
    result = subprocess.run(["git", "ls-files", "--cached", "--others",
                             "--exclude-standard", "-z"], cwd=root,
                            capture_output=True, check=True)
    files = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    errors = []
    python_count = 0
    large = []
    for name in files:
        path = root / name
        if not path.is_file():
            errors.append(f"Missing tracked file: {name}")
            continue
        if path.suffix == ".py":
            python_count += 1
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            except (SyntaxError, UnicodeError, OSError) as exc:
                errors.append(f"{name}: {exc}")
        if path.stat().st_size > 10 * 1024 * 1024:
            large.append((path.stat().st_size, name))
    for name, entry in ENTRIES.items():
        if not (root / entry).is_file():
            errors.append(f"Missing entry {name}: {entry}")
    diff = subprocess.run(["git", "diff", "--check", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=False)
    if diff.returncode:
        errors.append(diff.stdout + diff.stderr)
    return python_count, sorted(large, reverse=True), errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        count, large, errors = inspect(args.repo.resolve())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Cannot inspect source checkout: {exc}", file=sys.stderr)
        return 1
    print(f"Parsed {count} Python files; checked launch entries and diff whitespace")
    for size, name in large[:10]:
        print(f"Large asset (informational): {size / 1024**2:.1f} MiB {name}")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    # Also supports invoking this file directly from outside the repository.
    raise SystemExit(main())
