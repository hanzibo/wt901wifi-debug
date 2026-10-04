#!/usr/bin/env python3
"""Analyze an existing guided RGB-D session without changing runtime state."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optusbproj_ubuntu.visual_wearing import analyze_wearing


def main():
    parser = argparse.ArgumentParser(description='双臂/单臂视觉辅助穿戴标定（只读原录制）')
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.session.resolve() in args.output.resolve().parents:
        parser.error('输出目录必须尚不存在，且在原录制目录之外')
    try:
        report = analyze_wearing(args.session)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
        for part, result in report['parts'].items():
            if 'basis_matrix' in result:
                (args.output / f'{part}.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
            print(part, result['status'], '; '.join(result['warnings']), flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
