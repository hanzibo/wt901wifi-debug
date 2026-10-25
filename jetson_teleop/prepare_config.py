#!/usr/bin/env python3
"""Explicitly export only six device assignments and paths, not PC secrets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'OptUSBProj-ubuntu')]
from jetson_teleop.config import PARTS
from optusbproj_ubuntu.real_robot_checkpoints import validate


def configuration(local, left, bilateral):
    for mode, data in (('left_shared_ik', left), ('bilateral_shared_ik', bilateral)):
        validate(data, mode)
        if not data['input_settings']['manual_mapping']:
            raise ValueError('Jetson 入口只接受手动轴映射预设')
    return dict(version=1, mode='left_shared_ik',
                assignments={k: local['assignments'][k] for k in PARTS},
                axis_mappings={k: local['axis_mappings'][k] for k in PARTS},
                frequency=20, signs={s: [1]*6 for s in ('left', 'right')},
                palm_offsets={s: dict(axis='z', degrees=90) for s in ('left', 'right')},
                checkpoints={'left_shared_ik': 'checkpoints/left_shared_ik/'+left['id']+'.json',
                             'bilateral_shared_ik': 'checkpoints/bilateral_shared_ik/'+bilateral['id']+'.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, default=ROOT/'OptUSBProj-ubuntu/local_config.json')
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--bilateral', type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    # stdout only: caller decides where to save. No overwrite or device action.
    print(json.dumps(configuration(read(args.local), read(args.left), read(args.bilateral)),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
