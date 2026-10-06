#!/usr/bin/env python3
"""GUI-owned cleanup worker. Only a named completed capture under recordings is accepted."""
import argparse
import os
from pathlib import Path
import sys

from optusbproj_ubuntu.recording_cleanup import delete_capture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('--started-mono-ns',type=int,required=True)
    args=parser.parse_args()
    try:
        os.nice(10)
        target=delete_capture(args.capture,args.started_mono_ns)
        print(f'已永久删除原始录制：{target}',flush=True)
    except (OSError,ValueError,TypeError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':main()
