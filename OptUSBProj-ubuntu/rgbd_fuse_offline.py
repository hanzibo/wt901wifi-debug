#!/usr/bin/env python3
"""Experimental single/bilateral arm heading fusion; never modifies captures."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optusbproj_ubuntu.rgbd_camera_analysis import analyze_camera, capture_parts, window, metrics
from optusbproj_ubuntu.rgbd_fusion import ARM_SEGMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-state',action='store_true',help='Fit actual GUI gyro poses, preserving its reference epoch')
    parser.add_argument('--parts',nargs='+',choices=tuple(ARM_SEGMENTS),help='Explicit single/bilateral participants; otherwise follow guided capture metadata')
    parser.add_argument('--max-offset-deg',type=float,default=30.,help='累计纠偏范围5～90度，默认30；不改变创新门限或速率')
    parser.add_argument('--unlimited-offset',action='store_true',help='实验：不限累计角度，其他安全门限不变')
    parser.add_argument('--static-train-only',action='store_true',help='联合标定：仅首次中立/前举/侧举静止窗口拟合，重复动作独立验收')
    parser.add_argument('--neutral', type=window, default=(3.,6.), help='seconds relative to recording start; default 3:6')
    parser.add_argument('--train', type=window, default=(3.,18.), help='calibration prefix; default 3:18; correction only starts afterwards')
    args = parser.parse_args()
    if not np.isfinite(args.max_offset_deg) or not 5<=args.max_offset_deg<=90:parser.error('max-offset-deg必须在5～90之间')
    if args.unlimited_offset:args.max_offset_deg=None
    if args.output.resolve().is_relative_to(args.session.resolve()) or args.output.exists():
        parser.error('输出必须为原始录制之外、尚不存在的目录')
    if not args.train[0] <= args.neutral[0] < args.neutral[1] <= args.train[1]:
        parser.error('neutral 必须位于 train 内')
    try:
        analyze_camera(args)
    except (ValueError,KeyError,OSError,TypeError) as exc:
        parser.exit(1,str(exc)+'\n')


if __name__=='__main__':
    main()
