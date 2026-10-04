#!/usr/bin/env python3
"""Analyze an existing RGB-D capture without cameras, UDP or source modifications."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optusbproj_ubuntu.fusion_capture import CameraObservation, RGBDData
from optusbproj_ubuntu.rgbd_pose import DepthPoseEstimator, NAMES
from optusbproj_ubuntu.rgbd_pose_worker import load_detector, infer_observation
from optusbproj_ubuntu.rgbd_pose_gui import draw_pose_rgb
from optusbproj_ubuntu.pose_2d import Pose2DConfig,DEFAULT_YOLO
from optusbproj_ubuntu.pose_quality import PoseQualityStats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent analysis directory")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 analyzes all frames")
    parser.add_argument('--pose-backend',choices=('rtmw','yolo26'),default='rtmw')
    parser.add_argument('--yolo-model',type=Path,default=DEFAULT_YOLO)
    parser.add_argument('--pose-imgsz',type=int,choices=(640,960),default=640)
    parser.add_argument('--warmup-frames',type=int,default=5,help='Exclude first N measured frames from timing statistics only')
    args = parser.parse_args()
    if args.every < 1 or args.max_frames < 0 or args.warmup_frames<0:
        parser.error("every >= 1, max-frames >= 0")
    if args.output.resolve().is_relative_to(args.session.resolve()):
        parser.error("分析输出不能覆盖原始录制")
    calibration = json.loads((args.session/"camera_calibration.json").read_text())
    if args.output.exists():parser.error('分析输出目录已存在；不覆盖')
    config=Pose2DConfig(args.pose_backend,args.yolo_model,args.pose_imgsz)
    detector = load_detector(config)
    args.output.mkdir(parents=True, exist_ok=False)
    estimator = DepthPoseEstimator()
    reasons, counts, times = Counter(), np.zeros(8, int), []
    n = 0
    manifest=json.loads((args.session/'manifest.json').read_text())
    guide_path=args.session/'camera_guide.json'
    events=json.loads(guide_path.read_text()).get('events',[]) if guide_path.is_file() else []
    quality=PoseQualityStats();side_raise=PoseQualityStats();failed_images=0
    with (args.session/"camera.jsonl").open() as source, (args.output/"results.jsonl").open("w") as target:
        for index, line in enumerate(source):
            if index % args.every:
                continue
            row = json.loads(line)
            d = row.get("rgbd")
            if d is None:
                continue
            bgr = cv2.imread(str(args.session/row["frame"]))
            if bgr is None:
                raise OSError(f"无法读取 {row['frame']}")
            obs = CameraObservation(int(row["captured_mono_ns"]), "orbbec", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, None,
                                    RGBDData(np.load(args.session/d["raw_depth"], allow_pickle=False),
                                             np.load(args.session/d["aligned_depth_mm"], allow_pickle=False), calibration, d["timing"]))
            result = infer_observation(detector, estimator, obs)
            t=(obs.captured_mono_ns-manifest['started_mono_ns'])/1e9
            quality.update(result,time_s=t,include_timing=n>=args.warmup_frames)
            if any(e['phase']=='hold' and e['step'] in ('lateral','lateral_repeat') and e.get('end_s') is not None and e['start_s']<=t<=e['end_s'] for e in events):
                side_raise.update(result,time_s=t,include_timing=n>=args.warmup_frames)
            else:side_raise.break_sequence()
            value = result.to_record()
            value["source_session"] = str(args.session.resolve())
            value["frame"] = row["frame"]
            target.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
            if n < 5:
                cv2.imwrite(str(args.output/f"overlay-{n:03d}.jpg"), cv2.cvtColor(draw_pose_rgb(result), cv2.COLOR_RGB2BGR))
            elif failed_images<5 and not result.valid[[2,3,4,5]].all():
                cv2.imwrite(str(args.output/f'failure-{n:05d}.jpg'),cv2.cvtColor(draw_pose_rgb(result),cv2.COLOR_RGB2BGR));failed_images+=1
            counts += result.valid
            reasons.update(result.reasons)
            times.append(result.inference_ms)
            n += 1
            if n % 10 == 0:
                print(f"已分析 {n} 帧；当前有效点 {int(result.valid.sum())}/8", flush=True)
            if args.max_frames and n >= args.max_frames:
                break
    report = {"source_session": str(args.session.resolve()), "frames": n,
              "valid_counts": dict(zip(NAMES, counts.tolist())), "reasons": dict(reasons),
              "median_inference_ms": float(np.median(times[args.warmup_frames:])) if len(times)>args.warmup_frames else None,
              'vision_config':detector.metadata,'quality':quality.report(),'side_raise':side_raise.report(),
              'selection':{'every':args.every,'max_frames':args.max_frames,'warmup_frames':args.warmup_frames},
              "note": "Offline surface observations, not joint-centre ground truth; no IMU fusion or source changes"}
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
