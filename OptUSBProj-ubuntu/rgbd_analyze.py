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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent analysis directory")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 analyzes all frames")
    args = parser.parse_args()
    if args.every < 1 or args.max_frames < 0:
        parser.error("every >= 1, max-frames >= 0")
    if args.output.resolve() == args.session.resolve():
        parser.error("分析输出不能覆盖原始录制")
    calibration = json.loads((args.session/"camera_calibration.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    detector = load_detector()
    estimator = DepthPoseEstimator()
    reasons, counts, times = Counter(), np.zeros(8, int), []
    n = 0
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
            value = result.to_record()
            value["source_session"] = str(args.session.resolve())
            value["frame"] = row["frame"]
            target.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
            if n < 5:
                cv2.imwrite(str(args.output/f"overlay-{n:03d}.jpg"), cv2.cvtColor(draw_pose_rgb(result), cv2.COLOR_RGB2BGR))
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
              "median_inference_ms": float(np.median(times)) if times else None,
              "note": "Offline surface observations, not joint-centre ground truth; no IMU fusion or source changes"}
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
