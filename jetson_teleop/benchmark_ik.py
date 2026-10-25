"""Offline two-arm objective benchmark. Never imports/open serial or UDP.

Run: python -m jetson_teleop.benchmark_ik --frames 40 [--profile]
Targets exercise both scene geometries, palms and arm-segment orientation.
This is a reproducible synthetic benchmark, not live-input acceptance.
"""
import argparse
import cProfile
import json
import pstats
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'OptUSBProj-ubuntu'))
from optusbproj_ubuntu.real_robot_arm import load_profile, ArmLimits
from optusbproj_ubuntu.scene_robot_model import SceneArm


def model(side):
    data, _, _ = load_profile(side)
    arm = SceneArm(data, [-np.pi]*6, [np.pi]*6)
    limits = (ArmLimits(side=side, lower=(0,90,0,0,0,50),
                        upper=(235,240,180,118,180,121))
              if side=='right' else ArmLimits())
    arm.lower, arm.upper = limits.to_sim(limits.lower), limits.to_sim(limits.upper)
    return arm


def tasks(arm, count):
    for t in np.linspace(0, 2*np.pi, count):
        q = arm.neutral + np.array([.25*np.sin(t), .3+.15*np.sin(t),
                                    .2*np.sin(t+.3), .25*np.cos(t),
                                    .25*np.sin(t), .15*np.cos(t)])
        q = np.clip(q, arm.lower+.01, arm.upper-.01)
        s,e,w = arm.points(q)
        yield dict(directions=[(e-s)/arm.lengths[0], (w-e)/arm.lengths[1]],
                   positions=(e,w), rotation=arm.tool_frame(q)[:3,:3],
                   segment_rotations={k:arm.segment_frame(q,k)[:3,:3]
                                      for k in ('upper_arm','forearm')})


def run(compiled, frames):
    arms = {side:model(side) for side in ('right','left')}
    warm = time.monotonic()
    if compiled:
        from jetson_teleop.fast_geometry import enable
        for arm in arms.values(): enable(arm)
    warm = (time.monotonic()-warm)*1000
    previous = {s:np.clip(m.neutral,m.lower,m.upper) for s,m in arms.items()}
    inputs = {s:list(tasks(m,frames)) for s,m in arms.items()}
    elapsed, outputs, errors = [], [], []
    for i in range(frames):
        start = time.monotonic(); output=[]; error=[]
        for s, arm in arms.items():
            q, pos, rot = arm.fit(previous=previous[s], step=np.full(6,2*np.pi),
                                  solver_mode='staged', **inputs[s][i])
            previous[s] = q
            output.extend(q); error.extend([pos,rot])
        elapsed.append((time.monotonic()-start)*1000)
        outputs.append(output); errors.append(error)
    return dict(backend='numba_geometry' if compiled else 'reference',
                frames=frames, warmup_ms=warm,
                median_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed,95)),
                max_ms=max(elapsed)), np.asarray(outputs), np.asarray(errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=40)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.frames<2: parser.error('frames must be >=2')
    profile=cProfile.Profile() if args.profile else None
    if profile: profile.enable()
    if profile:
        run(False,args.frames)
        profile.disable(); pstats.Stats(profile).sort_stats('cumtime').print_stats(20)
    # Profiling overhead must never contaminate the reference timing comparison.
    reference,q0,e0=run(False,args.frames)
    compiled,q1,e1=run(True,args.frames)
    print(json.dumps(dict(reference=reference,compiled=compiled,
                         max_joint_difference_deg=float(np.rad2deg(np.max(np.abs(q1-q0)))),
                         max_error_difference=float(np.max(np.abs(e1-e0)))),indent=2))


if __name__=='__main__': main()
