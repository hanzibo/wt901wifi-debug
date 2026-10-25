"""Replay recorded pose snapshots through BOTH IK backends, without hardware.

Usage: python -m jetson_teleop.replay_ik recordings/jetson_ik/.../diagnostic.jsonl
This measures compute time, not a live network/control schedule simulation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'OptUSBProj-ubuntu'))
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from optusbproj_ubuntu.real_robot_shared import SharedArmIK
from optusbproj_ubuntu.real_robot_protocol import encode


def human(raw):
    return {key:({k:np.asarray(v,dtype=float) for k,v in value.items()} if key=='rotations'
                 else None if value is None else np.asarray(value,dtype=float))
            for key,value in raw.items()}


def solver(settings,compiled):
    arms={}
    for side,raw in settings['limits'].items():
        profile=settings['profiles'][side]
        profile=(profile[0],np.asarray(profile[1],dtype=float),profile[2])
        arm=SharedArmIK(encode([90.]*15,[0]*10),ArmLimits(**raw),side=side,
            profile=profile,palm=settings['palms'][side],offset=settings['offsets'][side])
        if compiled:
            from .fast_geometry import prepare
            prepare(arm)
        arms[side]=arm
    return arms


def replay(path,max_frames=300):
    # Bounded memory: at most max_frames snapshots; no pickle or eval.
    settings=None;frames=[];network={};complete=False;live=[]
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);event=r['event']
            if event=='metadata':
                if r.get('schema_version')!=1:raise ValueError('不支持的诊断格式')
                settings=r['settings']
            elif event=='imu_parsed':
                row=network.setdefault(r['device'],dict(packets=0,gaps=[],queues=[]))
                row['packets']+=1
                if r.get('gap_ms') is not None:row['gaps'].append(r['gap_ms'])
                if r.get('socket_queue_ms') is not None:row['queues'].append(r['socket_queue_ms'])
            elif event=='ik_solved':live.append(r['ik_ms'])
            elif event=='recording_end':complete=r.get('complete',False)
            elif event=='pose_snapshot' and settings and len(frames)<max_frames:
                entries=r['entries']; sides=settings['limits']
                if not all(entries.get(s) is not None for s in sides):continue
                observed=r['monotonic_ns']/1e9
                if any(max(entries[s][0]['device_ages_ms'].values())+
                       (observed-entries[s][0]['at'])*1000>250 for s in sides):continue
                frames.append({s:human(entries[s][0]['sides'][s]) for s in sides})
    if not settings or not frames:raise ValueError('没有可回放姿态；请先归零并保持设备在线后录制')
    def stats(values):
        if not values:return None
        return dict(n=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(max(values)))
    results={};outputs=[]
    for compiled in (False,True):
        arms=solver(settings,compiled)  # Includes warmup, excluded from solve timings.
        times=[];angles=[];positions=[]
        for frame in frames:
            before=time.monotonic();q=[];errors=[]
            for s,arm in arms.items():
                out=arm.solve(frame[s]);q.extend(out['target']);errors.append(out['position_error_mm'])
            times.append((time.monotonic()-before)*1000);angles.append(q);positions.append(errors)
        results['compiled' if compiled else 'reference']=dict(solve_ms=stats(times),max_position_error_mm=float(np.max(positions)))
        outputs.append(np.asarray(angles))
    return dict(recording=str(path),complete=complete,frames=len(frames),
        note='offline compute replay; not live control acceptance; first bounded snapshot window',
        max_joint_difference_deg=float(np.max(np.abs(outputs[0]-outputs[1]))),
        live_ik_ms=stats(live),backends=results,
        receive={k:dict(packets=v['packets'],read_gap_ms=stats(v['gaps']),kernel_to_read_ms=stats(v['queues'])) for k,v in network.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--max-frames',type=int,default=300)
    args=parser.parse_args()
    if not 1<=args.max_frames<=3000:parser.error('max-frames must be 1..3000')
    print(json.dumps(replay(args.recording,args.max_frames),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
