"""Offline comparison of recorded candidate snapshots; no ROS or motor output."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from x1_collision_model import CollisionModel


def check_with_prefixes(model, start, goal):
    began = time.monotonic()
    original = None
    for alpha in (1., .5, .25, .125, .0625):
        step = {k:v+alpha*(goal[k]-v) for k,v in start.items()}
        result = model.check_motion(start, step,
            budget=min(.025, max(0., .09-(time.monotonic()-began))))
        if original is None:
            original = result.copy()
        if result['safe']:
            result['alpha'] = alpha
            break
        if time.monotonic()-began >= .09:
            break
    result['requested_result'] = original
    result['total_elapsed_ms'] = (time.monotonic()-began)*1000
    return result


def analyze(session, count=60):
    with (session/'frames.csv').open() as f:
        rows = list(csv.DictReader(f))
    snapshots = []
    seen = set()
    for row in rows:
        status = json.loads(row.get('collision_status_json') or '{}')
        key = (status.get('configuration_id'),status.get('result',{}).get('id'))
        if not status.get('enabled') or not status.get('accepted') or key in seen:
            continue
        start = status['accepted']
        goal = start.copy()
        try:
            for joint in ('1','2','3','4','5','6','51','52','53','54','55','56'):
                goal[joint] = float(row['joint_'+joint])
        except (KeyError,ValueError):
            continue
        seen.add(key)
        snapshots.append((int(row['frame']), start, goal))
    if not snapshots:
        raise ValueError('没有启用碰撞检查的已接受构型与候选角度')
    indices = np.unique(np.linspace(0,len(snapshots)-1,min(count,len(snapshots)),dtype=int))
    report = dict(session=str(session.resolve()),snapshot_count=len(indices),
        note='相同录制快照的起点/候选对照；候选与异步结果非严格同一请求。'
             '不重新求IK，不是整段闭环随动回放，也不是实体机器人安全验收。', models={})
    for label, compound in [('single_hull',False),('compound_hulls',True)]:
        model = CollisionModel(compound=compound)
        results = []
        for index in indices:
            frame, start, goal = snapshots[index]
            result = check_with_prefixes(model,start,goal)
            results.append(dict(frame=frame, **result))
        times = [r['total_elapsed_ms'] for r in results]
        report['models'][label] = dict(version=model.version,
            full_steps=sum(r['safe'] and r.get('alpha') == 1 for r in results),
            shortened_steps=sum(r['safe'] and r.get('alpha',1) < 1 for r in results),
            blocked_steps=sum(not r['safe'] for r in results),
            final_categories=dict(Counter(r['category'] for r in results)),
            median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
            results=results)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--samples',type=int,default=60)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.samples < 1:parser.error('samples must be positive')
    if args.output and (args.output.exists() or
                       args.output.resolve().is_relative_to(args.session.resolve())):
        parser.error('输出应为录制目录之外尚不存在的文件')
    report = analyze(args.session,args.samples)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='models'},ensure_ascii=False))
    for label,data in report['models'].items():
        print(label,json.dumps({k:v for k,v in data.items() if k!='results'},ensure_ascii=False))


if __name__=='__main__':main()
