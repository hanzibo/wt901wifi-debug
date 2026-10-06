#!/usr/bin/env python3
"""Compare two offline 2D/depth analyses of exactly the same source frames."""
import argparse
import json
from pathlib import Path


def compare(a,b):
    reports=[json.loads((p/'report.json').read_text()) for p in (a,b)]
    if reports[0]['source_session']!=reports[1]['source_session']:raise ValueError('必须为同一原始录制')
    rows=[]
    for p in (a,b):
        with (p/'results.jsonl').open() as stream:rows.append([json.loads(s) for s in stream])
    keys=[[(r['captured_mono_ns'],r['frame']) for r in group] for group in rows]
    if not keys[0] or keys[0]!=keys[1]:raise ValueError('必须使用完全相同且顺序一致的源帧；every/max-frames需一致')
    gates=[[r.get('diagnostics',{}).get('gates') for r in group] for group in rows]
    if gates[0]!=gates[1]:raise ValueError('两次分析深度/二维质量门限不一致，不能直接对比')
    return {'source_session':reports[0]['source_session'],'paired_frames':len(keys[0]),
            'analyses':[{'path':str(p.resolve()),'vision_config':r.get('vision_config'),
                         'quality':r['quality'],'side_raise':r['side_raise']} for p,r in zip((a,b),reports)],
            'note':'Same-frame availability comparison only, not pose accuracy or independent calibration validation.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_a',type=Path);parser.add_argument('analysis_b',type=Path)
    parser.add_argument('--output',type=Path,help='Optional new JSON file; never overwrites')
    args=parser.parse_args()
    try:result=compare(args.analysis_a,args.analysis_b)
    except (OSError,KeyError,ValueError) as exc:parser.error(str(exc))
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print('配对帧数：',result['paired_frames'])
    for block in result['analyses']:
        print('\n',block['vision_config']['backend'],block['path'])
        for title,key in (('全部帧','quality'),('侧平举静止窗口','side_raise')):
            stats=block[key];print(title,'帧数',stats['frames'])
            for name in ('left_elbow','right_elbow','left_wrist','right_wrist'):
                p=stats['joints'][name]
                if p['available_2d_rate'] is not None:print(name,f"2D {p['available_2d_rate']:.1%} / 3D {p['available_3d_rate']:.1%}")
        print('耗时中位/P95(ms)：',block['quality']['median_inference_ms'],block['quality']['p95_inference_ms'])


if __name__=='__main__':main()
