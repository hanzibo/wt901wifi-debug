#!/usr/bin/env python3
"""Read a guided capture once per backend; emit separate wearing and camera reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from optusbproj_ubuntu.combined_calibration import build_joint_report, load_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wearing-method',choices=('imu','visual'),default='imu')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.session.resolve()):parser.error('输出必须在原录制外且尚不存在')
    try:
        manifest,guide,parts,windows = load_capture(args.session)
        args.output.mkdir(parents=True,exist_ok=False)
        origin = manifest['started_mono_ns']
        seconds = lambda ns:(ns-origin)/1e9
        neutral = ':'.join(str(seconds(ns)) for ns in windows['neutral'])
        train = f'{seconds(windows["neutral"][0])}:{seconds(windows["lateral"][1])}'
        command = [sys.executable,'-u',str(Path(__file__).with_name('rgbd_fuse_offline.py')),str(args.session),
            '--output',str(args.output/'camera'),'--live-state','--static-train-only','--parts',*parts,'--neutral',neutral,'--train',train]
        limit = guide.get('heading_max_offset_deg',30.)
        command += ['--unlimited-offset'] if limit is None else ['--max-offset-deg',str(limit)]
        process = subprocess.run(command)
        if process.returncode == 0:
            camera = json.loads((args.output/'camera'/'report.json').read_text())
        else:
            (args.output/'camera').mkdir(exist_ok=True)
            camera = {'format':'rgbd-heading-fusion-experiment-v1','orientation_source':'live_runtime_pose',
                      'camera_epoch':manifest['camera_epoch'],'vision_epoch':manifest.get('vision_epoch'),
                      'vision_config':manifest.get('vision_config'),'source_session':str(args.session.resolve()),
                      'parts':{p:{'accepted':False,'device_id':guide['devices'][p],
                                 'reason':'相机分析失败，见进程日志；穿戴分析仍独立执行'} for p in parts}}
        report = build_joint_report(args.session,camera,args.wearing_method)
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
        (args.output/'wearing.json').write_text(json.dumps(report['wearing'],ensure_ascii=False,allow_nan=False,indent=2))
        (args.output/'camera'/'report.json').write_text(json.dumps(report['camera'],ensure_ascii=False,allow_nan=False,indent=2))
        print('联合分析完成；未应用到实时模型',flush=True)
    except (ValueError,KeyError,OSError,TypeError,subprocess.CalledProcessError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':main()
