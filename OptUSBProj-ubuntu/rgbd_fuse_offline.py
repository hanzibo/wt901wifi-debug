#!/usr/bin/env python3
"""Experimental right-arm heading fusion. Never modifies captures or GUI settings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optusbproj_ubuntu.rgbd_fusion import (SEGMENTS, unit, angle_deg, fit_mounting,
                                         HeadingCorrector, load_six_axis, visual_direction)
from optusbproj_ubuntu.rgbd_fusion import load_live_series


def window(text):
    a, b = map(float, text.split(':'))
    if not 0 <= a < b:
        raise argparse.ArgumentTypeError('要求 0 <= 起点 < 终点')
    return a, b


def metrics(values):
    return {'samples': len(values), 'median_deg': float(np.median(values)) if values else None,
            'p90_deg': float(np.percentile(values, 90)) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-state',action='store_true',help='Fit actual GUI gyro poses, preserving its reference epoch')
    parser.add_argument('--neutral', type=window, default=(3.,6.), help='seconds relative to recording start; default 3:6')
    parser.add_argument('--train', type=window, default=(3.,18.), help='calibration prefix; default 3:18; correction only starts afterwards')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.session.resolve()) or args.output.exists():
        parser.error('输出必须为原始录制之外、尚不存在的目录')
    if not args.train[0] <= args.neutral[0] < args.neutral[1] <= args.train[1]:
        parser.error('neutral 必须位于 train 内')
    manifest = json.loads((args.session/'manifest.json').read_text())
    rows = [json.loads(l) for l in (args.session/'vision.jsonl').open()]
    if not rows:
        parser.error('没有 RGB-D 三维观测')
    origin = int(manifest['started_mono_ns'])
    def seconds(row):
        return (row['captured_mono_ns']-origin)/1e9
    up_vectors = []
    for row in rows:
        if args.neutral[0] <= seconds(row) <= args.neutral[1] and all(row['valid'][i] for i in [0,1,6,7]):
            xyz = np.asarray(row['points_camera_m'], float)
            up_vectors.append(unit((xyz[0]+xyz[1]-xyz[6]-xyz[7])/2))
    if len(up_vectors) < 5:
        parser.error('中立窗口缺少完整肩髋观测')
    up_camera = unit(np.mean(up_vectors, axis=0))
    up_spread = float(np.percentile([angle_deg(v,up_camera) for v in up_vectors],90))
    if up_spread > 8:
        parser.error('中立窗口躯干方向不稳定；请选择站直、静止的窗口')
    report = {'format':'rgbd-heading-fusion-experiment-v1', 'source_session':str(args.session.resolve()),
              'orientation_source':'live_runtime_pose' if args.live_state else 'recomputed_from_raw_imu',
              'camera_epoch':manifest.get('camera_epoch'),
              'neutral_window_s':args.neutral, 'training_window_s':args.train,
              'camera_up_assumption':'upright torso shoulder-to-hip direction in neutral window; NOT a measured floor gravity',
              'camera_up':up_camera.tolist(), 'neutral_up_spread_p90_deg':up_spread,
              'clock_note':'device time for gyro integration, host arrival for matching; lag=0 not physically synchronized',
              'measurement_note':'direction errors are against noisy RGB-D surfaces, NOT ground-truth pose accuracy; no axial twist correction',
              'parts':{}}
    outputs = [{'captured_mono_ns':r['captured_mono_ns'],'completed_mono_ns':r['completed_mono_ns'],
                'time_s':seconds(r),'frame':r['frame'],'vision':r, 'parts':{}} for r in rows]
    args.output.mkdir(parents=True)
    for part in SEGMENTS:
        device = manifest['assignments_at_start'].get(part)
        if not device:
            report['parts'][part] = {'accepted':False,'reason':'部位未分配设备'}
            continue
        try:
            neutral_ns=(origin+int(args.neutral[0]*1e9),origin+int(args.neutral[1]*1e9))
            if args.live_state:series,up_world,imu_samples,bootstrap=load_live_series(args.session,device,neutral_ns)
            else:series, up_world, imu_samples, bootstrap = load_six_axis(args.session, device,neutral_ns=neutral_ns)
            train = [r for r in rows if args.train[0] <= seconds(r) <= args.train[1]]
            lengths = [r['bone_lengths_m'][part] for r in train if r['bone_lengths_m'].get(part) is not None]
            if not lengths:
                raise ValueError('没有标定骨长')
            expected = float(np.median(lengths))
            matrices, directions = [], []
            for r in train:
                matrix = series.at(r['captured_mono_ns'])
                d, _ = visual_direction(r,part,expected)
                if matrix is not None and d is not None:
                    matrices.append(matrix);directions.append(d)
            fit = fit_mounting(matrices,directions,up_world,up_camera)
        except ValueError as exc:
            report['parts'][part] = {'accepted':False,'device_id':device,'reason':str(exc)}
            continue
        summary = {'device_id':device,'accepted':fit.accepted,'calibration_samples':len(matrices),
                   'reference_epoch':imu_samples[0].get('epoch') if args.live_state else None,
                   'up_world':up_world.tolist(),
                   'Q_camera_from_imu_world':fit.camera_from_imu_world.tolist(),
                   'bone_axis_sensor':fit.bone_axis_sensor.tolist(),'bone_length_m':expected,
                   'fit_median_deg':fit.median_error_deg,'fit_p90_deg':fit.p90_error_deg,
                   'fit_jacobian_singular_values':fit.jacobian_singular_values.tolist(),
                   'horizontal_calibration_samples':fit.horizontal_samples,
                   'gravity_initialization':bootstrap,
                   'imu_packets':len(imu_samples),'max_so3_distance':max(x['so3_distance'] for x in imu_samples)}
        report['parts'][part] = summary
        corrector = HeadingCorrector(up_camera)
        gates, baseline_errors, fused_before_errors, fused_after_errors = Counter(), [], [], []
        events = []
        for r, output in zip(rows,outputs):
            matrix = series.at(r['captured_mono_ns'])
            if matrix is None:
                output['parts'][part]={'reason':'无可插值 IMU / 间隔超过100ms'}
                continue
            pure = fit.camera_from_imu_world @ matrix
            predicted = pure @ fit.bone_axis_sensor
            before = corrector.rotation() @ pure
            d, reason = visual_direction(r,part,expected)
            applied, innovation = False, None
            validation = seconds(r) > args.train[1]
            if validation and fit.accepted and d is not None:
                applied, reason, innovation = corrector.update(seconds(r),predicted,d)
                baseline_errors.append(angle_deg(predicted,d))
                fused_before_errors.append(angle_deg(before @ fit.bone_axis_sensor,d))
            elif not validation:
                reason='标定前段：不执行融合'
            elif not fit.accepted:
                reason='标定未通过：只输出纯IMU，不纠偏'
            after = corrector.rotation() @ pure
            if validation and fit.accepted and d is not None:
                fused_after_errors.append(angle_deg(after @ fit.bone_axis_sensor,d))
            gates[reason]+=1
            output['parts'][part]={'rotation_pure_camera':pure.tolist(),'rotation_fused_camera':after.tolist(),
                                   'direction_pure':predicted.tolist(),'direction_fused':(after @ fit.bone_axis_sensor).tolist(),
                                   'vision_direction':d.tolist() if d is not None else None,
                                   'offset_deg':float(np.degrees(corrector.offset_rad)),
                                   'correction_applied':applied,'reason':reason,'innovation_deg':innovation,
                                   'validation':validation}
            # Completed time marks when a correction could become available.
            events.append((int(r['completed_mono_ns']),corrector.offset_rad))
        summary['validation_pure_direction_error']=metrics(baseline_errors)
        summary['validation_fused_before_current_observation_error']=metrics(fused_before_errors)
        summary['validation_fused_after_current_observation_error']=metrics(fused_after_errors)
        summary['gates']=dict(gates)
        summary['final_heading_offset_deg']=float(np.degrees(corrector.offset_rad))
        # High-frequency matrices hold the most recently completed correction;
        # never feed a future observation to an earlier IMU packet.
        event_index, offset = 0, 0.
        with (args.output/(part+'_imu.jsonl')).open('w') as stream:
            for sample in imu_samples:
                while event_index < len(events) and events[event_index][0] <= sample['host_recv_mono_ns']:
                    offset=events[event_index][1];event_index+=1
                pure=fit.camera_from_imu_world @ np.array(sample['rotation_imu_world'])
                fused=Rotation.from_rotvec(up_camera*offset).as_matrix() @ pure
                sample.update(rotation_pure_camera=pure.tolist(),rotation_fused_camera=fused.tolist(),
                              offset_deg=float(np.degrees(offset)))
                stream.write(json.dumps(sample,allow_nan=False)+'\n')
        print(part, 'accepted=',fit.accepted,'fit median=',round(fit.median_error_deg,2),
              'validation pure/fused(before)=',summary['validation_pure_direction_error'],summary['validation_fused_before_current_observation_error'],flush=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
    with (args.output/'comparison.jsonl').open('w') as stream:
        for row in outputs:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print('分析目录：',args.output.resolve())


if __name__=='__main__':
    main()
