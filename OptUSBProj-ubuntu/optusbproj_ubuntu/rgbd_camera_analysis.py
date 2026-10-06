#!/usr/bin/env python3
"""Experimental single/bilateral arm heading fusion; never modifies captures."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .rgbd_fusion import (SEGMENTS, ARM_SEGMENTS, unit, angle_deg, fit_mounting,
                                         HeadingCorrector, load_six_axis, visual_direction)
from .rgbd_fusion import load_live_series
from .camera_protocol import camera_phases, OBLIQUE_PROTOCOL
from .analysis_context import AnalysisContext


def window(text):
    a, b = map(float, text.split(':'))
    if not 0 <= a < b:
        raise argparse.ArgumentTypeError('要求 0 <= 起点 < 终点')
    return a, b


def metrics(values):
    return {'samples': len(values), 'median_deg': float(np.median(values)) if values else None,
            'p90_deg': float(np.percentile(values, 90)) if values else None}

def capture_parts(session, requested=None, *, guide=None):
    """Honor explicit scope, not other devices incidentally online in a capture."""
    guide_path=session/'camera_guide.json'
    if guide is None:guide=json.loads(guide_path.read_text()) if guide_path.exists() else {}
    parts=requested if requested is not None else guide.get('participants',list(guide.get('devices',{})) or list(SEGMENTS))
    if not parts or len(parts)!=len(set(parts)) or any(p not in ARM_SEGMENTS for p in parts):
        raise ValueError('参与部位必须为不重复的左右上臂/前臂')
    return list(parts)


def analyze_camera(args, *, context=None):
    """Original scalar CPU analysis, with a per-run read-only data context."""
    context = context or AnalysisContext(args.session)
    context.check_session(args.session)
    if args.output.resolve().is_relative_to(args.session.resolve()) or args.output.exists():
        raise ValueError('输出必须为原始录制之外、尚不存在的目录')
    if args.max_offset_deg is not None and (not np.isfinite(args.max_offset_deg) or not 5<=args.max_offset_deg<=90):
        raise ValueError('max-offset-deg必须在5～90之间')
    if not args.train[0] <= args.neutral[0] < args.neutral[1] <= args.train[1]:
        raise ValueError('neutral 必须位于 train 内')
    manifest = context.manifest
    try:participants=capture_parts(args.session,args.parts,guide=context.guide)
    except ValueError as exc:raise ValueError(str(exc))
    rows = context.vision
    if not rows:
        raise ValueError('没有 RGB-D 三维观测')
    origin = int(manifest['started_mono_ns'])
    guide_path=args.session/'camera_guide.json'
    guide=context.guide
    visible_protocol=guide.get('camera_pose_protocol')==OBLIQUE_PROTOCOL
    if guide.get('camera_pose_protocol') is not None:camera_phases(guide)
    if visible_protocol and (guide.get('status')!='complete' or not args.live_state):
        raise ValueError('斜前举协议要求完整录制及实时IMU姿态记录')
    static_windows = []
    if args.static_train_only or visible_protocol:
        train_phases,_=camera_phases(guide)
        for step in train_phases:
            matches = [e for e in guide['events'] if e['step']==step and e['phase']=='hold' and e.get('end_s') is not None]
            if len(matches)!=1:raise ValueError('首次静止训练窗口缺失或重复')
            e = matches[0];static_windows.append((e['start_s']+.5,e['end_s']-.2))
    def seconds(row):
        return (row['captured_mono_ns']-origin)/1e9
    up_vectors = []
    for row in rows:
        if args.neutral[0] <= seconds(row) <= args.neutral[1] and all(row['valid'][i] for i in [0,1,6,7]):
            xyz = np.asarray(row['points_camera_m'], float)
            up_vectors.append(unit((xyz[0]+xyz[1]-xyz[6]-xyz[7])/2))
    if len(up_vectors) < 5:
        raise ValueError('中立窗口缺少完整肩髋观测')
    up_camera = unit(np.mean(up_vectors, axis=0))
    up_spread = float(np.percentile([angle_deg(v,up_camera) for v in up_vectors],90))
    if up_spread > 8:
        raise ValueError('中立窗口躯干方向不稳定；请选择站直、静止的窗口')
    report = {'format':'rgbd-heading-fusion-experiment-v1', 'source_session':str(args.session.resolve()),
              'orientation_source':'live_runtime_pose' if args.live_state else 'recomputed_from_raw_imu',
              'camera_epoch':manifest.get('camera_epoch'),
              'vision_config':manifest.get('vision_config'),'vision_epoch':manifest.get('vision_epoch'),
              'heading_max_offset_deg':args.max_offset_deg,
              'neutral_window_s':args.neutral, 'training_window_s':args.train,
              'camera_up_assumption':'upright torso shoulder-to-hip direction in neutral window; NOT a measured floor gravity',
              'camera_up':up_camera.tolist(), 'neutral_up_spread_p90_deg':up_spread,
              'clock_note':'device time for gyro integration, host arrival for matching; lag=0 not physically synchronized',
              'measurement_note':'direction errors are against noisy RGB-D surfaces, NOT ground-truth pose accuracy; no axial twist correction',
              'participants':participants,'parts':{}}
    outputs = [{'captured_mono_ns':r['captured_mono_ns'],'completed_mono_ns':r['completed_mono_ns'],
                'time_s':seconds(r),'frame':r['frame'],'vision':r, 'parts':{}} for r in rows]
    args.output.mkdir(parents=True)
    for part in participants:
        device = manifest['assignments_at_start'].get(part)
        if not device:
            report['parts'][part] = {'accepted':False,'reason':'部位未分配设备'}
            continue
        try:
            neutral_ns=(origin+int(args.neutral[0]*1e9),origin+int(args.neutral[1]*1e9))
            if args.live_state:series,up_world,imu_samples,bootstrap=load_live_series(args.session,device,neutral_ns,context=context)
            else:series, up_world, imu_samples, bootstrap = load_six_axis(args.session, device,neutral_ns=neutral_ns)
            train = [r for r in rows if args.train[0] <= seconds(r) <= args.train[1] and
                     (not (args.static_train_only or visible_protocol) or any(a<=seconds(r)<=b for a,b in static_windows))]
            lengths = [r['bone_lengths_m'][part] for r in train if r['bone_lengths_m'].get(part) is not None]
            if not lengths:
                raise ValueError('没有标定骨长')
            expected = float(np.median(lengths))
            matrices, directions = [], []
            phase_directions={p:[] for p in camera_phases(guide)[0]} if visible_protocol else {}
            for r in train:
                matrix = series.at(r['captured_mono_ns'])
                d, _ = visual_direction(r,part,expected)
                if matrix is not None and d is not None:
                    matrices.append(matrix);directions.append(d)
                    for phase,(a,b) in zip(phase_directions,static_windows):
                        if a<=seconds(r)<=b:phase_directions[phase].append(d)
            if visible_protocol:
                for phase,values in phase_directions.items():
                    if len(values)<8:raise ValueError(f'{phase} 拟合同期可信观测不足8帧，请调整姿态让肩肘腕分开')
                means={p:unit(np.mean(v,axis=0)) for p,v in phase_directions.items()}
                separation=angle_deg(means['oblique'],means['lateral'])
                if separation<20:raise ValueError('斜前举与侧举方向相差不足20°，不能提供足够独立信息')
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
        if visible_protocol:
            summary['training_phase_samples']={p:len(v) for p,v in phase_directions.items()}
            summary['oblique_lateral_separation_deg']=separation
        corrector = HeadingCorrector(up_camera,max_offset_deg=args.max_offset_deg)
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
    if visible_protocol:
        from optusbproj_ubuntu.combined_calibration import validate_camera
        report['camera_pose_protocol']=OBLIQUE_PROTOCOL
        report['training_phases']=list(camera_phases(guide)[0])
        report=validate_camera(args.session,report,context=context)
    context.check_session(args.session)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
    with (args.output/'comparison.jsonl').open('w') as stream:
        for row in outputs:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print('分析目录：',args.output.resolve())
    return report
