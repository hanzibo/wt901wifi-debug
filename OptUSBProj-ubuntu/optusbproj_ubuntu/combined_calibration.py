"""One capture, separate mounting/heading fits; never use held-out poses for fitting."""
from copy import deepcopy
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from .calibration_reuse import proper
from .camera_protocol import camera_phases
from .guided_calibration import fit_guided_arm, neutral_matrices_by_mode, reference_motion_deg
from .pose import GYRO_TILT
from .calibration_options import capture_prior_weight
from .rgbd_fusion import ARM_SEGMENTS, angle_deg, gravity_alignment, load_live_series, visual_direction
from .visual_wearing import PASS, REVIEW, analyze_wearing, apply_wearing_report

FORMAT = 'wt901-combined-arm-camera-calibration-v1'
PHASES = {'neutral': 'neutral', 'forward': 'forward', 'lateral': 'lateral',
          'check_forward': 'forward_repeat', 'check_neutral': 'neutral_return'}


def load_capture(session):
    session = Path(session)
    manifest = json.loads((session/'manifest.json').read_text())
    guide = json.loads((session/'camera_guide.json').read_text())
    if manifest.get('error') or guide.get('status') != 'complete':raise ValueError('录制未完整结束，不能联合标定')
    parts = guide.get('participants', list(guide.get('devices', {})))
    if not parts or len(set(parts)) != len(parts) or not set(parts).issubset(ARM_SEGMENTS):raise ValueError('参与部位无效')
    origin = int(manifest['started_mono_ns']);cutoff = int(manifest['retained_until_mono_ns'])
    windows = {}
    for e in guide['events']:
        if e['phase'] == 'hold' and e.get('end_s') is not None:
            if e['step'] in windows:raise ValueError('静止窗口重复，不能确定拟合/复测数据')
            start = origin+round((e['start_s']+.5)*1e9)
            end = min(cutoff, origin+round((e['end_s']-.2)*1e9))
            if end <= start:raise ValueError('静止窗口太短')
            windows[e['step']] = (start, end)
    required = set(PHASES.values()) | {'lateral_repeat'}
    train_phases, check_phases = camera_phases(guide)
    required |= set(train_phases) | set(check_phases)
    if not required.issubset(windows):raise ValueError('缺少联合标定训练/复测窗口')
    return manifest, guide, parts, windows


def imu_wearing(session, *, prior_weight=None):
    """Reuse original guided fitter with gravity-aligned FIT poses and RAW neutral streams."""
    session = Path(session);manifest, guide, parts, windows = load_capture(session)
    prior_weight = capture_prior_weight(guide, prior_weight)
    raw = {}
    with (session/'imu.csv').open() as stream:
        for row in csv.DictReader(stream):raw[(int(row['host_recv_mono_ns']), row['device_id'])] = row
    with (session/'runtime_pose.jsonl').open() as stream:runtime = [json.loads(line) for line in stream]
    report = {'format': 'wt901-combined-imu-wearing-v1', 'session': str(session.resolve()),
              'camera_epoch': manifest['camera_epoch'], 'vision_epoch': manifest.get('vision_epoch'),
              'vision_config': manifest.get('vision_config'), 'prior_weight': prior_weight, 'parts': {}}
    for part in parts:
        device = guide['devices'][part]
        try:
            neutral_ns = windows['neutral']
            _, up, rows, _ = load_live_series(session, device, neutral_ns)
            aligned = gravity_alignment(up, [0, 0, 1]);samples = {step: [] for step in PHASES}
            context = {step: [] for step in PHASES}
            torso = manifest['assignments_at_start'].get('torso')
            for step, phase in PHASES.items():
                start, end = windows[phase]
                for row in runtime:
                    stamp = row['host_recv_mono_ns']
                    if not start <= stamp <= end or row['device_id'] not in (device, torso):continue
                    packet = raw.get((stamp, row['device_id']))
                    if packet is None:continue
                    sample = {name: float(packet[name]) for name in ('roll','pitch','yaw','gyro_x','gyro_y','gyro_z','acc_x','acc_y','acc_z')}
                    sample['received_ns'] = stamp
                    sample['orientation_by_mode'] = row['orientation_by_mode']
                    if row['device_id'] == device:
                        if row['epoch'] != rows[0]['epoch']:raise ValueError('积分参考系变化')
                        sample['orientation_mode'] = GYRO_TILT
                        sample['orientation_matrix'] = (aligned@np.asarray(row['orientation_by_mode'][GYRO_TILT])).tolist()
                        samples[step].append(sample)
                    else:context[step].append(sample)
            result = fit_guided_arm(part, device, samples, tuple(manifest['axis_mappings_at_start'][part]), prior_weight=prior_weight)
            entry = asdict(result)
            paired = {m: n.tolist() for m,n in neutral_matrices_by_mode(samples['neutral']).items()}
            if GYRO_TILT not in paired:raise ValueError('缺少原始陀螺中立基准')
            entry.update(neutral_matrix=paired[GYRO_TILT], neutral_matrices_by_mode=paired,
                         reference_epoch=rows[0]['epoch'], up_world=up.tolist())
            if torso and all(context.values()):
                motions = reference_motion_deg(context);entry['torso_motion_deg'] = motions
                if max(motions.values()) > 12:entry['status'] = '需重录';entry['warnings'].append('躯干转动超过12°')
                elif max(motions.values()) >= 5:entry['warnings'].append('提示：躯干移动超过建议5°')
            if part.endswith('forearm') and result.fit_error_deg['lateral'] >= 15:
                entry['warnings'].append('提示：前臂侧举误差≥15°，检查肘伸直和前臂旋转')
            entry['accepted'] = entry['status'] == '通过'
            report['parts'][part] = entry
        except (ValueError, KeyError) as exc:
            report['parts'][part] = {'device_id': device, 'accepted': False, 'status': '需重录', 'warnings': [str(exc)]}
    return report


def validate_camera(session, report):
    """Independent repeat/return observation tests, not GT and not folded into the fit."""
    if not any(d.get('accepted') for d in report['parts'].values()):return report
    session = Path(session);_, guide, parts, windows = load_capture(session)
    _, check_phases = camera_phases(guide)
    with (session/'vision.jsonl').open() as stream:vision = [json.loads(line) for line in stream]
    for part in parts:
        data = report['parts'].get(part)
        if data is None:continue
        data['independent_validation'] = {};failures = []
        if not data.get('accepted'):continue
        try:
            series, _, _, _ = load_live_series(session, data['device_id'], windows['neutral'])
            q = proper(data['Q_camera_from_imu_world']);bone = np.asarray(data['bone_axis_sensor'])
            for phase in check_phases:
                errors = []
                for row in vision:
                    if not windows[phase][0] <= row['captured_mono_ns'] <= windows[phase][1]:continue
                    r = series.at(row['captured_mono_ns']);d, _ = visual_direction(row, part, data['bone_length_m'])
                    if r is not None and d is not None:errors.append(angle_deg(q@r@bone, d))
                metrics = {'samples': len(errors)}
                if len(errors) < 8:failures.append(f'{phase} 同期可信观测不足8帧')
                else:
                    metrics.update(median_deg=float(np.median(errors)), p90_deg=float(np.percentile(errors,90)))
                    if metrics['median_deg'] >= 12 or metrics['p90_deg'] >= 25:failures.append(f'{phase} 方向误差过大')
                data['independent_validation'][phase] = metrics
        except (ValueError, KeyError) as exc:failures.append(str(exc))
        if failures:data['accepted'] = False;data['reason'] = '；'.join(failures)
    return report


def build_joint_report(session, camera_report, method='imu', *, prior_weight=None):
    if method not in ('imu','visual'):raise ValueError('未知穿戴标定方法')
    try:wearing = imu_wearing(session, prior_weight=prior_weight) if method == 'imu' else analyze_wearing(Path(session), prior_weight=prior_weight)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        manifest,guide,parts,_ = load_capture(session)
        wearing = {'camera_epoch':manifest['camera_epoch'],'vision_epoch':manifest.get('vision_epoch'),
                   'vision_config':manifest.get('vision_config'),'parts':{
                       p:{'device_id':guide['devices'][p],'accepted':False,'status':'需重录','warnings':[str(exc)]} for p in parts}}
    camera = deepcopy(camera_report)
    try:camera = validate_camera(session, camera)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        for data in camera['parts'].values():data.update(accepted=False,reason='相机独立复测失败：'+str(exc))
    for data in wearing['parts'].values():
        if method == 'visual':data['accepted'] = data.get('status') in (PASS,REVIEW) and not data.get('hard_failures')
    return {'format': FORMAT, 'session': str(Path(session).resolve()), 'wearing_method': method,
            'wearing': wearing, 'camera': camera, 'note': '同一次采集，两路独立拟合和验收；不自动应用'}


def apply_joint(owner, report, wearing_parts, camera_parts, *, review_acknowledged=False):
    """Validate on copies, then transactional install: mounting first, camera second."""
    if report.get('format') != FORMAT:raise ValueError('联合报告格式错误')
    parts = set(wearing_parts) | set(camera_parts)
    if not parts:raise ValueError('请选择至少一项合格结果')
    if len(set(wearing_parts)) != len(wearing_parts) or len(set(camera_parts)) != len(camera_parts):raise ValueError('重复选择部位')
    if owner.replay_mode or owner.session is not None or owner._record_start_deadline is not None:raise ValueError('录制、预约或回放中不能应用')
    if owner.x1_preview_dialog is not None and owner.x1_preview_dialog.sender is not None:raise ValueError('请先停止X1随动')
    wearing = report['wearing'];camera = report['camera']
    owner._check_vision_report(wearing);owner._check_vision_report(camera)
    epoch = getattr(owner.camera_worker,'capture_epoch',None)
    if epoch is None or any(r.get('camera_epoch') != epoch for r in (wearing,camera)):raise ValueError('相机会话已变化')
    for part in parts:
        if part not in ARM_SEGMENTS or owner.engine.algorithm_modes[part] != GYRO_TILT:raise ValueError('参与部位须使用陀螺＋加计')
        if not owner.engine.is_live(owner.engine.assignments[part],max_age=2):raise ValueError('参与设备不在线')
    camera_selected = deepcopy(camera);camera_selected['parts'] = {p:camera['parts'][p] for p in camera_parts}
    if any(not d.get('accepted') for d in camera_selected['parts'].values()):raise ValueError('不能应用未通过的相机结果')

    def install_wearing(engine):
        if not wearing_parts:return
        if report['wearing_method'] == 'visual':
            apply_wearing_report(engine,wearing,epoch,wearing_parts,review_acknowledged=review_acknowledged)
        else:
            for part in wearing_parts:
                data = wearing['parts'][part];device = data['device_id']
                if not data.get('accepted') or engine.assignments[part] != device or engine.epochs.get(device) != data.get('reference_epoch'):
                    raise ValueError('穿戴结果未通过或设备/参考系已变化')
                estimator = engine._tilt_filters.get(device)
                if estimator is None or estimator.neutral_up is None or angle_deg(estimator.neutral_up,data['up_world']) > 5:raise ValueError('穿戴重力参考不一致')
                engine.apply_calibration(part,device,proper(data['basis_matrix']),proper(data['neutral_matrix']),data['neutral_matrices_by_mode'])
    from .live_rgbd_fusion import LiveRGBDFusion
    trial = deepcopy(owner.engine);install_wearing(trial)
    trial_fusion = LiveRGBDFusion(trial);trial_fusion.max_offset_deg = owner.live_fusion.max_offset_deg
    if camera_parts:trial_fusion.install(camera_selected)
    fields = ('continuous_basis','calibration_references','natural_hand_references','natural_hand_previews','mode_neutral','current','neutral','mode_current','fusion_corrections','fusion_ready')
    backup = {k:deepcopy(getattr(owner.engine,k)) for k in fields}
    fusion_backup = {k:deepcopy(getattr(owner.live_fusion,k)) for k in ('bindings','states','last_stamp','events')}
    camera_epoch = owner._live_camera_epoch
    try:
        install_wearing(owner.engine)
        for part in set(wearing_parts)-set(camera_parts):
            owner.engine.fusion_ready.discard(owner.engine.assignments[part])
            owner.live_fusion.states[part] = {'reason':'本轮仅应用穿戴映射，旧相机标定已停用','offset_deg':0.}
        if camera_parts:owner.apply_camera_report(camera_selected)
    except Exception:
        for k,v in backup.items():setattr(owner.engine,k,v)
        for k,v in fusion_backup.items():setattr(owner.live_fusion,k,v)
        owner._live_camera_epoch = camera_epoch
        raise
    owner.live_fusion.events.append({'type':'combined_calibration_application','time_ns':time.monotonic_ns(),
        'wearing_parts':list(wearing_parts),'camera_parts':list(camera_parts),'source_session':report['session']})
    owner._render_dirty = True
