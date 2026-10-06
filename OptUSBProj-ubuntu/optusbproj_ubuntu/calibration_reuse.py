"""Portable mounting geometry; new-session references must be measured, not relabelled."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .pose import GYRO_TILT, GYRO_RGBD
from .rgbd_fusion import ARM_SEGMENTS, angle_deg, gravity_alignment, unit, visual_direction
from .visual_wearing import body_frame

FORMAT = 'wt901-calibration-profile-v1'


def proper(value):
    r = np.asarray(value, float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('历史矩阵不是有效 SO(3) 旋转')
    return r


def camera_signature(calibration):
    if not calibration:raise ValueError('缺少相机硬件/内参信息')
    return hashlib.sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()


def profile_from_state(engine, fusion, calibration, vision):
    entries = {}
    for part, (data, _) in fusion.bindings.items():
        device = data['device_id']
        if (part not in engine.continuous_basis or device not in engine.fusion_ready or
                engine.assignments.get(part) != device or engine.epochs.get(device) != data.get('reference_epoch')):
            continue
        neutral = engine.neutral_for(device, GYRO_TILT)
        if neutral is None:continue
        basis = proper(engine.continuous_basis[part])
        q = proper(data['Q_camera_from_imu_world'])
        entries[part] = {'device_id': device, 'basis_matrix': basis.tolist(),
                         'bone_axis_sensor': unit(data['bone_axis_sensor']).tolist(),
                         'bone_length_m': float(data['bone_length_m']),
                         'camera_from_model_neutral': (q @ proper(neutral) @ basis.T).tolist(),
                         'camera_up': unit(q @ unit(data['up_world'])).tolist(),
                         'source_reference_epoch': data['reference_epoch']}
    profile = {'format': FORMAT, 'created_at': datetime.now().astimezone().isoformat(),
               'parts': entries, 'camera_calibration': deepcopy(calibration),
               'camera_signature': camera_signature(calibration), 'vision_config': deepcopy(vision),
               'note': '安装几何快照；不恢复旧积分器、零偏、中立位或累计纠偏角'}
    validate_profile(profile)
    return profile


def validate_profile(profile):
    if profile.get('format') != FORMAT or not profile.get('parts'):raise ValueError('没有可复用的已应用穿戴＋相机标定')
    if camera_signature(profile.get('camera_calibration')) != profile.get('camera_signature'):raise ValueError('历史相机信息校验失败')
    if not profile.get('vision_config', {}).get('signature'):raise ValueError('历史配置缺少视觉模型签名')
    ids = []
    for part, data in profile['parts'].items():
        if part not in ARM_SEGMENTS:raise ValueError('仅支持左右上臂/前臂')
        device = data.get('device_id')
        if not isinstance(device, str) or not device or device in ids:raise ValueError('历史设备分配无效或重复')
        ids.append(device)
        proper(data['basis_matrix']);proper(data['camera_from_model_neutral'])
        for name in ('bone_axis_sensor', 'camera_up'):
            v = np.asarray(data[name], float)
            if v.shape != (3,) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1, atol=1e-6):raise ValueError('历史方向向量无效')
        if not np.isfinite(data['bone_length_m']) or not .08 <= data['bone_length_m'] <= .8:raise ValueError('历史骨长无效')
    return profile


def write_json(path, value):
    """Atomic local write; callers use fresh names, never replace captures."""
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():raise ValueError('目标文件已存在，请另选文件名')
    tmp = path.with_suffix(path.suffix + '.tmp')
    created = False
    try:
        with tmp.open('x', encoding='utf-8') as stream:
            created = True
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        tmp.replace(path)
    finally:
        if created and tmp.exists():tmp.unlink()


def load_profile(path):
    with Path(path).open(encoding='utf-8') as stream:return validate_profile(json.load(stream))


def import_recording(path):
    """Import a complete applied snapshot, not arbitrary standalone old reports."""
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    bases = manifest.get('continuous_basis_at_start', {})
    bindings = manifest.get('fusion_at_start', {})
    neutrals = {}
    with (path/'runtime_pose.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line);device = row['device_id']
            if device not in neutrals and row.get('neutral_by_mode', {}).get(GYRO_TILT) is not None:
                neutrals[device] = row
            if all(v.get('calibration', {}).get('device_id') in neutrals for v in bindings.values()):break
    from .pose import PoseEngine
    from .live_rgbd_fusion import LiveRGBDFusion
    engine = PoseEngine(manifest.get('assignments_at_start', {}))
    fusion = LiveRGBDFusion(engine)
    for part, saved in bindings.items():
        data = saved['calibration'];device = data['device_id'];row = neutrals.get(device)
        if part not in bases or row is None or not row.get('fusion_ready') or row['epoch'] != data.get('reference_epoch'):continue
        engine.set_continuous_basis(part, bases[part]);engine.epochs[device] = row['epoch']
        engine.set_neutral_for_mode(device, GYRO_TILT, row['neutral_by_mode'][GYRO_TILT])
        engine.fusion_ready.add(device);fusion.bindings[part] = (data, None)
    # The stored SDK calibration may live in camera_calibration.json.
    calibration = manifest.get('camera_calibration')
    if not calibration:
        calibration = json.loads((path/'camera_calibration.json').read_text())
    profile = profile_from_state(engine, fusion, calibration, manifest.get('vision_config', {}))
    profile['source_recording'] = str(path.resolve())
    if manifest.get('started_wall_ns'):
        profile['source_started_at'] = datetime.fromtimestamp(manifest['started_wall_ns']/1e9).astimezone().isoformat()
    return profile


def import_camera_report(path):
    """Pair a historical camera report with its own capture's applied wearing snapshot."""
    path = Path(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('format') != 'rgbd-heading-fusion-experiment-v1' or report.get('orientation_source') != 'live_runtime_pose':
        raise ValueError('仅支持包含实时参考系的历史相机标定 report.json')
    source = Path(report['source_session'])
    manifest = json.loads((source/'manifest.json').read_text())
    if report.get('camera_epoch') != manifest.get('camera_epoch') or report.get('vision_epoch') != manifest.get('vision_epoch'):
        raise ValueError('相机报告与原录制会话不匹配')
    from .pose import PoseEngine
    from .live_rgbd_fusion import LiveRGBDFusion
    engine = PoseEngine(manifest.get('assignments_at_start', {}));fusion = LiveRGBDFusion(engine)
    snapshots = {}
    with (source/'runtime_pose.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line);device = row['device_id']
            if device not in snapshots and row.get('neutral_by_mode', {}).get(GYRO_TILT) is not None:snapshots[device] = row
            devices = {d.get('device_id') for d in report['parts'].values() if d.get('accepted')}
            if devices and devices.issubset(snapshots):break
    for part, data in report['parts'].items():
        if not data.get('accepted') or part not in manifest.get('continuous_basis_at_start', {}):continue
        device = data['device_id'];row = snapshots.get(device)
        if row is None or row['epoch'] != data.get('reference_epoch'):continue
        engine.set_continuous_basis(part, manifest['continuous_basis_at_start'][part])
        engine.epochs[device] = row['epoch'];engine.fusion_ready.add(device)
        engine.set_neutral_for_mode(device, GYRO_TILT, row['neutral_by_mode'][GYRO_TILT])
        fusion.bindings[part] = (data, None)
    calibration = json.loads((source/'camera_calibration.json').read_text())
    profile = profile_from_state(engine, fusion, calibration, report.get('vision_config', {}))
    profile['source_camera_report'] = str(path.resolve());profile['source_recording'] = str(source.resolve())
    if manifest.get('started_wall_ns'):
        profile['source_started_at'] = datetime.fromtimestamp(manifest['started_wall_ns']/1e9).astimezone().isoformat()
    return profile


def context_error(profile, owner, *, epochs=None, camera_epoch=None, vision_epoch=None):
    if owner.replay_mode or owner.camera_source != 'orbbec':return '仅支持实时 Gemini RGB-D 模式'
    if owner.session is not None or owner._record_start_deadline is not None:return '请先结束录制/预约'
    robot = owner.x1_preview_dialog
    if robot is not None and robot.sender is not None:return '请先停止 X1 随动'
    if not owner.rgbd_pose_enabled.isChecked():return '请先开启三维观测'
    if owner.vision_metadata.get('signature') != profile['vision_config']['signature']:return '视觉模型、权重或输入尺寸与历史配置不同，请切回历史设置或重标定'
    camera = owner.camera_worker
    if camera is None or getattr(camera, 'calibration', None) is None:return '相机尚未就绪'
    if camera_signature(camera.calibration) != profile['camera_signature']:return '相机硬件/内参配置与历史不一致'
    if camera_epoch is not None and camera.capture_epoch != camera_epoch:return '恢复期间相机采集会话已变化'
    if vision_epoch is not None and owner.vision_epoch != vision_epoch:return '恢复期间视觉会话已变化'
    for part, data in profile['parts'].items():
        device = data['device_id']
        if owner.engine.assignments.get(part) != device:return f'{part} 设备 ID 与历史配置不一致'
        if not owner.engine.is_live(device, max_age=2):return f'{device} 没有实时数据'
        if owner.engine.algorithm_modes[part] != GYRO_TILT:return f'{part} 请先选择陀螺＋加计（不使用旧视觉纠偏流）'
        if owner.engine.current_for(device, GYRO_TILT) is None or device not in owner.engine._tilt_filters:return f'{device} 缺少完整陀螺/加计姿态流'
        if epochs is not None and owner.engine.epochs.get(device) != epochs[device]:return f'{device} 积分参考系已变化'
    return None


def learn_still_bias(service, devices, start_ns, end_ns):
    """Explicit trusted still window, same strict gyro/acc thresholds as filter."""
    proposed = {}
    for device in devices:
        rows = [r for r in service.gravity_history.get(device, ()) if start_ns <= r['stamp'] <= end_ns and r['epoch'] == service.engine.epochs[device]]
        if len(rows) < 40 or (rows[-1]['stamp']-rows[0]['stamp'])/1e9 < .8:raise ValueError(f'{device} 静止采样不足')
        if max(np.diff([r['stamp'] for r in rows])) > 100_000_000:raise ValueError(f'{device} 静止采样断流超过100ms')
        gyro = np.array([r['gyro'] for r in rows]);acc = np.array([r['acc'] for r in rows])
        norms = np.linalg.norm(acc, axis=1)
        directions = acc/norms[:, None]
        if (not np.isfinite(gyro).all() or not np.isfinite(directions).all() or
                np.max(np.abs(norms-1)) > .08 or np.max(np.linalg.norm(gyro, axis=1)) > .8 or
                np.max(np.degrees(np.arccos(np.clip(np.sum(directions[1:]*directions[:-1], axis=1), -1, 1)))) > 1):
            raise ValueError(f'{device} 未达到静止零偏学习条件：支撑双臂、避免晃动；原零偏不覆盖')
        proposed[device] = gyro.mean(axis=0)
    for device, bias in proposed.items():
        estimator = service.engine._tilt_filters[device]
        estimator.gyro_bias_dps = bias.copy();estimator.bias_ready = True
        estimator.static_rate_sum[:] = 0;estimator.static_samples = 0;estimator.static_duration_s = 0
    service.events.append({'type': 'reuse_bias_learning', 'time_ns': int(end_ns),
                           'bias_dps': {d: b.tolist() for d, b in proposed.items()}})
    return {d: b.tolist() for d, b in proposed.items()}


def plan_bias_reuse(service, devices, *, force=False, skip=False):
    """Reuse only the LIVE estimator checked in this integration epoch."""
    if force and skip:raise ValueError('强制重估与跳过重估不能同时启用')
    pending = [];outcomes = {}
    for device in devices:
        estimator = service.engine._tilt_filters[device]
        bias = np.asarray(estimator.gyro_bias_dps, float)
        if bias.shape != (3,) or not np.isfinite(bias).all():raise ValueError(f'{device} 当前零偏状态无效，不能复用')
        ready = bool(estimator.bias_ready)
        result = {'updated': False, 'bias_ready': ready, 'bias_dps': bias.tolist(),
                  'previous_bias_dps': bias.tolist(), 'window_ns': None,
                  'reference_epoch': service.engine.epochs[device]}
        if ready and not force:
            result.update(status='reused_current', reused=True, skipped=True,
                          message='复用当前会话已学习零偏，不重复重估（不代表精度认证）')
        elif skip:
            result.update(status='unlearned', reused=False, skipped=True,
                          message='零偏尚未学习：用户跳过重估，后续验收不放宽')
        else:
            pending.append(device)
            result.update(status='pending', reused=False, skipped=False,
                          message='强制重新估计零偏' if force else '尚未学习，将尝试寻找安静片段')
        outcomes[device] = result
    service.events.append({'type': 'reuse_bias_plan', 'time_ns': time.monotonic_ns(),
                           'force': force, 'skip': skip, 'pending': pending, 'devices': outcomes})
    return pending, outcomes


def attempt_still_bias(service, devices, start_ns, end_ns, *, skip=False):
    """Optional, independent rest learning over the WHOLE preparation window.

    The strict learner remains unchanged. A missing quiet interval is a warning,
    never a fake success, and never a reason to reset the estimator or old bias.
    """
    outcomes = {}
    for device in devices:
        estimator = service.engine._tilt_filters[device]
        before = estimator.gyro_bias_dps.copy()
        result = {'updated': False, 'skipped': bool(skip), 'window_ns': None}
        if not skip:
            rows = [r for r in service.gravity_history.get(device, ()) if
                    start_ns <= r['stamp'] <= end_ns and r['epoch'] == service.engine.epochs[device]]
            runs = [];run = [];previous = None
            for row in rows:
                acc = np.asarray(row['acc'], float);gyro = np.asarray(row['gyro'], float)
                norm = np.linalg.norm(acc)
                good = np.isfinite(acc).all() and np.isfinite(gyro).all() and abs(norm-1) <= .08 and np.linalg.norm(gyro) <= .8
                if good and previous is not None:
                    good = (0 < row['stamp']-previous['stamp'] <= 100_000_000 and
                            angle_deg(acc, previous['acc']) <= 1)
                if not good:
                    if run:runs.append(run)
                    run = [];previous = None
                    # A gap or direction change starts a fresh interval; that
                    # boundary sample is deliberately not used for bias.
                    continue
                run.append(row);previous = row
            if run:runs.append(run)
            candidates = [r for r in runs if len(r) >= 40 and r[-1]['stamp']-r[0]['stamp'] >= 800_000_000]
            candidates.sort(key=lambda r: r[-1]['stamp']-r[0]['stamp'], reverse=True)
            for candidate in candidates:
                try:
                    learn_still_bias(service, [device], candidate[0]['stamp'], candidate[-1]['stamp'])
                    result.update(updated=True, window_ns=[candidate[0]['stamp'], candidate[-1]['stamp']], samples=len(candidate))
                    break
                except ValueError:
                    continue
        result.update(bias_dps=estimator.gyro_bias_dps.tolist(), bias_ready=bool(estimator.bias_ready),
                      previous_bias_dps=before.tolist())
        if result['updated']:
            result.update(status='relearned', message='找到可信安静片段，已重新学习零偏')
        elif estimator.bias_ready:
            result.update(status='preserved_ready', message='未显式重估：保留当前已学习零偏，继续恢复验证')
        else:
            result.update(status='unlearned', message='零偏尚未学习：沿用当前估计值，可能漂移更快；后续验收不放宽')
        outcomes[device] = result
    service.events.append({'type': 'reuse_bias_attempt', 'time_ns': int(end_ns), 'devices': outcomes})
    return outcomes


def fit_fixed_bone(rotations, directions, bone, up_world, up_camera):
    """One parameter only: yaw of Q. Do not refit the historical bone axis."""
    q0 = gravity_alignment(up_world, up_camera);up = unit(up_camera)
    p = np.einsum('ij,njk,k->ni', q0, np.asarray(rotations), unit(bone))
    d = np.array([unit(v) for v in directions])
    ph = p-(p@up)[:, None]*up;dh = d-(d@up)[:, None]*up
    keep = (np.linalg.norm(ph, axis=1) >= .35) & (np.linalg.norm(dh, axis=1) >= .35)
    if np.count_nonzero(keep) < 10:raise ValueError('非竖直有效观测不足，无法恢复航向')
    angles = np.arctan2(np.cross(ph[keep], dh[keep])@up, np.sum(ph[keep]*dh[keep], axis=1))
    mean = math.atan2(np.sin(angles).mean(), np.cos(angles).mean())
    # Circular robust centre; reject the >90 degree outliers, not average opposite phases.
    delta = np.arctan2(np.sin(angles-mean), np.cos(angles-mean))
    yaw = mean+float(np.median(delta))
    return Rotation.from_rotvec(up*yaw).as_matrix()@q0


def analyze_recovery(profile, samples, engine, gravity, *, camera_epoch, vision_epoch, vision_config):
    validate_profile(profile)
    neutral_rows = [s['vision'] for s in samples if s['phase'] == 'neutral']
    k, body_quality = body_frame(neutral_rows)
    up_camera = k.T @ np.array([0., 0., 1.])
    historical_up = unit(np.mean([d['camera_up'] for d in profile['parts'].values()], axis=0))
    if angle_deg(up_camera, historical_up) > 5:raise ValueError('相机/站姿竖直方向与历史相差超过5°，请站直或重新标定')
    for phase in ('forward', 'lateral', 'forward_check', 'return'):
        frame, _ = body_frame([s['vision'] for s in samples if s['phase'] == phase])
        if np.degrees(Rotation.from_matrix(frame@k.T).magnitude()) > 10:
            raise ValueError(f'{phase} 躯干/肩线相对中立位转动超过10°；保持身体朝向后重试')
    report = {'format': 'rgbd-heading-fusion-experiment-v1', 'orientation_source': 'live_runtime_pose',
              'camera_epoch': camera_epoch, 'vision_epoch': vision_epoch, 'vision_config': vision_config,
              'camera_up': up_camera.tolist(), 'parts': {}, 'workflow': 'historical_calibration_recovery',
              'profile_created_at': profile['created_at'], 'body_frame_quality': body_quality,
              'measurement_note': '短动作 RGB-D 一致性验收，不是真值精度；不观测长轴扭转'}
    for part, saved in profile['parts'].items():
        device = saved['device_id']
        try:
            bins = {}
            for phase in ('neutral', 'forward', 'lateral', 'forward_check', 'return'):
                rows = []
                for sample in samples:
                    if sample['phase'] != phase or device not in sample['imu']:continue
                    direction, _ = visual_direction(sample['vision'], part, saved['bone_length_m'])
                    if direction is not None:rows.append((sample, direction))
                if len(rows) < 8:raise ValueError(f'{phase} 同期可信观测不足8帧')
                bins[phase] = rows
            n = Rotation.from_matrix([r[0]['imu'][device] for r in bins['neutral']]).mean().as_matrix()
            basis = proper(saved['basis_matrix']);bone = unit(saved['bone_axis_sensor'])
            up = unit(gravity[device]['up_world'])
            train = bins['forward']+bins['lateral']
            q = fit_fixed_bone([r[0]['imu'][device] for r in train], [r[1] for r in train], bone, up, up_camera)
            errors = {}
            for phase, rows in bins.items():
                errors[phase] = float(np.percentile([angle_deg(q@np.asarray(s['imu'][device])@bone, d) for s, d in rows], 90))
            separation = angle_deg(np.mean([d for _, d in bins['forward']], axis=0), np.mean([d for _, d in bins['lateral']], axis=0))
            neutral_spread = max(float(np.percentile(Rotation.from_matrix([n.T@np.asarray(s['imu'][device]) for s, _ in rows]).magnitude()*180/np.pi, 90)) for phase, rows in bins.items() if phase == 'neutral')
            returned = Rotation.from_matrix([s['imu'][device] for s, _ in bins['return']]).mean().as_matrix()
            return_error = float(np.degrees(Rotation.from_matrix(n.T@returned).magnitude()))
            neutral_alignment = float(np.degrees(Rotation.from_matrix(proper(saved['camera_from_model_neutral']).T@q@n@basis.T).magnitude()))
            failures = []
            for phase, rows in bins.items():
                mean = Rotation.from_matrix([s['imu'][device] for s, _ in rows]).mean().as_matrix()
                spread = np.percentile(Rotation.from_matrix([mean.T@np.asarray(s['imu'][device]) for s, _ in rows]).magnitude()*180/np.pi, 90)
                if spread > 5:failures.append(f'{phase} 未充分静止（姿态波动P90>5°）')
            if max(errors.values()) > 15:failures.append('历史骨轴与短动作不一致（P90>15°）')
            if separation < 35:failures.append('前举/侧举方向分离不足35°')
            if neutral_spread > 3:failures.append('中立姿态波动超过3°')
            if return_error > 10:failures.append('返回中立位与首次相差超过10°')
            if neutral_alignment > 15:failures.append('中立位/轴向姿势或相机位置与历史不一致（>15°）；确认掌心、拇指方向')
            paired = {}
            modes = set.intersection(*(set(s.get('modes', {}).get(device, {})) for s, _ in bins['neutral']))
            for mode in modes:
                if mode != GYRO_RGBD:paired[mode] = Rotation.from_matrix([s['modes'][device][mode] for s, _ in bins['neutral']]).mean().as_matrix().tolist()
            paired[GYRO_TILT] = n.tolist()
            report['parts'][part] = {'accepted': not failures, 'device_id': device,
                'reference_epoch': engine.epochs[device], 'Q_camera_from_imu_world': q.tolist(),
                'up_world': up.tolist(), 'bone_axis_sensor': bone.tolist(), 'bone_length_m': saved['bone_length_m'],
                'basis_matrix': basis.tolist(), 'neutral_matrix': n.tolist(), 'neutral_matrices_by_mode': paired,
                'phase_p90_deg': errors, 'neutral_alignment_deg': neutral_alignment,
                'return_error_deg': return_error, 'warnings': failures}
        except (ValueError, KeyError) as exc:
            report['parts'][part] = {'accepted': False, 'device_id': device, 'warnings': [str(exc)]}
    return report


def apply_recovery(owner, report, parts):
    """Preflight on a detached engine; apply selected accepted parts as one transaction."""
    if not parts or len(parts) != len(set(parts)):raise ValueError('请选择至少一个合格部位')
    if owner.replay_mode or owner.session is not None or owner._record_start_deadline is not None:raise ValueError('录制、预约或回放中不能恢复标定')
    if owner.x1_preview_dialog is not None and owner.x1_preview_dialog.sender is not None:raise ValueError('请先停止 X1 随动')
    selected = deepcopy(report);selected['parts'] = {p: deepcopy(report['parts'][p]) for p in parts}
    for part, data in selected['parts'].items():
        if not data.get('accepted') or owner.engine.algorithm_modes[part] != GYRO_TILT:raise ValueError('只能应用合格且使用陀螺＋加计的部位')
    owner._check_vision_report(report)
    if owner.camera_worker.capture_epoch != report['camera_epoch']:raise ValueError('相机会话已变化')
    from .live_rgbd_fusion import LiveRGBDFusion
    trial = deepcopy(owner.engine);service = LiveRGBDFusion(trial)
    service.max_offset_deg = owner.live_fusion.max_offset_deg
    service.install(selected)  # Epoch/gravity/SO(3) checks before any real changes.
    for part, data in selected['parts'].items():
        trial.apply_calibration(part, data['device_id'], proper(data['basis_matrix']), proper(data['neutral_matrix']), data['neutral_matrices_by_mode'])
    fields = ('continuous_basis', 'calibration_references', 'natural_hand_references', 'natural_hand_previews', 'mode_neutral', 'current', 'neutral', 'mode_current', 'fusion_corrections', 'fusion_ready')
    backup = {name: deepcopy(getattr(owner.engine, name)) for name in fields}
    fusion_fields = ('bindings', 'states', 'last_stamp', 'events')
    fusion_backup = {name: deepcopy(getattr(owner.live_fusion, name)) for name in fusion_fields}
    old_camera_epoch = owner._live_camera_epoch
    try:
        for part, data in selected['parts'].items():
            owner.engine.apply_calibration(part, data['device_id'], data['basis_matrix'], data['neutral_matrix'], data['neutral_matrices_by_mode'])
        owner.apply_camera_report(selected)
    except Exception:
        for name, value in backup.items():setattr(owner.engine, name, value)
        for name, value in fusion_backup.items():setattr(owner.live_fusion, name, value)
        owner._live_camera_epoch = old_camera_epoch
        raise
    owner.live_fusion.events.append({'type': 'historical_recovery_applied', 'parts': list(parts),
                                    'time_ns': time.monotonic_ns(),
                                    'profile_created_at': report.get('profile_created_at')})
    owner._render_dirty = True
