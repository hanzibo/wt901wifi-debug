"""RGB-D assisted wearing calibration, independent of live heading feedback.

Fit only the first labelled holds. Repeats are held out. Model neutral is a
chosen convention, not a claim that the person's resting bones are vertical.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .axis_mapping import DEFAULT_AXIS_MAPPING, mapping_matrix
from .guided_calibration import fit_guided_arm, _hold, neutral_matrices_by_mode
from .pose import GYRO_TILT, ORIENTATION_MODES
from .rgbd_fusion import (SEGMENTS, angle_deg, fit_mounting, gravity_alignment,
                          load_live_series, unit, visual_direction)

FORMAT = "rgbd-assisted-wearing-v1"
TRAIN_STEPS = ("neutral", "forward", "lateral")
CHECK_STEPS = ("forward_repeat", "lateral_repeat", "neutral_return")
DOWN = np.array([0., 0., -1.])


def proper_rotation(value):
    r = np.asarray(value, float)
    if (r.shape != (3, 3) or not np.isfinite(r).all() or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(r), 1., atol=1e-6)):
        raise ValueError("矩阵不是有效 SO(3) 旋转")
    return r


def body_frame(neutral_rows):
    """Fixed camera->model axes: X front, Y anatomical left, Z upright torso.

    This is an anatomical/upright assumption, not measured camera gravity.
    Project the shoulder line perpendicular to up and enforce right handedness.
    """
    ups, lefts = [], []
    for row in neutral_rows:
        ids = (0, 1, 6, 7)
        if not all(row['valid'][i] and row['support'][i] >= .7 for i in ids):
            continue
        if min(row['confidence'][i] for i in (5, 6, 11, 12)) < .6:
            continue
        xyz = np.asarray(row['points_camera_m'], float)
        if not np.isfinite(xyz[list(ids)]).all():
            continue
        if np.linalg.norm(xyz[0] - xyz[1]) < .1:
            continue
        ups.append(unit((xyz[0] + xyz[1] - xyz[6] - xyz[7]) / 2))
        lefts.append(unit(xyz[0] - xyz[1]))
    if len(ups) < 5:
        raise ValueError("中立窗口至少需要5帧可信肩髋三维点")
    z = unit(np.mean(ups, axis=0))
    y_raw = np.mean(lefts, axis=0)
    y = unit(y_raw - z * (z @ y_raw))
    x = unit(np.cross(y, z))
    y = unit(np.cross(z, x))
    up_spread = float(np.percentile([angle_deg(v, z) for v in ups], 90))
    side_spread = float(np.percentile([angle_deg(v, y) for v in lefts], 90))
    if up_spread > 8 or side_spread > 12:
        raise ValueError("中立躯干/肩线不稳定，请站直并保持朝向")
    return proper_rotation(np.column_stack((x, y, z)).T), {
        'valid_frames': len(ups), 'up_spread_p90_deg': up_spread,
        'shoulder_spread_p90_deg': side_spread,
        'assumption': 'upright torso and anatomical shoulder line; not physical gravity GT'}


def normalized_basis(q, k, r0, bone):
    """B=L K Q R0, with the shortest L mapping actual neutral bone to -Z.

    Shortest-rotation L chooses the unresolved transverse neutral convention;
    it does not observe anatomical axial twist.
    """
    q, k, r0 = map(proper_rotation, (q, k, r0))
    n0 = unit(k @ q @ r0 @ unit(bone))
    normalization = gravity_alignment(n0, DOWN)
    return proper_rotation(normalization @ k @ q @ r0), normalization, n0


def hold_windows(guide, origin):
    if guide.get('status') != 'complete':
        raise ValueError("仅接受完整的引导录制，不使用取消/未完成记录")
    windows = {}
    for event in guide['events']:
        if event['phase'] != 'hold' or event.get('end_s') is None:
            continue
        name = event['step']
        if name in windows:
            raise ValueError("引导动作窗口重复")
        start, end = float(event['start_s']) + .5, float(event['end_s']) - .2
        if end <= start:
            raise ValueError("静止窗口过短")
        windows[name] = (origin + round(start * 1e9), origin + round(end * 1e9))
    if not set(TRAIN_STEPS + CHECK_STEPS).issubset(windows):
        raise ValueError("缺少首次动作或独立复测窗口，请重新引导录制")
    return windows


def in_window(stamp, window):
    return window[0] <= stamp <= window[1]


def direction_metrics(rotations, directions, q, bone):
    errors = [angle_deg(q @ r @ bone, d) for r, d in zip(rotations, directions)]
    if not errors:
        return None
    return {'frames': len(errors), 'median_deg': float(np.median(errors)),
            'p90_deg': float(np.percentile(errors, 90))}


def _read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze_wearing(session: Path):
    """Read a guided session. Never mutate originals, device state or maps."""
    session = session.resolve()
    manifest = json.loads((session / 'manifest.json').read_text())
    guide = json.loads((session / 'camera_guide.json').read_text())
    if manifest.get('error') or not manifest.get('camera_epoch'):
        raise ValueError("录制不完整或缺少相机会话标识")
    origin = int(manifest['started_mono_ns'])
    windows = hold_windows(guide, origin)
    cutoff = int(manifest.get('retained_until_mono_ns') or (2**63 - 1))
    if windows['neutral_return'][1] > cutoff:
        raise ValueError("有效录制范围未覆盖回中立复测")
    vision = [r for r in _read_jsonl(session / 'vision.jsonl')
              if r['captured_mono_ns'] <= cutoff]
    runtime = [r for r in _read_jsonl(session / 'runtime_pose.jsonl')
               if r['host_recv_mono_ns'] <= cutoff]
    with (session / 'imu.csv').open(newline='', encoding='utf-8') as stream:
        packets = {(int(r['host_recv_mono_ns']), r['device_id']): r
                   for r in csv.DictReader(stream)}
    neutral_vision = [r for r in vision if in_window(r['captured_mono_ns'], windows['neutral'])]
    k, body_quality = body_frame(neutral_vision)
    up_c = k.T @ np.array([0., 0., 1.])
    report = {'format': FORMAT, 'session': str(session),
              'camera_epoch': manifest['camera_epoch'], 'K_model_from_camera': k.tolist(),
              'body_frame_quality': body_quality, 'parts': {},
              'train_steps': list(TRAIN_STEPS), 'check_steps': list(CHECK_STEPS),
              'windows_ns': {name: list(win) for name, win in windows.items()},
              'measurement_note': 'RGB-D surface directions, not joint-centre GT; no axial twist validation',
              'neutral_policy': 'shortest per-segment neutral normalization to model -Z',
              'clock_note': 'host-time SLERP, max gap 100ms, no hardware synchronization'}
    for part in SEGMENTS:
        device = guide['devices'].get(part)
        try:
            if not device or manifest['assignments_at_start'].get(part) != device:
                raise ValueError("设备分配与引导录制不一致")
            if manifest.get('algorithm_modes_at_start', {}).get(part) != GYRO_TILT:
                raise ValueError("首版请使用陀螺＋加计录制，避免在线纠偏参与安装拟合")
            series, up_w, _, bootstrap = load_live_series(session, device, windows['neutral'])
            device_rows = [r for r in runtime if r['device_id'] == device]
            epoch = device_rows[0]['epoch']
            holds, sample_rows = {}, {}
            alignment = gravity_alignment(up_w, [0, 0, 1])
            for step in TRAIN_STEPS + CHECK_STEPS:
                rows = []
                for row in device_rows:
                    stamp = row['host_recv_mono_ns']
                    if not in_window(stamp, windows[step]):
                        continue
                    packet = packets.get((stamp, device))
                    if packet is None:
                        raise ValueError("实时姿态与原始IMU包缺少对应关系")
                    values = {name: float(packet[name]) for name in
                              ('gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z')}
                    values.update(received_ns=stamp, orientation_mode=GYRO_TILT,
                                  orientation_matrix=row['orientation_by_mode'][GYRO_TILT],
                                  orientation_by_mode=row['orientation_by_mode'])
                    rows.append(values)
                mean, spread, gyro = _hold(rows)
                if spread > 5 or gyro > 20:
                    raise ValueError(f"{step} 未充分静止：波动{spread:.1f}° / 陀螺P90 {gyro:.1f}°/s")
                acc_norm = np.linalg.norm([[r['acc_x'], r['acc_y'], r['acc_z']] for r in rows], axis=1)
                if abs(float(np.median(acc_norm)) - 1) > .2:
                    raise ValueError(f"{step} 加速度模长偏离1g")
                holds[step] = mean
                sample_rows[step] = rows
            lengths = [r['bone_lengths_m'][part] for r in vision
                       if any(in_window(r['captured_mono_ns'], windows[s]) for s in TRAIN_STEPS)
                       and r['bone_lengths_m'].get(part) is not None]
            if not lengths:
                raise ValueError("首次动作缺少有效骨长")
            bone_length = float(np.median(lengths))
            paired = {}
            for step in TRAIN_STEPS + CHECK_STEPS + ('validation',):
                matrices, directions, frames = [], [], []
                if step in windows:
                    for row in vision:
                        stamp = row['captured_mono_ns']
                        if not in_window(stamp, windows[step]):
                            continue
                        r = series.at(stamp)
                        d, _ = visual_direction(row, part, bone_length)
                        if r is not None and d is not None:
                            matrices.append(r); directions.append(d); frames.append(row)
                paired[step] = (matrices, directions, frames)
                if step != 'validation' and len(matrices) < 5:
                    raise ValueError(f"{step} 少于5帧可信视觉/IMU对应观测")
            train_r = sum((paired[s][0] for s in TRAIN_STEPS), [])
            train_d = sum((paired[s][1] for s in TRAIN_STEPS), [])
            fit = fit_mounting(train_r, train_d, up_w, up_c)
            q, bone = fit.camera_from_imu_world, fit.bone_axis_sensor
            r0 = holds['neutral']
            basis, normalization, n0 = normalized_basis(q, k, r0, bone)
            checks = {s: direction_metrics(paired[s][0], paired[s][1], q, bone)
                      for s in CHECK_STEPS + ('validation',)}
            warnings, notes = [], []
            if not fit.accepted:
                warnings.append("首次动作骨轴/航向拟合或可观测性未通过")
            for step in CHECK_STEPS:
                metrics = checks[step]
                if metrics['median_deg'] >= 12 or metrics['p90_deg'] >= 25:
                    warnings.append(f"{step} 独立方向复测未通过")
            if checks['validation'] is None or checks['validation']['frames'] < 10:
                warnings.append("独立自由动作至少需要10帧可信观测")
            elif checks['validation']['median_deg'] >= 15 or checks['validation']['p90_deg'] >= 30:
                warnings.append("独立自由动作方向误差过大")
            return_angle = float(np.degrees(Rotation.from_matrix(r0.T @ holds['neutral_return']).magnitude()))
            if return_angle > 15:
                warnings.append(f"设备回中立整体旋转偏差{return_angle:.1f}°，检查漂移/绑带/姿势")
            # Check body motion during static raises separately from segment fit.
            torso_angles, elbow_angles = [], []
            for step in ('forward', 'lateral', 'forward_repeat', 'lateral_repeat'):
                body_count = 0
                for row in paired[step][2]:
                    try:
                        instantaneous, _ = body_frame([row] * 5)
                        torso_angles.append(float(np.degrees(
                            Rotation.from_matrix(instantaneous @ k.T).magnitude())))
                        body_count += 1
                    except ValueError:
                        continue
                    upper, _ = visual_direction(row, 'right_upper_arm')
                    fore, _ = visual_direction(row, 'right_forearm')
                    if upper is not None and fore is not None:
                        elbow_angles.append(angle_deg(upper, fore))
                if body_count < 5:
                    warnings.append(f"{step} 缺少可信肩髋观测，无法验收身体朝向")
            torso_motion = float(np.percentile(torso_angles, 90)) if torso_angles else 0.
            elbow_bend = float(np.percentile(elbow_angles, 90)) if elbow_angles else 0.
            if torso_motion > 12:
                warnings.append(f"静止平举时躯干/肩线变化P90 {torso_motion:.1f}°，请保持身体朝向")
            elif torso_motion > 5:
                notes.append(f"躯干/肩线变化{torso_motion:.1f}°，建议复核")
            if elbow_bend > 25:
                warnings.append(f"平举时视觉肘弯曲P90 {elbow_bend:.1f}°，请伸直肘部")
            if not elbow_angles:
                warnings.append("缺少可信肩肘腕同时观测，无法检查肘伸直")
            mapping = tuple(manifest.get('axis_mappings_at_start', {}).get(part, DEFAULT_AXIS_MAPPING))
            prior_basis = np.asarray(manifest.get('continuous_basis_at_start', {}).get(part, mapping_matrix(mapping)), float)
            legacy_samples = {}
            for new, old in (('neutral', 'neutral'), ('forward', 'forward'), ('lateral', 'lateral'),
                             ('forward_repeat', 'check_forward'), ('neutral_return', 'check_neutral')):
                legacy_samples[old] = [dict(row, orientation_matrix=(alignment @ np.array(row['orientation_matrix'])).tolist())
                                       for row in sample_rows[new]]
            try:
                legacy = asdict(fit_guided_arm(part, device, legacy_samples, mapping))
            except ValueError as exc:
                legacy = {'status': '无法拟合', 'warnings': [str(exc)]}
            assumed_bone = (alignment @ r0).T @ DOWN
            visual_n0 = unit(k @ unit(np.mean(paired['neutral'][1], axis=0)))
            offset = float(np.degrees(Rotation.from_matrix(basis @ prior_basis.T).magnitude()))
            if offset > 45:
                notes.append(f"候选映射与现有映射相差{offset:.1f}°，须人工检查轴向转动")
            candidate = {'format': FORMAT, 'method': 'visual_bone_v1', 'session': str(session),
                         'part': part, 'device_id': device, 'reference_epoch': epoch,
                         'attitude_mode': GYRO_TILT, 'basis_matrix': basis.tolist(),
                         'neutral_matrix': r0.tolist(),
                         'neutral_matrices_by_mode': {mode: r.tolist() for mode, r in
                                                     neutral_matrices_by_mode(sample_rows['neutral']).items()},
                         'status': '通过' if not warnings else '需重录', 'warnings': warnings,
                         'notes': notes, 'Q_camera_from_imu_world': q.tolist(),
                         'up_world': up_w.tolist(), 'bone_axis_sensor': bone.tolist(),
                         'L_neutral_normalization': normalization.tolist(),
                         'K_model_from_camera': k.tolist(), 'bone_length_m': bone_length,
                         'neutral_actual_model_direction': visual_n0.tolist(),
                         'neutral_tilt_deg': angle_deg(visual_n0, DOWN),
                         'bone_axis_difference_deg': angle_deg(assumed_bone, bone),
                         'mapping_offset_deg': offset, 'return_rotation_deg': return_angle,
                         'torso_motion_p90_deg': torso_motion,
                         'elbow_bend_p90_deg': elbow_bend,
                         'fit_median_deg': fit.median_error_deg, 'fit_p90_deg': fit.p90_error_deg,
                         'jacobian_singular_values': fit.jacobian_singular_values.tolist(),
                         'checks': checks, 'gravity_initialization': bootstrap,
                         'existing_basis_matrix': prior_basis.tolist(), 'legacy_fit': legacy,
                         'twist_policy': 'shortest neutral normalization; anatomical twist not observed'}
            report['parts'][part] = candidate
        except (ValueError, KeyError, TypeError) as exc:
            report['parts'][part] = {'part': part, 'device_id': device, 'status': '需重录',
                                     'warnings': [str(exc)]}
    return report


def apply_wearing_report(engine, report, camera_epoch, parts):
    """Validate everything before writing. Does not enable visual feedback."""
    if report.get('format') != FORMAT or camera_epoch is None or report.get('camera_epoch') != camera_epoch:
        raise ValueError("相机采集会话已变化或标定格式不匹配")
    selected = []
    for part in parts:
        result = report['parts'].get(part, {})
        device = result.get('device_id')
        if part not in SEGMENTS or result.get('status') != '通过':
            raise ValueError("只能应用合格的右臂部位结果")
        if not result.get('reference_epoch') or engine.assignments.get(part) != device or engine.epochs.get(device) != result.get('reference_epoch'):
            raise ValueError("设备绑定或积分参考系已变化，请重录")
        if engine.algorithm_modes[part] != GYRO_TILT or not engine.is_live(device, max_age=2):
            raise ValueError("请先选择陀螺＋加计并确认设备在线")
        estimator = engine._tilt_filters.get(device)
        if estimator is None or estimator.neutral_up is None or angle_deg(estimator.neutral_up, result['up_world']) > 5:
            raise ValueError("实时重力参考与标定不一致")
        basis = proper_rotation(result['basis_matrix'])
        neutral = proper_rotation(result['neutral_matrix'])
        paired = {mode: proper_rotation(value) for mode, value in result['neutral_matrices_by_mode'].items()}
        if not set(paired).issubset(ORIENTATION_MODES):
            raise ValueError("同步中立位含有未知姿态模式")
        if GYRO_TILT not in paired or not np.allclose(paired[GYRO_TILT], neutral, atol=1e-6):
            raise ValueError("中立姿态与原始积分流不一致")
        selected.append((part, device, basis, neutral, paired))
    if not selected or len(set(parts)) != len(parts):
        raise ValueError("请至少选择一个合格部位，不可重复")
    for part, device, basis, neutral, paired in selected:
        engine.apply_calibration(part, device, basis, neutral, paired)
