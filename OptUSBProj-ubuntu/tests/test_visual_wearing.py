import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT, GYRO_RGBD
from optusbproj_ubuntu.visual_wearing import (FORMAT, DOWN, analyze_wearing, body_frame,
                                              normalized_basis, apply_wearing_report,
                                              assess_body_motion, PASS, REVIEW)


def fixture(path, *, corrupt_repeat=False, bilateral=False, corrupt_left_repeat=False):
    """Nonvertical rest bones and arbitrarily tilted camera; no RPY fitting."""
    origin = 10_000_000_000
    c_from_model = Rotation.from_euler('xyz', [77, 12, -16], degrees=True).as_matrix()
    q = c_from_model @ Rotation.from_euler('z', 25, degrees=True).as_matrix()
    r0 = Rotation.from_euler('xyz', [20, -15, 50], degrees=True).as_matrix()
    n0 = Rotation.from_euler('x', 12, degrees=True).apply(DOWN)
    bone = r0.T @ q.T @ c_from_model @ n0
    steps = ['neutral', 'forward', 'down', 'lateral', 'forward_repeat',
             'lateral_repeat', 'neutral_return', 'validation']
    events = [{'step': s, 'phase': 'hold', 'start_s': i*3, 'end_s': (i+1)*3}
              for i, s in enumerate(steps)]
    ids = {'right_upper_arm': 'A', 'right_forearm': 'B'}
    if bilateral:
        ids.update(left_upper_arm='C', left_forearm='D')
    sensor_configs = {device: (q, r0, bone) for device in ('A', 'B')}
    left_n0 = Rotation.from_euler('x', -9, degrees=True).apply(DOWN)
    for device, yaw in (('C', -35), ('D', 60)):
        lq = c_from_model @ Rotation.from_euler('z', yaw, degrees=True).as_matrix()
        lr0 = Rotation.from_euler('xyz', [-30, 20, yaw], degrees=True).as_matrix()
        lb = lr0.T @ lq.T @ c_from_model @ left_n0
        sensor_configs[device] = (lq, lr0, lb)
    manifest = {'started_mono_ns': origin, 'camera_epoch': 'camera', 'error': None,
                'retained_until_mono_ns': origin + 24_000_000_000,
                'assignments_at_start': ids,
                'algorithm_modes_at_start': {p: GYRO_TILT for p in ids},
                'axis_mappings_at_start': {p: ['+X', '+Y', '+Z'] for p in ids}}
    initial_basis, _, _ = normalized_basis(q, c_from_model.T, r0, bone)
    manifest['continuous_basis_at_start'] = {
        p: normalized_basis(sensor_configs[d][0], c_from_model.T, sensor_configs[d][1], sensor_configs[d][2])[0].tolist()
        for p, d in ids.items()}
    guide = {'status': 'complete', 'devices': ids, 'events': events,
             'gravity_reference': {d: {'epoch': 'epoch'+d, 'up_world': [0, 0, 1],
                                       'locked_at_ns': origin} for d in ids.values()}}
    if bilateral:
        guide.update(participants=list(ids), capture_scope='bilateral')
    (path/'manifest.json').write_text(json.dumps(manifest))
    (path/'camera_guide.json').write_text(json.dumps(guide))
    runtime, vision, imu = [], [], []
    for i in range(1200):
        t = i * .02
        step = steps[min(int(t // 3), 7)]
        if step in ('forward', 'forward_repeat'):
            movement = Rotation.from_euler('y', -90, degrees=True).as_matrix()
        elif step in ('lateral', 'lateral_repeat'):
            movement = Rotation.from_euler('x', -90, degrees=True).as_matrix()
        elif step == 'validation':
            movement = Rotation.from_euler('xy', [-35, -45], degrees=True).as_matrix()
        else:
            movement = np.eye(3)
        left_movement = (Rotation.from_euler('x', 90, degrees=True).as_matrix()
                         if step in ('lateral', 'lateral_repeat') else movement)
        stamp = origin + round(t*1e9)
        for device in ids.values():
            dq, dr0, _ = sensor_configs[device]
            dm = left_movement if device in ('C','D') else movement
            r = dq.T @ c_from_model @ dm @ c_from_model.T @ dq @ dr0
            modes = {GYRO_TILT: r.tolist(), GYRO_RGBD: r.tolist(), 'vendor_rpy': r.tolist()}
            runtime.append({'device_id': device, 'epoch': 'epoch'+device,
                            'host_recv_mono_ns': stamp, 'orientation_by_mode': modes,
                            'up_world': [0, 0, 1]})
            acc = r.T @ np.array([0, 0, 1.])
            imu.append(dict(host_recv_mono_ns=stamp, device_id=device,
                            gyro_x=0, gyro_y=0, gyro_z=0,
                            acc_x=acc[0], acc_y=acc[1], acc_z=acc[2]))
        if i % 5 == 0:
            direction = movement @ n0
            left_direction = left_movement @ left_n0
            if corrupt_repeat and step == 'forward_repeat':
                direction = Rotation.from_euler('z', 40, degrees=True).apply(direction)
            if corrupt_left_repeat and step == 'forward_repeat':
                left_direction = Rotation.from_euler('z', -40, degrees=True).apply(left_direction)
            xyz = np.array([[0, .18, 1.], [0, -.18, 1.], [0, .18, .75],
                            np.array([0, -.18, 1.]) + .26*direction,
                            [0, .18, .5], np.array([0, -.18, 1.]) + .53*direction,
                            [0, .12, .4], [0, -.12, .4]])
            if bilateral:
                xyz[2] = xyz[0] + .26*left_direction
                xyz[4] = xyz[0] + .53*left_direction
            xyz = (c_from_model @ xyz.T).T + [0, 0, 2]
            vision.append({'captured_mono_ns': stamp, 'valid': [True]*8,
                           'support': [1.]*8, 'confidence': [1.]*133,
                           'points_camera_m': xyz.tolist(),
                           'bone_lengths_m': {'right_upper_arm': .26, 'right_forearm': .27,
                                              'left_upper_arm': .26, 'left_forearm': .27}})
    for name, rows in (('runtime_pose.jsonl', runtime), ('vision.jsonl', vision)):
        (path/name).write_text('\n'.join(json.dumps(r) for r in rows))
    with (path/'imu.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(imu[0]))
        writer.writeheader(); writer.writerows(imu)
    return q, bone, r0, c_from_model.T


class VisualWearingTests(unittest.TestCase):
    def test_bilateral_independent_bases_and_left_side_sign(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path, bilateral=True)
            report = analyze_wearing(path)
            self.assertEqual(set(report['parts']), {'left_upper_arm','left_forearm','right_upper_arm','right_forearm'})
            manifest = json.loads((path/'manifest.json').read_text())
            for part, result in report['parts'].items():
                self.assertEqual(result['status'], PASS, result['warnings'] + result['review_reasons'])
                np.testing.assert_allclose(result['basis_matrix'], manifest['continuous_basis_at_start'][part], atol=1e-5)
                self.assertLess(result['fit_median_deg'], .001)
            self.assertFalse(np.allclose(report['parts']['left_upper_arm']['basis_matrix'],
                                         report['parts']['left_forearm']['basis_matrix']))

    def test_left_failure_does_not_contaminate_right_results(self):
        with TemporaryDirectory() as normal_dir, TemporaryDirectory() as bad_dir:
            fixture(Path(normal_dir), bilateral=True)
            fixture(Path(bad_dir), bilateral=True, corrupt_left_repeat=True)
            normal = analyze_wearing(Path(normal_dir)); bad = analyze_wearing(Path(bad_dir))
            for part in ('right_upper_arm','right_forearm'):
                self.assertEqual(bad['parts'][part]['status'], PASS)
                np.testing.assert_allclose(normal['parts'][part]['basis_matrix'], bad['parts'][part]['basis_matrix'], atol=1e-12)
            for part in ('left_upper_arm','left_forearm'):
                self.assertEqual(bad['parts'][part]['status'], '需重录')

    def test_four_part_apply_is_atomic_and_partial_apply_keeps_left_unchanged(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path, bilateral=True)
            report = analyze_wearing(path)
            assignments = {p:r['device_id'] for p,r in report['parts'].items()}
            engine = PoseEngine(assignments, algorithm_modes={p:GYRO_TILT for p in assignments})
            engine.epochs.update({d:'epoch'+d for d in assignments.values()})
            engine.is_live = lambda *a, **k: True
            engine._tilt_filters = {d:SimpleNamespace(neutral_up=np.array([0.,0.,1.])) for d in assignments.values()}
            parts = list(report['parts'])
            last = parts[-1]; old_epoch = report['parts'][last]['reference_epoch']
            report['parts'][last]['reference_epoch'] = 'stale'
            with self.assertRaisesRegex(ValueError, '参考系'):
                apply_wearing_report(engine, report, 'camera', parts)
            self.assertFalse(engine.continuous_basis)
            report['parts'][last]['reference_epoch'] = old_epoch
            engine.set_continuous_basis('left_upper_arm', np.eye(3))
            apply_wearing_report(engine, report, 'camera', ['right_upper_arm','right_forearm'])
            np.testing.assert_allclose(engine.continuous_basis['left_upper_arm'], np.eye(3))
            self.assertNotIn('left_forearm', engine.continuous_basis)
            apply_wearing_report(engine, report, 'camera', parts)
            for part in parts:
                np.testing.assert_allclose(engine.continuous_basis[part], report['parts'][part]['basis_matrix'])
            self.assertFalse(engine.fusion_ready)

    def test_old_right_only_guides_ignore_uninvolved_left_assignments(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path)
            manifest = json.loads((path/'manifest.json').read_text())
            manifest['assignments_at_start']['left_upper_arm'] = 'UNINVOLVED'
            (path/'manifest.json').write_text(json.dumps(manifest))
            report = analyze_wearing(path)
            self.assertEqual(set(report['parts']), {'right_upper_arm','right_forearm'})

    def test_shoulder_only_turn_requires_review_not_failure(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path)
            rows = [json.loads(line) for line in (path/'vision.jsonl').read_text().splitlines()]
            guide = json.loads((path/'camera_guide.json').read_text())
            from optusbproj_ubuntu.visual_wearing import hold_windows
            windows = hold_windows(guide, 10_000_000_000)
            neutral = [r for r in rows if windows['neutral'][0] <= r['captured_mono_ns'] <= windows['neutral'][1]]
            k, _ = body_frame(neutral)
            turn = Rotation.from_rotvec(k.T[:,2]*np.radians(20)).as_matrix()
            for row in rows:
                if any(windows[s][0] <= row['captured_mono_ns'] <= windows[s][1] for s in ('forward', 'forward_repeat')):
                    xyz = np.array(row['points_camera_m']); centre = (xyz[0]+xyz[1])/2
                    for i in (0,1): xyz[i] = centre + turn @ (xyz[i]-centre)
                    row['points_camera_m'] = xyz.tolist()
            quality = assess_body_motion(rows, windows, k)
            self.assertEqual(quality['hard_failures'], [])
            self.assertIn('不能据此判定', ' '.join(quality['review_reasons']))
            self.assertLess(quality['windows']['forward']['tilt']['p90_abs_deg'], .001)

    def test_body_turn_missing_hips_and_sustained_tilt_are_distinguished(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path)
            original = [json.loads(line) for line in (path/'vision.jsonl').read_text().splitlines()]
            from copy import deepcopy
            from optusbproj_ubuntu.visual_wearing import hold_windows
            windows = hold_windows(json.loads((path/'camera_guide.json').read_text()), 10_000_000_000)
            neutral = [r for r in original if windows['neutral'][0] <= r['captured_mono_ns'] <= windows['neutral'][1]]
            k, _ = body_frame(neutral)
            for kind in ('turn', 'missing', 'tilt'):
                rows = deepcopy(original)
                axis = k.T[:,2] if kind == 'turn' else k.T[:,0]
                motion = Rotation.from_rotvec(axis*np.radians(20)).as_matrix()
                for row in rows:
                    if windows['forward'][0] <= row['captured_mono_ns'] <= windows['forward'][1]:
                        if kind == 'missing': row['valid'][6] = row['valid'][7] = False
                        else:
                            xyz = np.array(row['points_camera_m']); centre = (xyz[6]+xyz[7])/2
                            xyz = (motion @ (xyz-centre).T).T + centre
                            row['points_camera_m'] = xyz.tolist()
                quality = assess_body_motion(rows, windows, k)
                if kind == 'tilt': self.assertIn('持续倾斜', ' '.join(quality['hard_failures']))
                else:
                    self.assertEqual(quality['hard_failures'], [])
                    word = '共同转向' if kind == 'turn' else '无法可靠核验'
                    self.assertIn(word, ' '.join(quality['review_reasons']))

    def test_review_acknowledgement_is_enforced_before_any_mutation(self):
        with TemporaryDirectory() as folder:
            fixture(Path(folder)); report = analyze_wearing(Path(folder))
            engine = PoseEngine({'right_upper_arm': 'A', 'right_forearm': 'B'},
                algorithm_modes={'right_upper_arm': GYRO_TILT, 'right_forearm': GYRO_TILT})
            engine.epochs.update(A='epochA', B='epochB')
            engine.is_live = lambda *a, **k: True
            engine._tilt_filters = {d: SimpleNamespace(neutral_up=np.array([0.,0.,1.])) for d in ('A','B')}
            report['parts']['right_forearm'].update(status=REVIEW, review_reasons=['肩线变化'])
            with self.assertRaisesRegex(ValueError, '明确确认'):
                apply_wearing_report(engine, report, 'camera', ['right_upper_arm','right_forearm'])
            self.assertFalse(engine.continuous_basis)
            apply_wearing_report(engine, report, 'camera', ['right_upper_arm','right_forearm'], review_acknowledged=True)
            self.assertIn('right_forearm', engine.continuous_basis)

    def test_nonvertical_neutral_is_recovered_and_normalized(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); q, bone, r0, k = fixture(path)
            before = (path/'runtime_pose.jsonl').read_bytes()
            report = analyze_wearing(path)
            self.assertEqual(before, (path/'runtime_pose.jsonl').read_bytes())
            np.testing.assert_allclose(report['K_model_from_camera'], k, atol=1e-8)
            for result in report['parts'].values():
                self.assertEqual(result['status'], '通过', result['warnings'])
                np.testing.assert_allclose(result['bone_axis_sensor'], bone, atol=1e-5)
                self.assertAlmostEqual(result['neutral_tilt_deg'], 12, places=4)
                self.assertAlmostEqual(result['bone_axis_difference_deg'], 12, places=4)
                self.assertLess(result['fit_median_deg'], .001)
                b = np.array(result['basis_matrix'])
                np.testing.assert_allclose(b @ bone, DOWN, atol=1e-5)

    def test_repeats_do_not_train_and_can_reject_a_perfect_training_fit(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            fixture(Path(first)); fixture(Path(second), corrupt_repeat=True)
            normal = analyze_wearing(Path(first)); bad = analyze_wearing(Path(second))
            for part in normal['parts']:
                a, b = normal['parts'][part], bad['parts'][part]
                np.testing.assert_allclose(a['bone_axis_sensor'], b['bone_axis_sensor'], atol=1e-10)
                np.testing.assert_allclose(a['Q_camera_from_imu_world'], b['Q_camera_from_imu_world'], atol=1e-10)
                self.assertEqual(b['status'], '需重录')
                self.assertTrue(any('独立方向' in w for w in b['warnings']))

    def test_basis_identity_for_arbitrary_motion_and_reference_gauge(self):
        q = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_matrix()
        k = Rotation.from_euler('z', 15, degrees=True).as_matrix()
        r0 = Rotation.from_euler('y', 50, degrees=True).as_matrix()
        bone = np.array([.2, .3, -.9]); bone /= np.linalg.norm(bone)
        b, l, _ = normalized_basis(q, k, r0, bone)
        r1 = r0 @ Rotation.from_euler('xyz', [65, -72, 33], degrees=True).as_matrix()
        mapped = b @ (r0.T @ r1) @ b.T
        np.testing.assert_allclose(mapped @ DOWN, l @ k @ q @ r1 @ bone, atol=1e-12)
        gauge = Rotation.from_euler('x', 70, degrees=True).as_matrix()
        bb, _, _ = normalized_basis(q @ gauge.T, k, gauge @ r0, bone)
        np.testing.assert_allclose(bb, b, atol=1e-12)

    def test_partial_and_atomic_application_preserves_other_parts_and_filters(self):
        with TemporaryDirectory() as folder:
            fixture(Path(folder)); report = analyze_wearing(Path(folder))
            engine = PoseEngine({'right_upper_arm': 'A', 'right_forearm': 'B', 'left_hand': 'C'},
                                algorithm_modes={'right_upper_arm': GYRO_TILT, 'right_forearm': GYRO_TILT})
            engine.epochs.update(A='epochA', B='epochB')
            engine.is_live = lambda *a, **k: True
            filters = {d: SimpleNamespace(neutral_up=np.array([0., 0., 1.])) for d in ('A', 'B')}
            engine._tilt_filters = filters.copy()
            engine.set_continuous_basis('left_hand', np.eye(3))
            report['parts']['right_forearm']['reference_epoch'] = 'wrong'
            with self.assertRaisesRegex(ValueError, '参考系'):
                apply_wearing_report(engine, report, 'camera', ['right_upper_arm', 'right_forearm'])
            self.assertNotIn('right_upper_arm', engine.continuous_basis)
            apply_wearing_report(engine, report, 'camera', ['right_upper_arm'])
            self.assertNotIn('right_forearm', engine.continuous_basis)
            np.testing.assert_allclose(engine.continuous_basis['left_hand'], np.eye(3))
            self.assertIs(engine._tilt_filters['A'], filters['A'])
            self.assertEqual(engine.fusion_ready, set())
            with self.assertRaisesRegex(ValueError, '相机'):
                apply_wearing_report(engine, report, 'new-camera', ['right_upper_arm'])

    def test_degenerate_body_and_incomplete_capture_rejected(self):
        with self.assertRaises(ValueError):
            body_frame([])
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path)
            guide = json.loads((path/'camera_guide.json').read_text()); guide['status'] = 'cancelled'
            (path/'camera_guide.json').write_text(json.dumps(guide))
            with self.assertRaisesRegex(ValueError, '完整'):
                analyze_wearing(path)


if __name__ == '__main__':
    unittest.main()
