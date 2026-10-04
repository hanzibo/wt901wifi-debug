"""Cross-session geometry reuse never copies epoch, bias or old correction."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.calibration_reuse import (analyze_recovery, apply_recovery,
    attempt_still_bias, camera_signature, context_error, fit_fixed_bone, import_camera_report, import_recording,
    learn_still_bias, load_profile, plan_bias_reuse, profile_from_state, validate_profile, write_json)
from optusbproj_ubuntu.calibration_reuse_gui import CalibrationReuseDialog
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.live_rgbd_fusion import LiveRGBDFusion
from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT, GYRO_RGBD
from optusbproj_ubuntu.rgbd_fusion import ARM_SEGMENTS
from test_fusion_capture import packet
from test_pose_backend_gui import Source


CAMERA = {'serial': 'test', 'color_intrinsic': {'fx': 600}}
VISION = {'signature': 'test-model'}


def system():
    assignments = {p: str(i) for i, p in enumerate(ARM_SEGMENTS)}
    engine = PoseEngine(assignments, algorithm_modes={p: GYRO_TILT for p in assignments})
    fusion = LiveRGBDFusion(engine)
    for device in assignments.values():
        engine.feed(replace(packet(), device_id=device), received_at=time.monotonic())
        engine._tilt_filters[device].neutral_up = np.array([0., 0., 1.])
        engine.set_continuous_basis(next(p for p, d in assignments.items() if d == device), np.eye(3))
        engine.set_neutral_for_mode(device, GYRO_TILT, np.eye(3))
    report = {'orientation_source': 'live_runtime_pose', 'camera_up': [0, 0, 1], 'parts': {
        p: {'device_id': d, 'reference_epoch': engine.epochs[d], 'accepted': True,
            'up_world': [0, 0, 1], 'Q_camera_from_imu_world': np.eye(3).tolist(),
            'bone_axis_sensor': [0, 0, -1], 'bone_length_m': .25}
        for p, d in assignments.items()}}
    fusion.install(report)
    return engine, fusion, profile_from_state(engine, fusion, CAMERA, VISION)


def recovery_samples(profile, gauge=0):
    samples = [];change = Rotation.from_euler('z', gauge, degrees=True).as_matrix()
    for phase in ('neutral', 'forward', 'lateral', 'forward_check', 'return'):
        for i in range(20):
            xyz = np.zeros((8, 3));xyz[0] = [0, .2, 2];xyz[1] = [0, -.2, 2]
            xyz[6] = [0, .15, 1.5];xyz[7] = [0, -.15, 1.5]
            matrices = {}
            for part, (a, b) in ARM_SEGMENTS.items():
                direction = np.array([0., 0., -1.])
                if phase in ('forward', 'forward_check'):direction = np.array([1., 0., 0.])
                if phase == 'lateral':direction = np.array([0., 1. if part.startswith('left') else -1., 0.])
                r = Rotation.align_vectors([direction], [[0, 0, -1]])[0].as_matrix()
                xyz[b] = xyz[a]+.25*direction
                matrices[profile['parts'][part]['device_id']] = (change@r).tolist()
            row = {'valid': [True]*8, 'confidence': [1.]*133, 'support': [1.]*8, 'points_camera_m': xyz.tolist()}
            samples.append({'phase': phase, 'vision': row, 'imu': matrices,
                            'modes': {d: {GYRO_TILT: r} for d, r in matrices.items()}})
    return samples


def analyze(profile, samples, engine):
    return analyze_recovery(profile, samples, engine,
        {d['device_id']: {'up_world': [0, 0, 1]} for d in profile['parts'].values()},
        camera_epoch='new-camera', vision_epoch='new-vision', vision_config=VISION)


class ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_profile_atomic_roundtrip_no_old_state(self):
        engine, fusion, profile = system()
        fusion.bindings['left_upper_arm'][1].offset_rad = 1.2
        profile = profile_from_state(engine, fusion, CAMERA, VISION)
        self.assertNotIn('offset_rad', json.dumps(profile));self.assertNotIn('neutral_matrix', json.dumps(profile))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile.json';write_json(path, profile)
            self.assertEqual(load_profile(path), profile)
            with self.assertRaises(ValueError):write_json(path, profile)
            self.assertFalse(path.with_suffix('.json.tmp').exists())
        bad = deepcopy(profile);bad['parts']['left_upper_arm']['basis_matrix'][0][0] = -1
        with self.assertRaises(ValueError):validate_profile(bad)
        bad = deepcopy(profile);bad['camera_calibration']['serial'] = 'other'
        with self.assertRaises(ValueError):validate_profile(bad)

    def test_four_new_gauges_restore_only_q_and_validate(self):
        engine, fusion, profile = system()
        result = analyze(profile, recovery_samples(profile, gauge=75), engine)
        for p, data in result['parts'].items():
            self.assertTrue(data['accepted'], data)
            self.assertAlmostEqual(data['neutral_alignment_deg'], 0, places=5)
            q = Rotation.from_euler('z', -75, degrees=True).as_matrix()
            np.testing.assert_allclose(data['Q_camera_from_imu_world'], q, atol=1e-12)
            np.testing.assert_allclose(data['basis_matrix'], profile['parts'][p]['basis_matrix'])

    def test_validation_not_used_for_fit_and_wrong_mount_rejected(self):
        engine, _, profile = system();samples = recovery_samples(profile)
        for s in samples:
            if s['phase'] == 'forward_check':
                s['imu']['0'] = Rotation.from_euler('y', 40, degrees=True).as_matrix().tolist()
        result = analyze(profile, samples, engine)
        self.assertFalse(result['parts']['left_upper_arm']['accepted'])
        self.assertTrue(result['parts']['right_upper_arm']['accepted'])
        np.testing.assert_allclose(result['parts']['left_upper_arm']['Q_camera_from_imu_world'], np.eye(3), atol=1e-12)
        samples = recovery_samples(profile)
        twist = Rotation.from_euler('z', 40, degrees=True).as_matrix()
        for s in samples:s['imu']['0'] = (np.asarray(s['imu']['0'])@twist).tolist()
        result = analyze(profile, samples, engine)
        self.assertFalse(result['parts']['left_upper_arm']['accepted'])
        self.assertGreater(result['parts']['left_upper_arm']['neutral_alignment_deg'], 30)

    def test_each_device_can_have_a_new_tilted_world_reference(self):
        engine, _, profile = system();samples = recovery_samples(profile)
        gauges = {};gravity = {}
        for i, data in enumerate(profile['parts'].values()):
            d = data['device_id'];g = Rotation.from_euler('xyz', [20+i*13, -35+i*7, 60-i*47], degrees=True).as_matrix()
            gauges[d] = g;gravity[d] = {'up_world': (g@np.array([0., 0., 1.])).tolist()}
            for s in samples:
                s['imu'][d] = (g@np.asarray(s['imu'][d])).tolist()
                s['modes'][d][GYRO_TILT] = s['imu'][d]
        report = analyze_recovery(profile, samples, engine, gravity, camera_epoch='new-camera', vision_epoch='new-vision', vision_config=VISION)
        for data in report['parts'].values():
            self.assertTrue(data['accepted'], data)
            np.testing.assert_allclose(data['Q_camera_from_imu_world'], gauges[data['device_id']].T, atol=1e-12)

    def test_vertical_only_and_body_turn_rejected(self):
        with self.assertRaisesRegex(ValueError, '非竖直'):
            fit_fixed_bone([np.eye(3)]*20, [[0, 0, -1]]*20, [0, 0, -1], [0, 0, 1], [0, 0, 1])
        engine, _, profile = system();samples = recovery_samples(profile)
        turn = Rotation.from_euler('z', 25, degrees=True).as_matrix()
        for s in samples:
            if s['phase'] == 'lateral':s['vision']['points_camera_m'] = (np.asarray(s['vision']['points_camera_m'])@turn.T).tolist()
        with self.assertRaisesRegex(ValueError, '躯干'):analyze(profile, samples, engine)

    def test_strict_bias_window_all_parts_prevalidated_no_pose_reset(self):
        engine, fusion, _ = system()
        for d in engine.epochs:
            fusion.gravity_history[d] = [{'stamp': i*20_000_000, 'epoch': engine.epochs[d], 'gyro': np.array([.1, -.1, .2]), 'acc': np.array([0, 0, 1.])} for i in range(60)]
        before = engine.current_for('0', GYRO_TILT).copy()
        bias = learn_still_bias(fusion, engine.epochs, 0, 2_000_000_000)
        np.testing.assert_allclose(bias['0'], [.1, -.1, .2]);np.testing.assert_equal(engine.current_for('0', GYRO_TILT), before)
        fusion.gravity_history['1'][0]['gyro'] = np.array([2, 0, 0.])
        engine._tilt_filters['0'].gyro_bias_dps[:] = 0
        with self.assertRaisesRegex(ValueError, '静止'):learn_still_bias(fusion, engine.epochs, 0, 2_000_000_000)
        np.testing.assert_equal(engine._tilt_filters['0'].gyro_bias_dps, np.zeros(3))

    def owner(self, engine, fusion):
        owner = SimpleNamespace(engine=engine, live_fusion=fusion, replay_mode=False, session=None,
            _record_start_deadline=None, x1_preview_dialog=None, _live_camera_epoch=None,
            camera_source='orbbec', camera_worker=SimpleNamespace(calibration=CAMERA, capture_epoch='new-camera'),
            vision_metadata=VISION, vision_epoch='new-vision', rgbd_pose_enabled=SimpleNamespace(isChecked=lambda: True))
        owner._check_vision_report = lambda report: None
        owner.apply_camera_report = lambda report: fusion.install(report)
        return owner

    def test_optional_bias_search_whole_window_and_independent_devices(self):
        engine, fusion, _ = system()
        for d in engine.epochs:
            fusion.gravity_history[d] = [{'stamp': i*20_000_000, 'epoch': engine.epochs[d],
                'gyro': np.array([.1, 0., 0.]) if i < 60 else np.array([2., 0., 0.]),
                'acc': np.array([0., 0., 1.])} for i in range(200)]
        for r in fusion.gravity_history['1']:r['gyro'] = np.array([2., 0., 0.])
        engine._tilt_filters['1'].bias_ready = True;engine._tilt_filters['1'].gyro_bias_dps[:] = [.2, -.3, .4]
        for r in fusion.gravity_history['2']:r['gyro'] = np.array([2., 0., 0.])
        before = engine.current_for('0', GYRO_TILT).copy()
        result = attempt_still_bias(fusion, engine.epochs, 0, 4_000_000_000)
        self.assertTrue(result['0']['updated']);self.assertLess(result['0']['window_ns'][1], 2_000_000_000)
        self.assertEqual(result['1']['status'], 'preserved_ready')
        np.testing.assert_equal(engine._tilt_filters['1'].gyro_bias_dps, [.2, -.3, .4])
        self.assertEqual(result['2']['status'], 'unlearned');self.assertFalse(engine._tilt_filters['2'].bias_ready)
        np.testing.assert_equal(engine.current_for('0', GYRO_TILT), before)

    def test_skip_explicit_learning_preserves_bias_and_readiness(self):
        engine, fusion, _ = system()
        engine._tilt_filters['0'].gyro_bias_dps[:] = [.1, .2, .3];engine._tilt_filters['0'].bias_ready = True
        result = attempt_still_bias(fusion, engine.epochs, 0, 4_000_000_000, skip=True)
        self.assertTrue(all(r['skipped'] and not r['updated'] for r in result.values()))
        np.testing.assert_equal(engine._tilt_filters['0'].gyro_bias_dps, [.1, .2, .3])
        self.assertEqual(result['1']['status'], 'unlearned')

    def test_auto_bias_plan_partial_ready_force_and_reset(self):
        engine, fusion, _ = system()
        for d in ('0','1'):
            engine._tilt_filters[d].bias_ready = True
            engine._tilt_filters[d].gyro_bias_dps[:] = [.1,.2,.3]
        before = engine.current_for('0',GYRO_TILT).copy()
        pending, result = plan_bias_reuse(fusion, engine.epochs)
        self.assertEqual(pending,['2','3']);self.assertEqual(result['0']['status'],'reused_current')
        np.testing.assert_equal(engine.current_for('0',GYRO_TILT),before)
        self.assertEqual(plan_bias_reuse(fusion,engine.epochs,force=True)[0],list(engine.epochs))
        self.assertEqual(plan_bias_reuse(fusion,engine.epochs,skip=True)[0],[])
        with self.assertRaises(ValueError):plan_bias_reuse(fusion,engine.epochs,skip=True,force=True)
        old_epoch = engine.epochs['0'];engine.invalidate_neutral('0')
        engine.feed(replace(packet(),device_id='0'),received_at=time.monotonic())
        self.assertNotEqual(engine.epochs['0'],old_epoch)
        pending,result = plan_bias_reuse(fusion,['0'])
        self.assertEqual(pending,['0']);self.assertFalse(result['0']['bias_ready'])

    def test_gui_all_ready_skips_stage_force_keeps_stage(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w = FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w.engine,w.live_fusion,profile = system()
                for f in w.engine._tilt_filters.values():f.bias_ready=True
                w.camera_worker=SimpleNamespace(calibration=CAMERA,capture_epoch='new-camera')
                w.vision_epoch='new-vision';w.vision_metadata=VISION
                w.rgbd_pose_enabled.blockSignals(True);w.rgbd_pose_enabled.setChecked(True);w.rgbd_pose_enabled.blockSignals(False)
                w.last_pose_estimate=SimpleNamespace(captured_mono_ns=time.monotonic_ns())
                w._show_calibration_reuse();d=w._reuse_calibration_dialog
                d.set_profile(profile,'test');d.confirm.setChecked(True);d.delay.setValue(0)
                d.begin();d.timer.stop()
                self.assertEqual(d.steps[0][0],'neutral');self.assertEqual(d.bias_pending,[])
                self.assertEqual(d.state,'prepare');self.assertTrue(all(r['status']=='reused_current' for r in d.bias_outcomes.values()))
                d.abort();d.force_bias.setChecked(True);d.begin();d.timer.stop()
                self.assertEqual(d.steps[0][0],'still');self.assertEqual(len(d.bias_pending),4)
                d.abort();d.skip_bias.setChecked(True)
                self.assertFalse(d.force_bias.isChecked())
            finally:w.camera_worker=None;w.close()

    def test_context_mismatch_and_atomic_apply_rollback(self):
        engine, fusion, profile = system();owner = self.owner(engine, fusion)
        self.assertIsNone(context_error(profile, owner))
        owner.camera_worker.calibration = {'serial': 'other'}
        self.assertIn('相机', context_error(profile, owner));owner.camera_worker.calibration = CAMERA
        owner.vision_metadata = {'signature': 'other'}
        self.assertIn('视觉模型', context_error(profile, owner));owner.vision_metadata = VISION
        result = analyze(profile, recovery_samples(profile, gauge=50), engine)
        before = deepcopy(engine.mode_neutral)
        def failed(report):raise ValueError('simulated failure')
        owner.apply_camera_report = failed
        with self.assertRaisesRegex(ValueError, 'simulated'):apply_recovery(owner, result, list(profile['parts']))
        for d in before:np.testing.assert_equal(before[d][GYRO_TILT], engine.mode_neutral[d][GYRO_TILT])
        owner.apply_camera_report = lambda report: fusion.install(report)
        fusion.bindings['left_upper_arm'][1].offset_rad = 2
        apply_recovery(owner, result, list(profile['parts']))
        self.assertEqual(fusion.bindings['left_upper_arm'][1].offset_rad, 0)
        self.assertTrue(all(engine.algorithm_modes[p] == GYRO_TILT for p in profile['parts']))
        old = deepcopy(engine.mode_neutral);result['parts']['right_upper_arm']['reference_epoch'] = 'old'
        with self.assertRaises(ValueError):apply_recovery(owner, result, list(profile['parts']))
        for d in old:np.testing.assert_equal(old[d][GYRO_TILT], engine.mode_neutral[d][GYRO_TILT])

    def test_historical_record_import_requires_complete_applied_snapshot(self):
        engine, fusion, profile = system()
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = {'assignments_at_start': engine.assignments, 'continuous_basis_at_start': {p: r.tolist() for p, r in engine.continuous_basis.items()},
                        'fusion_at_start': {p: {'calibration': d} for p, (d, _) in fusion.bindings.items()}, 'vision_config': VISION}
            write_json(path/'manifest.json', manifest);write_json(path/'camera_calibration.json', CAMERA)
            with (path/'runtime_pose.jsonl').open('w') as stream:
                for device in engine.epochs:stream.write(json.dumps(engine.runtime_snapshot(device, 1))+'\n')
            imported = import_recording(path)
            self.assertEqual(set(imported['parts']), set(profile['parts']))
            manifest['camera_epoch'] = 'old-camera';manifest['vision_epoch'] = 'old-vision'
            report = {'format': 'rgbd-heading-fusion-experiment-v1', 'orientation_source': 'live_runtime_pose',
                      'camera_epoch': 'old-camera', 'vision_epoch': 'old-vision', 'source_session': str(path),
                      'vision_config': VISION, 'parts': {p: d for p, (d, _) in fusion.bindings.items()}}
            (path/'manifest.json').write_text(json.dumps(manifest))
            write_json(path/'report.json', report)
            self.assertEqual(set(import_camera_report(path/'report.json')['parts']), set(profile['parts']))
            report['camera_epoch'] = 'mismatch';(path/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, '不匹配'):import_camera_report(path/'report.json')
            manifest['continuous_basis_at_start'] = {}
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, '没有可复用'):import_recording(path)

    def test_gui_defaults_load_and_block_before_capture(self):
        with TemporaryDirectory() as tmp, patch.object(FusionWindow, 'start_camera'):
            w = FusionWindow(Source(), Path(tmp)/'settings.json', {}, {}, camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_calibration_reuse();d = w._reuse_calibration_dialog
                self.assertIsInstance(d, CalibrationReuseDialog);self.assertIs(w.camera_calibration_dialog, d)
                _, _, profile = system();d.set_profile(profile, 'test')
                d.begin();self.assertFalse(d.active);self.assertIn('确认', d.summary.toPlainText())
                d.confirm.setChecked(True);d.begin();self.assertFalse(d.active)
                self.assertFalse(d.apply.isEnabled());self.assertFalse(w.engine.continuous_basis)
                d.shutdown()
            finally:w.close()

    def test_gui_full_recovery_pipeline_no_depth_capture_or_auto_mode_change(self):
        with TemporaryDirectory() as tmp, patch.object(FusionWindow, 'start_camera'):
            w = FusionWindow(Source(), Path(tmp)/'settings.json', {}, {}, camera_source='orbbec');w.poll_timer.stop()
            try:
                w.engine, w.live_fusion, profile = system()
                w.camera_worker = SimpleNamespace(calibration=CAMERA, capture_epoch='new-camera')
                w.vision_epoch = 'new-vision';w.vision_metadata = VISION
                w.rgbd_pose_enabled.blockSignals(True);w.rgbd_pose_enabled.setChecked(True);w.rgbd_pose_enabled.blockSignals(False)
                w.last_pose_estimate = SimpleNamespace(captured_mono_ns=time.monotonic_ns())
                w._show_calibration_reuse();d = w._reuse_calibration_dialog;d.root = Path(tmp)/'profiles'
                d.set_profile(profile, 'test');d.confirm.setChecked(True);d.delay.setValue(0);d.begin();d.timer.stop()
                self.assertTrue(d.active);self.assertIsNone(w.session)
                now = time.monotonic_ns()
                for device in w.engine.epochs:
                    w.live_fusion.gravity_history[device] = [{'stamp': now-1_200_000_000+i*20_000_000,
                        'epoch': w.engine.epochs[device], 'gyro': np.array([0., 0., 0.]), 'acc': np.array([0., 0., 1.]),
                        'rotation': np.eye(3)} for i in range(60)]
                # One worn sensor never meets rest-learning conditions: this
                # must not abort the neutral/visual recovery pipeline.
                for row in w.live_fusion.gravity_history['0']:row['gyro'] = np.array([1.2, 0., 0.])
                d.deadline = 0;d.tick()  # still prepare -> hold
                d.windows[-1]['start_ns'] = now-2_000_000_000
                d.deadline = 0;d.tick()  # strict still learning -> neutral prepare
                self.assertEqual(d.steps[d.index][0], 'neutral')
                d.deadline = 0;d.tick();d.windows[-1]['start_ns'] = now-2_000_000_000
                d.deadline = 0;d.tick()  # neutral gravity lock -> forward prepare
                d.samples = recovery_samples(profile)
                while d.active:
                    d.deadline = 0;d.tick()
                self.assertEqual(d.state, 'complete', d.summary.toPlainText())
                self.assertEqual(d.bias_outcomes['0']['status'], 'unlearned')
                self.assertTrue(all(v['accepted'] for v in d.result['parts'].values()))
                self.assertTrue(d.report_path.is_file());self.assertFalse(list(d.root.rglob('*.npy')))
                d.apply_selected();self.assertFalse(d.apply.isEnabled())
                self.assertTrue(all(w.engine.algorithm_modes[p] == GYRO_TILT for p in profile['parts']))
                self.assertTrue(all(c.offset_rad == 0 for _, c in w.live_fusion.bindings.values()))
                self.assertEqual(w._live_camera_epoch, 'new-camera')
                self.assertIsNone(w.session)
            finally:w.camera_worker = None;w.close()


if __name__ == '__main__':unittest.main()
