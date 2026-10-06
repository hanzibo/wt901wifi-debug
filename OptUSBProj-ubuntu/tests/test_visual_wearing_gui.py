import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6 import QtWidgets
from optusbproj_ubuntu.visual_wearing import analyze_wearing, REVIEW, PART_LABELS
from optusbproj_ubuntu.visual_wearing_gui import VisualWearingDialog, wearing_steps
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT
from test_camera_calibration_gui import Owner
from test_visual_wearing import fixture


class Source:
    def seen_devices(self, max_age=3): return {}
    def close(self): pass


class VisualGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_bilateral_capture_locks_four_devices_and_saves_shared_windows(self):
        with TemporaryDirectory() as folder:
            clock = [100.]; owner = Owner(Path(folder)/'record', clock)
            owner.engine.assignments = dict(zip(PART_LABELS, ('C','D','A','B')))
            owner.engine.algorithm_modes = {p: GYRO_TILT for p in PART_LABELS}
            owner.engine.epochs = {d: 'epoch'+d for d in owner.engine.assignments.values()}
            owner.camera_worker = SimpleNamespace(capture_epoch='camera')
            owner.x1_preview_dialog = None
            dialog = VisualWearingDialog(owner); dialog.auto_analyze.setChecked(False)
            try:
                self.assertEqual(dialog.scope_combo.currentData(), 'bilateral')
                with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic', side_effect=lambda: clock[0]), \
                     patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns', side_effect=lambda: int(clock[0]*1e9)):
                    dialog.begin(); dialog.timer.stop()
                    self.assertFalse(dialog.scope_combo.isEnabled())
                    clock[0] = dialog.deadline+.1; owner.refresh(); dialog.tick()
                    while dialog.active:
                        clock[0] = dialog.deadline+.1; owner.refresh(); dialog.tick()
                guide = json.loads((owner.path/'camera_guide.json').read_text())
                self.assertEqual(guide['capture_scope'], 'bilateral')
                self.assertEqual(set(guide['participants']), set(PART_LABELS))
                self.assertEqual(len(owner.gravity_calls[0][0]), 4)
                self.assertEqual(len(guide['events']), 16)
                self.assertEqual(dialog.state, 'complete')
                self.assertTrue(dialog.scope_combo.isEnabled())
                self.assertIn('T字', next(s[2] for s in dialog.steps if s[0]=='lateral'))
            finally:
                dialog.close(); owner.close()

    def test_side_selection_and_partial_failure_application(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path, bilateral=True, corrupt_left_repeat=True)
            report = analyze_wearing(path)
            (path/'report.json').write_text(json.dumps(report))
            owner = Owner(path, [100.]); owner.x1_preview_dialog = None
            owner.camera_worker = SimpleNamespace(capture_epoch='camera')
            owner.engine.algorithm_modes = {p: GYRO_TILT for p in PART_LABELS}
            dialog = VisualWearingDialog(owner)
            try:
                dialog.scope_combo.setCurrentIndex(dialog.scope_combo.findData('bilateral'))
                with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns', return_value=100_000_000_000):
                    self.assertIn('left_upper_arm', dialog.readiness_error())
                    dialog.scope_combo.setCurrentIndex(dialog.scope_combo.findData('right'))
                    self.assertIsNone(dialog.readiness_error())
                dialog.analysis_path = path; dialog.read_log = lambda: None
                calls = []
                owner.apply_visual_wearing_report = lambda r, parts, **kw: calls.append(parts) or {}
                dialog.analysis_finished(0, None)
                self.assertFalse(dialog.part_checks['left_upper_arm'].isEnabled())
                self.assertFalse(dialog.part_checks['left_forearm'].isEnabled())
                self.assertTrue(dialog.part_checks['right_upper_arm'].isChecked())
                with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.StandardButton.Yes):
                    dialog.apply_live()
                self.assertEqual(set(calls[0]), {'right_upper_arm','right_forearm'})
                left = wearing_steps(3, 3, 5, 20, 'left')
                self.assertIn('左侧', next(s[2] for s in left if s[0]=='lateral'))
                self.assertFalse(any('右臂' in s[2] for s in left))
            finally:
                dialog.close(); owner.close()

    def test_failed_parts_cannot_be_selected_and_application_is_explicit(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path)
            report = analyze_wearing(path)
            report['parts']['right_forearm']['status'] = '需重录'
            report['parts']['right_forearm']['warnings'] = ['独立复测失败']
            (path/'report.json').write_text(json.dumps(report))
            owner = Owner(path, [100.]); owner.x1_preview_dialog = None
            dialog = VisualWearingDialog(owner)
            dialog.analysis_path = path; dialog.read_log = lambda: None
            calls = []
            owner.apply_visual_wearing_report = lambda r, parts: calls.append(parts) or {}
            try:
                dialog.analysis_finished(0, None)
                self.assertTrue(dialog.part_checks['right_upper_arm'].isChecked())
                self.assertFalse(dialog.part_checks['right_forearm'].isEnabled())
                self.assertEqual(calls, [])
                with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.StandardButton.Yes):
                    dialog.apply_live()
                self.assertEqual(calls, [['right_upper_arm']])
                self.assertTrue(dialog.restore_button.isEnabled())
            finally:
                dialog.close(); owner.close()

    def test_review_requires_preview_and_explicit_unchecked_confirmation(self):
        with TemporaryDirectory() as folder:
            path = Path(folder); fixture(path)
            report = analyze_wearing(path)
            report['parts']['right_upper_arm'].update(status=REVIEW, review_reasons=['肩线变化不等于转身'])
            (path/'report.json').write_text(json.dumps(report))
            owner = Owner(path, [100.]); owner.x1_preview_dialog = None
            dialog = VisualWearingDialog(owner); dialog.analysis_path = path
            dialog.read_log = lambda: None
            calls = []
            owner.apply_visual_wearing_report = lambda r, parts, **kw: calls.append((parts, kw)) or {}
            try:
                dialog.analysis_finished(0, None)
                self.assertFalse(dialog.apply.isEnabled())
                self.assertFalse(dialog.review_confirmation.isChecked())
                self.assertFalse(dialog.review_confirmation.isEnabled())
                with patch.object(QtWidgets.QMessageBox, 'warning') as warning:
                    dialog.apply_live(); warning.assert_called_once()
                self.assertEqual(calls, [])
                # Equivalent to the successful open_replay callback; merely
                # selecting a baseline or checking a part is insufficient.
                dialog._preview_opened = True; dialog.update_apply_gate()
                self.assertTrue(dialog.review_confirmation.isEnabled())
                self.assertFalse(dialog.apply.isEnabled())
                dialog.review_confirmation.setChecked(True)
                self.assertTrue(dialog.apply.isEnabled())
                with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.StandardButton.Yes):
                    dialog.apply_live()
                self.assertTrue(calls[0][1]['review_acknowledged'])
                self.assertFalse(dialog.apply.isEnabled())
            finally:
                dialog.close(); owner.close()

    def test_owner_apply_and_undo_are_guarded_and_preserve_unrelated_parts(self):
        with TemporaryDirectory() as folder, patch.object(FusionWindow, 'start_camera'):
            path = Path(folder); fixture(path); report = analyze_wearing(path)
            window = FusionWindow(Source(), path/'settings.json',
                {'right_upper_arm': 'A', 'right_forearm': 'B', 'left_hand': 'C'}, {},
                camera_source='orbbec', algorithm_modes={'right_upper_arm': GYRO_TILT, 'right_forearm': GYRO_TILT})
            window.poll_timer.stop()
            try:
                window.camera_worker = SimpleNamespace(capture_epoch='camera')
                engine = window.engine
                engine.epochs.update(A='epochA', B='epochB')
                engine.is_live = lambda *a, **k: True
                engine._tilt_filters = {d: SimpleNamespace(neutral_up=np.array([0., 0., 1.])) for d in ('A', 'B')}
                engine.mode_neutral['A'] = {GYRO_TILT: np.eye(3)}
                engine.set_continuous_basis('left_hand', np.eye(3))
                backup = window.apply_visual_wearing_report(report, ['right_upper_arm'])
                self.assertIn('right_upper_arm', engine.continuous_basis)
                self.assertNotIn('right_forearm', engine.continuous_basis)
                window.restore_visual_wearing(backup)
                self.assertNotIn('right_upper_arm', engine.continuous_basis)
                np.testing.assert_allclose(engine.mode_neutral['A'][GYRO_TILT], np.eye(3))
                np.testing.assert_allclose(engine.continuous_basis['left_hand'], np.eye(3))
                backup = window.apply_visual_wearing_report(report, ['right_upper_arm'])
                engine.set_continuous_basis('right_upper_arm', np.eye(3))
                with self.assertRaisesRegex(ValueError, '其他操作'):
                    window.restore_visual_wearing(backup)
                window.x1_preview_dialog = SimpleNamespace(sender=object())
                with self.assertRaisesRegex(ValueError, 'X1'):
                    window.apply_visual_wearing_report(report, ['right_upper_arm'])
            finally:
                window.x1_preview_dialog = None; window.camera_worker = None; window.close()

    def test_both_avatars_use_actual_runtime_rotations_not_reintegrated_rpy(self):
        # A lightweight timed source with a recorded rotation deliberately
        # different from the supplied RPY; no camera hardware is opened.
        r0 = Rotation.from_euler('x', 20, degrees=True).as_matrix()
        r1 = r0 @ Rotation.from_euler('y', 40, degrees=True).as_matrix()
        candidate = {'part': 'right_upper_arm', 'device_id': 'A', 'basis_matrix': np.eye(3).tolist(),
                     'neutral_matrix': r0.tolist(), 'neutral_matrices_by_mode': {GYRO_TILT: r0.tolist()}}
        class Replay(Source):
            is_replay = True
            def __init__(self): self.sent = False
            def take_camera(self): return []
            def take_vision(self): return []
            def poll_timed(self):
                if self.sent: return []
                self.sent = True
                return [(1_000_000_000, SimpleNamespace(device_id='A', roll=0, pitch=0, yaw=0))]
            def take_runtime(self):
                return {(1_000_000_000, 'A'): {'device_id': 'A', 'epoch': 'e',
                    'orientation_by_mode': {GYRO_TILT: r1.tolist()},
                    'neutral_by_mode': {GYRO_TILT: np.eye(3).tolist()},
                    'fusion_correction': np.eye(3).tolist()}}
        with TemporaryDirectory() as folder, patch.object(FusionWindow, 'start_camera'):
            # Generic replay source exercises the same packet/runtime path.
            source = Replay()
            window = FusionWindow(source, Path(folder)/'settings.json', {'right_upper_arm': 'A'}, {},
                                  algorithm_modes={'right_upper_arm': GYRO_TILT},
                                  calibration_candidates=[candidate])
            window.poll_timer.stop()
            try:
                window._poll()
                for engine in (window.engine, window.candidate_engine):
                    np.testing.assert_allclose(engine.current['A'], r1)
                    np.testing.assert_allclose(engine.neutral['A'], r0)
                    np.testing.assert_allclose(engine.transforms()[6][:3, :3], r0.T @ r1, atol=1e-12)
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
