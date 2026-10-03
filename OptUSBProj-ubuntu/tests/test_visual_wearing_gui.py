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
from optusbproj_ubuntu.visual_wearing import analyze_wearing
from optusbproj_ubuntu.visual_wearing_gui import VisualWearingDialog
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
