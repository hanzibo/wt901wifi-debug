import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6 import QtCore,QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.recording_cleanup import delete_capture,validate_capture
from optusbproj_ubuntu.fusion_gui import FusionWindow
from test_pose_backend_gui import Source


def capture(root):
    path=root/'20261004-123456-abcdef';path.mkdir()
    (path/'manifest.json').write_text(json.dumps({'format':'wt901-fusion-session-v2','started_mono_ns':123,
                                                'ended_mono_ns':456,'error':None}))
    (path/'camera_guide.json').write_text(json.dumps({'status':'complete','workflow':'combined_wearing_camera_calibration','recording_started_mono_ns':123}))
    (path/'large-depth.npy').write_bytes(b'test-only')
    return path


class CleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_exact_capture_delete_keeps_analysis_and_inner_symlink_target(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=capture(root);analysis=root/'analysis_combined';analysis.mkdir()
            outside=root/'keep.txt';outside.write_text('keep');(path/'link').symlink_to(outside)
            delete_capture(path,123,root=root)
            self.assertFalse(path.exists());self.assertTrue(analysis.exists());self.assertEqual(outside.read_text(),'keep')

    def test_reject_root_analysis_other_directory_identity_and_symlink(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=capture(root)
            for target in (root,root/'analysis_combined',root/'../other',root/'not-a-recording'):
                with self.assertRaises((ValueError,OSError)):validate_capture(target,123,root=root)
            with self.assertRaises(ValueError):validate_capture(path,999,root=root)
            alias=root/'20261004-123456-fedcba';alias.symlink_to(path,target_is_directory=True)
            with self.assertRaises(ValueError):validate_capture(alias,123,root=root)
            self.assertTrue(path.exists())

    def test_gui_computed_without_apply_enables_delete_and_cancel_preserves_data(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            root=Path(tmp);path=capture(root);analysis=root/'analysis_combined';analysis.mkdir()
            w=FusionWindow(Source(),root/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                self.assertFalse(d.delete_recording.isEnabled())
                d.recording_path=path;d.analysis_path=analysis;d.report={'session':str(path)};d.state='complete'
                with patch('optusbproj_ubuntu.combined_calibration_gui.validate_capture',side_effect=lambda p,t:validate_capture(p,t,root=root)):
                    d._authorize_cleanup()
                    self.assertFalse(d.applied);self.assertTrue(d.delete_recording.isEnabled())
                    with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.StandardButton.No):d.delete_current_capture()
                    self.assertTrue(path.exists());self.assertIsNone(d._cleanup_process)
                    d._camera_preview_processes=[object()]
                    self.assertIn('回放',d.cleanup_error());d._camera_preview_processes=[]
                    source=SimpleNamespace(is_replay=True,path=path,imu_file=SimpleNamespace(closed=False))
                    window=SimpleNamespace(source=source);w._dialogs.append(window)
                    self.assertIn('回放',d.cleanup_error());w._dialogs.remove(window)
            finally:w.close()

    def test_gui_background_success_disables_replay_but_keeps_result(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            root=Path(tmp);path=capture(root);analysis=root/'analysis_combined';analysis.mkdir()
            w=FusionWindow(Source(),root/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                d.recording_path=path;d.analysis_path=analysis;d.report={'session':str(path)};d.state='complete'
                with patch('optusbproj_ubuntu.combined_calibration_gui.validate_capture',side_effect=lambda p,t:validate_capture(p,t,root=root)):
                    d._authorize_cleanup()
                    with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.StandardButton.Yes),patch.object(QtCore.QProcess,'start'):
                        d.delete_current_capture()
                    self.assertFalse(d.delete_recording.isEnabled());self.assertFalse(d.analyze.isEnabled())
                    delete_capture(path,123,root=root);d._cleanup_finished(0,None)
                    self.assertFalse(d.delete_recording.isEnabled());self.assertFalse(d.replay.isEnabled())
                    self.assertIsNotNone(d.report);self.assertFalse(d.applied);self.assertTrue(analysis.exists())
                    self.assertTrue(d.start.isEnabled())
            finally:w.close()


if __name__=='__main__':unittest.main()
