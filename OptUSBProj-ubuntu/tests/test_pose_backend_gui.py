import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PySide6 import QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


class Source:
    def seen_devices(self,max_age=3):return {}
    def close(self):pass


class ConfigGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_switch_preserves_wearing_basis_and_correction_but_invalidates_fusion(self):
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{'left_upper_arm':'A'},{},camera_source='orbbec')
            w.poll_timer.stop()
            try:
                w.engine.continuous_basis['left_upper_arm']=np.eye(3)
                h=np.diag([-1.,-1.,1.]);w.engine.fusion_corrections['A']=h.copy();w.engine.fusion_ready.add('A')
                w.live_fusion.bindings['left_upper_arm']=({'device_id':'A'},SimpleNamespace(offset_rad=.2))
                epoch=w.vision_epoch;worker=Mock();w.pose_worker=worker
                w.pose_backend_combo.setCurrentIndex(w.pose_backend_combo.findData('yolo26'))
                self.assertEqual(w.vision_config.backend,'yolo26');self.assertNotEqual(w.vision_epoch,epoch)
                self.assertNotIn('A',w.engine.fusion_ready)
                np.testing.assert_equal(w.engine.fusion_corrections['A'],h)
                np.testing.assert_equal(w.engine.continuous_basis['left_upper_arm'],np.eye(3))
                worker.stop.assert_called_once()
            finally:w.pose_worker=None;w.close()

    def test_switch_blocked_during_record_calibration_and_x1(self):
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{},{});w.poll_timer.stop()
            try:
                for blocker in ('record','guide','robot'):
                    if blocker=='record':w._record_start_deadline=1
                    elif blocker=='guide':w.camera_calibration_dialog=SimpleNamespace(active=True,state='hold')
                    else:w.x1_preview_dialog=SimpleNamespace(sender=object())
                    epoch=w.vision_epoch
                    w.pose_backend_combo.setCurrentIndex(w.pose_backend_combo.findData('yolo26'))
                    self.assertEqual(w.vision_config.backend,'rtmw');self.assertEqual(w.vision_epoch,epoch)
                    w._record_start_deadline=None;w.camera_calibration_dialog=None;w.x1_preview_dialog=None
            finally:w.close()

    def test_report_model_epoch_and_legacy_rules(self):
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{},{});w.poll_timer.stop()
            try:
                w._check_vision_report({})
                report={'vision_config':copy.deepcopy(w.vision_metadata),'vision_epoch':w.vision_epoch}
                w._check_vision_report(report)
                report['vision_epoch']='old'
                with self.assertRaisesRegex(ValueError,'视觉会话'):w._check_vision_report(report)
                w.pose_backend_combo.setCurrentIndex(w.pose_backend_combo.findData('yolo26'))
                with self.assertRaisesRegex(ValueError,'旧报告'):w._check_vision_report({})
                w.pose_backend_combo.setCurrentIndex(w.pose_backend_combo.findData('rtmw'))
                with self.assertRaisesRegex(ValueError,'旧报告'):w._check_vision_report({})
            finally:w.close()

    def test_recording_manifest_and_replay_restore_model_configuration(self):
        with TemporaryDirectory() as folder:
            recorder=FusionSessionRecorder(Path(folder),tail_trim_seconds=0)
            recorder.vision_config={'backend':'yolo26','weights':[{'path':'/unavailable/yolo26m-pose.pt','sha256':'example'}],
                                    'imgsz':960,'signature':'example'}
            recorder.vision_epoch='vision';recorder.close()
            source=FusionSessionReplay(recorder.path)
            w=FusionWindow(source,Path(folder)/'settings.json',{},{});w.poll_timer.stop()
            try:
                self.assertEqual(w.vision_config.backend,'yolo26');self.assertEqual(w.vision_config.imgsz,960)
                self.assertEqual(w.vision_epoch,'vision');self.assertFalse(w.pose_backend_combo.isEnabled())
            finally:w.close()


if __name__=='__main__':unittest.main()
