import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.rgbd_fusion import HeadingCorrector
from optusbproj_ubuntu.live_rgbd_fusion import LiveRGBDFusion
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.fusion_gui import FusionWindow
from test_pose_backend_gui import Source


class LimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_default_limit_rejection_and_increase_resume_without_jump(self):
        c=HeadingCorrector([0,0,1]);c.offset_rad=np.radians(29.99)
        target=Rotation.from_euler('z',45,degrees=True).apply([1,0,0])
        applied,reason,_=c.update(0,[1,0,0],target)
        self.assertFalse(applied);self.assertIn('±30',reason);self.assertIn('已拒绝',c.limit_status())
        old=c.rotation().copy();c.set_max_offset(60)
        np.testing.assert_equal(c.rotation(),old)
        self.assertTrue(c.update(.1,[1,0,0],target)[0])
        self.assertGreater(np.degrees(c.offset_rad),30)
        self.assertLessEqual(np.degrees(c.offset_rad)-29.99,.3+1e-9)
        self.assertAlmostEqual(np.degrees(c.max_innovation),30)
        self.assertAlmostEqual(np.degrees(c.max_rate),3)
        self.assertIsNone(c.limit_status())

    def test_limits_symmetric_and_large_innovation_still_rejected(self):
        for sign in (-1,1):
            c=HeadingCorrector([0,0,1],max_offset_deg=60);c.offset_rad=np.radians(sign*59.99)
            target=Rotation.from_euler('z',sign*75,degrees=True).apply([1,0,0])
            self.assertFalse(c.update(0,[1,0,0],target)[0]);self.assertIn('±60',c.limit_status())
            target=Rotation.from_euler('z',sign*50,degrees=True).apply([1,0,0])
            self.assertTrue(c.update(.1,[1,0,0],target)[0])
            self.assertIn('接近',c.limit_status())
        c=HeadingCorrector([0,0,1],max_offset_deg=90)
        self.assertFalse(c.update(0,[1,0,0],[0,1,0])[0]);self.assertEqual(c.offset_rad,0)

    def test_unlimited_multiple_turns_and_safety_gates(self):
        for sign in (-1,1):
            c=HeadingCorrector([0,0,1],max_offset_deg=None)
            for i in range(1500):
                target=Rotation.from_rotvec(c.up*(c.offset_rad+sign*np.radians(20))).apply([1,0,0])
                old=c.offset_rad
                self.assertTrue(c.update(i*.1,[1,0,0],target)[0])
                self.assertLessEqual(abs(c.offset_rad-old),np.radians(.3)+1e-12)
            self.assertGreater(abs(np.degrees(c.offset_rad)),360)
            np.testing.assert_allclose(c.rotation().T@c.rotation(),np.eye(3),atol=1e-12)
            self.assertIn('不限',c.limit_status())
            old=c.offset_rad
            with self.assertRaises(ValueError):c.set_max_offset(90)
            self.assertIsNone(c.max_offset);self.assertEqual(c.offset_rad,old)
            target=Rotation.from_rotvec(c.up*(old+np.radians(40))).apply([1,0,0])
            self.assertFalse(c.update(150,[1,0,0],target)[0])
            self.assertFalse(c.update(150.1,[0,0,1],[0,0,1])[0])
            self.assertEqual(c.offset_rad,old)

    def test_gui_unlimited_apply_and_refusal_restore(self):
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{},{});w.poll_timer.stop()
            try:
                self.assertFalse(w.heading_unlimited.isChecked())
                c=HeadingCorrector([0,0,1]);c.offset_rad=np.radians(29)
                w.live_fusion.bindings={'left_upper_arm':({},c)}
                w.heading_unlimited.setChecked(True);w._apply_heading_limit()
                self.assertIsNone(w.live_fusion.max_offset_deg)
                self.assertIsNone(c.max_offset);self.assertAlmostEqual(np.degrees(c.offset_rad),29)
                c.offset_rad=np.radians(100)
                w.heading_unlimited.setChecked(False)
                with patch.object(QtWidgets.QMessageBox,'warning') as warning:w._apply_heading_limit();warning.assert_called_once()
                self.assertTrue(w.heading_unlimited.isChecked());self.assertIsNone(c.max_offset)
                c.offset_rad=np.radians(20);w.heading_unlimited.setChecked(False);w._apply_heading_limit()
                self.assertEqual(w.live_fusion.max_offset_deg,30)
            finally:w.live_fusion.bindings={};w.close()

    def test_atomic_service_change_and_invalid_ranges(self):
        service=LiveRGBDFusion(PoseEngine())
        a=HeadingCorrector([0,0,1]);b=HeadingCorrector([0,0,1]);b.offset_rad=np.radians(25)
        service.bindings={'left_upper_arm':({},a),'right_upper_arm':({},b)}
        with self.assertRaises(ValueError):service.set_max_offset(20)
        self.assertEqual(service.max_offset_deg,30)
        self.assertAlmostEqual(np.degrees(a.max_offset),30)
        service.set_max_offset(60)
        self.assertEqual(service.max_offset_deg,60);self.assertAlmostEqual(np.degrees(b.offset_rad),25)
        service.set_max_offset(None)
        self.assertIsNone(service.events[-1]['max_offset_deg'])
        self.assertIsNone(a.max_offset);self.assertIsNone(b.max_offset)
        service.set_max_offset(60)
        for value in (0,91,float('nan'),float('inf')):
            with self.assertRaises(ValueError):service.set_max_offset(value)
        self.assertEqual(service.max_offset_deg,60)

    def test_gui_apply_block_and_preserve_offset(self):
        with TemporaryDirectory() as folder,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(folder)/'settings.json',{},{});w.poll_timer.stop()
            try:
                c=HeadingCorrector([0,0,1]);c.offset_rad=np.radians(29)
                w.live_fusion.bindings={'left_upper_arm':({},c)}
                w.heading_limit.setValue(60);w._apply_heading_limit()
                self.assertEqual(w.live_fusion.max_offset_deg,60);self.assertAlmostEqual(np.degrees(c.offset_rad),29)
                w._record_start_deadline=1;w.heading_limit.setValue(90);w._apply_heading_limit()
                self.assertEqual(w.live_fusion.max_offset_deg,60);w._record_start_deadline=None
                w.heading_limit.setValue(20)
                with patch.object(QtWidgets.QMessageBox,'warning') as warning:w._apply_heading_limit();warning.assert_called_once()
                self.assertEqual(w.heading_limit.value(),60)
            finally:w.live_fusion.bindings={};w.close()


if __name__=='__main__':unittest.main()
