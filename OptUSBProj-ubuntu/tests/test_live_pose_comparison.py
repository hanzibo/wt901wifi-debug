"""Two read-only render views share integration, B, and paired zero frames."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets
from scipy.spatial.transform import Rotation

from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT, GYRO_RGBD
from optusbproj_ubuntu.rgbd_fusion import ARM_SEGMENTS
from optusbproj_ubuntu.skeleton import PARTS, PART_BY_KEY
from optusbproj_ubuntu.fusion_gui import FusionWindow


def rz(deg):
    return Rotation.from_euler('z', deg, degrees=True).as_matrix()


def populate(engine):
    for i, part in enumerate(ARM_SEGMENTS):
        device = str(i)
        engine.assignments[part] = device
        engine.feed(SimpleNamespace(device_id=device, roll=0., pitch=0., yaw=0.,
            hour=0, minute=0, second=0, millisecond=0,
            gyro_x=0., gyro_y=0., gyro_z=0., acc_x=0., acc_y=0., acc_z=1., battery_percent=100))
        engine.set_continuous_basis(part, np.eye(3))
        engine.mode_current[device][GYRO_TILT] = rz(30)
        engine.mode_current[device][GYRO_RGBD] = rz(50)
        engine.set_neutral_for_mode(device, GYRO_TILT, rz(10))
        engine.set_neutral_for_mode(device, GYRO_RGBD, rz(15))
        engine._sync_active(device)


def view(engine, mode):
    return engine.transforms(mode_overrides={p:mode for p in ARM_SEGMENTS})


class RenderViewTests(unittest.TestCase):
    def test_noncommuting_conjugation_and_constant_correction_cancellation(self):
        engine = PoseEngine();populate(engine)
        part = 'left_upper_arm';device = engine.assignments[part]
        n = Rotation.from_euler('xyz',[10,25,40],degrees=True).as_matrix()
        r = Rotation.from_euler('xyz',[55,-15,70],degrees=True).as_matrix()
        b = Rotation.from_euler('xyz',[-5,20,60],degrees=True).as_matrix()
        h0, h1 = rz(35), rz(65)
        engine.set_continuous_basis(part,b)
        engine.set_neutral_for_mode(device,GYRO_TILT,n)
        engine.set_neutral_for_mode(device,GYRO_RGBD,h0@n)
        engine.mode_current[device][GYRO_TILT] = r
        engine.mode_current[device][GYRO_RGBD] = h0@r
        index = PART_BY_KEY[part]
        np.testing.assert_allclose(view(engine,GYRO_TILT)[index][:3,:3],
                                   view(engine,GYRO_RGBD)[index][:3,:3],atol=1e-12)
        engine.mode_current[device][GYRO_RGBD] = h1@r
        expected = b @ (h0@n).T @ h1 @ r @ b.T
        np.testing.assert_allclose(view(engine,GYRO_RGBD)[index][:3,:3],expected,atol=1e-12)

    def test_paired_neutrals_and_readonly(self):
        engine = PoseEngine(); populate(engine)
        before = deepcopy((engine.algorithm_modes, engine.current, engine.neutral,
                           engine.mode_current, engine.mode_neutral))
        raw, fused = view(engine, GYRO_TILT), view(engine, GYRO_RGBD)
        for part in ARM_SEGMENTS:
            index = PART_BY_KEY[part]
            np.testing.assert_allclose(raw[index][:3,:3], rz(20), atol=1e-12)
            np.testing.assert_allclose(fused[index][:3,:3], rz(35), atol=1e-12)
        self.assertEqual(before[0], engine.algorithm_modes)
        for original, actual in zip(before[1:], (engine.current, engine.neutral,
                                                engine.mode_current, engine.mode_neutral)):
            for device, value in original.items():
                if isinstance(value, dict):
                    for mode in value:np.testing.assert_array_equal(value[mode], actual[device][mode])
                else:np.testing.assert_array_equal(value, actual[device])

    def test_zero_resets_both_views_and_does_not_change_basis(self):
        engine = PoseEngine(); populate(engine)
        before = deepcopy(engine.continuous_basis)
        engine.zero()
        for mode in (GYRO_TILT, GYRO_RGBD):
            for part in ARM_SEGMENTS:
                np.testing.assert_allclose(view(engine, mode)[PART_BY_KEY[part]][:3,:3], np.eye(3), atol=1e-12)
        for part in before:np.testing.assert_array_equal(before[part], engine.continuous_basis[part])

    def test_missing_override_does_not_fall_back_to_active_and_invalid_rejected(self):
        engine = PoseEngine(); populate(engine)
        part = 'left_upper_arm'; device = engine.assignments[part]
        engine.mode_current[device].pop(GYRO_RGBD)
        matrices = view(engine, GYRO_RGBD)
        np.testing.assert_array_equal(matrices[PART_BY_KEY[part]][:3,:3], np.eye(3))
        with self.assertRaises(ValueError):engine.transforms(mode_overrides={part:'unknown'})
        with self.assertRaises(ValueError):engine.transforms(mode_overrides={'unknown':GYRO_TILT})

    def test_default_view_unchanged_and_nonarm_sources_shared(self):
        engine = PoseEngine(); populate(engine)
        normal = engine.transforms()
        explicit = engine.transforms(mode_overrides={})
        for a,b in zip(normal, explicit):np.testing.assert_array_equal(a,b)
        raw, fused = view(engine, GYRO_TILT), view(engine, GYRO_RGBD)
        for i,part in enumerate(PARTS):
            if part.key in ('torso', 'head'):
                np.testing.assert_array_equal(raw[i],fused[i])


class Source:
    def seen_devices(self, max_age=3):return {}
    def seen_endpoints(self, max_age=3):return {}
    def close(self):pass


class LiveComparisonGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_toggle_matrices_axes_and_no_control_mutation(self):
        with TemporaryDirectory() as folder, patch.object(FusionWindow, 'start_camera'):
            w = FusionWindow(Source(), Path(folder)/'settings.json', {}, {}, camera_source='orbbec')
            w.poll_timer.stop();w.rgbd_timer.stop()
            try:
                populate(w.engine)
                original = w.engine.transforms()
                modes = deepcopy(w.engine.algorithm_modes)
                currents = deepcopy(w.engine.mode_current)
                filters = dict(w.engine._tilt_filters)
                self.assertFalse(w.live_compare_checkbox.isChecked())
                w.live_compare_checkbox.setChecked(True)
                for actors, mode, y in ((w.actors,GYRO_TILT,-1.), (w.live_compare_actors,GYRO_RGBD,1.)):
                    shift = np.eye(4);shift[1,3] = y
                    for actor,matrix in zip(actors,view(w.engine,mode)):
                        np.testing.assert_allclose(actor.user_matrix,shift@matrix,atol=1e-12)
                self.assertIn('未标定', w.live_compare_labels[1].text())
                w.axes_checkbox.setChecked(True)
                self.assertTrue(all(a.GetVisibility() for a in w.live_compare_axis_actors))
                w.live_compare_checkbox.setChecked(False)
                self.assertTrue(all(not a.GetVisibility() for a in w.live_compare_actors+w.live_compare_axis_actors))
                for actor,matrix in zip(w.actors,original):np.testing.assert_allclose(actor.user_matrix,matrix)
                self.assertEqual(modes,w.engine.algorithm_modes)
                self.assertEqual(filters,w.engine._tilt_filters)
                for device in currents:
                    for mode in currents[device]:np.testing.assert_array_equal(currents[device][mode],w.engine.mode_current[device][mode])
                w.live_compare_checkbox.setChecked(True)
                self.assertEqual(len(w.live_compare_actors),len(PARTS))  # no reallocation on toggles
                w._zero_all();w._render_pose()
                for actor in w.actors+w.live_compare_actors:np.testing.assert_allclose(actor.user_matrix[:3,:3],np.eye(3),atol=1e-12)
            finally:w.close()

    def test_status_for_standby_frozen_expired_and_invalidated(self):
        with TemporaryDirectory() as folder, patch.object(FusionWindow, 'start_camera'):
            w = FusionWindow(Source(),Path(folder)/'settings.json',{},{});w.poll_timer.stop();w.rgbd_timer.stop()
            try:
                populate(w.engine);w.engine.fusion_ready.update(w.engine.assignments[p] for p in ARM_SEGMENTS)
                w.live_compare_checkbox.setChecked(True)
                self.assertIn('纠偏待机',w.live_compare_labels[1].text())
                for part in ARM_SEGMENTS:w.engine.set_algorithm_mode(part,GYRO_RGBD)
                w.rgbd_pose_enabled.setChecked(True)
                w.last_pose_estimate=SimpleNamespace(captured_mono_ns=time.monotonic_ns())
                w.freeze_fusion.setChecked(True);w._update_live_comparison_status()
                self.assertIn('视觉冻结',w.live_compare_labels[1].text())
                w.freeze_fusion.setChecked(False)
                w.last_pose_estimate.captured_mono_ns=0;w._update_live_comparison_status()
                self.assertIn('过期',w.live_compare_labels[1].text())
                w.engine.fusion_ready.clear();w.engine.received_at={d:0 for d in w.engine.received_at}
                w._update_live_comparison_status()
                self.assertIn('已停用',w.live_compare_labels[1].text())
                self.assertIn('数据过期',w.live_compare_labels[0].text())
            finally:w.close()
