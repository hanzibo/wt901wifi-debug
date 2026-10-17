"""Lateral/front stop gestures: pure clocks, Qt widgets, virtual UART only."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QWidget, QMessageBox, QInputDialog
from optusbproj_ubuntu.real_robot_stop_gesture import (
    GESTURE_KINDS, EndGesture, GestureConfig, gesture_indices, axis_name)
from optusbproj_ubuntu.real_robot_checkpoints import validate, CheckpointStore, gesture_axes
from optusbproj_ubuntu.real_robot_checkpoint_manager import CheckpointManager
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.real_robot_shared import SharedArmControl
from optusbproj_ubuntu.real_robot_protocol import encode
from test_real_robot_end import APP, BASE, limits, wait_for
import test_real_robot_end as end_tests
from test_real_robot_checkpoints import preset


def front_preset(mode='left_shared_ik'):
    data = preset(mode)
    data['stop_signal']['kind'] = 'front_raise'
    data['gesture'] = {axis: dict(angle=90., tolerance=5.) for axis in gesture_axes(mode, 'front_raise')}
    return validate(data)


class DetectionTests(unittest.TestCase):
    def update(self, g, now, angles=None, **kw):
        return g.update(BASE if angles is None else encode(angles, [0]*10), now,
                        following=kw.get('following', True), fresh=kw.get('fresh', True),
                        enabled=kw.get('enabled', (True,)*15), max_gap=.3)

    def test_all_four_combinations_and_one_shot(self):
        for kind in GESTURE_KINDS:
            for bilateral in (False, True):
                indices = gesture_indices(kind, bilateral=bilateral)
                expected = ((1, 3, 7, 9) if bilateral else (7, 9)) if kind == 'lateral_raise' else (
                    (0, 1, 3, 6, 7, 9) if bilateral else (6, 7, 9))
                self.assertEqual(indices, expected)
                g = EndGesture(GestureConfig((90.,)*len(indices), (5.,)*len(indices), .5, indices, kind))
                for t in (0., .2, .4, .49): self.assertFalse(self.update(g, t))
                self.assertTrue(self.update(g, .5)); self.assertFalse(self.update(g, .6))

    def test_each_front_axis_required_and_other_axes_ignored(self):
        for bilateral in (False, True):
            indices = gesture_indices('front_raise', bilateral=bilateral)
            config = GestureConfig((90.,)*len(indices), (5.,)*len(indices), .5, indices, 'front_raise')
            for i in indices:
                g = EndGesture(config); self.update(g, 0.)
                q = [90.]*15; q[i] = 95.01
                self.update(g, .1, q); self.assertIsNone(g.started)
                q[i] = 95.; self.update(g, .2, q); self.assertEqual(g.started, .2)
                mask = [True]*15; mask[i] = False
                self.update(g, .3, enabled=mask); self.assertIsNone(g.started)
            g = EndGesture(config)
            q = [90. if i in indices else 120. for i in range(15)]
            mask = [i in indices for i in range(15)]
            self.update(g, 0., q, enabled=mask)
            self.update(g, .25, q, enabled=mask)
            self.assertTrue(self.update(g, .5, q, enabled=mask))

    def test_front_phase_freshness_and_gap_reset(self):
        config = GestureConfig((90.,)*3, (5.,)*3, .5, (6, 7, 9), 'front_raise')
        for kw in ({'fresh': False}, {'following': False}):
            g = EndGesture(config); self.update(g, 0.); self.update(g, .1, **kw)
            self.assertIsNone(g.started)
        g = EndGesture(config); self.update(g, 0.); self.update(g, .4)
        self.assertEqual(g.started, .4)
        with self.assertRaises(ValueError): gesture_indices('unknown')
        with self.assertRaises(ValueError): GestureConfig((90.,)*4, kind='front_raise')

    def test_front_virtual_uart_stops_and_records_kind(self):
        for bilateral in (False, True):
            with self.subTest(bilateral=bilateral): self.check_front_uart(bilateral)

    def check_front_uart(self, bilateral):
        with tempfile.TemporaryDirectory() as directory:
            master, slave, worker, errors, writes = end_tests.WorkerReturnTests().setup_worker(directory)
            try:
                indices = gesture_indices('front_raise', bilateral=bilateral)
                config = GestureConfig((90.,)*len(indices), (5.,)*len(indices), .15, indices, 'front_raise')
                source = limits() if bilateral else limits().left
                target = [90.]*len(source.indices)
                control = SharedArmControl(BASE, source, target)
                control.gesture = EndGesture(config)
                worker.begin_control(control, target, time.monotonic()+.25)
                wait_for(lambda: control.phase == 'await_confirmation')
                self.assertFalse(control.gesture.triggered)
                worker.confirm_follow()
                deadline = time.monotonic()+2.
                while worker.active and time.monotonic() < deadline:
                    worker.update_arm_target(target, time.monotonic()+.25)
                    APP.processEvents(); time.sleep(.01)
                self.assertTrue(control.gesture.triggered); self.assertFalse(worker.active)
                self.assertFalse(errors); self.assertTrue(writes)
            finally:
                worker.shutdown(); wait_for(lambda: not worker.isRunning())
                os.close(master); os.close(slave)
            rows = [json.loads(s) for s in (Path(directory)/'control.jsonl').read_text().splitlines()]
            triggered = [r for r in rows if r['event'] == 'end_gesture_triggered']
            self.assertEqual(len(triggered), 1)
            self.assertEqual(triggered[0]['kind'], 'front_raise')
            self.assertEqual(triggered[0]['indices'], list(indices))
            self.assertIn('前举', triggered[0]['reason'])


class GuiModeTests(unittest.TestCase):
    def setUp(self):
        self.viewer = QWidget(); self.viewer.engine = None; self.viewer.replay_mode = False
        self.d = RealRobotModesDialog(self.viewer); self.d.timer.stop()
        self.d.mode_select.setCurrentIndex(2)
        self.end = self.d.end_session
        self.temp = tempfile.TemporaryDirectory()
        self.d.checkpoints.store = CheckpointStore(self.temp.name)

    def tearDown(self):
        self.d.busy = False; self.d.close(); self.viewer.close(); self.temp.cleanup()

    def test_visible_axes_exclusive_and_confirmation_invalidated(self):
        for mode in (2, 3):
            self.d.mode_select.setCurrentIndex(mode)
            for kind in GESTURE_KINDS:
                self.end.kind.setCurrentIndex(self.end.kind.findData(kind))
                selected = gesture_indices(kind, bilateral=mode == 3)
                self.assertEqual(self.end.indices(), selected)
                for i, row in self.end.gesture_rows.items():
                    self.assertEqual(not row[0].isHidden(), i in selected)
                    self.assertEqual(row[0].isEnabled(), i in selected)
                with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes): self.end.configure(False)
                self.end.enable.setChecked(True)
                source = limits() if mode == 3 else limits().left
                self.assertEqual(self.end.gesture(source).config.kind, kind)
        self.end.kind.setCurrentIndex(0)
        self.assertFalse(self.end.configured); self.assertIsNone(self.end.reference)
        self.d.busy = True; self.d._controls()
        self.assertFalse(self.end.kind.isEnabled())

    def test_front_permissions_and_parameter_changes_require_confirmation(self):
        self.end.kind.setCurrentIndex(1); self.end.enable.setChecked(True)
        with patch.object(QMessageBox, 'warning', return_value=QMessageBox.Yes): self.end.configure(False)
        with self.assertRaises(ValueError):
            self.end.gesture(replace(limits().left, enabled=(False, True, True, True, True, True)))
        self.end.gesture_rows[6][0].setValue(100.)
        with self.assertRaises(ValueError): self.end.gesture(limits().left)

    def test_checkpoint_switch_kind_transaction_dirty_and_no_runtime(self):
        for mode, name in ((2, 'left_shared_ik'), (3, 'bilateral_shared_ik')):
            self.d.mode_select.setCurrentIndex(mode)
            data = front_preset(name); data['stop_signal']['enabled'] = True
            self.assertIn('侧举 → 前举', self.d.checkpoints.differences(data))
            self.d.checkpoints.apply(data)
            self.assertEqual(self.end.kind.currentData(), 'front_raise')
            self.assertEqual(self.end.checkpoint_values(), data['gesture'])
            self.assertFalse(self.end.configured); self.assertFalse(self.d.safety.isChecked())
            self.assertIsNone(self.d.worker); self.assertFalse(self.d.auto_start.authorized)
            self.assertIn('数值一致', self.d.checkpoints.label.text())
            self.end.kind.setCurrentIndex(0)
            self.assertIn('已修改', self.d.checkpoints.label.text())
            self.d.checkpoints.apply(data)
            self.assertIn('不再参与', self.d.checkpoints.differences(preset(name)))
            self.d.checkpoints.apply(preset(name))
            self.assertEqual(self.end.kind.currentData(), 'lateral_raise')

    def test_cross_kind_invalid_load_does_not_partially_change_ui(self):
        before = self.d.checkpoints.snapshot()
        data = front_preset(); data['joints']['L3']['speed'] = 20.
        self.end.gesture_rows[6][1].setMaximum(4.)
        with self.assertRaises(ValueError): self.d.checkpoints.apply(data)
        self.assertEqual(self.d.checkpoints.snapshot()['gesture'], before['gesture'])
        self.assertEqual(self.end.kind.currentData(), 'lateral_raise')
        self.assertEqual(self.d.axis_rows[2][3].value(), before['joints']['L3']['speed'])

    def test_manager_front_save_as_load_and_only_active_axes(self):
        for mode in (2, 3):
            self.d.mode_select.setCurrentIndex(mode)
            m = CheckpointManager(self.d.checkpoints)
            try:
                m.stop_kind.setCurrentIndex(1)
                m.gesture_widgets['L1']['angle'].setValue(105.)
                self.assertFalse(m.gesture_widgets['L1']['angle'].isHidden())
                m.save(); original = m.read()
                self.assertEqual(original['stop_signal']['kind'], 'front_raise')
                self.assertEqual(len(original['gesture']), 3 if mode == 2 else 6)
                with patch.object(QInputDialog, 'getText', return_value=('前举副本', True)): m.save_as()
                self.assertNotEqual(m.read()['id'], original['id'])
                with patch.object(QMessageBox, 'exec', return_value=QMessageBox.Yes): m.load()
                self.assertEqual(self.end.kind.currentData(), 'front_raise')
                self.assertEqual(self.end.gesture_rows[6][0].value(), 105.)
                m.stop_kind.setCurrentIndex(0)
                self.assertTrue(m.gesture_widgets['L1']['angle'].isHidden())
                self.assertNotIn('L1', m.read()['gesture'])
            finally:
                m.dirty = False; m.close()


class MigrationTests(unittest.TestCase):
    def test_versions_1_to_3_migrate_without_rewriting_or_arming(self):
        for version in (1, 2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                old = preset(); old['schema_version'] = version
                del old['input_settings']
                del old['stop_signal']['kind']
                if version == 1: del old['stop_signal']; del old['return_settings']
                if version < 3: del old['start_settings']
                store = CheckpointStore(directory); path = store.path(old['mode'], old['id'])
                path.parent.mkdir(parents=True); path.write_text(json.dumps(old))
                before = path.read_bytes(); migrated = store.load(old['mode'], old['id'])
                self.assertEqual(migrated['schema_version'], 5)
                self.assertEqual(migrated['stop_signal']['kind'], 'lateral_raise')
                self.assertEqual(migrated['gesture'], old['gesture'])
                self.assertEqual(path.read_bytes(), before)

    def test_schema_requires_exact_active_axes_and_known_kind(self):
        for mode in ('left_shared_ik', 'bilateral_shared_ik'):
            front = front_preset(mode)
            for mutate in (lambda d: d['stop_signal'].update(kind='unknown'),
                           lambda d: d['stop_signal'].update(kind='lateral_raise'),
                           lambda d: d['gesture'].pop('L1'),
                           lambda d: d['gesture'].update(L3=dict(angle=90., tolerance=5.)),
                           lambda d: d['gesture']['L1'].update(angle=300.)):
                bad = deepcopy(front); mutate(bad)
                with self.assertRaises(ValueError): validate(bad)
