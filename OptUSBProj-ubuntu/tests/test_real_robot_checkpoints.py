"""Checkpoint storage and GUI tests; never use physical serial devices."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PySide6 import QtWidgets
from optusbproj_ubuntu.real_robot_arm import ArmLimits
from optusbproj_ubuntu.real_robot_bilateral import right_defaults
from optusbproj_ubuntu.real_robot_checkpoints import (
    CheckpointStore, MODES, FIELDS, create, validate, gesture_axes)
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.real_robot_checkpoint_manager import CheckpointManager

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def preset(mode='left_shared_ik'):
    joints = {}
    for side in MODES[mode]:
        limits = ArmLimits() if side == 'left' else right_defaults()
        for i in range(6):
            joints[f'{side[0].upper()}{i+1}'] = {field: getattr(limits, field)[i] for field in FIELDS}
    return create(mode, joints, {axis: dict(angle=90., tolerance=5.) for axis in gesture_axes(mode)},
                  name='低速对比 A', notes='中文备注')


class StoreTests(unittest.TestCase):
    def test_roundtrip_mode_separation_and_recoverable_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            for mode in MODES:
                saved = store.save(preset(mode))
                self.assertEqual(store.load(mode, saved['id']), saved)
                self.assertEqual(len(store.list(mode)[0]), 1)
                saved['name'] = '修改后的名称'
                saved = store.save(saved, overwrite=True)
                self.assertEqual(store.load(mode, saved['id'])['name'], saved['name'])
                trash = store.delete(mode, saved['id'])
                self.assertTrue(trash.exists())
                self.assertEqual(store.list(mode), ([], []))
                self.assertEqual(json.loads(trash.read_text())['id'], saved['id'])

    def test_rejects_invalid_or_incomplete_values(self):
        changes = [lambda d: d.update(mode='wrong'), lambda d: d.update(schema_version=99),
                   lambda d: d.update(id='../escape'), lambda d: d.update(name=' '),
                   lambda d: d.update(notes='x'*2001), lambda d: d.update(created_at='bad'),
                   lambda d: d['joints'].pop('L1'),
                   lambda d: d['joints']['L1'].update(lower=101., upper=100.),
                   lambda d: d['joints']['L1'].update(upper=999.),
                   lambda d: d['joints']['L1'].update(speed=0.),
                   lambda d: d['joints']['L1'].update(speed=251.),
                   lambda d: d['joints']['L1'].update(speed=True),
                   lambda d: d['joints']['L1'].update(speed='20'),
                   lambda d: d['joints']['L1'].update(speed=float('nan')),
                   lambda d: d['joints']['L1'].update(speed=float('inf')),
                   lambda d: d['joints']['L1'].update(speed=10**1000),
                   lambda d: d['joints']['L1'].update(speed=1.234),
                   lambda d: d['gesture'].pop('L2'),
                   lambda d: d['gesture']['L2'].update(angle=80.),
                   lambda d: d['gesture']['L2'].update(tolerance=0.),
                   lambda d: d['gesture']['L2'].update(tolerance=31.),
                   lambda d: d.update(enabled=True)]
        for mutate in changes:
            data = preset(); mutate(data)
            with self.subTest(data=str(data)):
                with self.assertRaises(ValueError): validate(data)
        with self.assertRaises(ValueError): validate(preset(), 'bilateral_shared_ik')

    def test_invalid_files_do_not_hide_good_presets(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory); good = store.save(preset())
            folder = Path(directory)/good['mode']
            (folder/'broken.json').write_text('{broken')
            bad = preset(); path = store.path(bad['mode'], bad['id'])
            path.write_text(json.dumps(bad)[:-1]+', "name": "duplicate"}')
            rows, errors = store.list(good['mode'])
            self.assertEqual([d['id'] for d in rows], [good['id']])
            self.assertEqual(len(errors), 2)
            with self.assertRaises(ValueError): store.path(good['mode'], '../../bad')

    def test_atomic_save_failure_preserves_original_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory); saved = store.save(preset())
            path = store.path(saved['mode'], saved['id']); before = path.read_bytes()
            saved['name'] = '未写入'
            with patch('optusbproj_ubuntu.real_robot_checkpoints.os.replace', side_effect=OSError('磁盘故障')):
                with self.assertRaises(OSError): store.save(saved, overwrite=True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_concurrent_edit_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory); saved = store.save(preset())
            modified = deepcopy(saved); modified['name'] = '另一个窗口'
            store.save(modified, overwrite=True)
            with self.assertRaises(ValueError): store.save(saved, overwrite=True)
            with self.assertRaises(ValueError): store.save(saved)
            other = preset(); link = store.path(other['mode'], other['id'])
            link.symlink_to(store.path(saved['mode'], saved['id']))
            with self.assertRaises(ValueError): store.load(other['mode'], other['id'])

    def test_v1_migration_does_not_rewrite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            old = preset(); old['schema_version'] = 1
            del old['input_settings']
            del old['stop_signal']; del old['return_settings']; del old['start_settings']
            path = store.path(old['mode'], old['id']); path.parent.mkdir(parents=True)
            path.write_text(json.dumps(old))
            before = path.read_bytes()
            migrated = store.load(old['mode'], old['id'])
            self.assertEqual(migrated['schema_version'], 5)
            self.assertEqual(migrated['stop_signal'], dict(enabled=False, seconds=7., kind='lateral_raise'))
            self.assertEqual(migrated['return_settings']['order'], [1, 3, 5, 6, 4, 2])
            self.assertEqual(path.read_bytes(), before)
            store.save(migrated, overwrite=True)
            self.assertEqual(json.loads(path.read_text())['schema_version'], 5)

    def test_v2_start_settings_migration_and_strict_validation(self):
        old = preset(); old['schema_version'] = 2; del old['start_settings']
        del old['input_settings']
        del old['stop_signal']['kind']
        self.assertEqual(validate(old)['start_settings'], dict(tolerance=2., countdown=10., settle=2.))
        self.assertNotIn('start_settings', old)
        for key, value in (('tolerance', 0.), ('tolerance', 31.), ('countdown', 0.),
                           ('countdown', 121.), ('settle', -1.), ('settle', 31.),
                           ('settle', True), ('countdown', float('inf')), ('tolerance', 1.234)):
            data = preset(); data['start_settings'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): validate(data)
        data = preset(); data['start_settings']['authorized'] = True
        with self.assertRaises(ValueError): validate(data)

    def test_input_switches_strict_roundtrip_and_v4_defaults(self):
        for mode in MODES:
            data = preset(mode)
            data['input_settings'] = dict(manual_mapping=True,
                                         palm_offset_enabled={s:True for s in MODES[mode]})
            self.assertEqual(validate(data)['input_settings'], data['input_settings'])
            old = deepcopy(data); old['schema_version']=4; del old['input_settings']
            self.assertEqual(validate(old)['input_settings'], dict(manual_mapping=False,
                             palm_offset_enabled={s:False for s in MODES[mode]}))
            self.assertEqual(old['schema_version'],4)
            for change in (lambda d:d['input_settings'].update(manual_mapping=1),
                           lambda d:d['input_settings']['palm_offset_enabled'].update(left='true'),
                           lambda d:d['input_settings']['palm_offset_enabled'].pop('left'),
                           lambda d:d['input_settings'].update(authorized=True)):
                invalid=deepcopy(data); change(invalid)
                with self.assertRaises(ValueError):validate(invalid)

    def test_new_options_validation_and_roundtrip(self):
        changes = [('stop_signal', 'enabled', 1), ('stop_signal', 'seconds', 0.),
                   ('stop_signal', 'seconds', 120.01), ('stop_signal', 'seconds', True),
                   ('stop_signal', 'seconds', float('nan')), ('stop_signal', 'seconds', 1.234),
                   ('return_settings', 'order', [1, 2, 3, 4, 5, 5]),
                   ('return_settings', 'order', [True, 2, 3, 4, 5, 6]),
                   ('return_settings', 'order', '1,3,5,6,4,2'),
                   ('return_settings', 'speed', 0.), ('return_settings', 'acceleration', 251.),
                   ('return_settings', 'dwell', -.1), ('return_settings', 'dwell', 10.01)]
        for section, field, value in changes:
            with self.subTest(field=field, value=value):
                data = preset(); data[section][field] = value
                with self.assertRaises(ValueError): validate(data)
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            for mode in MODES:
                data = preset(mode)
                data['stop_signal'] = dict(enabled=True, seconds=2.5, kind='lateral_raise')
                data['return_settings'] = dict(order=[6, 5, 4, 3, 2, 1], speed=35., acceleration=60., dwell=1.2)
                saved = store.save(data)
                self.assertEqual(store.load(mode, saved['id']), saved)


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.viewer = QtWidgets.QWidget(); self.viewer.engine = None; self.viewer.replay_mode = False
        self.dialog = RealRobotModesDialog(self.viewer); self.dialog.timer.stop()
        self.dialog.mode_select.setCurrentIndex(2)
        self.session = self.dialog.checkpoints
        self.session.store = CheckpointStore(self.directory.name)

    def tearDown(self):
        self.dialog.worker = None; self.dialog.ik_worker = None; self.dialog.busy = False
        self.dialog.close(); self.viewer.close(); self.directory.cleanup()

    def test_load_is_parameter_only_preserves_permissions_and_other_options(self):
        d = self.dialog
        d.axis_rows[0][0].setChecked(False)
        d.axis_rows[5][5].setCurrentIndex(1)
        d.frequency.setValue(13.)
        d.shared_offset.setChecked(True); d.shared_offset_axis.setCurrentText('z')
        data = self.session.snapshot('测试')
        data['joints']['L1']['speed'] = 30.
        data['gesture']['L4']['angle'] = 5.
        d.end_session.enable.setChecked(True); d.end_session.configured = True
        d.safety.setChecked(True)
        d._arm_event = Mock()
        self.session.apply(data)
        self.assertEqual(d.axis_rows[0][3].value(), 30.)
        self.assertEqual(d.end_session.gesture_rows[9][0].value(), 5.)
        self.assertFalse(d.axis_rows[0][0].isChecked())
        self.assertEqual(d.axis_rows[5][5].currentIndex(), 1)
        self.assertEqual(d.frequency.value(), 13.)
        self.assertTrue(d.shared_offset.isChecked())
        self.assertIsNone(d.arm_capture); self.assertIsNone(d.capture)
        self.assertIsNone(d.arm_sample); self.assertFalse(d.safety.isChecked())
        self.assertFalse(d.end_session.enable.isChecked()); self.assertFalse(d.end_session.configured)
        self.assertIsNone(d.worker); self.assertIsNone(d.ik_worker)
        self.assertIn('数值一致', self.session.label.text())

    def test_input_switches_snapshot_apply_dirty_and_manager_copy(self):
        d=self.dialog
        for mode in (2,3):
            d.mode_select.setCurrentIndex(mode)
            d.manual.setChecked(True); d.shared_offset.setChecked(True)
            d.right_options[1].setChecked(True)
            d.shared_offset_axis.setCurrentText('z'); d.shared_offset_angle.setValue(85.)
            d.right_options[2].setCurrentText('y'); d.right_options[3].setValue(-70.)
            data=self.session.snapshot('开关测试')
            d.manual.setChecked(False); d.shared_offset.setChecked(False); d.right_options[1].setChecked(False)
            self.assertIn('使用手动映射',self.session.differences(data))
            self.session.apply(data)
            self.assertTrue(d.manual.isChecked()); self.assertTrue(d.shared_offset.isChecked())
            self.assertEqual(d.right_options[1].isChecked(),mode==3)
            self.assertEqual(d.shared_offset_axis.currentText(),'z'); self.assertEqual(d.shared_offset_angle.value(),85.)
            self.assertEqual(d.right_options[2].currentText(),'y'); self.assertEqual(d.right_options[3].value(),-70.)
            self.assertFalse(d.safety.isChecked()); self.assertIsNone(d.worker); self.assertIsNone(d.ik_worker)
            self.assertIn('数值一致',self.session.label.text())
            d.manual.setChecked(False); self.assertIn('已修改',self.session.label.text())
            d.manual.setChecked(True); d.shared_offset.setChecked(False)
            self.assertIn('已修改',self.session.label.text())
            d.shared_offset.setChecked(True)
            manager=CheckpointManager(self.session)
            try:
                self.assertTrue(manager.manual_mapping.isChecked())
                self.assertTrue(manager.palm_offsets['left'].isChecked())
                manager.manual_mapping.setChecked(False); manager.palm_offsets['left'].setChecked(False)
                manager.save(); saved=manager.read()
                with patch.object(QtWidgets.QInputDialog,'getText',return_value=('开关副本',True)):manager.save_as()
                copied=self.session.store.load(self.session.mode(),manager.selected_id)
                self.assertEqual(copied['input_settings'],saved['input_settings'])
                self.assertTrue(d.manual.isChecked())  # Saving does not apply.
                self.session.apply(copied)
                self.assertFalse(d.manual.isChecked()); self.assertFalse(d.shared_offset.isChecked())
            finally:
                manager.dirty=False; manager.close()

    def test_existing_dirty_label_updates(self):
        d=self.dialog
        self.session.apply(self.session.snapshot())
        original=d.axis_rows[0][3].value()
        d.axis_rows[0][3].setValue(31.)
        self.assertIn('已修改', self.session.label.text())
        d.axis_rows[0][3].setValue(original)
        self.assertIn('数值一致', self.session.label.text())

    def test_invalid_cross_mode_and_widget_bounds_leave_everything_unchanged(self):
        original = self.session.parameters()
        for mutate in (lambda d: d['joints']['L6'].update(speed=999.),
                       lambda d: d['gesture']['L4'].update(tolerance=0.)):
            data = self.session.snapshot(); mutate(data)
            with self.assertRaises(ValueError): self.session.apply(data)
            self.assertEqual(original, self.session.parameters())
        with self.assertRaises(ValueError): self.session.apply(preset('bilateral_shared_ik'))
        data = self.session.snapshot(); data['joints']['L1']['speed'] = 60.
        self.dialog.axis_rows[-1][3].setMaximum(99.)
        original = self.session.parameters()
        with self.assertRaises(ValueError): self.session.apply(data)
        self.assertEqual(original, self.session.parameters())

    def test_runtime_and_return_gates(self):
        d = self.dialog; data = self.session.snapshot()
        for owner, attr, value, reset in ((d, 'busy', True, False), (d, 'ik_worker', object(), None),
                                         (d.end_session, 'returning', True, False),
                                         (d.end_session, 'await_home', True, False)):
            setattr(owner, attr, value)
            self.assertFalse(self.session.available())
            with self.assertRaises(ValueError): self.session.apply(data)
            setattr(owner, attr, reset)
        d.auto_start.phase = 'countdown'
        with self.assertRaises(ValueError): self.session.apply(data)
        d.auto_start.phase = 'idle'
        worker = Mock(active=False, pending_start=None, quit_requested=False)
        worker.return_snapshot.return_value = (None, None, True); d.worker = worker
        with self.assertRaises(ValueError): self.session.apply(data)
        worker.begin_control.assert_not_called(); worker.update_arm_target.assert_not_called()
        d.worker = None
        for mode in (0, 1):
            d.mode_select.setCurrentIndex(mode)
            self.assertFalse(self.session.available())

    def test_bilateral_roundtrip_and_diff(self):
        self.dialog.mode_select.setCurrentIndex(3)
        data = self.session.snapshot()
        self.assertEqual(len(data['joints']), 12); self.assertEqual(len(data['gesture']), 4)
        data['joints']['R1']['speed'] = 35.; data['gesture']['R4']['angle'] = 10.
        self.assertIn('R1 速度', self.session.differences(data))
        self.session.apply(data)
        self.assertEqual(self.session.parameters(), (data['joints'], data['gesture']))

    def test_manager_selection_and_delete_are_parameter_only(self):
        data = self.session.store.save(preset())
        manager = CheckpointManager(self.session)
        original = self.session.parameters()
        try:
            manager.items.setCurrentRow(0)
            self.assertEqual(manager.selected_id, data['id'])
            self.assertEqual(self.session.parameters(), original)
            with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Cancel):
                manager.delete()
            self.assertEqual(len(self.session.store.list(self.session.mode())[0]), 1)
            with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
                manager.delete()
            self.assertEqual(self.session.store.list(self.session.mode())[0], [])
            self.assertEqual(len(list((Path(self.directory.name)/'trash').rglob('*.json'))), 1)
            self.assertEqual(self.session.parameters(), original)
        finally:
            manager.dirty = False; manager.close()

    def test_manager_warns_about_bad_files_and_confirms_discard(self):
        self.session.store.save(preset())
        (Path(self.directory.name)/self.session.mode()/'broken.json').write_text('bad')
        manager = CheckpointManager(self.session)
        try:
            self.assertIn('broken.json', manager.file_errors.text())
            manager.name.setText('未保存编辑')
            with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Cancel):
                manager.items.setCurrentRow(0)
            self.assertEqual(manager.name.text(), '未保存编辑')
            self.assertIsNone(manager.selected_id)
            with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
                manager.items.setCurrentRow(0)
            self.assertIsNotNone(manager.selected_id)
        finally:
            manager.dirty = False; manager.close()

    def test_invalid_current_values_can_be_corrected_in_editor(self):
        self.dialog.axis_rows[1][1].setValue(120.)  # T reference 90 is now invalid.
        manager = CheckpointManager(self.session)
        try:
            manager.save()
            self.assertIsNone(manager.selected_id)
            self.assertIn('参考角', manager.message.text())
            manager.gesture_widgets['L2']['angle'].setValue(130.)
            manager.save()
            self.assertIsNotNone(manager.selected_id)
            self.assertEqual(self.dialog.end_session.gesture_rows[7][0].value(), 90.)
        finally:
            manager.dirty = False; manager.close()

    def test_manager_new_save_edit_save_as_and_load(self):
        manager = CheckpointManager(self.session)
        try:
            manager.name.setText('实验甲'); manager.save()
            ident = manager.selected_id
            self.assertIsNotNone(ident)
            self.assertEqual(self.dialog.axis_rows[0][3].value(), 100.)
            manager.joint_widgets['L1']['speed'].setValue(40.)
            manager.save()
            self.assertEqual(self.session.store.load(self.session.mode(), ident)['joints']['L1']['speed'], 40.)
            with patch.object(QtWidgets.QInputDialog, 'getText', return_value=('实验乙', True)):
                manager.save_as()
            self.assertNotEqual(manager.selected_id, ident)
            self.assertEqual(len(self.session.store.list(self.session.mode())[0]), 2)
            with patch.object(QtWidgets.QMessageBox, 'exec', return_value=QtWidgets.QMessageBox.Yes):
                manager.load()
            self.assertEqual(self.dialog.axis_rows[0][3].value(), 40.)
            self.assertIsNone(self.dialog.worker)
        finally:
            manager.dirty = False; manager.close()

    def test_cancel_load_and_manager_edit_do_not_touch_owner(self):
        manager = CheckpointManager(self.session)
        try:
            manager.save()
            data = manager.read(); data['joints']['L1']['speed'] = 35.
            manager.populate(data); manager.save()
            original = self.session.parameters()
            with patch.object(QtWidgets.QMessageBox, 'exec', return_value=QtWidgets.QMessageBox.Cancel):
                manager.load()
            self.assertEqual(original, self.session.parameters())
            manager.joint_widgets['L1']['speed'].setValue(20.)
            manager.load()
            self.assertIn('先保存', manager.message.text())
            self.assertEqual(original, self.session.parameters())
        finally:
            manager.dirty = False; manager.close()

    def test_new_settings_apply_dirty_and_safety(self):
        d = self.dialog; end = d.end_session
        for mode in (2, 3):
            d.mode_select.setCurrentIndex(mode)
            data = self.session.snapshot()
            data['stop_signal'] = dict(enabled=True, seconds=2.5, kind='lateral_raise')
            data['start_settings'] = dict(tolerance=5., countdown=8., settle=3.)
            data['return_settings'] = dict(order=[6, 5, 4, 3, 2, 1], speed=35., acceleration=60., dwell=1.2)
            end.side_order.setCurrentIndex(1); end.manual.setChecked(False)
            self.assertIn('持续秒数', self.session.differences(data))
            self.session.apply(data)
            self.assertEqual(d.auto_start.values(), data['start_settings'])
            self.assertFalse(d.auto_start.authorized); self.assertFalse(d.auto_start.pending)
            self.assertEqual(end.checkpoint_options(), {k: data[k] for k in ('stop_signal', 'return_settings')})
            self.assertFalse(end.configured); self.assertFalse(d.safety.isChecked())
            self.assertIsNone(d.worker)
            self.assertEqual(end.side_order.currentIndex(), 1); self.assertFalse(end.manual.isChecked())
            with self.assertRaises(ValueError): end.gesture(d._limits())
            self.assertEqual(end.gesture_config().seconds, 2.5)
            self.assertIn('数值一致', self.session.label.text())
            d.auto_start.widgets['countdown'].setValue(9.)
            self.assertIn('已修改', self.session.label.text())
            d.auto_start.widgets['countdown'].setValue(8.)
            end.seconds.setValue(3.)
            self.assertIn('已修改', self.session.label.text())
            end.progress(dict(gesture_elapsed=1., gesture_seconds=3.))
            self.assertIn('/ 3 秒', end.gesture_label.text())
            data['return_settings']['dwell'] = 11.
            before = end.checkpoint_options()
            with self.assertRaises(ValueError): self.session.apply(data)
            self.assertEqual(end.checkpoint_options(), before)

    def test_manager_new_settings_save_as_and_correct_bad_order(self):
        end = self.dialog.end_session
        end.order.setText('bad order')
        manager = CheckpointManager(self.session)
        try:
            manager.save()
            self.assertIsNone(manager.selected_id)
            manager.return_order.setText('1,3,5,6,4,2')
            manager.stop_enabled.setChecked(True); manager.stop_seconds.setValue(4.25)
            manager.return_widgets['speed'].setValue(40.)
            manager.return_widgets['acceleration'].setValue(65.)
            manager.return_widgets['dwell'].setValue(2.)
            manager.start_widgets['tolerance'].setValue(6.)
            manager.start_widgets['countdown'].setValue(12.)
            manager.start_widgets['settle'].setValue(4.)
            manager.save()
            original = manager.read()
            with patch.object(QtWidgets.QInputDialog, 'getText', return_value=('副本', True)):
                manager.save_as()
            copied = self.session.store.load(self.session.mode(), manager.selected_id)
            for field in ('stop_signal', 'return_settings', 'start_settings'):
                self.assertEqual(copied[field], original[field])
            self.assertEqual(end.order.text(), 'bad order')  # Save never applies.
            with patch.object(QtWidgets.QMessageBox, 'exec', return_value=QtWidgets.QMessageBox.Yes):
                manager.load()
            self.assertEqual(end.seconds.value(), 4.25)
            self.assertEqual(end.speed.value(), 40.)
        finally:
            manager.dirty = False; manager.close()


if __name__ == '__main__': unittest.main()
