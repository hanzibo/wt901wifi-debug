import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.combined_calibration import apply_joint, build_joint_report, imu_wearing, load_capture
from optusbproj_ubuntu.fusion_gui import FusionWindow
from optusbproj_ubuntu.pose import GYRO_TILT, GYRO_RGBD
from test_calibration_reuse import system, CAMERA, VISION
from test_pose_backend_gui import Source
from test_visual_wearing import fixture


def capture(path, corrupt=False, **kwargs):
    fixture(path,bilateral=True,corrupt_left_repeat=corrupt,**kwargs)
    with (path/'imu.csv').open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:row.update(roll=0,pitch=0,yaw=0)
    with (path/'imu.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=json.loads((path/'manifest.json').read_text());manifest['vision_config']=VISION;manifest['vision_epoch']='vision'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with (path/'vision.jsonl').open() as stream:vision=[json.loads(line) for line in stream]
    for i,row in enumerate(vision):row.update(completed_mono_ns=row['captured_mono_ns']+10_000_000,frame=i)
    (path/'vision.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in vision))


class CombinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_cli_both_methods_and_camera_independent_repeat(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir();capture(session)
            script=Path(__file__).resolve().parents[1]/'combined_calibration_analyze.py'
            for method in ('imu','visual'):
                output=root/method
                result=subprocess.run([sys.executable,str(script),str(session),'--output',str(output),'--wearing-method',method],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                report=json.loads((output/'report.json').read_text())
                self.assertEqual(len(report['wearing']['parts']),4)
                self.assertTrue(all(d['accepted'] for d in report['camera']['parts'].values()),report['camera']['parts'])
                self.assertTrue(any(d['accepted'] for d in report['wearing']['parts'].values()))
                for d in report['camera']['parts'].values():self.assertEqual(len(d['independent_validation']),3)
            camera=json.loads((root/'imu'/'camera'/'report.json').read_text())
            capture(session,corrupt=True)
            report=build_joint_report(session,camera,'imu')
            self.assertFalse(report['camera']['parts']['left_upper_arm']['accepted'])
            self.assertTrue(report['camera']['parts']['right_upper_arm']['accepted'])

    def test_incomplete_capture_rejected(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp);capture(path)
            guide=json.loads((path/'camera_guide.json').read_text());guide['status']='cancelled'
            (path/'camera_guide.json').write_text(json.dumps(guide))
            with self.assertRaises(ValueError):load_capture(path)

    def test_camera_failure_does_not_discard_wearing_results(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir();capture(session)
            (session/'vision.jsonl').write_text('')
            script=Path(__file__).resolve().parents[1]/'combined_calibration_analyze.py'
            result=subprocess.run([sys.executable,str(script),str(session),'--output',str(root/'output')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'output'/'report.json').read_text())
            self.assertTrue(any(d['accepted'] for d in report['wearing']['parts'].values()))
            self.assertFalse(any(d['accepted'] for d in report['camera']['parts'].values()))

    def joint(self):
        engine,fusion,profile=system()
        camera={'orientation_source':'live_runtime_pose','camera_epoch':'camera','vision_epoch':'vision','vision_config':VISION,'camera_up':[0,0,1],
                'parts':{p:deepcopy(d) for p,(d,_) in fusion.bindings.items()}}
        wearing={'camera_epoch':'camera','vision_epoch':'vision','vision_config':VISION,'parts':{}}
        for p,data in camera['parts'].items():
            wearing['parts'][p]=dict(data,basis_matrix=np.eye(3).tolist(),neutral_matrix=np.eye(3).tolist(),
                neutral_matrices_by_mode={GYRO_TILT:np.eye(3).tolist()},status='通过',accepted=True)
        report={'format':'wt901-combined-arm-camera-calibration-v1','session':'capture','wearing_method':'imu','wearing':wearing,'camera':camera}
        owner=SimpleNamespace(engine=engine,live_fusion=fusion,replay_mode=False,session=None,_record_start_deadline=None,
            x1_preview_dialog=None,camera_worker=SimpleNamespace(capture_epoch='camera'),_live_camera_epoch='camera')
        owner._check_vision_report=lambda r:None;owner.apply_camera_report=lambda r:fusion.install(r)
        return owner,report

    def test_transaction_partial_results_and_no_auto_switch(self):
        owner,report=self.joint();apply_joint(owner,report,['left_upper_arm','right_upper_arm'],['right_upper_arm'])
        self.assertNotIn('0',owner.engine.fusion_ready);self.assertIn('2',owner.engine.fusion_ready)
        self.assertEqual(owner.engine.algorithm_modes['right_upper_arm'],GYRO_TILT)
        self.assertEqual(owner.live_fusion.bindings['right_upper_arm'][1].offset_rad,0)
        before=deepcopy(owner.engine.mode_neutral)
        report['camera']['parts']['left_forearm']['reference_epoch']='old'
        with self.assertRaises(ValueError):apply_joint(owner,report,['left_upper_arm'],['left_forearm'])
        for d in before:np.testing.assert_equal(before[d][GYRO_TILT],owner.engine.mode_neutral[d][GYRO_TILT])
        report['camera']['parts']['left_forearm']['reference_epoch']=owner.engine.epochs['1']
        owner.apply_camera_report=lambda r:(_ for _ in ()).throw(ValueError('failure'))
        with self.assertRaises(ValueError):apply_joint(owner,report,['left_forearm'],['left_forearm'])
        for d in before:np.testing.assert_equal(before[d][GYRO_TILT],owner.engine.mode_neutral[d][GYRO_TILT])

    def test_gui_entry_results_and_safety(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                self.assertEqual(d.method.currentData(),'imu')
                self.assertEqual((d.delay.value(),d.preparation.value(),d.hold.value(),d.neutral.value(),d.validation.value()),(5,4,4,5,10))
                owner,report=self.joint();d.report=report;d.devices={p:r['device_id'] for p,r in report['wearing']['parts'].items()}
                d.analysis_path=Path(tmp)
                report['wearing']['parts']['left_forearm']['accepted']=False;report['wearing']['parts']['left_forearm']['status']='需重录'
                d.show_results();self.assertFalse(d.wear_checks['left_forearm'].isEnabled())
                self.assertTrue(d.camera_checks['left_forearm'].isEnabled());self.assertFalse(w.engine.continuous_basis)
                self.assertEqual(len(d.capture_steps()),11)
                self.assertFalse(d.enable_rgbd.isEnabled())
            finally:w.close()

    def test_oneclick_rgbd_preflight_atomicity_and_safety(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                engine,fusion,_=system();w.engine=engine;w.live_fusion=fusion
                d.enable_bilateral_rgbd()
                self.assertTrue(all(engine.algorithm_modes[p]==GYRO_TILT for p in fusion.bindings))
                _,report=self.joint()
                for data in report['camera']['parts'].values():data['reference_epoch']=engine.epochs[data['device_id']]
                d.report=report;d.devices={p:r['device_id'] for p,r in report['wearing']['parts'].items()}
                d.analysis_path=Path(tmp)
                d.show_results()
                with patch('PySide6.QtWidgets.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes),patch('optusbproj_ubuntu.combined_calibration_gui.apply_joint',side_effect=ValueError('invalid')):
                    d.apply_live()
                self.assertFalse(d.enable_rgbd.isEnabled())
                with patch('PySide6.QtWidgets.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes),patch('optusbproj_ubuntu.combined_calibration_gui.apply_joint'):
                    d.apply_live()
                self.assertTrue(d.enable_rgbd.isEnabled())
                with patch('optusbproj_ubuntu.combined_calibration_gui.save_config',side_effect=OSError('disk')):
                    d.enable_bilateral_rgbd()
                self.assertTrue(all(engine.algorithm_modes[p]==GYRO_TILT for p in fusion.bindings))
                w.x1_preview_dialog=SimpleNamespace(sender=object())
                d.enable_bilateral_rgbd()
                self.assertTrue(all(engine.algorithm_modes[p]==GYRO_TILT for p in fusion.bindings))
                w.x1_preview_dialog=None
                last=engine.assignments['right_forearm'];engine.fusion_ready.discard(last)
                d.enable_bilateral_rgbd()
                for part in fusion.bindings:
                    self.assertEqual(engine.algorithm_modes[part],GYRO_TILT if part=='right_forearm' else GYRO_RGBD)
                self.assertIn('右前臂',d.status.text());self.assertIn('3/4',d.status.text())
                engine.fusion_ready.add(last)
                d.enable_bilateral_rgbd()
                for p in fusion.bindings:
                    self.assertEqual(engine.algorithm_modes[p],GYRO_RGBD)
                    self.assertEqual(w.algorithm_combos[p].currentData(),GYRO_RGBD)
                self.assertIsNone(w.x1_preview_dialog)
            finally:w.close()

    def test_oneclick_only_actual_applied_camera_selection(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                engine,fusion,_=system();w.engine=engine;w.live_fusion=fusion
                _,report=self.joint()
                for data in report['camera']['parts'].values():data['reference_epoch']=engine.epochs[data['device_id']]
                report['camera']['parts']['right_forearm']['accepted']=False
                d.report=report;d.devices={p:r['device_id'] for p,r in report['wearing']['parts'].items()};d.analysis_path=Path(tmp)
                d.show_results()
                with patch('PySide6.QtWidgets.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes),patch('optusbproj_ubuntu.combined_calibration_gui.apply_joint'):
                    d.apply_live()
                # Even old ready state and later checkbox edits cannot enroll
                # a failed/unapplied part in this round's one-click action.
                d.camera_checks['right_forearm'].setChecked(True)
                d.enable_bilateral_rgbd()
                self.assertEqual(engine.algorithm_modes['right_forearm'],GYRO_TILT)
                self.assertIn('右前臂：相机未通过',d.status.text())
                self.assertIn('3/4',d.status.text())
                # Safety gate failure for every applied part produces no write.
                engine.fusion_ready.clear()
                with patch('optusbproj_ubuntu.combined_calibration_gui.save_config') as save:
                    d.enable_bilateral_rgbd();save.assert_not_called()
                self.assertIn('没有本轮已应用且仍有效',d.status.text())
            finally:w.close()

    def test_wearing_only_application_keeps_fusion_button_disabled(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                _,report=self.joint();d.report=report;d.devices={p:r['device_id'] for p,r in report['wearing']['parts'].items()};d.analysis_path=Path(tmp)
                d.show_results()
                for check in d.camera_checks.values():check.setChecked(False)
                with patch('PySide6.QtWidgets.QMessageBox.question',return_value=QtWidgets.QMessageBox.StandardButton.Yes),patch('optusbproj_ubuntu.combined_calibration_gui.apply_joint'):
                    d.apply_live()
                self.assertFalse(d.enable_rgbd.isEnabled());self.assertEqual(d._applied_camera_parts,())
                self.assertIn('未应用相机',d.status.text())
            finally:w.close()

    def test_restart_preflight_preserves_results_and_shows_error(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                engine,fusion,_=system();w.engine=engine;w.live_fusion=fusion
                _,report=self.joint();d.report=report;d.devices={p:r['device_id'] for p,r in report['wearing']['parts'].items()};d.analysis_path=Path(tmp)
                d.show_results();d.applied=True;d._applied_camera_parts=tuple(fusion.bindings);d.enable_rgbd.setEnabled(True)
                d.state='complete';d.recording_path=Path(tmp)/'previous'
                for part in fusion.bindings:engine.set_algorithm_mode(part,GYRO_RGBD)
                w.rgbd_pose_enabled.setChecked(True)
                w.last_rgbd_observation=SimpleNamespace(captured_mono_ns=time.monotonic_ns())
                w.last_pose_estimate=SimpleNamespace(captured_mono_ns=time.monotonic_ns())
                with patch('PySide6.QtWidgets.QMessageBox.warning') as warning:
                    d.begin();warning.assert_called_once()
                self.assertIs(d.report,report);self.assertTrue(d.applied)
                self.assertEqual(d.table.rowCount(),4);self.assertTrue(d.enable_rgbd.isEnabled())
                self.assertEqual(d.recording_path,Path(tmp)/'previous')
                self.assertIn('一键切回',d.status.text());self.assertFalse(d.active)
                # A second check crossing a freshness boundary also preserves it.
                with patch.object(d,'readiness_error',side_effect=[None,'相机过期']),patch('PySide6.QtWidgets.QMessageBox.warning'):
                    d.begin()
                self.assertIs(d.report,report);self.assertTrue(d.applied)
                self.assertEqual(d.recording_path,Path(tmp)/'previous')
                d.enable_bilateral_gyro()
                with patch('PySide6.QtWidgets.QMessageBox.warning') as warning:
                    d.begin();warning.assert_not_called()
                d.timer.stop();self.assertTrue(d.active);self.assertEqual(d.state,'delay')
                self.assertIsNone(d.report);self.assertFalse(d.applied)
                d.abort('测试取消')
            finally:w.close()

    def test_gyro_button_atomicity_safety_and_no_reset(self):
        with TemporaryDirectory() as tmp,patch.object(FusionWindow,'start_camera'):
            w=FusionWindow(Source(),Path(tmp)/'settings.json',{}, {},camera_source='orbbec');w.poll_timer.stop()
            try:
                w._show_combined_calibration();d=w._combined_calibration_dialog
                engine,fusion,_=system();w.engine=engine;w.live_fusion=fusion
                for part in fusion.bindings:engine.set_algorithm_mode(part,GYRO_RGBD)
                geometry=deepcopy((engine.continuous_basis,engine.mode_neutral,engine.fusion_corrections))
                filters=dict(engine._tilt_filters);epochs=dict(engine.epochs)
                for case in ('save','robot','sample','recorder','paired'):
                    with patch('PySide6.QtWidgets.QMessageBox.warning') as warning:
                        if case=='save':
                            with patch('optusbproj_ubuntu.combined_calibration_gui.save_config',side_effect=OSError('disk')):d.enable_bilateral_gyro()
                        else:
                            if case=='robot':w.x1_preview_dialog=SimpleNamespace(sender=object())
                            if case=='sample':d.state='delay'
                            if case=='recorder':w.recorder=object()
                            if case=='paired':engine.mode_neutral['0'].pop(GYRO_TILT)
                            d.enable_bilateral_gyro()
                            if case=='paired':engine.mode_neutral['0'][GYRO_TILT]=geometry[1]['0'][GYRO_TILT].copy()
                            w.x1_preview_dialog=None;w.recorder=None;d.state='idle'
                        warning.assert_called_once()
                    self.assertTrue(all(engine.algorithm_modes[p]==GYRO_RGBD for p in fusion.bindings))
                d.enable_bilateral_gyro()
                self.assertTrue(all(engine.algorithm_modes[p]==GYRO_TILT for p in fusion.bindings))
                for part in fusion.bindings:self.assertEqual(w.algorithm_combos[part].currentData(),GYRO_TILT)
                for old,new in zip(geometry,(engine.continuous_basis,engine.mode_neutral,engine.fusion_corrections)):
                    for key,value in old.items():
                        if isinstance(value,dict):
                            for mode in value:np.testing.assert_array_equal(value[mode],new[key][mode])
                        else:np.testing.assert_array_equal(value,new[key])
                self.assertEqual(filters,engine._tilt_filters);self.assertEqual(epochs,engine.epochs)
                self.assertEqual(set(fusion.bindings),set(d.owner.live_fusion.bindings))
                self.assertIn('没有归零',d.status.text())
            finally:w.close()

    def test_oblique_protocol_front_occlusion_and_held_out_checks(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir()
            capture(session,camera_oblique=True,occluded_forward=True)
            script=Path(__file__).resolve().parents[1]/'combined_calibration_analyze.py'
            result=subprocess.run([sys.executable,str(script),str(session),'--output',str(root/'output')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            report=json.loads((root/'output'/'report.json').read_text())
            self.assertTrue(all(d['accepted'] for d in report['wearing']['parts'].values()))
            for data in report['camera']['parts'].values():
                self.assertTrue(data['accepted'],data)
                self.assertEqual(set(data['training_phase_samples']),{'neutral','oblique','lateral'})
                self.assertEqual(set(data['independent_validation']),{'oblique_repeat','lateral_repeat','neutral_return'})
            standalone=script.with_name('rgbd_fuse_offline.py')
            result=subprocess.run([sys.executable,str(standalone),str(session),'--output',str(root/'standalone'),
                '--live-state','--neutral','0.5:2.8','--train','0.5:14.8'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            standalone_report=json.loads((root/'standalone'/'report.json').read_text())
            self.assertTrue(all(d['accepted'] for d in standalone_report['parts'].values()))
            capture(session,corrupt=True,camera_oblique=True,occluded_forward=True)
            checked=build_joint_report(session,report['camera'])
            self.assertFalse(checked['camera']['parts']['left_upper_arm']['accepted'])
            self.assertTrue(checked['camera']['parts']['right_upper_arm']['accepted'])
            guide=json.loads((session/'camera_guide.json').read_text())
            guide['events']=[e for e in guide['events'] if e['step']!='oblique_repeat']
            (session/'camera_guide.json').write_text(json.dumps(guide))
            with self.assertRaises(ValueError):load_capture(session)

    def test_oblique_training_must_have_diverse_directions(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'capture';session.mkdir()
            capture(session,camera_oblique=True,degenerate_oblique=True)
            script=Path(__file__).resolve().parents[1]/'combined_calibration_analyze.py'
            result=subprocess.run([sys.executable,str(script),str(session),'--output',str(root/'output')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            report=json.loads((root/'output'/'report.json').read_text())
            for data in report['camera']['parts'].values():
                self.assertFalse(data['accepted'])
                self.assertIn('不足20°',data['reason'])


if __name__=='__main__':unittest.main()
