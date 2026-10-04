"""Bilateral heading feedback stays independent and bound to each IMU epoch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtWidgets

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optusbproj_ubuntu.camera_calibration_gui import CameraCalibrationDialog,CAPTURE_SCOPES
from optusbproj_ubuntu.live_rgbd_fusion import LiveRGBDFusion
from optusbproj_ubuntu.pose import PoseEngine,GYRO_TILT,GYRO_RGBD
from optusbproj_ubuntu.rgbd_fusion import ARM_SEGMENTS
from optusbproj_ubuntu.rgbd_pose import RGBDPoseEstimate
from test_fusion_capture import packet
from test_camera_calibration_gui import Owner
from test_visual_wearing import fixture
from dataclasses import replace
from rgbd_fuse_offline import capture_parts
from rgbd_fusion_viewer import ComparisonWindow


class BilateralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def system(self):
        ids={p:str(i) for i,p in enumerate(ARM_SEGMENTS)}
        engine=PoseEngine(ids,algorithm_modes={p:GYRO_TILT for p in ids})
        service=LiveRGBDFusion(engine)
        for d in ids.values():
            for i in range(30):
                engine.feed(replace(packet(),device_id=d,millisecond=i*20),received_at=1+i*.02)
                service.feed(d,1_000_000_000+i*20_000_000)
        report={'orientation_source':'live_runtime_pose','camera_up':[0,0,1],'parts':{
            p:{'device_id':d,'reference_epoch':engine.epochs[d],'accepted':True,
               'up_world':[0,0,1],'Q_camera_from_imu_world':np.eye(3).tolist(),
               'bone_axis_sensor':[1,0,0],'bone_length_m':.25}
            for p,d in ids.items()}}
        return engine,service,report

    def observation(self,invalid_left=False,stamp=1_400_000_000):
        xyz=np.zeros((8,3));xyz[0]=[0,.2,2];xyz[1]=[0,-.2,2]
        for part,(a,b) in ARM_SEGMENTS.items():
            yaw=-12 if part.startswith('left') else 10
            xyz[b]=xyz[a]+.25*Rotation.from_euler('z',yaw,degrees=True).apply([1,0,0])
        row={'valid':[True]*8,'confidence':[1.]*133,'support':[1.]*8,'points_camera_m':xyz.tolist()}
        if invalid_left:row['valid'][2]=False
        return SimpleNamespace(captured_mono_ns=stamp,to_record=lambda:row)

    def test_four_feedbacks_have_separate_offsets_and_keep_pure_poses(self):
        engine,service,report=self.system();service.install(report)
        pure={d:engine.current_for(d,GYRO_TILT).copy() for d in engine.assignments.values() if d}
        for p in ARM_SEGMENTS:engine.set_algorithm_mode(p,GYRO_RGBD)
        service.observe(self.observation(),now_ns=1_440_000_000)
        for p,d in ((p,engine.assignments[p]) for p in ARM_SEGMENTS):
            self.assertTrue(service.states[p]['applied'])
            self.assertLess(service.states[p]['offset_deg'],0) if p.startswith('left') else self.assertGreater(service.states[p]['offset_deg'],0)
            np.testing.assert_allclose(engine.current_for(d,GYRO_TILT),pure[d])
            np.testing.assert_allclose(engine.current[d],service.bindings[p][1].rotation()@pure[d])

    def test_left_occlusion_and_epoch_failure_do_not_block_right(self):
        engine,service,report=self.system();service.install(report)
        for p in ARM_SEGMENTS:engine.set_algorithm_mode(p,GYRO_RGBD)
        service.observe(self.observation(invalid_left=True),now_ns=1_440_000_000)
        for p in ('left_upper_arm','left_forearm'):self.assertFalse(service.states[p]['applied'])
        for p in ('right_upper_arm','right_forearm'):self.assertTrue(service.states[p]['applied'])
        d=engine.assignments['left_upper_arm'];engine.epochs[d]='new'
        service.observe(self.observation(stamp=1_500_000_000),now_ns=1_540_000_000)
        self.assertNotIn(d,engine.fusion_ready)
        self.assertTrue(service.states['right_upper_arm']['applied'])

    def test_install_atomic_and_right_only_preserves_left(self):
        engine,service,report=self.system();bad=copy.deepcopy(report)
        bad['parts']['right_forearm']['reference_epoch']='stale'
        with self.assertRaises(ValueError):service.install(bad)
        self.assertFalse(service.bindings);self.assertFalse(engine.fusion_ready)
        service.install(report);left=service.bindings['left_upper_arm']
        right=copy.deepcopy(report);right['parts']={p:r for p,r in right['parts'].items() if p.startswith('right')}
        service.install(right);self.assertIs(service.bindings['left_upper_arm'],left)
        failed=copy.deepcopy(report);failed['parts']['left_upper_arm']['accepted']=False
        service.install(failed);self.assertNotIn(engine.assignments['left_upper_arm'],engine.fusion_ready)
        self.assertIn(engine.assignments['right_upper_arm'],engine.fusion_ready)

    def test_bilateral_guide_locks_four_devices_and_scope_does_not_change_midrecord(self):
        with TemporaryDirectory() as folder:
            clock=[100.];owner=Owner(Path(folder)/'record',clock)
            owner.engine.assignments.update(left_upper_arm='C',left_forearm='D')
            dialog=CameraCalibrationDialog(owner);dialog.delay.setValue(0);dialog.auto_analyze.setChecked(False)
            self.assertEqual(dialog.scope_combo.currentData(),'bilateral')
            self.assertIn('同时向左右两侧',next(s[2] for s in dialog.capture_steps() if s[0]=='lateral'))
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic',side_effect=lambda:clock[0]),patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',side_effect=lambda:int(clock[0]*1e9)):
                dialog.begin();dialog.timer.stop();self.assertFalse(dialog.scope_combo.isEnabled())
                while dialog.active:
                    clock[0]=dialog.deadline+.1;owner.refresh();dialog.tick()
            saved=json.loads((owner.path/'camera_guide.json').read_text())
            self.assertEqual(saved['capture_scope'],'bilateral');self.assertEqual(set(saved['participants']),set(ARM_SEGMENTS))
            self.assertEqual(set(owner.gravity_calls[0][0]),{'A','B','C','D'})
            dialog.close();owner.close()

    def test_left_only_and_bilateral_missing_device_preflight(self):
        with TemporaryDirectory() as folder:
            owner=Owner(Path(folder),[100.])
            owner.engine.assignments={'left_upper_arm':'C','left_forearm':'D'}
            dialog=CameraCalibrationDialog(owner)
            self.assertEqual(dialog.scope_combo.currentData(),'left')
            self.assertIn('向左侧',next(s[2] for s in dialog.capture_steps() if s[0]=='lateral'))
            with patch('optusbproj_ubuntu.camera_calibration_gui.time.monotonic_ns',return_value=100_000_000_000):
                self.assertIsNone(dialog.readiness_error())
                dialog.scope_combo.setCurrentIndex(dialog.scope_combo.findData('bilateral'))
                self.assertIn('right_upper_arm',dialog.readiness_error())
            dialog.close();owner.close()

    def test_scope_metadata_and_full_offline_four_part_fit(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();fixture(source,bilateral=True)
            rows=[json.loads(s) for s in (source/'vision.jsonl').read_text().splitlines()]
            rows=[dict(RGBDPoseEstimate(r['captured_mono_ns'],np.zeros((133,2)),np.ones(133),
                   np.array(r['points_camera_m']),np.ones(8,bool),['synthetic']*8,np.zeros((8,2)),
                   np.ones(8),np.zeros(8),r['bone_lengths_m'],completed_mono_ns=r['captured_mono_ns']+10_000_000).to_record(),frame='unused.jpg') for r in rows]
            (source/'vision.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(set(capture_parts(source)),set(ARM_SEGMENTS))
            output=root/'output'
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'rgbd_fuse_offline.py'),str(source),'--output',str(output),'--live-state','--neutral','0.5:2.5','--train','0:20'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(set(report['parts']),set(ARM_SEGMENTS))
            for part,data in report['parts'].items():
                self.assertTrue(data['accepted'],data);self.assertLess(data['fit_median_deg'],.01)
                self.assertTrue((output/(part+'_imu.jsonl')).exists())
            viewer=ComparisonWindow(output);viewer.show_frame(40)
            for canvas in viewer.canvases:self.assertTrue(canvas.estimate.valid[:6].all())
            viewer.close()
            guide=json.loads((source/'camera_guide.json').read_text());guide.pop('participants');guide['devices']={p:d for p,d in guide['devices'].items() if p.startswith('right')}
            (source/'camera_guide.json').write_text(json.dumps(guide))
            self.assertEqual(set(capture_parts(source)),set(CAPTURE_SCOPES['right']))


if __name__=='__main__':unittest.main()
