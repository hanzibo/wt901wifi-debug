import json
from pathlib import Path
import sys
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_fusion_capture import packet
from optusbproj_ubuntu.pose import PoseEngine,GYRO_TILT,GYRO_RGBD
from optusbproj_ubuntu.live_rgbd_fusion import LiveRGBDFusion
from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
from optusbproj_ubuntu.fusion_replay import FusionSessionReplay


class LiveTests(unittest.TestCase):
    def setup_system(self):
        device=packet().device_id
        engine=PoseEngine({'right_upper_arm':device},algorithm_modes={'right_upper_arm':GYRO_TILT})
        service=LiveRGBDFusion(engine)
        for i in range(30):
            engine.feed(replace(packet(),millisecond=i*20),received_at=1+i*.02)
            service.feed(device,1_000_000_000+i*20_000_000)
        q=Rotation.from_euler('x',25,degrees=True).as_matrix()
        report={'orientation_source':'live_runtime_pose','camera_up':(q@np.array([0,0,1])).tolist(),'parts':{'right_upper_arm':{'device_id':device,'reference_epoch':engine.epochs[device],'accepted':True,'up_world':[0,0,1],'Q_camera_from_imu_world':q.tolist(),'bone_axis_sensor':[1,0,0],'bone_length_m':.25}}}
        service.install(report)
        return engine,service,device,q,report

    def observation(self,q,stamp=1_400_000_000):
        up=q@np.array([0,0,1]);d=Rotation.from_rotvec(up*np.radians(10)).apply(q@np.array([1,0,0]))
        xyz=np.zeros((8,3));xyz[1]=[0,0,2];xyz[3]=xyz[1]+.25*d
        row={'valid':[True]*8,'confidence':[1.]*133,'support':[1.]*8,'points_camera_m':xyz.tolist()}
        return SimpleNamespace(captured_mono_ns=stamp,to_record=lambda:row)

    def test_lock_missing_reference_preserves_pose_epoch_and_neutral(self):
        engine,service,device,_,_=self.setup_system()
        estimator=engine._tilt_filters[device];estimator.neutral_up=None
        rotation=engine.current_for(device,GYRO_TILT).copy()
        neutral=engine.neutral_for(device,GYRO_TILT).copy();epoch=engine.epochs[device]
        result=service.lock_gravity([device],1_000_000_000,1_600_000_000)
        np.testing.assert_allclose(estimator.neutral_up,[0,0,1])
        np.testing.assert_allclose(engine.current_for(device,GYRO_TILT),rotation)
        np.testing.assert_allclose(engine.neutral_for(device,GYRO_TILT),neutral)
        self.assertEqual(engine.epochs[device],epoch)
        self.assertEqual(result[device]['method'],'neutral_hold_bootstrap')

    def test_lock_preserves_existing_reference(self):
        engine,service,device,_,_=self.setup_system()
        up=Rotation.from_euler('x',2,degrees=True).apply([0,0,1])
        engine._tilt_filters[device].neutral_up=up.copy()
        result=service.lock_gravity([device],1_000_000_000,1_600_000_000)
        np.testing.assert_allclose(result[device]['up_world'],up)
        np.testing.assert_allclose(engine._tilt_filters[device].neutral_up,up)

    def test_lock_is_atomic_and_rejects_motion(self):
        engine,service,device,_,_=self.setup_system()
        estimator=engine._tilt_filters[device];estimator.neutral_up=None
        with self.assertRaisesRegex(ValueError,'采样不足'):
            service.lock_gravity([device,'missing'],1_000_000_000,1_600_000_000)
        self.assertIsNone(estimator.neutral_up)
        for r in service.gravity_history[device]:r['gyro']=np.array([0,0,10])
        with self.assertRaisesRegex(ValueError,'静止'):
            service.lock_gravity([device],1_000_000_000,1_600_000_000)
        self.assertIsNone(estimator.neutral_up)

    def test_conjugate_left_correction_does_not_change_pure_stream(self):
        engine,service,device,q,_=self.setup_system()
        pure=engine.current_for(device,GYRO_TILT).copy()
        engine.set_algorithm_mode('right_upper_arm',GYRO_RGBD)
        service.observe(self.observation(q),now_ns=1_440_000_000)
        self.assertTrue(service.states['right_upper_arm']['applied'])
        h=service.bindings['right_upper_arm'][1].rotation()
        np.testing.assert_allclose(engine.current[device],q.T@h@q@pure,atol=1e-12)
        np.testing.assert_allclose(engine.current_for(device,GYRO_TILT),pure)
        engine.zero('right_upper_arm')
        np.testing.assert_allclose(engine.rotation_for('right_upper_arm'),np.eye(3),atol=1e-12)
        np.testing.assert_allclose(engine.fusion_corrections[device],q.T@h@q,atol=1e-12)

    def test_diagnostic_mode_does_not_apply_feedback(self):
        engine,service,device,q,_=self.setup_system()
        service.observe(self.observation(q),now_ns=1_440_000_000)
        self.assertFalse(service.states['right_upper_arm']['applied'])
        np.testing.assert_allclose(engine.fusion_corrections[device],np.eye(3))

    def test_old_reference_rejected_and_disabling_keeps_offset(self):
        engine,service,device,q,report=self.setup_system()
        engine.set_algorithm_mode('right_upper_arm',GYRO_RGBD)
        service.observe(self.observation(q),now_ns=1_440_000_000)
        h=engine.fusion_corrections[device].copy()
        service.invalidate('manual')
        service.observe(self.observation(q,1_500_000_000),now_ns=1_540_000_000)
        np.testing.assert_allclose(engine.fusion_corrections[device],h)
        engine.set_algorithm_mode('right_upper_arm',GYRO_TILT)
        engine.invalidate_neutral(device)
        with self.assertRaises(ValueError):service.install(report)

    def test_expired_and_duplicate_observations_do_not_update(self):
        engine,service,device,q,_=self.setup_system();engine.set_algorithm_mode('right_upper_arm',GYRO_RGBD)
        obs=self.observation(q)
        service.observe(obs,now_ns=2_000_000_000)
        self.assertFalse(service.states['right_upper_arm']['applied'])
        service.observe(obs,now_ns=1_440_000_000)
        h=engine.fusion_corrections[device].copy()
        service.observe(obs,now_ns=1_450_000_000)
        self.assertIn('重复',service.states['right_upper_arm']['reason'])
        np.testing.assert_allclose(engine.fusion_corrections[device],h)

    def test_runtime_record_and_replay_preserves_fused_neutral(self):
        engine,service,device,q,_=self.setup_system();engine.set_algorithm_mode('right_upper_arm',GYRO_RGBD)
        service.observe(self.observation(q),now_ns=1_440_000_000);engine.zero('right_upper_arm')
        with TemporaryDirectory() as folder:
            recorder=FusionSessionRecorder(Path(folder),tail_trim_seconds=0)
            stamp=recorder.started_mono_ns+1
            state=engine.runtime_snapshot(device,stamp);recorder.write_imu(stamp,packet(),state);recorder.close()
            replay=FusionSessionReplay(recorder.path)
            try:
                replay.poll_timed(replay._started_at+1)
                saved=replay.take_runtime()[(stamp,device)]
                restored=PoseEngine({'right_upper_arm':device},algorithm_modes={'right_upper_arm':GYRO_RGBD})
                restored.restore_runtime(saved)
                np.testing.assert_allclose(restored.current[device],engine.current[device])
                np.testing.assert_allclose(restored.rotation_for('right_upper_arm'),np.eye(3),atol=1e-12)
                self.assertEqual(json.loads((recorder.path/'manifest.json').read_text())['runtime_pose_records'],1)
            finally:replay.close()
