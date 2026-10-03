from pathlib import Path
import sys
import unittest
import json
from tempfile import TemporaryDirectory
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from optusbproj_ubuntu.rgbd_fusion import HeadingCorrector,fit_mounting,angle_deg,visual_direction,load_live_series


class FusionTests(unittest.TestCase):
    def test_locked_reference_used_instead_of_early_neutral_candidate(self):
        with TemporaryDirectory() as folder:
            path=Path(folder);up=Rotation.from_euler('x',7.5,degrees=True).apply([0,0,1])
            rows=[{'device_id':'A','epoch':'epoch','host_recv_mono_ns':i*20_000_000,
                   'orientation_by_mode':{'gyro_tilt':np.eye(3).tolist()},
                   'up_world':None if i<30 else up.tolist()} for i in range(60)]
            (path/'runtime_pose.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            locked={'epoch':'epoch','up_world':up.tolist(),'locked_at_ns':600_000_000}
            (path/'camera_guide.json').write_text(json.dumps({'gravity_reference':{'A':locked}}))
            _,actual,_,info=load_live_series(path,'A',(0,500_000_000))
            np.testing.assert_allclose(actual,up)
            self.assertIn('locked',info['method'])
            rows[-1]['up_world']=[0,0,1]
            (path/'runtime_pose.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,'发生变化'):load_live_series(path,'A',(0,500_000_000))
            locked['epoch']='new-epoch'
            (path/'camera_guide.json').write_text(json.dumps({'gravity_reference':{'A':locked}}))
            with self.assertRaisesRegex(ValueError,'参考系不一致'):load_live_series(path,'A',(0,500_000_000))

    def test_mounting_recovery_without_rpy(self):
        up=np.array([0.,0.,1.]);q=Rotation.from_rotvec(up*.6).as_matrix()
        b=np.array([.3,.7,-.64]);b/=np.linalg.norm(b)
        rv=np.random.default_rng(9).normal(size=(70,3))*.5
        matrices=Rotation.from_rotvec(rv).as_matrix()
        dirs=np.einsum('ij,njk,k->ni',q,matrices,b)
        fit=fit_mounting(matrices,dirs,up,up)
        self.assertTrue(fit.accepted)
        self.assertLess(fit.median_error_deg,.001)
        self.assertLess(angle_deg(fit.bone_axis_sensor,b),.001)

    def test_heading_correction_preserves_gravity_and_so3(self):
        corrector=HeadingCorrector([0,0,1])
        target=Rotation.from_euler('z',12,degrees=True).apply([1,0,0])
        for t in np.arange(100)/10:
            applied,_,_=corrector.update(t,[1,0,0],target)
            self.assertTrue(applied)
        r=corrector.rotation()
        np.testing.assert_allclose(r@[0,0,1],[0,0,1])
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r),1)
        self.assertLess(angle_deg(r@[1,0,0],target),1)

    def test_vertical_and_large_innovation_rejected(self):
        corrector=HeadingCorrector([0,0,1])
        self.assertFalse(corrector.update(0,[0,0,1],[0,0,1])[0])
        self.assertFalse(corrector.update(.1,[1,0,0],[0,1,0])[0])
        self.assertEqual(corrector.offset_rad,0)
        with self.assertRaises(ValueError):corrector.update(.1,[1,0,0],[1,0,0])

    def test_no_catchup_jump_after_occlusion(self):
        corrector=HeadingCorrector([0,0,1])
        target=Rotation.from_euler('z',20,degrees=True).apply([1,0,0])
        corrector.update(0,[1,0,0],target)
        before=corrector.offset_rad
        corrector.update(100,[1,0,0],target)
        self.assertLessEqual(np.degrees(corrector.offset_rad-before),.75+1e-9)

    def test_gravity_only_motion_cannot_fit_heading(self):
        with self.assertRaises(ValueError):
            fit_mounting(np.tile(np.eye(3),(20,1,1)),np.tile([0,0,-1],(20,1)),[0,0,1],[0,0,1])

    def test_high_2d_confidence_does_not_override_bone_quality(self):
        row={'valid':[True]*8,'confidence':[1.]*133,'support':[1.]*8,
             'points_camera_m':[[0,0,0]]*8}
        row['points_camera_m'][3]=[.6,0,0]
        direction,reason=visual_direction(row,'right_upper_arm',.25)
        self.assertIsNone(direction);self.assertIn('骨长',reason)
