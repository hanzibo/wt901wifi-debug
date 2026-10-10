import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from optusbproj_ubuntu.wrist_diagnostics import WristDiagnostics


class WristDiagnosticTests(unittest.TestCase):
    def test_per_joint_step_diagnostics(self):
        t=WristDiagnostics();q=np.deg2rad([0,0,0,0,12,-6]);steps=np.deg2rad([6,6,6,6,12,12])
        row=t.measure('right',1,1,np.eye(3),np.eye(3),q,np.zeros(6),steps,np.full(6,-3.),np.full(6,3.))
        np.testing.assert_allclose(row['joint_step_limits_deg'],[6,6,6,6,12,12])
        self.assertEqual(row['wrist_step_limit_active'],[True,False])

    def sample(self,tracker,input_at,output_at,degrees,q=None,previous=None):
        q=np.zeros(6) if q is None else q
        previous=np.zeros(6) if previous is None else previous
        R=Rotation.from_euler('z',degrees,degrees=True).as_matrix()
        return tracker.measure('right',input_at,output_at,R,np.eye(3),q,previous,
            np.deg2rad(6),np.full(6,-3.),np.full(6,3.),R)

    def test_first_sample_is_unknown_not_zero(self):
        row=self.sample(WristDiagnostics(),1,1,30)
        self.assertIsNone(row['target_speed_deg_s'])
        self.assertIsNone(row['joint5_velocity_deg_s'])
        self.assertAlmostEqual(row['orientation_error_deg'],30)

    def test_known_speed_and_signed_readback_rates(self):
        tracker=WristDiagnostics();self.sample(tracker,1,2,0)
        q=np.deg2rad([0,0,0,0,6,-3])
        row=self.sample(tracker,1.05,2.1,9,q)
        self.assertAlmostEqual(row['target_speed_deg_s'],180)
        self.assertAlmostEqual(row['target_relative_forearm_speed_deg_s'],180)
        self.assertAlmostEqual(row['joint5_velocity_deg_s'],60)
        self.assertAlmostEqual(row['joint6_velocity_deg_s'],-30)
        self.assertEqual(row['wrist_step_limit_active'],[True,False])

    def test_same_input_timestamp_does_not_create_infinite_velocity(self):
        tracker=WristDiagnostics();self.sample(tracker,1,1,0)
        row=self.sample(tracker,1,1.05,0)
        self.assertIsNone(row['target_speed_deg_s'])
        row=self.sample(tracker,1.1,1.1,10)
        self.assertAlmostEqual(row['target_speed_deg_s'],100)

    def test_rotation_wrap_uses_so3_shortest_difference(self):
        tracker=WristDiagnostics();self.sample(tracker,1,1,179)
        row=self.sample(tracker,1.1,1.1,-179)
        self.assertAlmostEqual(row['target_speed_deg_s'],20)

    def test_gap_and_reset_do_not_report_resume_spikes(self):
        tracker=WristDiagnostics();self.sample(tracker,1,1,0)
        row=self.sample(tracker,2,2,90)
        self.assertIsNone(row['target_speed_deg_s']);self.assertIsNone(row['joint_velocity_deg_s'])
        tracker.reset();row=self.sample(tracker,2.1,2.1,100)
        self.assertIsNone(row['joint_velocity_deg_s'])

    def test_disabled_hand_and_joint_limit(self):
        t=WristDiagnostics();q=np.zeros(6);q[5]=3
        row=t.measure('left',1,1,None,np.eye(3),q,q,.1,np.full(6,-3.),np.full(6,3.))
        self.assertIsNone(row['orientation_error_deg']);self.assertIsNone(row['target_speed_deg_s'])
        self.assertEqual(row['wrist_joint_limit_active'],[False,True])


if __name__=='__main__':unittest.main()
