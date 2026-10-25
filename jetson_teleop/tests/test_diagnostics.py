"""Recording and offline replay never require a real UART or a second UDP socket."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'OptUSBProj-ubuntu'))
from jetson_teleop.diagnostics import DiagnosticLog
from jetson_teleop.replay_ik import replay
from optusbproj_ubuntu.real_robot_arm import ArmLimits,load_profile
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PART_BY_KEY


class DiagnosticTests(unittest.TestCase):
    def test_json_encoding_and_complete_footer(self):
        with tempfile.TemporaryDirectory() as root:
            log=DiagnosticLog(root,dict(matrix=np.eye(3),path=Path(root)))
            log.record('array',value=np.arange(3));log.close()
            rows=[json.loads(l) for l in log.path.read_text().splitlines()]
            self.assertEqual(rows[0]['schema_version'],1)
            self.assertEqual(rows[1]['value'],[0,1,2])
            self.assertTrue(rows[-1]['complete']);self.assertIsNone(log.error)

    def test_overflow_and_expiry_are_visible_without_blocking(self):
        with tempfile.TemporaryDirectory() as root:
            log=DiagnosticLog(root,{})
            with patch.object(log.queue,'put_nowait',side_effect=__import__('queue').Full):
                log.record('overflow')
            log.deadline=time.monotonic()-1;log.record('too_late');log.close()
            rows=[json.loads(l) for l in log.path.read_text().splitlines()]
            self.assertFalse(rows[-1]['complete']);self.assertEqual(rows[-1]['dropped'],1)
            self.assertNotIn('too_late',[r['event'] for r in rows])

    def test_pose_recording_replays_both_backends_without_device_access(self):
        transforms=PoseEngine().transforms(include_natural=False)
        frames=[transforms[PART_BY_KEY['left_'+p]] for p in ('upper_arm','forearm','hand')]
        points=[T[:3,3] for T in frames]
        directions=[(b-a)/np.linalg.norm(b-a) for a,b in zip(points,points[1:])]
        h=dict(points=points,directions=directions,hand=frames[2][:3,:3],
               rotations={p:T[:3,:3] for p,T in zip(('upper_arm','forearm'),frames)})
        settings=dict(limits={'left':vars(ArmLimits())},profiles={'left':load_profile('left')},
                      palms={'left':True},offsets={'left':None})
        with tempfile.TemporaryDirectory() as root:
            log=DiagnosticLog(root,settings)
            for i in range(2):
                log.record('pose_snapshot',entries={'left':[
                    dict(at=time.monotonic(),sides={'left':h},device_ages_ms={'left_hand':2}),[]]})
            log.close()
            with patch('socket.socket',side_effect=AssertionError('no networking')),patch(
                    'serial.Serial',side_effect=AssertionError('no serial')):
                result=replay(log.path)
            self.assertTrue(result['complete']);self.assertEqual(result['frames'],2)
            self.assertLess(result['max_joint_difference_deg'],.001)


if __name__=='__main__':unittest.main()
