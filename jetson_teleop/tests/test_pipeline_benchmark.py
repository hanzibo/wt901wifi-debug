import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from test_jetson import APP, raw_config, checkpoint, MODES, packet
from jetson_teleop.benchmark_pipeline import STAGES, RecordedSource, benchmark, load_recording
from jetson_teleop.diagnostics import json_value
from jetson_teleop.solve_timing import instrument
from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.real_robot_arm import ArmLimits, load_profile
from optusbproj_ubuntu.skeleton import PART_BY_KEY
from wt901.protocol import AttitudePacket


class TimingTests(unittest.TestCase):
    def test_instrumentation_is_idempotent_preserves_result_and_exception(self):
        def solve(x):
            if x<0:raise ValueError('original failure')
            time.sleep(.002)
            return x+1
        a=SimpleNamespace(solve=solve)
        root=SimpleNamespace(solvers={'left':a},solve=lambda x:a.solve(x))
        instrument(root);wrapped=root.solve;instrument(root)
        self.assertIs(wrapped,root.solve)
        self.assertEqual(root.solve(2),3)
        self.assertGreater(root.last_solve_timing['arms']['left']['wall_ms'],1)
        self.assertTrue(root.last_solve_timing['success'])
        self.assertGreaterEqual(root.last_solve_timing['thread_cpu_ms'],0)
        with self.assertRaisesRegex(ValueError,'original failure'):root.solve(-1)
        self.assertFalse(root.last_solve_timing['success'])
        self.assertFalse(a.last_solve_timing['success'])


class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        cfg=raw_config()
        for m in MODES:
            p=root/(m+'.json');p.write_text(json.dumps(checkpoint(m)))
            cfg['checkpoints'][m]=str(p)
        ts=PoseEngine().transforms(include_natural=False)
        points=[ts[PART_BY_KEY['left_'+p]][:3,3] for p in ('upper_arm','forearm','hand')]
        h=dict(points=points,directions=[(b-a)/np.linalg.norm(b-a) for a,b in zip(points,points[1:])],
               hand=ts[PART_BY_KEY['left_hand']][:3,:3],rotations={p:ts[PART_BY_KEY['left_'+p]][:3,:3] for p in ('upper_arm','forearm')})
        settings=dict(mode=MODES[0],config=cfg,limits={'left':vars(ArmLimits())},
                      profiles={'left':load_profile('left')},palms={'left':True},offsets={'left':None})
        packets=[]
        for tick in range(0,240,20):
            for device in cfg['assignments'].values():
                parsed=AttitudePacket.parse(packet(device,tick))
                packets.append((100+tick/1000,vars(parsed)))
        begins=[dict(event='ik_begin',monotonic_ns=int((100.02+i*.04)*1e9),
                     sequence=i+1,payload={'left':h}) for i in range(2)]
        rows=[dict(event='metadata',schema_version=1,settings=settings),
              *[dict(event='imu_packet',read_mono_ns=int(t*1e9),packet=p) for t,p in packets],
              *begins,dict(event='recording_end',complete=True)]
        path=root/'diagnostic.jsonl'
        path.write_text('\n'.join(json.dumps(r,default=json_value) for r in rows))
        return path

    def test_paced_source_opens_no_socket_and_rejects_device_command(self):
        with tempfile.TemporaryDirectory() as td:
            data=load_recording(self.fixture(Path(td)))
            with patch('socket.socket',side_effect=AssertionError('no socket')):
                source=RecordedSource(data['packets'],data['first'])
                source.started=200
                with patch('jetson_teleop.benchmark_pipeline.time.monotonic',return_value=200.001):
                    self.assertEqual(len(source.poll_timed()),6)
                    self.assertEqual(source.poll_timed(),[])
                with self.assertRaises(RuntimeError):source.switch_to_ap_udp('any')

    def test_four_groups_use_identical_inputs_without_uart_or_network(self):
        with tempfile.TemporaryDirectory() as td:
            path=self.fixture(Path(td));data=load_recording(path)
            with patch('serial.Serial',side_effect=AssertionError('no uart')),patch(
                    'socket.socket',side_effect=AssertionError('no networking')):
                result=benchmark(data)
            self.assertEqual([r['stage'] for r in result['runs']],list(STAGES))
            for r in result['runs']:
                self.assertEqual(r['frames'],2)
                self.assertLess(r['max_joint_difference_deg'],1e-8)
                self.assertEqual(r['arms']['left']['thread_cpu_ms']['n'],2)
            self.assertFalse(result['uart_opened']);self.assertFalse(result['udp_opened'])
            self.assertTrue(result['complete_recording'])

    def test_control_seed_requires_matching_session(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=self.fixture(root);control=root/'control.jsonl'
            control.write_text(json.dumps(dict(event='shared_transition',monotonic_ns=1,target=[90.]*12)))
            with self.assertRaisesRegex(ValueError,'不匹配'):load_recording(path,control)
            control.write_text(json.dumps(dict(event='shared_transition',monotonic_ns=int(100e9),target=[90.]*12)))
            self.assertEqual(load_recording(path,control)['initial'],[90.]*12)


if __name__=='__main__':unittest.main()
