"""No hardware, UDP or simulator; observer tests only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.robot_runtime_probe import Sampler, capture, distribution, summarize


class ProbeTests(unittest.TestCase):
    def test_distribution_and_missing_values(self):
        self.assertIsNone(distribution([None, float('nan')]))
        self.assertEqual(distribution([0., 10.])['p50'], 5.)
        self.assertEqual(distribution([150., 200.])['max'], 200.)
        self.assertIsNone(summarize([])['tree_cpu_percent'])

    def test_sampler_observes_self_without_mutation(self):
        sampler=Sampler(os.getpid()); row=sampler.sample()
        self.assertGreater(row['tree_rss_mib'],0.)
        self.assertGreaterEqual(row['process_count'],1)
        self.assertGreater(row['system_available_mib'],0.)
        json.dumps(row, allow_nan=False)

    def test_capture_does_not_stop_child_and_writes_summary(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
        try:
            with tempfile.TemporaryDirectory() as root:
                output=Path(root)/'probe'
                result=capture(child.pid,.15,.05,output)
                self.assertIsNone(child.poll())
                self.assertEqual(result['reason'],'duration_complete')
                rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['event'],'metadata')
                self.assertGreater(len(rows),1)
                self.assertTrue((output/'summary.json').exists())
                with self.assertRaises(FileExistsError):capture(child.pid,.1,.05,output)
        finally:
            child.terminate(); child.wait(timeout=3)

    def test_gui_protocol_saves_context_and_stops_only_observer(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
        try:
            with tempfile.TemporaryDirectory() as root:
                output=Path(root)/'probe'
                observer=subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[1]/
                    'scripts/robot_runtime_probe.py'), '--pid', str(child.pid), '--duration', '30',
                    '--output', str(output), '--gui-control'], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout,stderr=observer.communicate(json.dumps(dict(event='gui_context',
                    mode='left', control_log='/fake/control.jsonl'))+'\nstop\n', timeout=5)
                self.assertEqual(observer.returncode,0,(stdout,stderr))
                self.assertIsNone(child.poll())
                result=json.loads((output/'summary.json').read_text())
                self.assertEqual(result['reason'],'controller_stop_requested')
                rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
                self.assertEqual(rows[1]['control_log'],'/fake/control.jsonl')
        finally:
            child.terminate(); child.wait(timeout=3)


if __name__=='__main__': unittest.main()
