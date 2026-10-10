import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'urdf-model/script_python'))
from x1_angle_log import AngleLogger


class AngleLoggerTests(unittest.TestCase):
    def test_disabled_snapshot_restart_and_failure(self):
        with tempfile.TemporaryDirectory() as root:
            log=AngleLogger(root)
            log.emit('unused')
            self.assertFalse(list(Path(root).iterdir()))
            log.start(['1'])
            path=log.path
            q=[.2]
            log.emit('publish', positions=q)
            q[0]=.8
            log.start(['1'])
            self.assertEqual(log.path,path)
            log.close()
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[1]['positions'],[.2])
            self.assertEqual(rows[-1]['event'],'stop')
            log.start(['1']);self.assertNotEqual(log.path,path);log.close()
            invalid=AngleLogger(path)
            invalid.start(['1'])
            self.assertTrue(invalid.status()['error'])
            self.assertFalse(invalid.status()['enabled'])
