"""No-simulator regression tests for the reversible remote API probe."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_connectivity import probe


class FakeSim:
    simulation_stopped = 0

    def __init__(self, state=0, ignore_writes=False):
        self.state = state
        self.ignore_writes = ignore_writes
        self.position = [0.0, 0.0, 0.0]
        self.created = False
        self.removed = False

    def getSimulationState(self):
        return self.state

    def createDummy(self, size):
        self.created = True
        return 42

    def setObjectPosition(self, handle, position):
        if not self.ignore_writes:
            self.position = position.copy()

    def getObjectPosition(self, handle):
        return self.position.copy()

    def removeObject(self, handle):
        self.removed = True


class ConnectivityProbeTests(unittest.TestCase):
    def test_position_command_is_read_back_and_dummy_removed(self):
        sim = FakeSim()
        before, after = probe(sim)
        self.assertEqual(before, [0.0, 0.0, 0.4])
        self.assertEqual(after, [0.1, 0.0, 0.4])
        self.assertTrue(sim.removed)

    def test_running_simulation_is_not_modified(self):
        sim = FakeSim(state=1)
        with self.assertRaisesRegex(RuntimeError, "运行中"):
            probe(sim)
        self.assertFalse(sim.created)

    def test_failed_readback_still_removes_dummy(self):
        sim = FakeSim(ignore_writes=True)
        with self.assertRaisesRegex(RuntimeError, "未返回目标位置"):
            probe(sim)
        self.assertTrue(sim.removed)


if __name__ == "__main__":
    unittest.main()
