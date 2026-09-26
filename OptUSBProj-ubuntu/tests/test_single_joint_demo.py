from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from single_joint_demo import SingleJointMapper, TemporaryJoint, twist_angle_deg
from wt901_viz.orientation import rotation_matrix


def packet(roll=0.0, pitch=0.0, yaw=0.0):
    return SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw)


class FakeSim:
    simulation_stopped = 0
    joint_revolute = 10
    jointmode_kinematic = 11
    primitiveshape_cuboid = 12

    def __init__(self, state=0):
        self.state = state
        self.created = []
        self.removed = []
        self.position = 0.0

    def getSimulationState(self):
        return self.state

    def createJoint(self, *_args):
        self.created.append(1)
        return 1

    def createPrimitiveShape(self, *_args):
        self.created.append(2)
        return 2

    def setObjectAlias(self, *_args):
        pass

    def setObjectPosition(self, *_args):
        pass

    def setObjectParent(self, *_args):
        pass

    def setJointPosition(self, _handle, value):
        self.position = value

    def getJointPosition(self, _handle):
        return self.position

    def removeObjects(self, handles):
        self.removed.extend(handles)


class SingleJointTests(unittest.TestCase):
    def test_first_packet_zero_then_z_twist_controls_joint(self):
        mapper = SingleJointMapper()
        self.assertEqual(mapper.angle_deg(packet(yaw=20)), 0.0)
        self.assertAlmostEqual(mapper.angle_deg(packet(yaw=65)), 45.0)

    def test_other_euler_components_are_not_naively_used(self):
        mapper = SingleJointMapper("+Z")
        mapper.zero(packet())
        self.assertAlmostEqual(mapper.angle_deg(packet(pitch=45)), 0.0)
        self.assertAlmostEqual(mapper.angle_deg(packet(roll=30)), 0.0)
        self.assertAlmostEqual(twist_angle_deg(rotation_matrix(0, 0, 45), "-Z"), -45.0)

    def test_axis_mapping_and_limit(self):
        mapper = SingleJointMapper("+Z", ("+Y", "+Z", "+X"), limit_deg=60)
        mapper.zero(packet())
        self.assertAlmostEqual(mapper.angle_deg(packet(pitch=45)), 45.0)
        self.assertAlmostEqual(mapper.angle_deg(packet(pitch=80)), 60.0)

    def test_180_degree_perpendicular_swing_is_undefined(self):
        with self.assertRaisesRegex(ValueError, "不确定"):
            twist_angle_deg(rotation_matrix(180, 0, 0), "+Z")

    def test_temporary_joint_reads_back_and_cleans_up(self):
        sim = FakeSim()
        with TemporaryJoint(sim) as joint:
            self.assertAlmostEqual(joint.command_deg(35), 35)
            self.assertAlmostEqual(sim.position, math.radians(35))
        self.assertEqual(sim.created, [1, 2])
        self.assertEqual(sim.removed, [2, 1])

    def test_running_scene_is_not_modified(self):
        sim = FakeSim(state=1)
        with self.assertRaisesRegex(RuntimeError, "停止"):
            with TemporaryJoint(sim):
                pass
        self.assertFalse(sim.created)


if __name__ == "__main__":
    unittest.main()
