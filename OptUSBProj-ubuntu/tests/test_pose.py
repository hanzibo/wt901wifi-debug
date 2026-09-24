from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PARTS, PART_BY_KEY


def packet(device_id: str, roll: float = 0, pitch: float = 0, yaw: float = 0):
    return SimpleNamespace(device_id=device_id, roll=roll, pitch=pitch, yaw=yaw)


class PoseEngineTests(unittest.TestCase):
    def test_rest_pose_has_all_fifteen_parts_and_legacy_offsets(self):
        engine = PoseEngine()
        transforms = engine.transforms()
        self.assertEqual(len(transforms), 15)
        np.testing.assert_allclose(transforms[PART_BY_KEY["torso"]][:3, 3],
                                   [-0.00978, 0, 0.13049])
        self.assertEqual(PARTS[PART_BY_KEY["left_upper_arm"]].parent,
                         PART_BY_KEY["torso"])
        self.assertEqual(PARTS[PART_BY_KEY["right_hand"]].parent,
                         PART_BY_KEY["right_forearm"])

    def test_first_packet_is_neutral_and_child_inherits_parent(self):
        engine = PoseEngine({"torso": "WT5500011381"})
        engine.feed(packet("WT5500011381", yaw=30))
        neutral = engine.transforms()
        np.testing.assert_allclose(neutral[PART_BY_KEY["torso"]][:3, :3], np.eye(3), atol=1e-12)
        engine.feed(packet("WT5500011381", yaw=60))
        moved = engine.transforms()
        self.assertFalse(np.allclose(moved[PART_BY_KEY["torso"]][:3, :3], np.eye(3)))
        np.testing.assert_allclose(moved[PART_BY_KEY["head"]][:3, :3],
                                   moved[PART_BY_KEY["torso"]][:3, :3])
        np.testing.assert_allclose(engine.rotation_for("head"), np.eye(3), atol=1e-12)
        engine.zero("torso")
        np.testing.assert_allclose(engine.transforms()[PART_BY_KEY["torso"]][:3, :3],
                                   np.eye(3), atol=1e-12)

    def test_common_motion_does_not_create_relative_joint_rotation(self):
        engine = PoseEngine({"torso": "WT5500011381", "left_upper_arm": "WT5500011408"})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011408"))
        engine.feed(packet("WT5500011381", roll=40))
        engine.feed(packet("WT5500011408", roll=40))
        np.testing.assert_allclose(engine.rotation_for("left_upper_arm"), np.eye(3), atol=1e-12)

    def test_duplicate_assignment_is_rejected(self):
        engine = PoseEngine({"left_hand": "WT5500011381"})
        with self.assertRaises(ValueError):
            engine.assign("right_hand", "WT5500011381")

    def test_reassignment_waits_for_new_neutral_packet(self):
        engine = PoseEngine({"left_hand": "WT5500011381"})
        engine.feed(packet("WT5500011381"))
        engine.feed(packet("WT5500011408", yaw=40))
        engine.assign("left_hand", "WT5500011408")
        self.assertNotIn("WT5500011408", engine.neutral)
        engine.feed(packet("WT5500011408", yaw=40))
        np.testing.assert_allclose(engine.rotation_for("left_hand"), np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
