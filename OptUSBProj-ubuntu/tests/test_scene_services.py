"""Direct tests of Qt-free services, in addition to old GUI compatibility tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from optusbproj_ubuntu.pose import PoseEngine
from optusbproj_ubuntu.skeleton import PART_BY_KEY
from optusbproj_ubuntu.scene_robot_model import (
    SceneArm, body_axes, palm_target, position_targets, segment_targets,
)
from optusbproj_ubuntu.scene_robot_state import Mailbox, joint_steps
from optusbproj_ubuntu.scene_robot_solver import SceneSolver
from optusbproj_ubuntu.scene_robot_runtime import SceneRuntime
from optusbproj_ubuntu.scene_robot_log import SceneLog, SceneDiagnostics
from optusbproj_ubuntu.scene_hand import HandMailbox
from optusbproj_ubuntu.scene_hand_profiles import read_hand_profiles, write_hand_profiles


def scene_data():
    path = Path(__file__).parent / "fixtures/scene_retarget_baseline.json"
    cases = json.loads(path.read_text())["sides"]
    arms = {}
    for side, case in cases.items():
        arms[side] = copy.deepcopy(case["model"])
        arms[side]["limits"] = [[-3, 3]] * 6
    return {"arms": arms}


class SceneServiceTests(unittest.TestCase):
    def test_core_layers_import_without_qt(self):
        command = (
            "import sys; from optusbproj_ubuntu import scene_robot_runtime, "
            "scene_robot_solver, scene_robot_session, scene_robot_state, "
            "scene_hand_profiles; assert 'PySide6' not in sys.modules"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(Path(__file__).resolve().parents[1]),
            str(Path(__file__).resolve().parents[2]),
        ])
        subprocess.run([sys.executable, "-c", command], env=env, check=True)

    def test_solver_extraction_preserves_old_target_construction(self):
        models = {s: SceneArm(data, [-3] * 6, [3] * 6)
                  for s, data in scene_data()["arms"].items()}
        axes = body_axes(models)
        rest = PoseEngine().transforms(include_natural=False)
        solver = SceneSolver(models, axes, rest, clock=lambda: 10.)
        sides = {}
        for side in models:
            points = [rest[PART_BY_KEY[f"{side}_{p}"]][:3, 3].copy()
                      for p in ("upper_arm", "forearm", "hand")]
            points[1] += [.01, .02, .03]
            points[2] += [.03, .02, .04]
            sides[side] = {
                "points": points,
                "directions": [(b - a) / np.linalg.norm(b - a)
                               for a, b in zip(points, points[1:])],
                "hand": rest[PART_BY_KEY[f"{side}_hand"]][:3, :3].copy(),
                "rotations": {p: rest[PART_BY_KEY[f"{side}_{p}"]][:3, :3].copy()
                              for p in ("upper_arm", "forearm")},
            }
        for task in ("positions", "vectors"):
            for mode in ("staged", "joint"):
                with self.subTest(task=task, mode=mode):
                    sample = {"sides": sides, "task": task, "solver_mode": mode,
                              "segment_orientation": True, "segment_weight": .15,
                              "scale": "uniform"}
                    previous = {s: m.neutral.copy() for s, m in models.items()}
                    result = solver.solve(sample, previous, 120, 240, .05)
                    for side, model in models.items():
                        human = sides[side]
                        neutral = tuple(rest[PART_BY_KEY[f"{side}_{p}"]][:3, 3].copy()
                                        for p in ("upper_arm", "forearm", "hand"))
                        rotation = palm_target(axes, human["hand"],
                            rest[PART_BY_KEY[f"{side}_hand"]][:3, :3],
                            model.tool_frame(model.neutral)[:3, :3])
                        goals = segment_targets(model, axes, human["rotations"],
                            {p: rest[PART_BY_KEY[f"{side}_{p}"]][:3, :3]
                             for p in ("upper_arm", "forearm")})
                        positions = (position_targets(model, axes, neutral, human["points"])
                                     if task == "positions" else None)
                        q, error, orient = model.fit(
                            [axes @ v for v in human["directions"]], previous[side],
                            joint_steps(120, 240, .05), rotation, positions=positions,
                            segment_rotations=goals, segment_weight=.15, solver_mode=mode)
                        np.testing.assert_allclose(result.targets[side], q, rtol=0, atol=1e-12)
                        np.testing.assert_allclose(result.errors[side], [error, orient], atol=1e-12)
                        np.testing.assert_array_equal(previous[side], model.neutral)
                        self.assertEqual(result.side_ms[side], 0.)

    def test_runtime_idle_never_writes_and_always_closes(self):
        now = [10.]
        stopped = [False]
        reports, failures = [], []
        bridge = Mock()
        bridge.data = scene_data()
        bridge.hand_angles = {}
        bridge.exchange.return_value = {s: np.zeros(6) for s in bridge.data["arms"]}
        def sleep_ms(ms):
            now[0] += ms / 1000
            stopped[0] = now[0] > 10.35
        with tempfile.TemporaryDirectory() as folder:
            runtime = SceneRuntime(
                23000, Mailbox(), 120, None, True, 240, HandMailbox(),
                reports.append, failures.append, lambda: stopped[0], sleep_ms,
                bridge_factory=lambda port: bridge, clock=lambda: now[0], log_folder=folder)
            runtime.run()
            rows = [json.loads(line) for line in next(Path(folder).glob("*.jsonl")).read_text().splitlines()]
        self.assertFalse(failures)
        self.assertTrue(reports[0]["connected"])
        self.assertEqual(reports[0]["config"]["version"], 2)
        self.assertTrue(bridge.exchange.call_count >= 2)
        self.assertTrue(all((not call.args or (len(call.args) == 1 and call.args[0] is None))
                            and not call.kwargs for call in bridge.exchange.call_args_list))
        bridge.close.assert_called_once()
        self.assertEqual(rows[0]["config"]["wrist_speed_deg_s"], 240)
        self.assertEqual(rows[-1]["event"], "worker_closed")

    def test_runtime_initialization_failure_closes_connection_and_stops_hands(self):
        bridge = Mock()
        bridge.data = scene_data()
        bridge.data["arms"]["left"]["limits"] = [[0, 0]] * 6
        mailbox, hands = Mailbox(), HandMailbox()
        hands.set("left", np.zeros(15), 10., 120, "gesture")
        failures = []
        SceneRuntime(23000, mailbox, 120, None, False, 240, hands,
                     lambda row: None, failures.append, lambda: False, lambda ms: None,
                     bridge_factory=lambda port: bridge, clock=lambda: 10.).run()
        self.assertEqual(len(failures), 1)
        self.assertFalse(mailbox.get()[0])
        self.assertFalse(hands.snapshot()[0])
        bridge.close.assert_called_once()

    def test_log_throttling_and_disabled_log(self):
        now, reports = [10.], []
        with tempfile.TemporaryDirectory() as folder:
            log = SceneLog(True, {"version": 2}, folder=folder)
            diagnostics = SceneDiagnostics(reports.append, log, clock=lambda: now[0])
            diagnostics.emit("idle", "等待")
            now[0] += .2
            diagnostics.emit("idle", "等待")
            diagnostics.emit("stopped", "停止")
            now[0] += .6
            diagnostics.emit("stopped", "停止")
            log.close()
            rows = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(reports), 3)
            disabled = SceneLog(False, {}, folder=Path(folder) / "not-created")
            disabled.write({"arbitrary": object()})
            disabled.close()
            self.assertFalse((Path(folder) / "not-created").exists())

    def test_hand_storage_checks_whole_file_and_returns_unconfirmed_values(self):
        hands = {side: {"limits": [[0, 1.5]] * 15,
                       "objects": [{"path": side + str(i)} for i in range(15)]}
                 for side in ("left", "right")}
        sides = {side: {"open_deg": [0] * 15, "closed_deg": [30] * 15,
                        "speed_deg_s": 120, "paths": [obj["path"] for obj in hand["objects"]]}
                 for side, hand in hands.items()}
        original = copy.deepcopy(hands)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hands.json"
            write_hand_profiles(path, sides)
            checked = read_hand_profiles(path, hands)
            np.testing.assert_allclose(checked["left"][1], np.deg2rad([30] * 15))
            self.assertEqual(len(checked["right"]), 3)  # no authorization or confirmation
            for key, value in (("paths", ["wrong"] * 15), ("closed_deg", [180] * 15),
                               ("speed_deg_s", 721)):
                with self.subTest(key=key):
                    invalid = copy.deepcopy(sides)
                    invalid["right"][key] = value
                    write_hand_profiles(path, invalid)
                    with self.assertRaises(ValueError):
                        read_hand_profiles(path, hands)
            path.write_text(json.dumps({"version": 1, "sides": ["left"]}))
            with self.assertRaisesRegex(ValueError, "无效手侧"):
                read_hand_profiles(path, hands)
            with self.assertRaises(ValueError):
                write_hand_profiles(path, {})
        self.assertEqual(hands, original)
