"""No-camera, no-device, no-ROS checks for X1 live preview guardrails."""

from __future__ import annotations

import json
import csv
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path[:0] = [str(ROOT), str(PROJECT), str(ROOT / "urdf-model" / "script_python")]

from optusbproj_ubuntu.pose import GYRO_TILT, PoseEngine  # noqa: E402
from optusbproj_ubuntu.axis_mapping import mapping_matrix  # noqa: E402
from optusbproj_ubuntu.x1_live import (  # noqa: E402
    JOINT_NAMES, MAX_JOINT_SPEED_DEG_S, PARTS, LiveRetargeter, readiness,
)
from optusbproj_ubuntu.x1_live_recording import (  # noqa: E402
    FRAME_FIELDS, X1DiagnosticRecorder,
)
from x1_imu_live import parse_frame  # noqa: E402


URDF = ROOT / "urdf-model" / "pkg_robot_model" / "urdf" / "pkg_robot_model.urdf"


def prepared_engine() -> PoseEngine:
    engine = PoseEngine({part: f"WT_{part}" for part in PARTS})
    stamp = time.monotonic()
    for part in PARTS:
        device = engine.assignments[part]
        engine.apply_calibration(part, device, np.eye(3), np.eye(3))
        engine.current[device] = np.eye(3)
        engine.received_at[device] = stamp
    return engine


def with_manual_left_hand(engine: PoseEngine) -> PoseEngine:
    engine.assign("left_hand", "WT_left_hand")
    engine.set_axis_mapping("left_hand", ("+Y", "-X", "+Z"))
    engine.neutral["WT_left_hand"] = np.eye(3)
    engine.current["WT_left_hand"] = np.eye(3)
    engine.received_at["WT_left_hand"] = time.monotonic()
    return engine


class X1LiveTests(unittest.TestCase):
    def test_four_combinations_and_pose_recording_comparison(self):
        from x1_compare_ik import compare
        for backend in ("least_squares", "jacobian"):
            for mode in ("mimic", "pose"):
                engine = with_manual_left_hand(prepared_engine())
                engine.assign("right_hand", "WT_right_hand")
                engine.apply_calibration("right_hand", "WT_right_hand", np.eye(3), np.eye(3))
                engine.current["WT_right_hand"] = np.eye(3)
                engine.received_at["WT_right_hand"] = time.monotonic()
                bridge = LiveRetargeter(engine, URDF, follow_left_hand=True,
                    follow_right_hand=True, backend=backend, task_mode=mode)
                with tempfile.TemporaryDirectory() as directory:
                    recorder = X1DiagnosticRecorder(Path(directory), bridge, URDF)
                    stamp = time.monotonic()
                    for index in range(4):
                        moment = stamp + index * .05
                        for device in engine.assignments.values():
                            if device:
                                engine.received_at[device] = moment
                        q, status = bridge.step(now=moment)
                        self.assertTrue(np.isfinite(q).all())
                        self.assertLess(max(abs(v) for v in q), .03)
                        for side in ("left", "right"):
                            self.assertLess(status[side]["world_orientation_error_deg"], .2)
                            self.assertIsNotNone(status[side]["solver_ms"])
                        recorder.write_frame(bridge, q, status,
                            tick_mono_ns=round(moment * 1e9), send_seq=index+1,
                            ack_seq=None, ack_recv_mono_ns=None, ros_subscribers=1)
                    recorder.close()
                    if backend == "jacobian" and mode == "pose":
                        report = compare(recorder.path, Path(directory) / "comparison", URDF)
                        self.assertEqual(len(report["runs"]), 4)
                        for stats in report["runs"].values():
                            self.assertEqual(stats["right"]["valid_frames"], 4)

    def test_complete_pose_requires_hands_and_holds_side_on_stale_hand(self):
        self.assertTrue(readiness(prepared_engine(), task_mode="pose"))
        engine = with_manual_left_hand(prepared_engine())
        engine.assign("right_hand", "WT_right_hand")
        engine.neutral["WT_right_hand"] = np.eye(3)
        engine.current["WT_right_hand"] = np.eye(3)
        engine.received_at["WT_right_hand"] = time.monotonic()
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True,
                               follow_right_hand=True, task_mode="pose", backend="jacobian")
        stamp = time.monotonic()
        q, _ = bridge.step(now=stamp)
        engine.received_at["WT_right_hand"] = stamp - .4
        held, status = bridge.step(now=stamp)
        self.assertEqual(held[6:], q[6:])
        self.assertIn("保持该侧", status["right"]["state"])
        self.assertEqual(status["left"]["state"], "跟随")
        engine.received_at["WT_right_hand"] = stamp - 1.1
        with self.assertRaisesRegex(ValueError, "手背持续断流"):
            bridge.step(now=stamp)

    def test_absolute_target_uses_hand_not_robot_current_forearm(self):
        engine = with_manual_left_hand(prepared_engine())
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
        rotation = Rotation.from_euler("y", 20, degrees=True).as_matrix()
        basis = mapping_matrix(engine.axis_mappings["left_hand"])
        engine.current["WT_left_hand"] = basis.T @ rotation @ basis
        target = bridge.absolute_hand_target("left", np.zeros(3))
        expected = bridge.pose_axes["left"] @ rotation @ bridge.pose_axes["left"].T @ bridge.robot_hand_zero["left"]
        np.testing.assert_allclose(target[:3, :3], expected, atol=1e-10)
        bridge.previous["left"] = np.ones(6) * .4
        np.testing.assert_allclose(bridge.absolute_hand_target("left", np.zeros(3)), target)

    def test_absolute_right_forward_raise_is_not_mirrored_backwards(self):
        engine = prepared_engine()
        engine.assign("right_hand", "WT_RIGHT")
        engine.neutral["WT_RIGHT"] = np.eye(3)
        front = Rotation.from_euler("y", -60, degrees=True).as_matrix()
        engine.current["WT_RIGHT"] = front
        engine.received_at["WT_RIGHT"] = time.monotonic()
        bridge = LiveRetargeter(engine, URDF, follow_right_hand=True)
        target = bridge.absolute_hand_target("right", np.zeros(3))
        np.testing.assert_allclose(target[:3, :3] @ bridge.robot_hand_zero["right"].T,
                                   front, atol=1e-10)
    def test_preview_stops_when_orientation_algorithm_changes(self):
        engine = prepared_engine()
        bridge = LiveRetargeter(engine, URDF)
        engine.set_neutral_for_mode(engine.assignments["left_upper_arm"],
                                    GYRO_TILT, np.eye(3))
        engine.set_algorithm_mode("left_upper_arm", GYRO_TILT)
        self.assertTrue(bridge.calibration_changed())

    def test_requires_four_fresh_calibrated_devices(self):
        engine = prepared_engine()
        self.assertEqual(readiness(engine), [])
        engine.clear_calibration("left_forearm")
        self.assertTrue(any("left_forearm" in problem for problem in readiness(engine)))
        engine = prepared_engine()
        engine.received_at[engine.assignments["right_forearm"]] -= 2
        self.assertTrue(any("right_forearm" in problem for problem in readiness(engine)))

    def test_neutral_and_side_local_stale_hold(self):
        engine = prepared_engine()
        bridge = LiveRetargeter(engine, URDF)
        stamp = time.monotonic()
        q, status = bridge.step(now=stamp)
        self.assertEqual(len(q), 12)
        self.assertLess(max(abs(value) for value in q), 0.03)
        self.assertEqual(status["left"]["state"], "跟随")
        engine.received_at[engine.assignments["left_forearm"]] = stamp - 0.3
        q2, status2 = bridge.step(now=stamp)
        self.assertEqual(q2[:6], q[:6])
        self.assertIn("保持该侧", status2["left"]["state"])
        self.assertEqual(status2["right"]["state"], "跟随")

    def test_left_hand_mode_accepts_manual_or_calibrated_fresh_device(self):
        engine = prepared_engine()
        self.assertTrue(any("手背设备" in item for item in
                            readiness(engine, follow_left_hand=True)))
        with_manual_left_hand(engine)
        self.assertEqual(readiness(engine, follow_left_hand=True), [])
        engine.set_continuous_basis("left_hand", np.eye(3))
        self.assertEqual(readiness(engine, follow_left_hand=True), [])
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
        engine.set_continuous_basis("left_hand", Rotation.from_euler(
            "z", 5, degrees=True).as_matrix())
        self.assertTrue(bridge.calibration_changed())

    def test_both_hands_require_distinct_devices(self):
        engine = with_manual_left_hand(prepared_engine())
        with self.assertRaises(ValueError):
            engine.assign("right_hand", "WT_left_hand")
        # Readiness also guards a malformed/restored assignment snapshot.
        engine.assignments["right_hand"] = "WT_left_hand"
        self.assertTrue(readiness(engine, follow_left_hand=True,
                                  follow_right_hand=True))

    def test_left_hand_relative_target_and_stale_wrist_hold(self):
        engine = with_manual_left_hand(prepared_engine())
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
        self.assertLess(np.linalg.norm(bridge.left_hand_target() -
                                      bridge.robot_wrist_zero["left"]), 1e-8)
        basis = mapping_matrix(engine.axis_mappings["left_hand"])
        wanted = Rotation.from_euler("z", 25, degrees=True).as_matrix()
        engine.current["WT_left_hand"] = basis.T @ wanted @ basis
        stamp = time.monotonic()
        q = None
        for index in range(15):
            moment = stamp + 0.05 * index
            engine.received_at["WT_left_hand"] = moment
            for part in PARTS:
                engine.received_at[engine.assignments[part]] = moment
            q, status = bridge.step(now=moment)
        self.assertEqual(status["left"]["hand_state"], "跟随")
        self.assertIsNotNone(status["left"]["hand_orientation_error_deg"])
        self.assertGreater(np.linalg.norm(q[4:6]), 0.01)
        self.assertGreater(q[4], np.deg2rad(15))
        self.assertLess(abs(q[5]), np.deg2rad(2))
        stale_time = stamp + 1.1
        for part in PARTS:
            engine.received_at[engine.assignments[part]] = stale_time
        _, stale = bridge.step(now=stale_time)
        self.assertIn("手背断流", stale["left"]["hand_state"])
        engine.zero("left_hand")
        with self.assertRaisesRegex(ValueError, "中立位已改变"):
            bridge.step(now=stale_time)

    def test_hand_diagnostic_snapshots_mapping_target_and_error(self):
        engine = with_manual_left_hand(prepared_engine())
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
        with tempfile.TemporaryDirectory() as directory:
            recorder = X1DiagnosticRecorder(Path(directory), bridge, URDF)
            tick = time.monotonic_ns()
            q, status = bridge.step(now=tick / 1e9)
            recorder.write_frame(bridge, q, status, tick_mono_ns=tick,
                                 send_seq=1, ack_seq=None,
                                 ack_recv_mono_ns=None, ros_subscribers=0)
            recorder.close()
            manifest = json.loads((recorder.path / "manifest.json").read_text())
            self.assertEqual(manifest["left_hand"]["mapping"], ["+Y", "-X", "+Z"])
            with (recorder.path / "frames.csv").open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["left_hand_device_id"], "WT_left_hand")
            self.assertEqual(row["left_hand_state"], "跟随")
            self.assertAlmostEqual(float(row["left_hand_axial_deg"]), 0.0,
                                   places=2)
            self.assertAlmostEqual(float(row["left_hand_orientation_error_deg"]), 0.0,
                                   places=2)
            self.assertNotEqual(row["left_hand_target_rotvec_y"], "")

    def test_thumb_side_deviation_drives_second_wrist_joint(self):
        engine = with_manual_left_hand(prepared_engine())
        basis = mapping_matrix(engine.axis_mappings["left_hand"])
        thumb_side = Rotation.from_euler("x", -20, degrees=True).as_matrix()
        engine.current["WT_left_hand"] = basis.T @ thumb_side @ basis
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
        stamp = time.monotonic()
        for index in range(15):
            moment = stamp + 0.05 * index
            engine.received_at["WT_left_hand"] = moment
            for part in PARTS:
                engine.received_at[engine.assignments[part]] = moment
            q, status = bridge.step(now=moment)
        self.assertEqual(status["left"]["hand_state"], "跟随")
        self.assertGreater(q[5], np.deg2rad(4))
        self.assertLess(abs(q[4]), np.deg2rad(2))

    def test_wrist_discrimination_hand_only_limit_vs_common_rotation(self):
        basis = mapping_matrix(("+Y", "-X", "+Z"))
        for hand_and_forearm in (False, True):
            engine = with_manual_left_hand(prepared_engine())
            bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
            turn = Rotation.from_euler("z", 90, degrees=True).as_matrix()
            engine.current["WT_left_hand"] = basis.T @ turn @ basis
            if hand_and_forearm:
                engine.current[engine.assignments["left_forearm"]] = turn
            stamp = time.monotonic()
            for index in range(60):
                moment = stamp + 0.05 * index
                for device in engine.assignments.values():
                    if device:
                        engine.received_at[device] = moment
                q, status = bridge.step(now=moment)
            if hand_and_forearm:
                self.assertAlmostEqual(status["left"]["hand_axial_deg"], 0.0,
                                       delta=0.1)
                self.assertAlmostEqual(np.rad2deg(q[4]), 0.0, delta=0.5)
            else:
                self.assertAlmostEqual(status["left"]["hand_axial_deg"], 90.0,
                                       delta=0.1)
                self.assertGreater(np.rad2deg(q[4]), 75.0)
                self.assertLessEqual(np.rad2deg(q[4]), 90.0)
                self.assertLess(status["left"]["hand_orientation_error_deg"], 15.0)

    def test_both_hands_mirror_directions_and_independent_stale_hold(self):
        for axis, angle in (("z", 90), ("z", -90), ("x", 20)):
            engine = with_manual_left_hand(prepared_engine())
            engine.assign("right_hand", "WT_right_hand")
            basis = Rotation.from_euler("z", 17, degrees=True).as_matrix()
            engine.apply_calibration("right_hand", "WT_right_hand", basis, np.eye(3))
            engine.current["WT_right_hand"] = np.eye(3)
            engine.received_at["WT_right_hand"] = time.monotonic()
            bridge = LiveRetargeter(engine, URDF, follow_left_hand=True,
                                   follow_right_hand=True)
            turn = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
            engine.current["WT_right_hand"] = basis.T @ turn @ basis
            stamp = time.monotonic()
            for index in range(60):
                moment = stamp + 0.05 * index
                for device in engine.assignments.values():
                    if device:
                        engine.received_at[device] = moment
                q, status = bridge.step(now=moment)
            self.assertLess(max(abs(v) for v in q[:6]), 0.03)
            self.assertEqual(status["right"]["hand_state"], "跟随")
            if axis == "z":
                self.assertGreater(-np.sign(angle) * np.rad2deg(q[10]), 75)
                self.assertLess(abs(np.rad2deg(q[11])), 1)
            else:
                self.assertLess(np.rad2deg(q[11]), -15)
                self.assertLess(abs(np.rad2deg(q[10])), 1)
            engine.received_at["WT_right_hand"] = moment - 0.4
            _, status = bridge.step(now=moment)
            self.assertIn("手背断流", status["right"]["hand_state"])
            self.assertEqual(status["left"]["hand_state"], "跟随")
            with tempfile.TemporaryDirectory() as directory:
                recorder = X1DiagnosticRecorder(Path(directory), bridge, URDF)
                recorder.write_frame(bridge, q, status,
                    tick_mono_ns=round(moment * 1e9), send_seq=1, ack_seq=None,
                    ack_recv_mono_ns=None, ros_subscribers=1)
                recorder.close()
                manifest = json.loads((recorder.path / "manifest.json").read_text())
                self.assertTrue(manifest["right_hand"]["calibrated"])
                with (recorder.path / "frames.csv").open() as file:
                    row = next(csv.DictReader(file))
                self.assertEqual(row["right_hand_device_id"], "WT_right_hand")
                self.assertNotEqual(row["right_hand_target_rotvec_x"], "")
            engine.zero("right_hand")
            self.assertTrue(bridge.calibration_changed())

    def test_thumb_deviation_spatial_direction_at_neutral_and_raised_arm(self):
        bridge = LiveRetargeter(prepared_engine(), URDF)
        for side in ("left", "right"):
            model = bridge.models[side]
            names = model.joint_names
            home_forearm = model.frames(np.zeros(6))[names[3]][:3, :3]
            axes = bridge.hand_to_robot_axes_by_side[side]
            # Both sides' model +X must align with physical base +X at rest.
            np.testing.assert_allclose(home_forearm @ axes[:, 0], [1, 0, 0], atol=1e-4)
            np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(axes), 1.0, places=10)
            for arm in (np.zeros(6), np.array([.3, -.4, .2, -.2, 0, 0])):
                zero = model.frames(arm)
                forearm = zero[names[3]][:3, :3]
                for angle in (-20, 20):
                    human = Rotation.from_euler("x", angle, degrees=True).as_matrix()
                    target = bridge.hand_target(side, human)
                    target_world_delta = forearm @ target @ zero[names[5]][:3, :3].T
                    physical_axis = forearm @ home_forearm.T @ np.array([1., 0., 0.])
                    wanted = Rotation.from_rotvec(physical_axis * np.deg2rad(angle)).as_matrix()
                    np.testing.assert_allclose(target_world_delta, wanted, atol=1e-4)
                    q = arm.copy(); q[5] = np.deg2rad(-angle)
                    actual = model.frames(q)[names[5]][:3, :3] @ zero[names[5]][:3, :3].T
                    np.testing.assert_allclose(actual, wanted, atol=1e-4)

    def test_right_thumb_deviation_negative_joint56_with_both_backends(self):
        for backend in ("least_squares", "jacobian"):
            for angle in (-20, 20):
                engine = prepared_engine()
                engine.assign("right_hand", "WT_RIGHT")
                engine.neutral["WT_RIGHT"] = np.eye(3)
                engine.current["WT_RIGHT"] = Rotation.from_euler("x", angle, degrees=True).as_matrix()
                engine.received_at["WT_RIGHT"] = time.monotonic()
                bridge = LiveRetargeter(engine, URDF, follow_right_hand=True, backend=backend)
                stamp = time.monotonic()
                for index in range(25):
                    moment = stamp + index * .05
                    for device in engine.assignments.values():
                        if device:
                            engine.received_at[device] = moment
                    q, status = bridge.step(now=moment)
                self.assertGreater(-np.sign(angle) * np.rad2deg(q[11]), 15)
                self.assertLess(abs(np.rad2deg(q[10])), 1)
                self.assertLess(status["right"]["hand_orientation_error_deg"], 5)

    def test_wrist_axial_rotation_matches_physical_world_direction_on_both_sides(self):
        bridge = LiveRetargeter(prepared_engine(), URDF)
        for side in ("left", "right"):
            model = bridge.models[side]
            names = model.joint_names
            zero = model.frames(np.zeros(6))
            forearm = zero[names[3]][:3, :3]
            hand_zero = zero[names[5]][:3, :3]
            axes = bridge.hand_to_robot_axes_by_side[side]
            np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(axes), 1.0, places=10)
            # Both STL hands have +Z toward the elbow at neutral. Verify
            # actual spatial direction, not merely the sign of joint angles.
            np.testing.assert_allclose(forearm @ axes[:, 2], [0, 0, 1], atol=1e-4)
            for angle in (-30, 30):
                human = Rotation.from_euler("z", angle, degrees=True).as_matrix()
                target = bridge.hand_target(side, human)
                target_world = forearm @ target @ hand_zero.T
                wanted = Rotation.from_euler("z", angle, degrees=True).as_matrix()
                np.testing.assert_allclose(target_world, wanted, atol=1e-4)
                q = np.zeros(6)
                q[4] = np.deg2rad(angle) * (1 if side == "left" else -1)
                actual = model.frames(q)[names[5]][:3, :3] @ hand_zero.T
                np.testing.assert_allclose(actual, wanted, atol=1e-4)

    def test_palm_flexion_is_reported_as_unreachable_not_falsely_retargeted(self):
        engine = with_manual_left_hand(prepared_engine())
        basis = mapping_matrix(engine.axis_mappings["left_hand"])
        flexion = Rotation.from_euler("y", 25, degrees=True).as_matrix()
        engine.current["WT_left_hand"] = basis.T @ flexion @ basis
        bridge = LiveRetargeter(engine, URDF, follow_left_hand=True)
        stamp = time.monotonic()
        for index in range(15):
            moment = stamp + 0.05 * index
            engine.received_at["WT_left_hand"] = moment
            for part in PARTS:
                engine.received_at[engine.assignments[part]] = moment
            q, status = bridge.step(now=moment)
        self.assertLess(np.linalg.norm(q[4:6]), np.deg2rad(2))
        self.assertGreater(status["left"]["hand_orientation_error_deg"], 20.0)

    def test_zero_or_long_disconnect_stops_preview(self):
        engine = prepared_engine()
        bridge = LiveRetargeter(engine, URDF)
        engine.neutral[engine.assignments["right_forearm"]] = np.diag([1., -1., -1.])
        with self.assertRaisesRegex(ValueError, "中立位已改变"):
            bridge.step()
        engine = prepared_engine()
        bridge = LiveRetargeter(engine, URDF)
        engine.received_at[engine.assignments["left_upper_arm"]] -= 1.5
        with self.assertRaisesRegex(ValueError, "持续断流"):
            bridge.step()

    def test_live_joint_limit_uses_elapsed_time_and_caps_long_stalls(self):
        engine = prepared_engine()
        bridge = LiveRetargeter(engine, URDF)
        stamp = time.monotonic()
        limits = []

        def fake_fit(model, elbow, wrist, previous, *, human_bend_deg,
                     max_step_rad, diagnostics):
            limits.append(np.rad2deg(max_step_rad))
            q = previous.copy()
            q[3] += max_step_rad
            return q, 0.0, 0.0, True

        with patch("optusbproj_ubuntu.x1_live.fit_arm", side_effect=fake_fit):
            frames = [bridge.step(now=stamp)[0]]
            for delta in (0.025, 0.1, 0.1, 0.0):
                frames.append(bridge.step(now=stamp + delta)[0])

        self.assertEqual(len(limits), 2 * len(frames))
        for actual, expected in zip(limits[::2], (6.0, 3.0, 9.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=4)
        for before, after, expected in zip(frames, frames[1:], (3.0, 9.0, 0.0, 0.0)):
            self.assertAlmostEqual(np.rad2deg(after[3] - before[3]), expected,
                                   places=4)
        self.assertEqual(MAX_JOINT_SPEED_DEG_S, 120.0)

    def test_loopback_frame_validation(self):
        now = time.monotonic_ns()
        frame = {"version": 1, "type": "frame", "seq": 1,
                 "sent_mono_ns": now, "joint_names": JOINT_NAMES,
                 "positions": [0.0] * 12}
        seq, q = parse_frame(json.dumps(frame).encode(), now_ns=now)
        self.assertEqual(seq, 1)
        self.assertEqual(len(q), 12)
        for bad in ({**frame, "positions": [0.0] * 11},
                    {**frame, "sent_mono_ns": now - 600_000_000},
                    {**frame, "positions": [float("nan")] * 12}):
            with self.assertRaises(ValueError):
                parse_frame(json.dumps(bad).encode(), now_ns=now)

    def test_diagnostic_trace_has_same_tick_inputs_targets_and_outputs(self):
        engine = prepared_engine()
        bridge = LiveRetargeter(engine, URDF)
        with tempfile.TemporaryDirectory() as directory:
            recorder = X1DiagnosticRecorder(Path(directory), bridge, URDF)
            tick = time.monotonic_ns()
            q, status = bridge.step(now=tick / 1e9)
            recorder.write_frame(bridge, q, status, tick_mono_ns=tick,
                                 send_seq=7, ack_seq=6,
                                 ack_recv_mono_ns=tick - 3_000_000,
                                 ros_subscribers=1)
            recorder.close("user stopped")
            self.assertEqual(len(FRAME_FIELDS), len(set(FRAME_FIELDS)))
            with (recorder.path / "frames.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["send_seq"], "7")
            self.assertEqual(rows[0]["ack_seq"], "6")
            self.assertEqual(rows[0]["left_upper_arm_device_id"],
                             engine.assignments["left_upper_arm"])
            self.assertAlmostEqual(float(rows[0]["left_target_wrist_z"]),
                                   status["left"]["target_wrist"][2])
            self.assertAlmostEqual(float(rows[0]["joint_54"]), q[9])
            manifest = json.loads((recorder.path / "manifest.json").read_text())
            self.assertEqual(manifest["frames"], 1)
            self.assertEqual(manifest["stop_reason"], "user stopped")
            self.assertEqual(len(manifest["urdf_sha256"]), 64)
            self.assertEqual(manifest["solver"]["max_joint_speed_deg_s"], 120.0)

    def test_preview_dialog_automatically_opens_and_closes_diagnostics(self):
        from PySide6 import QtWidgets
        from optusbproj_ubuntu import x1_live_gui

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host = QtWidgets.QMainWindow()
        host.engine = prepared_engine()
        host.calibration_dialog = None
        host.settings_dialog = None
        host.x1_preview_button = QtWidgets.QPushButton()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                x1_live_gui, "RECORDINGS", Path(directory)), patch.object(
                x1_live_gui.X1PreviewDialog, "_start_ros_receiver"), patch.object(
                x1_live_gui.X1PreviewDialog, '_collision_ready', return_value=True):
            dialog = x1_live_gui.X1PreviewDialog(host)
            self.assertTrue(dialog.launch_rviz_button.isEnabled())
            self.assertFalse(dialog.close_rviz_button.isEnabled())
            with patch.object(dialog, "_external_display_launch_running", return_value=True):
                dialog.start_rviz()
            self.assertIsNone(dialog.rviz_process)
            self.assertIn("已有", dialog.rviz_status.text())
            dialog.show()
            dialog.start()
            self.assertIsNotNone(dialog.recorder)
            recorded = dialog.recorder.path
            self.assertGreaterEqual(dialog.recorder.count, 1)
            before_hide = dialog.recorder.count
            dialog.close()
            self.assertFalse(dialog.isVisible())
            self.assertIsNotNone(dialog.retargeter)
            self.assertIsNotNone(dialog.recorder)
            self.assertTrue(dialog.timer.isActive())
            self.assertIn("后台随动", host.x1_preview_button.text())
            dialog._tick()
            self.assertGreater(dialog.recorder.count, before_hide)
            dialog.show()
            dialog.stop()
            self.assertIsNone(dialog.recorder)
            self.assertTrue((recorded / "frames.csv").is_file())
            self.assertIn(str(recorded), dialog.recording_status.text())
            dialog.close()
        host.close()
        self.assertIsNotNone(app)

    def test_preview_dialog_hand_toggle_records_optional_fifth_device(self):
        self._preview_dialog_hand_toggles(False)

    def test_preview_dialog_both_hand_toggles_record_six_devices(self):
        self._preview_dialog_hand_toggles(True)

    def _preview_dialog_hand_toggles(self, both):
        from PySide6 import QtWidgets
        from optusbproj_ubuntu import x1_live_gui

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        host = QtWidgets.QMainWindow()
        host.engine = with_manual_left_hand(prepared_engine())
        if both:
            host.engine.assign("right_hand", "WT_right_hand")
            host.engine.apply_calibration("right_hand", "WT_right_hand",
                                          np.eye(3), np.eye(3))
            host.engine.current["WT_right_hand"] = np.eye(3)
            host.engine.received_at["WT_right_hand"] = time.monotonic()
        host.calibration_dialog = None
        host.settings_dialog = None
        host.x1_preview_button = QtWidgets.QPushButton()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                x1_live_gui, "RECORDINGS", Path(directory)), patch.object(
                x1_live_gui.X1PreviewDialog, "_start_ros_receiver"), patch.object(
                x1_live_gui.X1PreviewDialog, '_collision_ready', return_value=True):
            dialog = x1_live_gui.X1PreviewDialog(host)
            dialog.follow_left_hand.setChecked(True)
            dialog.follow_right_hand.setChecked(both)
            if both:
                dialog.precise_pose.setChecked(True)
                dialog.backend_combo.setCurrentIndex(1)
            self.assertEqual(dialog.scale_combo.currentData(), "uniform")
            dialog.scale_combo.setCurrentIndex(1)
            self.assertTrue(dialog.start_button.isEnabled())
            dialog.start()
            self.assertFalse(dialog.follow_left_hand.isEnabled())
            self.assertFalse(dialog.follow_right_hand.isEnabled())
            self.assertFalse(dialog.precise_pose.isEnabled())
            self.assertFalse(dialog.backend_combo.isEnabled())
            self.assertFalse(dialog.scale_combo.isEnabled())
            self.assertEqual(dialog.retargeter.scale_mode, "segments")
            self.assertEqual(dialog.recorder.manifest["solver"]["scale_mode"], "segments")
            self.assertTrue(dialog.recorder.manifest["left_hand"]["enabled"])
            self.assertEqual(dialog.recorder.manifest["right_hand"]["enabled"], both)
            if both:
                self.assertIn("右手掌朝向：跟随", dialog.right_hand_status.text())
            dialog.stop()
            self.assertTrue(dialog.follow_left_hand.isEnabled())
            self.assertTrue(dialog.follow_right_hand.isEnabled())
            self.assertTrue(dialog.precise_pose.isEnabled())
            self.assertTrue(dialog.backend_combo.isEnabled())
            self.assertTrue(dialog.scale_combo.isEnabled())
            dialog.close()
        host.close()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
