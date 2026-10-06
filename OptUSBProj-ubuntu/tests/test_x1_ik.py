"""Independent FK, finite-difference Jacobian and four-mode IK checks."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT.parent)]
from optusbproj_ubuntu.x1_kinematics import (PoEArmKinematics, inverse,
                                          adjoint, log_se3, exp_se3)
from optusbproj_ubuntu.x1_tasks import ArmTask
from optusbproj_ubuntu.x1_ik import solve

URDF = PROJECT.parent / "urdf-model/pkg_robot_model/urdf/pkg_robot_model.urdf"


class IKTests(unittest.TestCase):
    def test_space_body_and_urdf_fk_and_jacobians(self):
        rng = np.random.default_rng(31)
        for side in ("left", "right"):
            m = PoEArmKinematics(URDF, side)
            for _ in range(10):
                q = rng.uniform(-.7, .7, 6)
                T = m.frames(q)[m.joint_names[-1]]
                for name, frame in m.poe_frames(q).items():
                    np.testing.assert_allclose(frame, m.frames(q)[name], atol=1e-10)
                np.testing.assert_allclose(m.fk_space(q), T, atol=1e-10)
                np.testing.assert_allclose(m.fk_body(q), T, atol=1e-10)
                Js, Jb = m.space_jacobian(q), m.body_jacobian(q)
                np.testing.assert_allclose(Jb, adjoint(inverse(T)) @ Js, atol=1e-10)
                for i in range(6):
                    nextq = q.copy(); nextq[i] += 1e-6
                    nextT = m.fk_space(nextq)
                    np.testing.assert_allclose(log_se3(nextT @ inverse(T)) / 1e-6,
                                               Js[:, i], atol=1e-6)
                    np.testing.assert_allclose(log_se3(inverse(T) @ nextT) / 1e-6,
                                               Jb[:, i], atol=1e-6)

    def test_all_task_jacobians_match_central_differences(self):
        rng = np.random.default_rng(32)
        for side in ("left", "right"):
            m = PoEArmKinematics(URDF, side)
            for mode in ("mimic", "pose"):
                for _ in range(4):
                    q = rng.uniform(-.5, .5, 6)
                    targetq = rng.uniform(-.6, .6, 6)
                    _, elbow, wrist = m.points(targetq)
                    task = ArmTask(m, elbow, wrist, np.zeros(6), 20,
                        m.wrist_relative_rotation(targetq),
                        m.fk_space(targetq) if mode == "pose" else None)
                    r, J = task.evaluate(q, True)
                    numeric = np.empty_like(J)
                    for i in range(6):
                        delta = np.zeros(6); delta[i] = 1e-6
                        numeric[:, i] = (task.evaluate(q + delta) - task.evaluate(q - delta)) / 2e-6
                    np.testing.assert_allclose(J, numeric, atol=2e-6)

    def test_pose_solvers_reach_fk_generated_targets(self):
        for side in ("left", "right"):
            m = PoEArmKinematics(URDF, side)
            targetq = np.array([.25, -.2, .12, -.3, .4, .2])
            target = m.fk_space(targetq)
            _, elbow, wrist = m.points(targetq)
            for backend in ("least_squares", "jacobian"):
                q = targetq + np.array([.05, -.05, .04, .03, -.08, .04])
                for _ in range(12):
                    task = ArmTask(m, elbow, wrist, q, target_pose=target)
                    result = solve(task, backend, .15)
                    self.assertTrue(result.success)
                    self.assertLessEqual(np.max(np.abs(result.q - q)), .150001)
                    q = result.q
                self.assertLess(result.position_error_m, .001)
                self.assertLess(result.orientation_error_deg, .2)

    def test_mimic_residual_matches_existing_objective(self):
        from optusbproj_ubuntu.x1_retarget import fit_arm
        m = PoEArmKinematics(URDF, "left")
        q = np.array([.2, -.1, .1, -.2, .3, .1])
        _, elbow, wrist = m.points(q)
        task = ArmTask(m, elbow, wrist, np.zeros(6), 15,
                       m.wrist_relative_rotation(q))
        legacy, _, _, _ = fit_arm(m, elbow, wrist, np.zeros(6),
                                  human_bend_deg=15,
                                  target_wrist_rotation=task.relative_rotation)
        new = solve(task, "least_squares")
        np.testing.assert_allclose(new.q, legacy, atol=1e-5)
        analytic = solve(task, "jacobian")
        self.assertLess(np.linalg.norm(analytic.q - legacy), .015)

    def test_unreachable_target_and_near_pi_are_finite_and_bounded(self):
        m = PoEArmKinematics(URDF, "right")
        T = m.M.copy(); T[:3, 3] += [2., 1., .5]
        T[:3, :3] = Rotation.from_rotvec([np.pi - .01, 0, 0]).as_matrix() @ T[:3, :3]
        np.testing.assert_allclose(exp_se3(log_se3(T)), T, atol=1e-10)
        _, elbow, wrist = m.points(np.zeros(6))
        for backend in ("least_squares", "jacobian"):
            result = solve(ArmTask(m, elbow, wrist, np.zeros(6), target_pose=T), backend, .02)
            self.assertTrue(np.isfinite(result.q).all())
            self.assertLessEqual(np.max(np.abs(result.q)), .020001)
            self.assertGreater(result.position_error_m, 1.)


if __name__ == "__main__":
    unittest.main()
