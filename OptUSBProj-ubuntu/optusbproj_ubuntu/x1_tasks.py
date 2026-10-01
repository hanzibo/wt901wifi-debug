"""Shared objectives for fair numerical/analytic-Jacobian IK comparisons."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from .x1_kinematics import (adjoint, inverse, log_se3, se3_left_inverse,
                            so3_right_inverse)


@dataclass
class ArmTask:
    model: object
    elbow: np.ndarray
    wrist: np.ndarray
    previous: np.ndarray
    bend: float = 180.
    relative_rotation: np.ndarray | None = None
    target_pose: np.ndarray | None = None

    @property
    def mode(self):
        return "pose" if self.target_pose is not None else "mimic"

    def evaluate(self, q, derivative=False):
        m = self.model
        f = m.poe_frames(q)
        elbow = f[m.joint_names[3]]
        hand = f[m.joint_names[5]]
        if self.target_pose is not None:
            e = log_se3(inverse(hand) @ self.target_pose)
            weights = np.array([.30] * 3 + [5.] * 3)
            r = weights * e
            if not derivative:
                return r
            Jb = adjoint(inverse(hand)) @ m.space_jacobian(q)
            return r, weights[:, None] * (-se3_left_inverse(e) @ Jb)
        straight = .65 * np.clip((45. - self.bend) / 35., 0., 1.)
        wrist_bias = .06 if self.relative_rotation is not None else .5
        r = [5. * (hand[:3, 3] - self.wrist),
             2.5 * (elbow[:3, 3] - self.elbow),
             .18 * (q - self.previous), .08 * q,
             wrist_bias * q[4:], np.array([straight * q[3]])]
        if derivative:
            eye = np.eye(6)
            J = [5. * m.point_jacobian(q, 5, f),
                 2.5 * m.point_jacobian(q, 3, f), .18 * eye, .08 * eye,
                 wrist_bias * eye[4:], straight * eye[3:4]]
        if self.relative_rotation is not None:
            actual = elbow[:3, :3].T @ hand[:3, :3]
            e = Rotation.from_matrix(self.relative_rotation.T @ actual).as_rotvec()
            r.append(.30 * e)
            if derivative:
                angular = (m.space_jacobian(q)[:3] - m.space_jacobian(q, 3)[:3])
                J.append(.30 * so3_right_inverse(e) @ hand[:3, :3].T @ angular)
        residual = np.concatenate(r)
        return (residual, np.vstack(J)) if derivative else residual
