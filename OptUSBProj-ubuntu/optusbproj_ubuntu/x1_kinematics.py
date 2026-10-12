"""URDF-derived PoE and analytic screw Jacobians; [omega, v] convention."""
from __future__ import annotations

import numpy as np

from .x1_retarget import ArmKinematics
from .core.rigid import (skew, inverse, adjoint, so3_left, so3_right_inverse,
                         exp_se3, log_se3, se3_left_inverse)


class PoEArmKinematics(ArmKinematics):
    def __init__(self, urdf, side):
        super().__init__(urdf, side)
        T = np.eye(4)
        screws, homes = [], []
        for origin, axis in zip(self.origins, self.axes):
            T = T @ origin
            omega = T[:3, :3] @ axis
            screws.append(np.r_[omega, -np.cross(omega, T[:3, 3])])
            homes.append(T.copy())
        self.S = np.column_stack(screws)
        self.homes = homes
        self.M = homes[-1].copy()
        self.B = adjoint(inverse(self.M)) @ self.S

    def fk_space(self, q):
        T = np.eye(4)
        for i, angle in enumerate(q):
            T = T @ exp_se3(self.S[:, i] * angle)
        return T @ self.M

    def poe_frames(self, q):
        """Every link frame from the corresponding truncated space PoE."""
        T, frames = np.eye(4), {}
        for i, angle in enumerate(q):
            T = T @ exp_se3(self.S[:, i] * angle)
            frames[self.joint_names[i]] = T @ self.homes[i]
        return frames

    def fk_body(self, q):
        T = self.M.copy()
        for i, angle in enumerate(q):
            T = T @ exp_se3(self.B[:, i] * angle)
        return T

    def space_jacobian(self, q, last=5):
        T, J = np.eye(4), np.zeros((6, 6))
        for i in range(last + 1):
            J[:, i] = adjoint(T) @ self.S[:, i]
            T = T @ exp_se3(self.S[:, i] * q[i])
        return J

    def body_jacobian(self, q):
        return adjoint(inverse(self.fk_space(q))) @ self.space_jacobian(q)

    def point_jacobian(self, q, index, frames=None):
        if frames is None:
            frames = self.frames(q)
        p = frames[self.joint_names[index]][:3, 3]
        J = self.space_jacobian(q, index)
        return J[3:] - skew(p) @ J[:3]
