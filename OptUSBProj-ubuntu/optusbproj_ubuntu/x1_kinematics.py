"""URDF-derived PoE and analytic screw Jacobians; [omega, v] convention."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .x1_retarget import ArmKinematics


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew(p) @ R, R]])


def so3_left(phi):
    t = np.linalg.norm(phi)
    K = skew(phi)
    if t < 1e-5:
        return np.eye(3) + K / 2 + K @ K / 6
    return np.eye(3) + (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K


def so3_right_inverse(phi):
    return np.linalg.inv(so3_left(-np.asarray(phi)))


def exp_se3(xi):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix()
    out[:3, 3] = so3_left(xi[:3]) @ xi[3:]
    return out


def log_se3(T):
    phi = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.r_[phi, np.linalg.solve(so3_left(phi), T[:3, 3])]


def se3_left_inverse(xi):
    """Inverse dexp via convergent ad series, including translation coupling."""
    A = np.block([[skew(xi[:3]), np.zeros((3, 3))],
                  [skew(xi[3:]), skew(xi[:3])]])
    J = np.eye(6)
    term = np.eye(6)
    for n in range(1, 40):
        term = term @ A / (n + 1)
        J += term
        if np.linalg.norm(term) < 1e-14:
            break
    return np.linalg.inv(J)


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
