"""SO(3)/SE(3) primitives extracted without numerical changes.

Column vectors; T maps source coordinates to destination coordinates.
Twists are [omega, v], rotation vectors in radians, translations in metres
when callers use SI frames. These functions do not convert units or validate
input; callers retain responsibility for valid rigid transforms.
log_se3 uses SciPy's principal rotation logarithm, not an unwrapped angle.
The small-angle threshold (1e-5) and dexp series termination are unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation


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
