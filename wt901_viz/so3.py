"""Small NumPy-only SO(3) logarithm for live rotation-vector displays."""

from __future__ import annotations

import numpy as np


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the principal log vector (radians, length in [0, pi]).

    A unit quaternion avoids the sin(theta) denominator in the usual matrix
    logarithm, which is ill-conditioned close to 0 and 180 degrees.
    """
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("旋转矩阵必须是有限的 3×3 矩阵")
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(1.0 + trace)
        q = np.array([0.25 * scale,
                      (r[2, 1] - r[1, 2]) / scale,
                      (r[0, 2] - r[2, 0]) / scale,
                      (r[1, 0] - r[0, 1]) / scale])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + r[i, i] - r[j, j] - r[k, k]))
        q = np.zeros(4)
        q[i + 1] = 0.25 * scale
        q[0] = (r[k, j] - r[j, k]) / scale
        q[j + 1] = (r[j, i] + r[i, j]) / scale
        q[k + 1] = (r[k, i] + r[i, k]) / scale
    q /= np.linalg.norm(q)
    # q and -q describe the same rotation. Choose the shortest rotation.
    if q[0] < 0.0:
        q = -q
    xyz_norm = float(np.linalg.norm(q[1:]))
    if xyz_norm < 1e-12:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(xyz_norm, q[0])
    return q[1:] * (angle / xyz_norm)
