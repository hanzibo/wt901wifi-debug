"""Opt-in CPU JIT for the existing scene IK objective; no motion authority.

SciPy still owns optimization and SO(3) logarithms. No fastmath or float32:
retain the original SE(3) small-angle translation convention in float64.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def geometry(q, previous, reference, screws, homes, neutral, lengths,
             directions, positions, use_positions):
    prefixes = np.empty((7, 4, 4))
    prefixes[0] = np.eye(4)
    for i in range(6):
        delta = q[i] - reference[i]
        x, y, z = screws[0,i]*delta, screws[1,i]*delta, screws[2,i]*delta
        K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        K2 = K @ K
        t2 = x*x+y*y+z*z
        t = np.sqrt(t2)
        if t < 1e-5:
            # Stable Rodrigues coefficients; same rotation as SciPy rotvec.
            a = 1.-t2/6.+t2*t2/120.
            b = .5-t2/24.+t2*t2/720.
            V = np.eye(3)+.5*K+K2/6.
        else:
            a = np.sin(t)/t
            b = (1.-np.cos(t))/t2
            V = np.eye(3)+b*K+((t-np.sin(t))/(t2*t))*K2
        T = np.eye(4)
        T[:3,:3] = np.eye(3)+a*K+b*K2
        T[:3,3] = V @ (screws[3:,i]*delta)
        prefixes[i+1] = prefixes[i] @ T
    points = np.empty((3,3))
    for j in range(3):
        i = 1+2*j
        points[j] = (prefixes[i] @ homes[i])[:3,3]
    base = np.empty(18)
    if use_positions:
        base[:3] = 5.*(points[2]-positions[1])
        base[3:6] = 2.5*(points[1]-positions[0])
    else:
        base[:3] = 5.*(points[1]-points[0]-lengths[0]*directions[0])
        base[3:6] = 5.*(points[2]-points[1]-lengths[1]*directions[1])
    base[6:12] = .06*(q-previous)
    base[12:18] = .008*(q-neutral)
    return base, prefixes


def enable(model):
    """Compile and verify BEFORE first solve / any serial motion is authorized."""
    reference = np.ascontiguousarray(model.reference, dtype=np.float64)
    screws = np.ascontiguousarray(model.screws, dtype=np.float64)
    homes = np.ascontiguousarray(model.homes, dtype=np.float64)
    neutral = np.ascontiguousarray(model.neutral, dtype=np.float64)
    lengths = np.asarray(model.lengths, dtype=np.float64)

    def evaluate(q, previous, directions, positions):
        return geometry(np.asarray(q, dtype=np.float64), previous, reference,
                        screws, homes, neutral, lengths,
                        np.asarray(directions, dtype=np.float64),
                        np.zeros((2,3)) if positions is None else np.asarray(positions, dtype=np.float64),
                        positions is not None)

    # Compile the exact contiguous signatures used by residual calls. Also
    # compare against the original implementation; failure must block startup.
    for q in (neutral.copy(), reference.copy(), neutral + .123):
        _, prefixes = evaluate(q, neutral.copy(), np.zeros((2,3)), np.zeros((2,3)))
        if not np.allclose(prefixes, model.prefixes(q), rtol=1e-11, atol=1e-12):
            raise RuntimeError('CPU JIT geometry differs from reference PoE')
    model._compiled_geometry = evaluate


def prepare(solver):
    """SharedArmIK or BilateralIK; never silently fall back after a failure."""
    arms = solver.solvers.values() if hasattr(solver, 'solvers') else (solver,)
    for arm in arms:
        enable(arm.model)
    return solver


def warmup():
    """Run before UDP / GUI control starts, never compile on a live handover."""
    from optusbproj_ubuntu.real_robot_arm import load_profile
    from optusbproj_ubuntu.scene_robot_model import SceneArm
    data, _, _ = load_profile('left')
    enable(SceneArm(data, [-np.pi]*6, [np.pi]*6))
