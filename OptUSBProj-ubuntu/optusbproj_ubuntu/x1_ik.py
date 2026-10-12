"""Two bounded IK backends, both visualization-only, sharing task residuals."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation
from .core.contracts import (JointAnglesRad, Milliseconds, DistanceMeters,
                             AngleDegrees)


@dataclass
class IKResult:
    q: JointAnglesRad
    success: bool
    iterations: int
    elapsed_ms: Milliseconds
    cost: float
    position_error_m: DistanceMeters
    orientation_error_deg: AngleDegrees | None
    at_limit: bool
    reason: str


def joint_bounds(model):
    limits = np.deg2rad(np.array([[-110, 110], [-85, 85], [-90, 90],
                                 [-125, 125], [-90, 90], [-45, 45]]).T)
    lo = np.maximum(limits[0], model.lower_limits)
    hi = np.minimum(limits[1], model.upper_limits)
    if np.any(lo >= hi):
        raise ValueError("URDF 与预览限位无有效交集")
    return lo, hi


def solve(task, backend="least_squares", max_step_rad=np.deg2rad(9)):
    if backend not in ("least_squares", "jacobian"):
        raise ValueError("未知 X1 IK 后端")
    started = time.perf_counter()
    lo, hi = joint_bounds(task.model)
    initial = np.clip(task.previous, lo, hi)
    lower = np.maximum(lo, initial - max_step_rad)
    upper = np.minimum(hi, initial + max_step_rad)
    lower, upper = lower - 1e-10, upper + 1e-10
    initial_cost = np.linalg.norm(task.evaluate(initial))**2
    if backend == "least_squares":
        result = least_squares(task.evaluate, initial, bounds=(lower, upper),
                               max_nfev=50, ftol=1e-5, xtol=1e-5, gtol=1e-5)
        q, iterations = result.x, result.nfev
        reason = "收敛" if result.success else "迭代预算耗尽"
        success = bool(np.isfinite(q).all() and (result.success or
            np.linalg.norm(task.evaluate(q))**2 < initial_cost - 1e-12))
    else:
        q = initial.copy()
        reason = "迭代预算耗尽"
        for iterations in range(1, 21):
            r, J = task.evaluate(q, derivative=True)
            cost = r @ r
            if np.linalg.norm(J.T @ r, ord=np.inf) < 1e-5:
                reason = "收敛"
                break
            # Bound the whole tick relative to its start, not each inner step.
            damping = .01
            update = lsq_linear(np.vstack((J, damping * np.eye(6))),
                np.r_[-r, np.zeros(6)], bounds=(lower - q, upper - q),
                tol=1e-6, max_iter=30).x
            accepted = False
            for alpha in (1., .5, .25, .125, .0625):
                candidate = np.clip(q + alpha * update, lower, upper)
                rr = task.evaluate(candidate)
                if rr @ rr < cost - 1e-12:
                    q, accepted = candidate, True
                    break
            if not accepted or np.linalg.norm(update) < 1e-6:
                reason = "受限或局部稳定"
                break
            if time.perf_counter() - started > .020:
                reason = "时间预算耗尽"
                break
        # A tracking step may improve while the target remains unreachable
        # within this tick's speed limit. Report that separately from accuracy.
        success = np.isfinite(q).all() and np.linalg.norm(task.evaluate(q))**2 <= initial_cost + 1e-9
    q = np.clip(q, lo, hi)
    frames = task.model.frames(q)
    hand = frames[task.model.joint_names[5]]
    rotation_error = None
    if task.target_pose is not None:
        p_error = np.linalg.norm(hand[:3, 3] - task.target_pose[:3, 3])
        rotation_error = np.rad2deg(Rotation.from_matrix(
            task.target_pose[:3, :3].T @ hand[:3, :3]).magnitude())
    else:
        p_error = np.linalg.norm(hand[:3, 3] - task.wrist)
    return IKResult(q, success, iterations, (time.perf_counter() - started) * 1000,
                    float(np.linalg.norm(task.evaluate(q))**2), float(p_error),
                    None if rotation_error is None else float(rotation_error),
                    bool(np.any(np.minimum(q - lo, hi - q) < np.deg2rad(.5))), reason)
