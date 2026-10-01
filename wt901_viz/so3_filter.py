"""SO(3) numerical checks and an experimental gyro+accelerometer tilt filter.

Conventions: rotation maps sensor/body axes to the initial neutral frame;
WT acceleration is in g and points upward when stationary. The initial
gravity direction is learned from a still sample, not assumed to be +Z.
Acceleration only corrects inclination;
without an external heading observation this filter cannot correct yaw drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


IDENTITY = np.eye(3)


def rotation_step(body_rotvec_rad: np.ndarray) -> np.ndarray:
    """Rodrigues exponential of one body-frame rotation vector."""
    vector = np.asarray(body_rotvec_rad, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return IDENTITY.copy()
    x, y, z = vector / angle
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return IDENTITY + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class SO3Health:
    orthogonality_error: float
    determinant_error: float
    distance_to_so3: float
    projected: bool


def check_so3(rotation: np.ndarray, threshold: float = 1e-8) -> tuple[np.ndarray, SO3Health]:
    """Measure true Frobenius distance to SO(3), projecting only if needed.

    The nearest proper rotation is U diag(1,1,det(UV^T)) V^T, not simply
    U V^T when the input is a reflection. The SVD is only 3x3.
    """
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    orth_error = float(np.linalg.norm(r.T @ r - IDENTITY, ord="fro"))
    det_error = abs(float(np.linalg.det(r)) - 1.0)
    u, _singular_values, vt = np.linalg.svd(r)
    proper = np.diag((1.0, 1.0, 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0))
    projected = u @ proper @ vt
    distance = float(np.linalg.norm(r - projected, ord="fro"))
    needs_projection = distance > threshold
    return (projected if needs_projection else r.copy(),
            SO3Health(orth_error, det_error, distance, needs_projection))


@dataclass(frozen=True)
class TiltEstimate:
    rotation: np.ndarray
    gyro_bias_dps: tuple[float, float, float]
    bias_ready: bool
    static: bool
    accelerometer_weight: float
    health: SO3Health


class AdaptiveTiltFilter:
    """Gyro propagation, stillness-only bias learning, adaptive gravity tilt.

    This is an experimental six-axis estimator. It has no yaw observation.
    The first sample defines identity orientation; the user should begin in
    a neutral pose and remain still briefly to acquire a gyro bias.
    """

    def __init__(self, gravity_gain_per_s: float = 0.8,
                 bias_hold_s: float = 0.8,
                 projection_threshold: float = 1e-8) -> None:
        if gravity_gain_per_s < 0 or bias_hold_s <= 0:
            raise ValueError("invalid filter gains")
        self.gravity_gain_per_s = gravity_gain_per_s
        self.bias_hold_s = bias_hold_s
        self.projection_threshold = projection_threshold
        self.rotation = IDENTITY.copy()
        self.gyro_bias_dps = np.zeros(3)
        self.bias_ready = False
        self.previous_time_s: float | None = None
        self.previous_gyro: np.ndarray | None = None
        self.previous_acc_unit: np.ndarray | None = None
        self.neutral_up: np.ndarray | None = None
        self.static_duration_s = 0.0
        self.static_rate_sum = np.zeros(3)
        self.static_samples = 0
        self.last_health = SO3Health(0.0, 0.0, 0.0, False)

    def update(self, time_s: float, gyro_xyz_dps: tuple[float, float, float],
               acc_xyz_g: tuple[float, float, float]) -> TiltEstimate:
        gyro = np.asarray(gyro_xyz_dps, dtype=float)
        acc = np.asarray(acc_xyz_g, dtype=float)
        if not (np.isfinite(time_s) and np.isfinite(gyro).all()
                and np.isfinite(acc).all()):
            raise ValueError("non-finite IMU measurement")
        if self.previous_time_s is not None and time_s <= self.previous_time_s:
            raise ValueError("sample times must increase")
        acc_norm = float(np.linalg.norm(acc))
        acc_unit = acc / acc_norm if acc_norm > 1e-6 else None
        dt = 0.0 if self.previous_time_s is None else time_s - self.previous_time_s
        gyro_norm = float(np.linalg.norm(gyro))
        direction_change = 0.0
        if acc_unit is not None and self.previous_acc_unit is not None:
            direction_change = math.degrees(math.acos(float(np.clip(
                np.dot(acc_unit, self.previous_acc_unit), -1.0, 1.0))))
        static_candidate = (acc_unit is not None and abs(acc_norm - 1.0) <= 0.08
                            and gyro_norm <= 0.8 and direction_change <= 1.0)
        if static_candidate:
            self.static_duration_s += dt
            self.static_rate_sum += gyro
            self.static_samples += 1
            if self.static_duration_s >= self.bias_hold_s:
                measured_bias = self.static_rate_sum / self.static_samples
                if not self.bias_ready:
                    self.gyro_bias_dps = measured_bias
                    self.bias_ready = True
                else:
                    alpha = 1.0 - math.exp(-dt / 20.0)
                    self.gyro_bias_dps += alpha * (measured_bias - self.gyro_bias_dps)
        else:
            self.static_duration_s = 0.0
            self.static_rate_sum[:] = 0.0
            self.static_samples = 0

        if self.neutral_up is None and self.static_duration_s >= 0.3:
            # Our rotation starts at I in the *sensor's initial frame*. The
            # gravity direction there is generally not world [0, 0, 1].
            self.neutral_up = self.rotation @ acc_unit

        magnitude_weight = float(np.clip((0.18 - abs(acc_norm - 1.0)) / 0.15, 0.0, 1.0))
        motion_weight = float(np.clip((60.0 - gyro_norm) / 50.0, 0.0, 1.0))
        weight = (magnitude_weight * motion_weight
                  if acc_unit is not None and self.neutral_up is not None else 0.0)
        if dt > 0:
            average_gyro = (self.previous_gyro + gyro) * 0.5 - self.gyro_bias_dps
            predicted = self.rotation @ rotation_step(np.deg2rad(average_gyro) * dt)
            if weight > 0:
                predicted_up_body = predicted.T @ self.neutral_up
                # Body-frame measured_up x predicted_up rotates the estimate
                # toward the measured gravity direction, without yaw feedback.
                tilt_error = np.cross(acc_unit, predicted_up_body)
                predicted = predicted @ rotation_step(
                    tilt_error * self.gravity_gain_per_s * weight * dt)
            self.rotation, self.last_health = check_so3(
                predicted, self.projection_threshold)
        self.previous_time_s = time_s
        self.previous_gyro = gyro
        self.previous_acc_unit = acc_unit
        return TiltEstimate(self.rotation.copy(),
                            tuple(float(value) for value in self.gyro_bias_dps),
                            self.bias_ready, static_candidate,
                            weight, self.last_health)
