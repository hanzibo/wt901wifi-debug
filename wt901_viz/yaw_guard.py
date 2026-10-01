"""Pure-NumPy gyro guard for unreliable ZYX yaw near the pitch singularity.

The correction is deliberately local to the GUI. It does not write WT901
registers or pretend that Euler yaw is observable at a 90-degree pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .orientation import rotation_matrix
from .so3_filter import SO3Health, check_so3


def _z_rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _rotation_step(body_gyro_dps: np.ndarray, dt: float) -> np.ndarray:
    vector = np.deg2rad(body_gyro_dps) * dt
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _yaw_degrees(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))


def _near_branch(value: float, reference: float) -> float:
    return reference + (value - reference + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class GuardedPose:
    rotation: np.ndarray
    active: bool
    entered: bool
    exited: bool
    yaw_offset_deg: float
    heading_deg: float
    gap_clamped: bool
    rejoin_jump_deg: float


class YawGuard:
    """Guard one sensor's pose using body-frame 3-axis angular velocity.

    Enter at ``abs(pitch) >= enter_deg`` and leave at
    ``abs(pitch) <= exit_deg``. After leaving, retain a world-Z offset so the
    next raw Euler sample does not immediately undo the gyro-protected pose.
    ``heading_deg`` is an integrated world-vertical angular displacement, not
    ZYX Euler yaw while pitch is near 90 degrees.
    """

    def __init__(self, enter_deg: float = 80.0, exit_deg: float = 75.0,
                 max_gap_s: float = 0.1) -> None:
        if not (0.0 < exit_deg < enter_deg < 90.0):
            raise ValueError("require 0 < exit_deg < enter_deg < 90")
        self.enter_deg = float(enter_deg)
        self.exit_deg = float(exit_deg)
        self.max_gap_s = max_gap_s
        self.rotation: np.ndarray | None = None
        self.active = False
        self.yaw_offset_deg = 0.0
        self.heading_deg = 0.0
        self._heading_offset_deg = 0.0
        self._raw_yaw_unwrapped_deg = 0.0
        self._previous_yaw_deg: float | None = None
        self._previous_gyro: np.ndarray | None = None
        self._previous_time_s: float | None = None
        self.last_health = SO3Health(0.0, 0.0, 0.0, False)

    def update(self, time_s: float, roll_deg: float, pitch_deg: float,
               yaw_deg: float, gyro_xyz_dps: tuple[float, float, float]) -> GuardedPose:
        gyro = np.asarray(gyro_xyz_dps, dtype=float)
        if not (np.isfinite(gyro).all() and np.isfinite(
                (time_s, roll_deg, pitch_deg, yaw_deg)).all()):
            raise ValueError("non-finite IMU value")
        if self._previous_time_s is not None and time_s <= self._previous_time_s:
            raise ValueError("sample times must increase")
        raw = rotation_matrix(roll_deg, pitch_deg, yaw_deg, "ZYX")
        if self._previous_yaw_deg is None:
            self._raw_yaw_unwrapped_deg = yaw_deg
        else:
            self._raw_yaw_unwrapped_deg = _near_branch(
                yaw_deg, self._raw_yaw_unwrapped_deg)
        entered = exited = gap_clamped = False
        jump = 0.0
        if self.rotation is None:
            self.rotation = raw.copy()
            self.heading_deg = self._raw_yaw_unwrapped_deg
            self.active = abs(pitch_deg) >= self.enter_deg
            entered = self.active
        else:
            dt = time_s - self._previous_time_s  # type: ignore[operator]
            if dt > self.max_gap_s:
                dt = self.max_gap_s
                gap_clamped = True
            averaged_gyro = (self._previous_gyro + gyro) * 0.5  # type: ignore[operator]
            if not self.active and abs(pitch_deg) >= self.enter_deg:
                self.active = True
                entered = True
            if self.active:
                previous = self.rotation
                self.rotation = previous @ _rotation_step(averaged_gyro, dt)
                self.heading_deg += float((previous @ averaged_gyro)[2]) * dt
                if abs(pitch_deg) <= self.exit_deg:
                    self.yaw_offset_deg = _near_branch(
                        _yaw_degrees(self.rotation) - yaw_deg, self.yaw_offset_deg)
                    self._heading_offset_deg = self.heading_deg - self._raw_yaw_unwrapped_deg
                    rejoined = _z_rotation(self.yaw_offset_deg) @ raw
                    delta = self.rotation.T @ rejoined
                    jump = math.degrees(math.acos(float(np.clip(
                        (np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))))
                    self.rotation = rejoined
                    self.active = False
                    exited = True
            else:
                self.rotation = _z_rotation(self.yaw_offset_deg) @ raw
                self.heading_deg = self._raw_yaw_unwrapped_deg + self._heading_offset_deg
        self.rotation, self.last_health = check_so3(self.rotation)
        self._previous_time_s = time_s
        self._previous_gyro = gyro
        self._previous_yaw_deg = yaw_deg
        return GuardedPose(self.rotation.copy(), self.active, entered, exited,
                           self.yaw_offset_deg, self.heading_deg, gap_clamped, jump)


class PacketClock:
    """Monotonic elapsed time from the WT packet clock, including midnight."""

    def __init__(self) -> None:
        self._first_tick: float | None = None
        self._last_tick: float | None = None
        self._last_elapsed = 0.0
        self._day_offset = 0.0

    def elapsed(self, packet) -> float:
        tick = (packet.hour * 3600 + packet.minute * 60 + packet.second
                + packet.millisecond / 1000.0)
        if self._last_tick is not None and tick + self._day_offset < self._last_tick - 43200:
            self._day_offset += 86400.0
        tick += self._day_offset
        if self._first_tick is None:
            self._first_tick = tick
            elapsed = 0.0
        else:
            elapsed = tick - self._first_tick
            if elapsed <= self._last_elapsed or elapsed - self._last_elapsed > 1.0:
                # A duplicate/resynchronized device timestamp should not
                # integrate an arbitrarily long gyro interval.
                elapsed = self._last_elapsed + 0.02
                self._first_tick = tick - elapsed
        self._last_tick = tick
        self._last_elapsed = elapsed
        return elapsed


class GyroOnly:
    """Relative SO(3) attitude from body-frame gyro alone; never reads RPY.

    The first sample defines the identity orientation. A missing interval is
    integrated using the reported sample timestamps, but callers should flag
    large gaps because unobserved motion cannot be reconstructed reliably.
    """

    def __init__(self) -> None:
        self.rotation = np.eye(3)
        self.previous_time_s: float | None = None
        self.previous_gyro: np.ndarray | None = None
        self.last_health = SO3Health(0.0, 0.0, 0.0, False)

    def update(self, time_s: float,
               gyro_xyz_dps: tuple[float, float, float]) -> np.ndarray:
        gyro = np.asarray(gyro_xyz_dps, dtype=float)
        if not (np.isfinite(time_s) and np.isfinite(gyro).all()):
            raise ValueError("non-finite gyro value")
        if self.previous_time_s is not None:
            dt = time_s - self.previous_time_s
            if dt <= 0:
                raise ValueError("sample times must increase")
            average_gyro = (self.previous_gyro + gyro) * 0.5
            self.rotation = self.rotation @ _rotation_step(average_gyro, dt)
            self.rotation, self.last_health = check_so3(self.rotation)
        self.previous_time_s = time_s
        self.previous_gyro = gyro
        return self.rotation.copy()
