"""Per-device neutral frames and hierarchical body transforms.

No translation is integrated from accelerometer readings. Segment positions
come from the legacy STL hierarchy and its fixed anthropometric offsets.
"""

from __future__ import annotations

import time

import numpy as np

from wt901.protocol import AttitudePacket
from wt901_viz.orientation import rotation_matrix

from .axis_mapping import DEFAULT_AXIS_MAPPING, mapping_matrix, normalized_mapping
from .skeleton import PARTS, PART_BY_KEY, SENSOR_KEYS, transform


class PoseEngine:
    def __init__(self, assignments: dict[str, str] | None = None,
                 axis_mappings: dict[str, tuple[str, str, str]] | None = None) -> None:
        self.assignments: dict[str, str] = {key: "" for key in SENSOR_KEYS}
        self.axis_mappings = {key: DEFAULT_AXIS_MAPPING for key in SENSOR_KEYS}
        self.continuous_basis: dict[str, np.ndarray] = {}
        self.neutral: dict[str, np.ndarray] = {}
        self.current: dict[str, np.ndarray] = {}
        self.packets: dict[str, AttitudePacket] = {}
        self.received_at: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        if assignments:
            for key, device_id in assignments.items():
                if key in self.assignments and device_id:
                    self.assign(key, device_id)
        if axis_mappings:
            for key, mapping in axis_mappings.items():
                self.set_axis_mapping(key, mapping)

    def set_axis_mapping(self, key: str, mapping: tuple[str, str, str]) -> None:
        if key not in self.axis_mappings:
            raise KeyError(f"未知身体部位：{key}")
        self.axis_mappings[key] = normalized_mapping(mapping)
        self.continuous_basis.pop(key, None)

    def set_continuous_basis(self, key: str, matrix: np.ndarray) -> None:
        """Temporarily override the signed-axis map with a calibrated SO(3) basis."""
        if key not in self.axis_mappings:
            raise KeyError(f"未知身体部位：{key}")
        basis = np.asarray(matrix, dtype=float)
        if (basis.shape != (3, 3) or not np.isfinite(basis).all() or
                not np.allclose(basis.T @ basis, np.eye(3), atol=1e-3) or
                not np.isclose(np.linalg.det(basis), 1.0, atol=1e-3)):
            raise ValueError("连续轴映射必须是右手正交旋转矩阵")
        self.continuous_basis[key] = basis.copy()

    def apply_calibration(self, key: str, device_id: str, basis: np.ndarray,
                          neutral: np.ndarray) -> None:
        """Atomically activate a fit together with the neutral frame it used."""
        if self.assignments.get(key) != device_id or not device_id:
            raise ValueError("标定的设备 ID 与当前部位分配不一致")
        reference = np.asarray(neutral, dtype=float)
        if (reference.shape != (3, 3) or not np.isfinite(reference).all() or
                not np.allclose(reference.T @ reference, np.eye(3), atol=1e-3) or
                not np.isclose(np.linalg.det(reference), 1.0, atol=1e-3)):
            raise ValueError("标定中立位不是有效旋转矩阵")
        self.set_continuous_basis(key, basis)
        self.neutral[device_id] = reference.copy()

    def clear_calibration(self, key: str) -> None:
        self.continuous_basis.pop(key, None)

    def assign(self, key: str, device_id: str) -> None:
        if key not in self.assignments:
            raise KeyError(f"未知身体部位：{key}")
        if device_id and any(other != key and bound == device_id
                             for other, bound in self.assignments.items()):
            raise ValueError(f"{device_id} 已分配到其他身体部位")
        old_id = self.assignments[key]
        self.assignments[key] = device_id
        if old_id != device_id:
            self.continuous_basis.pop(key, None)
        if old_id != device_id and device_id:
            # Reassignment starts from the next sensor pose, not its old slot's frame.
            self.neutral.pop(device_id, None)

    def feed(self, packet: AttitudePacket, received_at: float | None = None) -> None:
        device_id = packet.device_id
        orientation = rotation_matrix(packet.roll, packet.pitch, packet.yaw)
        self.current[device_id] = orientation
        self.packets[device_id] = packet
        self.received_at[device_id] = time.monotonic() if received_at is None else received_at
        self.counts[device_id] = self.counts.get(device_id, 0) + 1
        if device_id in self.assignments.values() and device_id not in self.neutral:
            self.neutral[device_id] = orientation.copy()

    def zero(self, key: str | None = None) -> None:
        """Use the current pose as neutral without changing sensor-to-body axes."""
        keys = SENSOR_KEYS if key is None else (key,)
        for selected_key in keys:
            if selected_key not in self.assignments:
                raise KeyError(f"未知身体部位：{selected_key}")
            device_id = self.assignments[selected_key]
            if device_id in self.current:
                self.neutral[device_id] = self.current[device_id].copy()

    def is_live(self, device_id: str, now: float | None = None, max_age: float = 3.0) -> bool:
        if device_id not in self.received_at:
            return False
        moment = time.monotonic() if now is None else now
        return moment - self.received_at[device_id] <= max_age

    def transforms(self) -> tuple[np.ndarray, ...]:
        """Return model transforms in PARTS order, preserving unassigned subtrees."""
        absolute_rotations: list[np.ndarray] = []
        world_transforms: list[np.ndarray] = []
        identity = np.eye(3)
        for part in PARTS:
            parent_index = part.parent
            parent_rotation = identity if parent_index is None else absolute_rotations[parent_index]
            device_id = self.assignments.get(part.key, "")
            if device_id and device_id in self.current and device_id in self.neutral:
                # Express the change in the sensor's *neutral local* frame.
                # A chosen column maps each neutral sensor axis to the model.
                delta_local = self.neutral[device_id].T @ self.current[device_id]
                basis = self.continuous_basis.get(part.key)
                if basis is None:
                    basis = mapping_matrix(self.axis_mappings[part.key])
                absolute_rotation = basis @ delta_local @ basis.T
            else:
                # Missing/unassigned parts follow their parent, including the head.
                absolute_rotation = parent_rotation.copy()
            local_rotation = parent_rotation.T @ absolute_rotation
            local_transform = transform(local_rotation, part.offset)
            world_transform = local_transform if parent_index is None else world_transforms[parent_index] @ local_transform
            absolute_rotations.append(absolute_rotation)
            world_transforms.append(world_transform)
        return tuple(world_transforms)

    def rotation_for(self, key: str) -> np.ndarray:
        """Current relative rotation of a body part in model coordinates."""
        index = PART_BY_KEY[key]
        world = self.transforms()
        parent_index = PARTS[index].parent
        if parent_index is None:
            return world[index][:3, :3]
        return world[parent_index][:3, :3].T @ world[index][:3, :3]
