"""Per-device neutral frames and hierarchical body transforms.

No translation is integrated from accelerometer readings. Segment positions
come from the legacy STL hierarchy and its fixed anthropometric offsets.
"""

from __future__ import annotations

import time
import math
import uuid
from copy import deepcopy

import numpy as np

from wt901.protocol import AttitudePacket
from wt901_viz.orientation import rotation_matrix
from wt901_viz.so3_filter import AdaptiveTiltFilter, rotation_step
from wt901_viz.yaw_guard import PacketClock

from .axis_mapping import DEFAULT_AXIS_MAPPING, mapping_matrix, normalized_mapping
from .skeleton import PARTS, PART_BY_KEY, SENSOR_KEYS, transform
from .natural_hand import validate_reference, rotation as natural_rotation, proper


VENDOR_RPY = "vendor_rpy"
GYRO_TILT = "gyro_tilt"
GYRO_RGBD = "gyro_rgbd"
ORIENTATION_MODES = (VENDOR_RPY, GYRO_TILT, GYRO_RGBD)
MODE_LABELS = {VENDOR_RPY:"厂商 R/P/Y", GYRO_TILT:"陀螺＋加计", GYRO_RGBD:"陀螺＋加计＋RGB-D"}


class PoseEngine:
    def __init__(self, assignments: dict[str, str] | None = None,
                 axis_mappings: dict[str, tuple[str, str, str]] | None = None,
                 algorithm_modes: dict[str, str] | None = None) -> None:
        self.assignments: dict[str, str] = {key: "" for key in SENSOR_KEYS}
        self.axis_mappings = {key: DEFAULT_AXIS_MAPPING for key in SENSOR_KEYS}
        self.continuous_basis: dict[str, np.ndarray] = {}
        # Immutable (B, N) pairs, distinct from ordinary display zero frames.
        self.calibration_references = {}
        self.natural_hand_references = {}
        self.natural_hand_previews = {}
        self.neutral: dict[str, np.ndarray] = {}
        self.current: dict[str, np.ndarray] = {}
        self.algorithm_modes = {key: VENDOR_RPY for key in SENSOR_KEYS}
        self.mode_current: dict[str, dict[str, np.ndarray]] = {}
        self.mode_neutral: dict[str, dict[str, np.ndarray]] = {}
        self._tilt_filters: dict[str, AdaptiveTiltFilter] = {}
        self._device_clocks: dict[str, PacketClock] = {}
        self.epochs = {}
        self.fusion_ready = set()
        self.fusion_corrections = {}
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
        if algorithm_modes:
            for key, mode in algorithm_modes.items():
                if key not in self.algorithm_modes or mode not in ORIENTATION_MODES:
                    raise ValueError(f"无效姿态算法：{key} / {mode}")
                self.algorithm_modes[key] = mode

    def _active_mode(self, device_id: str) -> str:
        return next((self.algorithm_modes[key] for key, bound in self.assignments.items()
                     if bound == device_id), VENDOR_RPY)

    def _sync_active(self, device_id: str) -> None:
        mode = self._active_mode(device_id)
        current = self.mode_current.get(device_id, {}).get(mode)
        neutral = self.mode_neutral.get(device_id, {}).get(mode)
        if current is None:
            self.current.pop(device_id, None)
        else:
            self.current[device_id] = current.copy()
        if neutral is None:
            self.neutral.pop(device_id, None)
        else:
            self.neutral[device_id] = neutral.copy()

    def set_algorithm_mode(self, key: str, mode: str) -> None:
        if key not in self.algorithm_modes or mode not in ORIENTATION_MODES:
            raise ValueError(f"无效姿态算法：{key} / {mode}")
        device_id = self.assignments[key]
        if mode == GYRO_RGBD and (key not in ("left_upper_arm","left_forearm","right_upper_arm","right_forearm") or device_id not in self.fusion_ready):
            raise ValueError("需先应用本次实时参考系的相机标定；仅支持左右上臂/前臂")
        if (device_id and device_id in self.packets and
                mode not in self.mode_current.get(device_id, {})):
            raise ValueError("该设备尚无陀螺＋加计数据；请等待完整数据包")
        if (device_id and key in self.continuous_basis and
                mode not in self.mode_neutral.get(device_id, {})):
            raise ValueError("当前穿戴标定没有此算法的同步中立位；请在该算法下重新标定")
        if self.algorithm_modes[key]!=mode:
            self.natural_hand_previews.pop(key,None)
            if key in self.natural_hand_references:self.natural_hand_references[key]['enabled']=False
        self.algorithm_modes[key] = mode
        if device_id:
            self._sync_active(device_id)

    def neutral_for(self, device_id: str, mode: str) -> np.ndarray | None:
        return self.mode_neutral.get(device_id, {}).get(mode)

    def current_for(self, device_id: str, mode: str) -> np.ndarray | None:
        return self.mode_current.get(device_id, {}).get(mode)

    def calibration_orientation_for(self, device_id: str,
                                    mode: str) -> np.ndarray | None:
        """Return a gravity-vertical attitude for fitting, not model playback.

        The gyro filter's world frame is the sensor frame at startup. Its
        relative rotations are already correct, but guided fitting assumes
        world +Z points upward. A fixed left rotation supplies that gauge;
        the neutral matrix used for live playback remains unaligned.
        """
        current = self.current_for(device_id, mode)
        if current is None or mode not in (GYRO_TILT,GYRO_RGBD):
            return current
        estimator = self._tilt_filters.get(device_id)
        if estimator is None or estimator.neutral_up is None:
            return None
        up = np.asarray(estimator.neutral_up, dtype=float)
        up /= np.linalg.norm(up)
        target = np.array((0.0, 0.0, 1.0))
        axis = np.cross(up, target)
        sine = float(np.linalg.norm(axis))
        cosine = float(np.clip(np.dot(up, target), -1.0, 1.0))
        if sine < 1e-10:
            alignment = (np.eye(3) if cosine > 0 else
                         np.diag((1.0, -1.0, -1.0)))
        else:
            alignment = rotation_step(axis / sine * math.atan2(sine, cosine))
        return alignment @ current

    def invalidate_neutral(self, device_id: str) -> None:
        for key,assigned in self.assignments.items():
            if assigned==device_id:
                self.clear_natural_hand(key);self.calibration_references.pop(key,None)
        self.epochs[device_id] = uuid.uuid4().hex
        self.fusion_ready.discard(device_id)
        self.fusion_corrections.pop(device_id,None)
        self.mode_neutral.pop(device_id, None)
        self.neutral.pop(device_id, None)
        self.mode_current.pop(device_id, None)
        self.current.pop(device_id, None)
        self._tilt_filters.pop(device_id, None)
        self._device_clocks.pop(device_id, None)

    def set_neutral_for_mode(self, device_id: str, mode: str,
                             neutral: np.ndarray) -> None:
        if mode not in ORIENTATION_MODES:
            raise ValueError(f"无效姿态算法：{mode}")
        reference = np.asarray(neutral, dtype=float)
        if (reference.shape != (3, 3) or not np.isfinite(reference).all() or
                not np.allclose(reference.T @ reference, np.eye(3), atol=1e-3) or
                not np.isclose(np.linalg.det(reference), 1.0, atol=1e-3)):
            raise ValueError("中立位不是有效旋转矩阵")
        self.mode_neutral.setdefault(device_id, {})[mode] = reference.copy()
        self._sync_active(device_id)

    def set_axis_mapping(self, key: str, mapping: tuple[str, str, str]) -> None:
        if key not in self.axis_mappings:
            raise KeyError(f"未知身体部位：{key}")
        self.axis_mappings[key] = normalized_mapping(mapping)
        self.clear_natural_hand(key)
        self.calibration_references.pop(key,None)
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
        self.clear_natural_hand(key)
        self.calibration_references.pop(key,None)

    def apply_calibration(self, key: str, device_id: str, basis: np.ndarray,
                          neutral: np.ndarray,
                          paired_neutrals: dict[str, np.ndarray] | None = None) -> None:
        """Atomically activate a fit together with the neutral frame it used."""
        if self.assignments.get(key) != device_id or not device_id:
            raise ValueError("标定的设备 ID 与当前部位分配不一致")
        reference = np.asarray(neutral, dtype=float)
        if (reference.shape != (3, 3) or not np.isfinite(reference).all() or
                not np.allclose(reference.T @ reference, np.eye(3), atol=1e-3) or
                not np.isclose(np.linalg.det(reference), 1.0, atol=1e-3)):
            raise ValueError("标定中立位不是有效旋转矩阵")
        validated = {self.algorithm_modes[key]: reference.copy()}
        for paired_mode, paired in (paired_neutrals or {}).items():
            if paired_mode not in ORIENTATION_MODES:
                raise ValueError(f"无效姿态算法：{paired_mode}")
            candidate = np.asarray(paired, dtype=float)
            if (candidate.shape != (3, 3) or not np.isfinite(candidate).all() or
                    not np.allclose(candidate.T @ candidate, np.eye(3), atol=1e-3) or
                    not np.isclose(np.linalg.det(candidate), 1.0, atol=1e-3)):
                raise ValueError("同步中立位不是有效旋转矩阵")
            # The fit may use a gravity-aligned world gauge. The paired
            # stream neutral is in the raw estimator gauge used by playback.
            validated[paired_mode] = candidate.copy()
        self.set_continuous_basis(key, basis)
        if GYRO_TILT in validated and GYRO_RGBD not in validated:
            validated[GYRO_RGBD] = self.fusion_corrections.get(device_id,np.eye(3)) @ validated[GYRO_TILT]
        self.mode_neutral[device_id] = validated
        self.calibration_references[key] = dict(device_id=device_id,epoch=self.epochs.get(device_id),
            basis=self.continuous_basis[key].tolist(),source='wearing_calibration',
            neutral_by_mode={m:n.tolist() for m,n in validated.items()})
        self._sync_active(device_id)

    def clear_calibration(self, key: str) -> None:
        self.continuous_basis.pop(key, None)
        self.clear_natural_hand(key)
        self.calibration_references.pop(key,None)

    def clear_natural_hand(self,key):
        self.natural_hand_references.pop(key,None);self.natural_hand_previews.pop(key,None)

    def make_natural_hand_reference(self,key,anchor,neutral_pose):
        if key!='left_hand':raise ValueError('首版仅支持左手')
        device=self.assignments[key];mode=self.algorithm_modes[key]
        if not device or mode not in (VENDOR_RPY,GYRO_TILT):raise ValueError('左手需分配设备并选择厂商RPY或陀螺＋加计')
        standard=self.calibration_references.get(key)
        if standard is None:raise ValueError('缺少独立标准朝向基准：请应用左手穿戴标定；手动映射请先采集标准朝向，不能用普通归零代替')
        if standard['device_id']!=device or standard['epoch']!=self.epochs.get(device):raise ValueError('标准标定参考已失效，请重新建立')
        if mode not in standard['neutral_by_mode']:raise ValueError('标准标定缺少当前算法的中立姿态，请重新标定或在当前算法采集手动标准基准')
        r0=proper(standard['neutral_by_mode'][mode])
        b=self.continuous_basis.get(key,mapping_matrix(self.axis_mappings[key]))
        if not np.allclose(b,standard['basis'],atol=1e-6):raise ValueError('标准基准与当前映射不匹配')
        anchor=proper(anchor)
        # Always use the immutable calibration N, NEVER the last display zero.
        rebased=b @ r0.T @ anchor
        return validate_reference(dict(format='natural-palm-reference-v1',part=key,device_id=device,
            epoch=self.epochs[device],mode=mode,anchor=anchor.tolist(),basis=rebased.tolist(),
            mapping_basis=b.tolist(),standard_neutral=r0.tolist(),standard_source=standard['source'],
            neutral_pose=proper(neutral_pose).tolist(),enabled=True,source='manual-long-axis-offset'))

    def capture_manual_standard_hand(self,key,neutral):
        if key!='left_hand':raise ValueError('首版仅支持左手')
        if key in self.continuous_basis:raise ValueError('已采用穿戴标定，请重新应用原标定结果，不能单独替换与B配对的N')
        device=self.assignments[key];mode=self.algorithm_modes[key]
        if not device or not self.epochs.get(device) or mode not in (VENDOR_RPY,GYRO_TILT):raise ValueError('缺少设备或有效算法')
        n=proper(neutral);self.clear_natural_hand(key)
        self.calibration_references[key]=dict(device_id=device,epoch=self.epochs[device],
            basis=mapping_matrix(self.axis_mappings[key]).tolist(),source='explicit_manual_standard',
            neutral_by_mode={mode:n.tolist()})

    def natural_hand_error(self,data):
        key=data['part'];device=data['device_id']
        if self.assignments.get(key)!=device:return '设备分配已变化'
        if self.epochs.get(device)!=data['epoch']:return '积分参考已变化，请重新采样'
        if self.algorithm_modes[key]!=data['mode']:return '姿态算法已变化；请切回采样算法或重新采样'
        b=self.continuous_basis.get(key,mapping_matrix(self.axis_mappings[key]))
        if not np.allclose(b,data['mapping_basis'],atol=1e-6):return '穿戴映射已变化'
        standard=self.calibration_references.get(key)
        if standard is not None and 'standard_neutral' in data:
            n=standard['neutral_by_mode'].get(data['mode'])
            if n is None or not np.allclose(n,data['standard_neutral'],atol=1e-6):return '标准朝向基准已变化，请重采自然参考'
        return None

    def set_natural_hand_reference(self,data,*,preview=False):
        data=validate_reference(data);error=self.natural_hand_error(data)
        if error:raise ValueError(error)
        target=self.natural_hand_previews if preview else self.natural_hand_references
        target[data['part']]=data

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
            self.clear_natural_hand(key)
            self.calibration_references.pop(key,None)
        if old_id != device_id and device_id:
            # Reassignment starts from the next sensor pose, not its old slot's frame.
            self.invalidate_neutral(device_id)
            self._sync_active(device_id)

    def feed(self, packet: AttitudePacket, received_at: float | None = None) -> None:
        device_id = packet.device_id
        self.epochs.setdefault(device_id,uuid.uuid4().hex)
        orientation = rotation_matrix(packet.roll, packet.pitch, packet.yaw)
        currents = self.mode_current.setdefault(device_id, {})
        currents[VENDOR_RPY] = orientation
        if all(hasattr(packet, name) for name in (
                "hour", "minute", "second", "millisecond",
                "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")):
            estimator = self._tilt_filters.setdefault(device_id, AdaptiveTiltFilter())
            clock = self._device_clocks.setdefault(device_id, PacketClock())
            estimate = estimator.update(clock.elapsed(packet),
                                        (packet.gyro_x, packet.gyro_y, packet.gyro_z),
                                        (packet.acc_x, packet.acc_y, packet.acc_z))
            currents[GYRO_TILT] = estimate.rotation.copy()
            currents[GYRO_RGBD] = self.fusion_corrections.get(device_id,np.eye(3)) @ estimate.rotation
        self.packets[device_id] = packet
        self.received_at[device_id] = time.monotonic() if received_at is None else received_at
        self.counts[device_id] = self.counts.get(device_id, 0) + 1
        if device_id in self.assignments.values():
            if device_id not in self.mode_neutral:
                self.mode_neutral[device_id] = {mode: value.copy()
                                                for mode, value in currents.items()}
            self._sync_active(device_id)

    def set_fusion_correction(self, device_id, matrix):
        value=np.asarray(matrix,float)
        if value.shape!=(3,3) or not np.allclose(value.T@value,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(value),1):
            raise ValueError("纠偏矩阵必须属于SO(3)")
        self.fusion_corrections[device_id]=value.copy()
        base=self.current_for(device_id,GYRO_TILT)
        if base is not None:
            self.mode_current[device_id][GYRO_RGBD]=value@base
            self._sync_active(device_id)

    def runtime_snapshot(self, device_id, received_ns):
        estimator=self._tilt_filters.get(device_id)
        return {"host_recv_mono_ns":int(received_ns),"device_id":device_id,"epoch":self.epochs.get(device_id),
                "fusion_ready":device_id in self.fusion_ready,
                "orientation_by_mode":{k:v.tolist() for k,v in self.mode_current.get(device_id,{}).items()},
                "neutral_by_mode":{k:v.tolist() for k,v in self.mode_neutral.get(device_id,{}).items()},
                "up_world":None if estimator is None or estimator.neutral_up is None else estimator.neutral_up.tolist(),
                "fusion_correction":self.fusion_corrections.get(device_id,np.eye(3)).tolist(),
                "natural_hand_references":{key:validate_reference(data) for key,data in self.natural_hand_references.items()
                                           if data['device_id']==device_id},
                "calibration_references":{key:deepcopy(data)
                                          for key,data in self.calibration_references.items() if data['device_id']==device_id}}

    def restore_runtime(self,row):
        device=row['device_id']
        self.mode_current[device]={k:np.asarray(v,float) for k,v in row['orientation_by_mode'].items()}
        self.mode_neutral[device]={k:np.asarray(v,float) for k,v in row['neutral_by_mode'].items()}
        self.epochs[device]=row['epoch']
        if row.get('fusion_ready'):self.fusion_ready.add(device)
        self.fusion_corrections[device]=np.asarray(row['fusion_correction'],float)
        if 'calibration_references' in row:
            from copy import deepcopy
            for key,assigned in self.assignments.items():
                if assigned==device:self.calibration_references.pop(key,None)
            for key,data in row['calibration_references'].items():
                if key not in self.assignments or self.assignments[key]!=device or data['device_id']!=device or data['epoch']!=row['epoch']:raise ValueError('回放标准参考绑定不一致')
                b=proper(data['basis']);neutrals={m:proper(n).tolist() for m,n in data['neutral_by_mode'].items()}
                if any(m not in ORIENTATION_MODES for m in neutrals):raise ValueError('回放标准参考算法无效')
                current_basis=self.continuous_basis.get(key,mapping_matrix(self.axis_mappings[key]))
                if np.allclose(b,current_basis,atol=1e-6):self.calibration_references[key]=deepcopy(dict(data,basis=b.tolist(),neutral_by_mode=neutrals))
        if 'natural_hand_references' in row:
            for key,assigned in self.assignments.items():
                if assigned==device:self.clear_natural_hand(key)
            for key,data in row['natural_hand_references'].items():
                if data.get('device_id')!=device or data.get('part')!=key:raise ValueError('回放自然参考设备不一致')
                data=validate_reference(data)
                # A calibration-comparison replay intentionally uses another B.
                # Do not attach the old display reference to that new mapping.
                if self.natural_hand_error(data) is None:self.set_natural_hand_reference(data)
        self._sync_active(device)

    def zero(self, key: str | None = None) -> None:
        """Use the current pose as neutral without changing sensor-to-body axes."""
        keys = SENSOR_KEYS if key is None else (key,)
        for selected_key in keys:
            if selected_key not in self.assignments:
                raise KeyError(f"未知身体部位：{selected_key}")
            device_id = self.assignments[selected_key]
            self.natural_hand_previews.pop(selected_key,None)
            if selected_key in self.natural_hand_references:
                self.natural_hand_references[selected_key]['enabled']=False
            if device_id in self.current:
                # Preserve callers that update ``current`` directly (replay/tests).
                active = self._active_mode(device_id)
                self.mode_current.setdefault(device_id, {})[active] = self.current[device_id].copy()
                self.mode_neutral[device_id] = {
                    mode: value.copy() for mode, value in
                    self.mode_current[device_id].items()}
                self._sync_active(device_id)

    def is_live(self, device_id: str, now: float | None = None, max_age: float = 3.0) -> bool:
        if device_id not in self.received_at:
            return False
        moment = time.monotonic() if now is None else now
        return moment - self.received_at[device_id] <= max_age

    def transforms(self, *, include_natural: bool = True) -> tuple[np.ndarray, ...]:
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
                natural=self.natural_hand_previews.get(part.key,self.natural_hand_references.get(part.key))
                if include_natural and natural and natural['enabled'] and self.natural_hand_error(natural) is None:
                    absolute_rotation=natural_rotation(natural,self.current[device_id])
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
