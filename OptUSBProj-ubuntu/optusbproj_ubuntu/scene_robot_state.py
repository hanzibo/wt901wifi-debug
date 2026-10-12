"""Qt-free snapshot, freshness and output authorization helpers.

Snapshot creation runs on the GUI thread; published arrays are copied.
"""
import threading
import time
from typing import NotRequired, TypedDict
import numpy as np
from .skeleton import PART_BY_KEY
from .core.contracts import (DirectionVector, PointMeters, RotationMatrix,
                             Seconds, Milliseconds)
from .x1_live import MAX_JOINT_SPEED_DEG_S, MAX_JOINT_STEP_DEG, NOMINAL_TICK_S


class HumanSideSnapshot(TypedDict):
    directions: list[DirectionVector]
    hand: RotationMatrix | None
    points: NotRequired[tuple[PointMeters, ...]]
    rotations: NotRequired[dict[str, RotationMatrix]]


class SceneSnapshot(TypedDict):
    """Dictionary-compatible contract; monotonic seconds and positions in metres."""
    at: Seconds
    sides: dict[str, HumanSideSnapshot]
    device_ages_ms: NotRequired[dict[str, Milliseconds]]
    gui_interval_ms: NotRequired[Milliseconds]
    task: NotRequired[str]
    scale: NotRequired[str]
    solver_mode: NotRequired[str]
    segment_orientation: NotRequired[bool]
    segment_weight: NotRequired[float]
    palm_offsets: NotRequired[dict[str, dict]]


def joint_step(speed,dt):
    return np.deg2rad(min(MAX_JOINT_STEP_DEG,speed*max(0.,dt)))


def joint_steps(arm_speed,wrist_speed,dt):
    # Scale the wrist increment ceiling too: a fixed 9-degree ceiling would
    # otherwise silently cap a 240 deg/s wrist at 180 deg/s at 20 Hz.
    wrist_cap=MAX_JOINT_STEP_DEG*wrist_speed/MAX_JOINT_SPEED_DEG_S
    wrist_step=np.deg2rad(min(wrist_cap,wrist_speed*max(0.,dt)))
    return np.r_[np.full(4,joint_step(arm_speed,dt)),np.full(2,wrist_step)]


def cycle_wait_ms(start,now):
    return max(0,int((NOMINAL_TICK_S-(now-start))*1000))


class Mailbox:
    def __init__(self):
        self.lock=threading.Lock()
        self.enabled=False
        self.sample=None
        self.generation=0
        self.blocked=False
        self.stop_reason='尚未启动'
    def set(self,enabled,sample: SceneSnapshot | None=None,reason=None):
        with self.lock:
            if enabled and self.blocked:return
            if self.enabled != enabled:
                self.generation+=1
            self.enabled=enabled
            self.sample=sample
            if reason is not None:self.stop_reason=reason
    def pause(self):
        with self.lock:
            self.blocked=True;self.enabled=False;self.sample=None;self.generation+=1
    def resume(self,sample):
        with self.lock:
            self.blocked=False;self.enabled=True;self.sample=sample;self.generation+=1
    def get(self):
        with self.lock:
            return self.enabled,self.sample,self.generation


def dispatch_reason(enabled,generation,current_generation,current,sample,now):
    if not enabled:return 'sending_stopped'
    if generation!=current_generation:return 'generation_changed'
    if current is None:return 'missing_snapshot'
    if now-sample['at']>=.25:return 'solve_snapshot_expired'
    return None


class StaleImuInput(ValueError):
    def __init__(self,part,device,age):
        self.age=age
        super().__init__(f'IMU 数据不新鲜：{part} / {device}，年龄 {age*1000:.0f} ms（允许 -100…250 ms）')


def temporary_imu_gap(exc):
    # Never accept old poses: only allow waiting for new input for up to 1 s.
    return isinstance(exc,StaleImuInput) and .25<exc.age<1.


def pose_sample(engine,sides,hand_sides,manual=False) -> tuple[SceneSnapshot, tuple]:
    """Runs only on GUI thread; no worker shares mutable PoseEngine state."""
    now=time.monotonic()
    parts=[f'{s}_{segment}' for s in sides for segment in ('upper_arm','forearm')]
    parts += [f'{s}_hand' for s in hand_sides if s in sides]
    ids=[engine.assignments.get(p,'') for p in parts]
    if not all(ids) or len(set(ids))!=len(ids):
        raise ValueError('所选部位须分配不同的在线设备')
    signature=[]
    ages=[(part,device,now-engine.received_at.get(device,-1e9)) for part,device in zip(parts,ids)]
    for part,device,age in sorted(ages,key=lambda item:item[2],reverse=True):
        if not -.1<=age<=.25:raise StaleImuInput(part,device,age)
    for part,device in zip(parts,ids):
        if device not in engine.neutral or device not in engine.current:
            raise ValueError(f'{part} 缺少当前姿态或中立位')
        if not part.endswith('_hand') and not manual and part not in engine.continuous_basis:
            raise ValueError(f'{part} 未应用穿戴标定')
        basis=engine.continuous_basis.get(part)
        signature.append((part,device,engine.algorithm_modes[part],
                          repr(engine.axis_mappings[part]),
                          engine.neutral[device].tobytes(),
                          None if basis is None else basis.tobytes()))
    transforms=engine.transforms(include_natural=False)
    result={}
    for side in sides:
        s,e,w=[transforms[PART_BY_KEY[f'{side}_{seg}']][:3,3] for seg in ('upper_arm','forearm','hand')]
        vectors=[e-s,w-e]
        if min(np.linalg.norm(v) for v in vectors)<1e-8:
            raise ValueError('人体骨段长度无效')
        hand=transforms[PART_BY_KEY[f'{side}_hand']][:3,:3].copy() if side in hand_sides else None
        result[side]={'directions':[v/np.linalg.norm(v) for v in vectors],'hand':hand,
                      'points':tuple(v.copy() for v in (s,e,w)),
                      'rotations':{p:transforms[PART_BY_KEY[f'{side}_{p}']][:3,:3].copy()
                                   for p in ('upper_arm','forearm')}}
    return {'at':now,'sides':result,'device_ages_ms':{p:(now-engine.received_at[d])*1000 for p,d in zip(parts,ids)}},tuple(signature)
