"""Scene task construction and IK; no Qt or simulator RPC."""
from dataclasses import dataclass
import time
import numpy as np
from .scene_robot_model import position_targets, segment_targets, palm_target
from .scene_robot_state import SceneSnapshot, joint_steps
from .skeleton import PART_BY_KEY


@dataclass
class SolveResult:
    targets: dict
    errors: dict
    segment_errors: dict
    palm_targets: dict
    relative_targets: dict
    side_ms: dict
    step: np.ndarray


class SceneSolver:
    def __init__(self, models, axes, human_rest, *, clock=time.monotonic):
        self.models = models
        self.axes = axes
        self.human_rest = human_rest
        self.clock = clock
        self.human_neutral = {
            side: tuple(human_rest[PART_BY_KEY[f"{side}_{part}"]][:3, 3].copy()
                        for part in ("upper_arm", "forearm", "hand"))
            for side in models
        }

    def solve(self, sample: SceneSnapshot, previous, arm_speed, wrist_speed, dt) -> SolveResult:
        models, C = self.models, self.axes
        human_rest, human_neutral = self.human_rest, self.human_neutral
        side_ms = {}
        targets={};errors={};segment_errors={};palm_targets={};relative_targets={}
        step=joint_steps(arm_speed,wrist_speed,dt)
        for side,human in sample['sides'].items():
            side_start=self.clock()
            directions=[C@v for v in human['directions']]
            rotation=None
            if human['hand'] is not None:
                rotation=palm_target(C,human['hand'],human_rest[PART_BY_KEY[f'{side}_hand']][:3,:3],
                    models[side].tool_frame(models[side].neutral)[:3,:3],sample.get('palm_offsets',{}).get(side))
            positions=None
            if sample.get('task','positions')=='positions':
                positions=position_targets(models[side],C,human_neutral[side],human['points'],
                                           sample.get('scale','uniform'))
            segment_rotations=None
            if sample.get('segment_orientation',False):
                human_zero={p:human_rest[PART_BY_KEY[f'{side}_{p}']][:3,:3]
                            for p in ('upper_arm','forearm')}
                segment_rotations=segment_targets(models[side],C,human['rotations'],human_zero)
            palm_targets[side]=rotation
            relative_targets[side]=(segment_rotations['forearm'].T@rotation
                if rotation is not None and segment_rotations is not None else None)
            q,err,orient=models[side].fit(directions,previous[side],
                step,rotation,positions=positions,
                segment_rotations=segment_rotations,segment_weight=sample.get('segment_weight',.15),
                solver_mode=sample.get('solver_mode','staged'))
            targets[side]=q
            errors[side]=(err,orient)
            segment_errors[side]=models[side].segment_errors(q,segment_rotations)
            side_ms[side]=(self.clock()-side_start)*1000
        return SolveResult(targets, errors, segment_errors, palm_targets,
                           relative_targets, side_ms, step)
