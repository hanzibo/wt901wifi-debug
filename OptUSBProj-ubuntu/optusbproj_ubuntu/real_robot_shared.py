"""Simulation task reuse and explicit transition gates; no hardware access."""
from copy import deepcopy
import numpy as np
from .real_robot_arm import ArmControl, RelativeArmIK, load_profile
from .real_robot_protocol import decode
from .scene_robot_model import SceneArm
from .scene_robot_solver import SceneSolver
from .pose import PoseEngine


class SharedArmIK:
    """Use SceneSolver's absolute STL-rest targets, without a second solver path."""
    def __init__(self, baseline, limits, *, profile=None, human_rest=None,
                 palm=True, offset=None, side='left'):
        self.side = side
        if limits.side != side:
            raise ValueError('运动学侧别与关节配置不一致')
        arm, C, self.provenance = load_profile(side) if profile is None else profile
        self.model = SceneArm(arm, [-np.pi]*6, [np.pi]*6)
        a, b = limits.to_sim(limits.lower), limits.to_sim(limits.upper)
        self.model.lower, self.model.upper = np.minimum(a,b), np.maximum(a,b)
        self.limits = limits
        # IK's rest seed is NOT the confirmed physical starting command. In
        # particular, all-zero scene q lies on L2's bound and can trap TRF at
        # its tiny initial trust radius. The trajectory starts at baseline.
        self.previous = np.clip(self.model.neutral, self.model.lower, self.model.upper)
        self.solver = SceneSolver({side:self.model}, C,
                                 PoseEngine().transforms(include_natural=False)
                                 if human_rest is None else human_rest)
        self.palm, self.offset = palm, deepcopy(offset)

    def solve(self, side):
        RelativeArmIK._validate(side)
        human = deepcopy(side)
        if not self.palm:
            human['hand'] = None
        sample = dict(sides={self.side:human}, task='positions', scale='uniform',
                      solver_mode='staged', segment_orientation=True, segment_weight=.15,
                      palm_offsets={self.side:self.offset} if self.offset else {})
        # The command planner, not IK iterations, owns physical rate limits.
        result = self.solver.solve(sample, {self.side:self.previous}, 100., 100., 1.,
                                   step_override=np.full(6, 2*np.pi))
        self.previous = result.targets[self.side].copy()
        error, orient = result.errors[self.side]
        return dict(target=self.limits.to_command(self.previous).tolist(),
                    position_error_mm=error*1000, orientation_error_deg=orient)


class SharedArmControl(ArmControl):
    """Freeze one startup target, wait for human confirmation, then follow.

    Every input update still refreshes the transport freshness deadline. Neither
    reaching the command endpoint nor a successful write proves physical arrival.
    """
    def __init__(self, baseline, limits, initial_target):
        if any(x != 90. for x in decode(baseline)[0]) or any(decode(baseline)[1]):
            raise ValueError('同源接管须重新确认全部 90° / 手指 0 的上电姿态')
        super().__init__(baseline, limits)
        super().set_target(initial_target)
        self.phase = 'transition'
        self.initial_target = self.target.copy()
        self.follow_requested = False

    def set_target(self, values):
        if self.phase == 'following':
            super().set_target(values)

    def request_follow(self):
        if self.phase != 'await_confirmation' or self.mode != 'active':
            raise ValueError('过渡指令尚未完成或已暂停')
        self.follow_requested = True

    def step(self):
        if self.follow_requested and self.mode == 'active':
            self.phase = 'following'
            self.follow_requested = False
        frame = super().step()
        if (self.phase == 'transition' and self.mode == 'active'
                and np.max(np.abs(self.q-self.initial_target)) < 1e-6
                and np.max(np.abs(self.v)) < 1e-6):
            self.phase = 'await_confirmation'
        return frame

    def status(self):
        gesture = getattr(self,'gesture',None)
        return {**super().status(), 'phase':self.phase,
                **({'gesture_elapsed':gesture.elapsed, 'gesture_triggered':gesture.triggered,
                    'gesture_seconds':gesture.config.seconds, 'gesture_kind':gesture.config.kind,
                    'gesture_indices':gesture.config.indices} if gesture is not None else {}),
                'initial_target':self.initial_target.tolist(), 'feedback':False}
