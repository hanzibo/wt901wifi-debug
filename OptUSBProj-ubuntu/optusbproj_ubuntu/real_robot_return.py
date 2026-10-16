"""Ordered, permission-preserving host-command homing. No IMU or serial IO."""
from dataclasses import dataclass
import numpy as np
from .real_robot_arm import ArmControl
from .real_robot_protocol import decode

ORDER = (1,3,5,6,4,2)


@dataclass(frozen=True)
class ReturnConfig:
    sides: tuple = ('left','right')
    order: tuple = ORDER
    dwell: float = .5
    manual: bool = True

    def __post_init__(self):
        if self.sides not in (('left','right'),('right','left'),('left',)):
            raise ValueError('回位先后须为双臂各一次，或仅左臂')
        if len(self.order) != 6 or set(self.order) != set(range(1,7)):
            raise ValueError('回位顺序须包含关节 1–6 各一次')
        if not np.isfinite(self.dwell) or self.dwell < 0 or type(self.manual) is not bool:
            raise ValueError('回位步间停留或确认模式无效')


class ReturnControl(ArmControl):
    task = 'return'

    def __init__(self, baseline, limits, config=ReturnConfig()):
        super().__init__(baseline,limits)
        if not (self.size == 12 and len(config.sides) == 2 or
                self.size == 6 and limits.side == 'left' and config.sides == ('left',)):
            raise ValueError('回位配置须与双臂或仅左臂许可一致')
        for i,allowed in enumerate(limits.enabled):
            if allowed and not limits.lower[i] <= 90. <= limits.upper[i]:
                raise ValueError(f'{self.names[i]} 范围不包含 90°，禁止限幅冒充回位')
            if not allowed and self.q0[i] != 90.:
                raise ValueError(f'{self.names[i]} 不在 90° 且未获许可，不能完整回位')
        self.config = config
        order = [self.indices.index((6 if side == 'left' else 0)+j-1)
                 for side in config.sides for j in config.order]
        self.sequence = [i for i in order if limits.enabled[i] and self.q0[i] != 90.]
        self.cursor = 0
        self.phase = 'await_step' if config.manual else 'moving'
        self.wait_ticks = 0
        self.phase_before_brake = None
        if not self.sequence:
            self.phase = 'complete'
        elif not config.manual:
            self._select_target()

    def _select_target(self):
        target = self.q.copy()
        target[self.sequence[self.cursor]] = 90.
        super().set_target(target)

    def set_target(self, values):
        # Incoming IMU / old queued IK can never change an autonomous target.
        return

    def advance(self):
        if self.mode != 'active' or self.phase != 'await_step':
            raise ValueError('当前不是等待人工执行下一步')
        self.phase = 'moving'
        self._select_target()

    def brake(self, reason):
        if self.mode == 'active':
            self.phase_before_brake = self.phase
            self.phase = 'aborted'
        super().brake(reason)

    def step(self):
        first = self.first
        frame = super().step()
        if first:
            return frame
        if self.mode != 'active':
            return frame
        if self.phase == 'moving' and np.max(np.abs(self.q-self.target)) < 1e-7 and np.max(np.abs(self.v)) < 1e-7:
            self.phase = 'dwell'
            self.wait_ticks = 0
        elif self.phase == 'dwell':
            self.wait_ticks += 1
            if self.wait_ticks / self.limits.frequency >= self.config.dwell:
                self.cursor += 1
                if self.cursor == len(self.sequence):
                    self.phase = 'complete'
                elif self.config.manual:
                    self.phase = 'await_step'
                else:
                    self.phase = 'moving'
                    self._select_target()
        if self.phase == 'complete':
            angles,_ = decode(frame)
            if any(x != 90. for x in angles):
                raise RuntimeError('回位结束帧不是全部 90°')
            self.mode = 'paused'
            self.reason = '回位指令完成；不是实际到位反馈，等待人工确认'
        return frame

    def status(self):
        return {**super().status(), 'task':'return', 'phase':self.phase,
                'return_axis':None if self.cursor >= len(self.sequence) else self.names[self.sequence[self.cursor]],
                'return_step':self.cursor+1, 'return_count':len(self.sequence),
                'return_order':[self.names[i] for i in self.sequence], 'feedback':False}
