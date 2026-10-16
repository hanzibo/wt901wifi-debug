"""Both arms share one packet, clock, startup gate and braking decision.

Protocol order is R1..R6, L1..L6. Fingers/head/waist remain frozen. Geometry
is exported CAD, not measured feedback or certified collision protection.
"""
from dataclasses import dataclass
from .real_robot_arm import ArmLimits
from .real_robot_shared import SharedArmIK


def right_defaults():
    # Historical PC software envelope; right mechanical limits not certified.
    return ArmLimits(side='right', lower=(0.,90.,0.,0.,0.,50.),
                     upper=(235.,240.,180.,118.,180.,121.))


@dataclass(frozen=True)
class BilateralLimits:
    right: ArmLimits
    left: ArmLimits
    indices = tuple(range(12))

    def __post_init__(self):
        if self.right.side != 'right' or self.left.side != 'left':
            raise ValueError('双臂配置顺序须为右、左')
        if self.right.frequency != self.left.frequency:
            raise ValueError('双臂须共用同一发送频率')

    @property
    def frequency(self):
        return self.left.frequency

    @property
    def sides(self):
        return tuple(s for s in ('right','left') if any(getattr(self,s).enabled))

    @property
    def mirror_signs(self):
        return {s:getattr(self,s).signs for s in ('left','right')}

    def __getattr__(self, key):
        if key in ('speed','acceleration','lower','upper','enabled','signs'):
            return getattr(self.right,key)+getattr(self.left,key)
        raise AttributeError(key)

    def as_dict(self):
        return {s:vars(getattr(self,s)) for s in ('right','left')}


class BilateralIK:
    """Reuse the same SceneSolver objective per arm; never run a second writer."""
    def __init__(self, baseline, limits, *, palms=None, offsets=None):
        self.limits = limits
        palms, offsets = palms or {}, offsets or {}
        self.solvers = {s:SharedArmIK(baseline, getattr(limits,s), side=s,
                                    palm=palms.get(s,True), offset=offsets.get(s))
                        for s in limits.sides}
        if not self.solvers:
            raise ValueError('至少勾选一侧的一个关节')
        self.provenance = {s:v.provenance for s,v in self.solvers.items()}

    def solve(self, human):
        if set(human) != set(self.solvers):
            raise ValueError('双臂输入侧别与输出许可不一致')
        target, errors = [], {}
        for side in ('right','left'):
            if side in self.solvers:
                result = self.solvers[side].solve(human[side])
                target.extend(result['target'])
                errors[side] = {k:v for k,v in result.items() if k != 'target'}
            else:
                target.extend([90.]*6)
        return dict(target=target, errors=errors,
                    position_error_mm=max(e['position_error_mm'] for e in errors.values()),
                    orientation_error_deg=None)
