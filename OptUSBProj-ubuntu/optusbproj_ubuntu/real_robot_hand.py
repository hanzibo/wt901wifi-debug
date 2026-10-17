"""Five protocol quantities per hand; pure planning, no serial or simulator IO."""
from dataclasses import dataclass
import math
import ruckig
from .real_robot_protocol import decode, encode
from .real_robot_trajectory import stopping_boundary


@dataclass(frozen=True)
class HandConfig:
    opened: tuple = (0,)*5
    closed: tuple = (200,)*5
    lower: tuple = (0,)*5
    upper: tuple = (200,)*5
    speed: tuple = (200.,)*5
    acceleration: tuple = (400.,)*5
    enabled: tuple = (False,)*5

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = tuple(getattr(self, name))
            if len(value) != 5:
                raise ValueError('每手配置须包含五个值')
            object.__setattr__(self, name, value)
        for i in range(5):
            if type(self.enabled[i]) is not bool:
                raise ValueError('手指许可须为布尔值')
            if any(type(v) is not int or not 0 <= v <= 4095 for v in
                   (self.opened[i], self.closed[i], self.lower[i], self.upper[i])):
                raise ValueError('手指端点与范围须为 0–4095 整数，不是角度')
            if not self.lower[i] < self.upper[i] or not all(
                    self.lower[i] <= v <= self.upper[i] for v in (self.opened[i], self.closed[i])):
                raise ValueError('端点须位于手指软件区间')
            if not all(type(v) in (int, float) and math.isfinite(v) and 0 < v <= cap for v, cap in
                       ((self.speed[i], 4095), (self.acceleration[i], 16380))):
                raise ValueError('手指速度或加速度无效')


def observation_valid(data, now, *, allow_residual=False):
    return (data.get('valid') is True and type(data.get('state')) is bool
            and type(data.get('stamp')) in (float, int) and math.isfinite(data['stamp'])
            and -.1 <= now-data['stamp'] <= .25 and bool(data.get('signature'))
            and type(data.get('residual_deg')) in (float,int)
            and math.isfinite(data['residual_deg']) and data['residual_deg'] >= 0
            and (data['residual_deg'] <= 20. or allow_residual))


class HandMotion:
    def __init__(self, config, values, observation, frequency, now, *, allow_residual=False, autonomous_return=False):
        self.allow_residual = bool(allow_residual)
        self.autonomous_return=autonomous_return
        if autonomous_return and any(q and e for q,e in zip(config.opened,config.enabled)):
            raise ValueError('自主手指回位目标必须为协议 0')
        if not any(config.enabled) or (not autonomous_return and not observation_valid(observation, now,allow_residual=self.allow_residual)):
            raise ValueError('须许可手指并有新鲜、已判定的手势')
        self.config, self.signature = config, None if autonomous_return else observation['signature']
        self.q = list(map(float, values)); self.v = [0.]*5; self.a = [0.]*5
        if any(allowed and not lo <= q <= hi for q,lo,hi,allowed in
               zip(self.q,config.lower,config.upper,config.enabled)):
            raise ValueError('最后指令起点不在手指区间')
        self.dt = 1/frequency; self.planner = ruckig.Ruckig(1,self.dt)
        self.mode = 'active'; self.reason = ''; self.first = True
        self.last_good = now if autonomous_return else observation['stamp']; self.last_seen = self.last_good
        self.target = list(config.opened if autonomous_return or not observation['state'] else config.closed)

    def brake(self, reason, *, temporary=False):
        if self.mode in ('braking','paused'): return
        self.target = [q+v*abs(v)/(2*a) for q,v,a in zip(self.q,self.v,self.config.acceleration)]
        self.mode = 'gap' if temporary else 'braking'; self.reason = reason

    def observe(self, data, now):
        if self.autonomous_return:return
        if self.mode in ('braking','paused'): return
        if data.get('signature') != self.signature:
            self.brake('设备、算法或开合参考已改变'); return
        if now > self.last_good+.5:
            self.brake('手势超过 500 ms 未恢复；需重新启用'); return
        stamp = data.get('stamp',0.)
        if observation_valid(data, now,allow_residual=self.allow_residual) and stamp > self.last_seen:
            self.last_seen = self.last_good = stamp
            self.target = list(self.config.closed if data['state'] else self.config.opened)
            self.mode, self.reason = 'active', ''
        elif not observation_valid(data, now,allow_residual=self.allow_residual):
            self.brake('手势无效，减速等待', temporary=True)

    def _plan(self, i):
        c = self.config
        plan = stopping_boundary(self.q[i],self.v[i],self.a[i],self.target[i],c.speed[i],c.acceleration[i])
        if plan is None:
            inp = ruckig.InputParameter(1)
            inp.current_position=[self.q[i]]; inp.current_velocity=[self.v[i]]; inp.current_acceleration=[self.a[i]]
            inp.target_position=[self.target[i]]; inp.target_velocity=inp.target_acceleration=[0.]
            inp.max_velocity=[c.speed[i]]; inp.max_acceleration=[c.acceleration[i]]
            plan = ruckig.Trajectory(1)
            if int(self.planner.calculate(inp,plan)) < 0:
                raise RuntimeError('手指轨迹生成失败；指令状态不确定')
        return plan

    def step(self, now):
        if not self.autonomous_return and self.mode in ('active','gap'):
            if now > self.last_good+.5: self.brake('手势超过 500 ms；需重新启用')
            elif now > self.last_good+.25: self.brake('手势短暂断流，减速等待', temporary=True)
        if self.first:
            self.first=False
            return list(map(round,self.q))
        plans = {i:self._plan(i) for i,e in enumerate(self.config.enabled) if e}
        if any(p.position_extrema[0].min < self.config.lower[i]-1e-7 or
               p.position_extrema[0].max > self.config.upper[i]+1e-7 for i,p in plans.items()):
            self.brake('手指目标反转预测越界；需重新启用')
            plans = {i:self._plan(i) for i in plans}
        for i,p in plans.items():
            if p.position_extrema[0].min < self.config.lower[i]-1e-7 or p.position_extrema[0].max > self.config.upper[i]+1e-7:
                raise RuntimeError('手指制动路径越界；停止发送')
            (q,),(v,),(a,) = p.at_time(self.dt)
            self.q[i],self.v[i],self.a[i] = min(self.config.upper[i],max(self.config.lower[i],q)),v,a
        if self.mode=='braking' and max(map(abs,self.v)) < 1e-7:
            self.mode='paused'
        return list(map(round,self.q))

    def status(self):
        return dict(mode=self.mode,reason=self.reason,residual_bypass=self.allow_residual,command=list(map(round,self.q)),
                    target=self.target.copy(),velocity=self.v.copy(),acceleration=self.a.copy())


class HandBank:
    """Owned by the UART thread lock. Merge only authorized fingers into arm frames."""
    def __init__(self):
        self.motions={}; self.observations={}; self.values=None

    def reset(self, baseline):
        self.motions.clear(); self.observations.clear(); self.values=decode(baseline)[1]

    def observe(self, side, data, now):
        if side not in ('left','right'): raise ValueError('未知手侧')
        self.observations[side]=dict(data)
        if side in self.motions: self.motions[side].observe(data,now)

    def enable(self, side, config, frequency, now, *, allow_residual=False):
        if side not in ('left','right') or self.values is None: raise ValueError('手指起点未知')
        if side in self.motions and self.motions[side].mode!='paused': raise ValueError('该手仍在输出或减速')
        start=0 if side=='right' else 5
        self.motions[side]=HandMotion(config,self.values[start:start+5],self.observations.get(side,{}),frequency,now,
                                     allow_residual=allow_residual)

    def stop(self, reason, side=None):
        for s,m in self.motions.items():
            if side is None or side==s: m.brake(reason)

    def merge(self, frame, now, *, arm_mode, following):
        angles,fingers=decode(frame)
        if self.values is None: self.values=fingers
        if not following or arm_mode in ('braking','paused'):
            self.stop('双臂暂停、接管或回位；手指停止跟随')
        elif arm_mode=='gap':
            for m in self.motions.values(): m.brake('臂侧短暂断流',temporary=True)
        for side,m in self.motions.items():
            offset=0 if side=='right' else 5
            self.values[offset:offset+5]=m.step(now)
        return encode(angles,self.values)

    @property
    def stopped(self):
        return all(m.mode=='paused' for m in self.motions.values())

    def status(self):
        return {s:m.status() for s,m in self.motions.items()}
