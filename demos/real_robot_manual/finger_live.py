"""Latest-only five-finger command trajectory with an independent input watchdog."""
import math
import time
import ruckig
from .protocol import decode, encode


class FingerTrajectory:
    live = True
    kind = 'fingers'
    axis = -1
    steps = duration = None

    def __init__(self, baseline, side, maps, speed, acceleration, frequency_hz):
        if side not in ('left','right') or len(maps) != 5 or not any(m.enabled for m in maps):
            raise ValueError('选择一侧并至少许可一根手指')
        if not all(math.isfinite(x) for x in (speed,acceleration,frequency_hz)) or not (
                0 < speed <= 4095 and 0 < acceleration <= 16380 and 1 <= frequency_hz <= 100):
            raise ValueError('手指速度/加速度/频率无效')
        self.start=decode(baseline);self.baseline=baseline;self.side=side;self.maps=maps
        self.offset=0 if side=='right' else 5
        self.q=[float(v) for v in self.start[1][self.offset:self.offset+5]]
        self.v=[0.]*5;self.a=[0.]*5;self.target=self.q.copy()
        self.lower=[min(m.output_open,m.output_closed) for m in maps]
        self.upper=[max(m.output_open,m.output_closed) for m in maps]
        for i,m in enumerate(maps):
            if m.enabled and not self.lower[i] <= self.q[i] <= self.upper[i]:
                raise ValueError('许可手指起点不在输出区间；先确认或调整区间')
        self.limits=(speed,acceleration);self.frequency_hz=frequency_hz;self.period=1/frequency_hz
        self.planner=ruckig.Ruckig(1,self.period)
        self.deadline=0.;self.stopping=False;self.done=False;self.reason=''

    def set_target(self, payload):
        if self.stopping:return
        if payload.get('stop'):
            self.brake(payload.get('reason','用户停止电位器随动'));return
        values=payload['values'];deadline=payload['deadline']
        if len(values)!=5 or not all(math.isfinite(v) for v in (*values,deadline)):
            raise ValueError('手指目标或有效期无效')
        for i,m in enumerate(self.maps):
            if m.enabled and not self.lower[i] <= values[i] <= self.upper[i]:
                raise ValueError('手指目标超出本次区间')
        self.target=[float(values[i]) if m.enabled else self.q[i] for i,m in enumerate(self.maps)]
        self.deadline=deadline

    def brake(self, reason):
        if self.stopping:return
        self.stopping=True;self.reason=reason
        self.stop_targets=[q+v*abs(v)/(2*self.limits[1]) for q,v in zip(self.q,self.v)]

    def _plan(self,i,target):
        inp=ruckig.InputParameter(1)
        inp.current_position=[self.q[i]];inp.current_velocity=[self.v[i]];inp.current_acceleration=[self.a[i]]
        inp.target_position=[target];inp.target_velocity=inp.target_acceleration=[0.]
        inp.max_velocity=[self.limits[0]];inp.max_acceleration=[self.limits[1]]
        trajectory=ruckig.Trajectory(1)
        if int(self.planner.calculate(inp,trajectory))<0:raise RuntimeError('手指轨迹生成失败')
        return trajectory

    def frame(self, step):
        if step==0:return self.baseline
        if time.monotonic()>self.deadline:self.brake('电位器输入超过 500 ms；减速停止，不自动恢复')
        if step==1 and not self.stopping:return self.baseline
        for i,m in enumerate(self.maps):
            if not m.enabled:continue
            stop=self.q[i]+self.v[i]*abs(self.v[i])/(2*self.limits[1])
            target=self.stop_targets[i] if self.stopping else self.target[i]
            trajectory=self._plan(i,target)
            extent=trajectory.position_extrema[0]
            if extent.min<self.lower[i]-1e-6 or extent.max>self.upper[i]+1e-6:
                if not self.lower[i]-1e-6<=stop<=self.upper[i]+1e-6:
                    raise RuntimeError('手指制动路径越界；指令状态不确定')
                trajectory=self._plan(i,min(self.upper[i],max(self.lower[i],stop)))
            (self.q[i],),(self.v[i],),(self.a[i],)=trajectory.at_time(self.period)
            if not self.lower[i]-1e-6 <= self.q[i] <= self.upper[i]+1e-6:
                raise RuntimeError('手指轨迹越界；停止发送')
        if self.stopping and max(abs(v) for v in self.v)<1e-6:
            self.done=True
        fingers=self.start[1].copy()
        for i,m in enumerate(self.maps):
            if m.enabled:fingers[self.offset+i]=round(self.q[i])
        return encode(self.start[0],fingers)
