"""Explicit autonomous finger-zero task, only after the arms' return task.

No sensor feedback and no serial IO. Keep every arm/head/waist angle fixed.
Only previously authorized finger channels may return, within their envelope.
"""
from dataclasses import replace
from .real_robot_arm import ArmControl
from .real_robot_hand import HandMotion
from .real_robot_protocol import encode


class FingerReturnControl(ArmControl):
    controls_fingers=True

    def __init__(self, baseline, limits, configs):
        super().__init__(baseline,limits,preserve_fingers=True)
        if any(q!=90. for q in self.angles):raise ValueError('须先完成手臂回位，所有协议角为 90°')
        if not any(self.fingers):raise ValueError('手指已为协议 0，无需回零')
        self.phase='moving';self.tick_count=0;self.motions={}
        for side,offset in (('right',0),('left',5)):
            values=self.fingers[offset:offset+5]
            if not any(values):continue
            if side=='right' and self.size!=12:raise ValueError('仅左臂模式不能回位右手')
            c=configs.get(side)
            if c is None or any(q and not e for q,e in zip(values,c.enabled)):
                raise ValueError('非零手指没有本次会话内的输出许可记录')
            enabled=tuple(q!=0 for q in values)
            if any(e and lo!=0 for e,lo in zip(enabled,c.lower)):
                raise ValueError('已授权手指区间不含 0，禁止越界冒充回位')
            # Disabled channels are already zero; retain their original valid
            # endpoints. They are never planned or changed by this task.
            opened=tuple(0 if e else q for e,q in zip(enabled,c.opened))
            motion_config=replace(c,opened=opened,enabled=enabled)
            self.motions[side]=HandMotion(motion_config,values,None,limits.frequency,0.,autonomous_return=True)
        self.values=self.fingers.copy()

    def set_target(self, values):return  # IMU/IK cannot retarget a return task.

    def brake(self, reason):
        if self.mode in ('braking','paused'):return
        self.phase='aborted';self.mode='braking';self.reason=reason
        for m in self.motions.values():m.brake(reason)

    def step(self):
        self.tick_count+=1
        for side,m in self.motions.items():
            offset=0 if side=='right' else 5
            self.values[offset:offset+5]=m.step(self.tick_count/self.limits.frequency)
        if self.mode=='braking' and all(m.mode=='paused' for m in self.motions.values()):self.mode='paused'
        if self.mode=='active' and all(max(map(abs,m.q))<1e-7 and max(map(abs,m.v))<1e-7 for m in self.motions.values()):
            self.phase='complete';self.mode='paused';self.reason='手指回零指令完成；须人工确认真实到位'
        return encode(self.angles,self.values)

    def status(self):
        return {**super().status(),'task':'return','return_kind':'fingers','phase':self.phase,
                'return_axis':'手指协议 0','return_step':1,'return_count':1,
                'return_order':['手指协议 0'],'feedback':False,
                'finger_return':{s:m.status() for s,m in self.motions.items()}}
