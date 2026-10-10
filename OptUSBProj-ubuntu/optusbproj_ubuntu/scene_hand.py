"""Explicit hand-pose endpoints and leased discrete input (simulation only)."""
import copy
import threading
import numpy as np


def hand_pose(values, limits):
    q=np.asarray(values,dtype=float)
    bounds=np.asarray(limits,dtype=float)
    if q.shape!=(15,) or bounds.shape!=(15,2) or not np.isfinite(q).all():
        raise ValueError('手型必须包含十五个有限关节角')
    if np.any(q<bounds[:,0]) or np.any(q>bounds[:,1]):
        raise ValueError('手型超出当前场景关节区间')
    return q.copy()


def hand_step(current,target,speed,dt):
    # One scalar fraction preserves the configured path between poses.
    delta=np.asarray(target)-np.asarray(current)
    length=float(np.max(np.abs(delta)))
    amount=np.deg2rad(speed)*min(max(dt,0.),.1)
    return np.asarray(current)+delta*min(1.,amount/length) if length else np.asarray(current).copy()


class HandMailbox:
    def __init__(self):
        self.lock=threading.Lock();self.requests={};self.revision=0

    def set(self,side,target,stamp,speed,source):
        if side not in ('left','right') or not np.isfinite(stamp) or not 5<=speed<=720:
            raise ValueError('无效手指控制参数')
        q=np.asarray(target,dtype=float)
        if q.shape!=(15,) or not np.isfinite(q).all():raise ValueError('无效手指目标')
        with self.lock:
            old=self.requests.get(side)
            if old is None or old['source']!=source or old['speed']!=speed or not np.array_equal(old['target'],q):self.revision+=1
            self.requests[side]={'target':q.copy(),'stamp':stamp,'speed':speed,'source':source}

    def stop(self,side=None):
        with self.lock:
            if side is None:self.requests.clear()
            else:self.requests.pop(side,None)
            self.revision+=1

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.requests),self.revision

    def targets(self,current,now,dt):
        requests,revision=self.snapshot()
        active={s:r for s,r in requests.items() if -.1<=now-r['stamp']<=.3}
        return {s:hand_step(current[s],r['target'],r['speed'],dt) for s,r in active.items() if s in current},active,revision
