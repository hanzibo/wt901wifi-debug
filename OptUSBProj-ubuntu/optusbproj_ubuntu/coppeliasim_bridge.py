"""Local-only, stopped-scene batched API. Never starts/saves a simulation."""
import json
import numpy as np


class SceneBridge:
    def __init__(self, port=23000):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        import zmq
        self.client=RemoteAPIClient('127.0.0.1',port)
        for option in (zmq.RCVTIMEO,zmq.SNDTIMEO):
            self.client.socket.setsockopt(option,2000)
        self.client.socket.setsockopt(zmq.LINGER,0)
        self._fast_idle=False
        try:
            self.sim=self.client.require('sim')
            self.data=self.run('''
local base=sim.getObject('/BaseLinkHull')
local r={base=base,uid=sim.getObjectUid(base),baseMatrix=sim.getObjectMatrix(base,sim.handle_world),arms={}}
for _,side in ipairs({'L','R'}) do
 local arm={q={},joints={},objects={},limits={}}
 for i=1,6 do
  local path='/BaseLinkHull/joint'..side..i
  local h=sim.getObject(path)
  assert(sim.getJointType(h)==sim.joint_revolute_subtype,'非旋转关节')
  assert(sim.getJointDependency(h)<0,'从属关节不可独立控制')
  local cyclic,iv=sim.getJointInterval(h)
  arm.q[i]=sim.getJointPosition(h)
  arm.joints[i]=sim.getObjectMatrix(h,base)
  arm.objects[i]={path=path,h=h,uid=sim.getObjectUid(h)}
  arm.limits[i]=cyclic and {-math.pi,math.pi} or {iv[1],iv[1]+iv[2]}
 end
 local palm=sim.getObject('/BaseLinkHull/Link'..side..'HMainHull')
 arm.palm=sim.getObjectMatrix(palm,base)
 local ancestor=palm
 for i=6,1,-1 do
  while ancestor~=-1 and ancestor~=arm.objects[i].h do ancestor=sim.getObjectParent(ancestor) end
  assert(ancestor~=-1,'关节链层级不匹配')
  ancestor=sim.getObjectParent(ancestor)
 end
 r.arms[side=='L' and 'left' or 'right']=arm
end
return r
''')
        except Exception:
            self.close()
            raise

    def run(self,code):
        ok,result=self.sim.executeLuaCode("local sim=require('sim')\nassert(sim.getSimulationState()==sim.simulation_stopped,'请先停止仿真')\n"+code)
        if not ok: raise RuntimeError(str(result))
        return result

    def exchange(self,targets=None):
        changes=[]
        for side,q in (targets or {}).items():
            if side not in self.data['arms'] or len(q)!=6 or not np.isfinite(q).all():
                raise ValueError('无效角度批次')
            for obj,value,limits in zip(self.data['arms'][side]['objects'],q,self.data['arms'][side]['limits']):
                if not limits[0]<=value<=limits[1]: raise ValueError('超出场景区间')
                changes.append('{h=%d,v=%.17g}'%(obj['h'],value))
        objects=[o for a in self.data['arms'].values() for o in a['objects']]
        rows=','.join('{h=%d,uid=%d,path=%s}'%(o['h'],o['uid'],json.dumps(o['path'])) for o in objects)
        code='''
local base=sim.getObject('/BaseLinkHull')
assert(sim.getObjectUid(base)==%d,'场景已更换')
local expected={%s}
local bm=sim.getObjectMatrix(base,sim.handle_world)
for i,v in ipairs(expected) do assert(math.abs(bm[i]-v)<1e-6,'底座已移动，请重新连接') end
local objects={%s}
for _,o in ipairs(objects) do
 assert(sim.getObject(o.path)==o.h and sim.getObjectUid(o.h)==o.uid,'关节对象已改变')
 assert(sim.getJointDependency(o.h)<0,'关节已变为从属关节')
end
local changes={%s}
for _,o in ipairs(changes) do
 local cyclic,iv=sim.getJointInterval(o.h)
 assert(cyclic or (o.v>=iv[1] and o.v<=iv[1]+iv[2]),'关节区间已改变')
end
for _,o in ipairs(changes) do sim.setJointPosition(o.h,o.v) end
local q={}
for _,o in ipairs(objects) do q[o.path]=sim.getJointPosition(o.h) end
return q
'''%(self.data['uid'],','.join(map(str,self.data['baseMatrix'])),rows,','.join(changes))
        raw=self.run(code)
        return {side:np.array([raw[o['path']] for o in a['objects']]) for side,a in self.data['arms'].items()}

    def fast_idle(self,enabled):
        if enabled==self._fast_idle:return
        self.run(self._idle_code(enabled))
        self._fast_idle=enabled

    @staticmethod
    def _idle_code(enabled):
        # App-level marker lets reconnect adopt a lease left by a lost REQ reply,
        # rather than incrementing CoppeliaSim's fast-idle reference count again.
        key='__WT901_PREVIEW_FAST_IDLE__'
        if enabled:
            return ("local lease=sim.readCustomStringData(sim.handle_app,'%s'); if not lease or #lease==0 then "
                    "sim.fastIdleLoop(true); sim.writeCustomStringData(sim.handle_app,'%s','1') end\nreturn true")%(key,key)
        return ("local lease=sim.readCustomStringData(sim.handle_app,'%s'); if lease and #lease>0 then "
                "sim.fastIdleLoop(false); sim.writeCustomStringData(sim.handle_app,'%s','') end\nreturn true")%(key,key)

    def close(self):
        if getattr(self,'_fast_idle',False):
            # Restore even if simulation was started; no stopped-scene assertion here.
            try:self.sim.executeLuaCode("local sim=require('sim');\n"+self._idle_code(False))
            except Exception:pass  # A lost server cannot be restored over a dead connection.
            self._fast_idle=False
        self.client.socket.close(0)
        self.client.context.term()

    def validate_identity(self, old):
        """Reconnect is read-only; never blindly replay an ambiguously timed-out write."""
        if self.data['uid']!=old['uid'] or not np.allclose(self.data['baseMatrix'],old['baseMatrix'],atol=1e-6,rtol=0):
            raise ValueError('恢复失败：场景更换或底座移动，请重新连接')
        for side,arm in old['arms'].items():
            current=self.data['arms'][side]
            if current['objects']!=arm['objects'] or current['limits']!=arm['limits']:
                raise ValueError('恢复失败：关节身份或区间改变，请重新连接')
