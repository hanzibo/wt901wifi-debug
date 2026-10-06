"""Session-bound visual feedback; pure and fused poses remain in IMU world W."""
from collections import deque
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .pose import GYRO_TILT,GYRO_RGBD
from .rgbd_fusion import ARM_SEGMENTS,HeadingCorrector,visual_direction,unit,angle_deg


class LiveRGBDFusion:
    def __init__(self,engine):
        self.engine=engine;self.history={};self.bindings={};self.states={};self.last_stamp={}
        self.gravity_history={}
        self.events=deque()
        self.max_offset_deg=30.

    def set_max_offset(self,degrees):
        # Validate ALL parts before changing any limit. Never clamp a pose.
        HeadingCorrector([0,0,1],max_offset_deg=degrees)
        if degrees is not None and any(abs(c.offset_rad)>np.radians(degrees)+1e-12 for _,c in self.bindings.values()):
            raise ValueError('有部位的累计修正已超过新范围，不能缩小；请保持原范围或在重标定后设置')
        for _,corrector in self.bindings.values():corrector.set_max_offset(degrees)
        self.max_offset_deg=None if degrees is None else float(degrees)
        self.events.append({'time_ns':time.monotonic_ns(),'type':'heading_limit_changed','max_offset_deg':self.max_offset_deg})

    def feed(self,device,stamp):
        base=self.engine.current_for(device,GYRO_TILT)
        if base is None:return
        history=self.history.setdefault(device,deque())
        epoch=self.engine.epochs[device]
        if history and history[-1][2]!=epoch:history.clear()
        history.append((int(stamp),base.copy(),epoch))
        while history and stamp-history[0][0]>3_000_000_000:history.popleft()
        packet=self.engine.packets.get(device)
        if packet is not None:
            samples=self.gravity_history.setdefault(device,deque())
            if samples and samples[-1]['epoch']!=epoch:samples.clear()
            samples.append({'stamp':int(stamp),'rotation':base.copy(),'acc':np.array([packet.acc_x,packet.acc_y,packet.acc_z]),
                            'gyro':np.array([packet.gyro_x,packet.gyro_y,packet.gyro_z]),'epoch':epoch})
            while samples and stamp-samples[0]['stamp']>35_000_000_000:samples.popleft()

    def lock_gravity(self,devices,start_ns,end_ns):
        """Atomically bootstrap missing gravity BEFORE non-neutral motions.

        Preserve existing references, clock, R and gyro bias. A validated
        explicit still window replaces only the missing-reference case.
        """
        proposed={}
        for device in devices:
            epoch=self.engine.epochs.get(device)
            samples=[r for r in self.gravity_history.get(device,()) if start_ns<=r['stamp']<=end_ns and r['epoch']==epoch]
            if len(samples)<30:raise ValueError(f'{device} 中立采样不足，请重新标定')
            good=[r for r in samples if abs(np.linalg.norm(r['acc'])-1)<.12]
            if len(good)<30 or len(good)<.8*len(samples):raise ValueError(f'{device} 加速度不稳定，请保持静止重录')
            gyro_p90=float(np.percentile([np.linalg.norm(r['gyro']) for r in good],90))
            if gyro_p90>5:raise ValueError(f'{device} 中立位未充分静止（角速度P90 {gyro_p90:.1f}°/s），请重录')
            vectors=[unit(r['rotation']@unit(r['acc'])) for r in good]
            candidate=unit(np.mean(vectors,axis=0))
            spread=float(np.percentile([angle_deg(v,candidate) for v in vectors],90))
            if spread>3:raise ValueError(f'{device} 中立重力方向波动{spread:.1f}°，请重录')
            estimator=self.engine._tilt_filters.get(device)
            if estimator is None:raise ValueError(f'{device} 实时积分器不可用')
            existing=estimator.neutral_up
            if existing is not None and angle_deg(existing,candidate)>5:raise ValueError(f'{device} 已有重力参考与当前静止观测不一致，请重新启动姿态链路后标定')
            selected=unit(existing) if existing is not None else candidate
            proposed[device]={'epoch':epoch,'up_world':selected.tolist(),'method':'preserved_live_reference' if existing is not None else 'neutral_hold_bootstrap',
                              'spread_p90_deg':spread,'gyro_p90_dps':gyro_p90,'samples':len(good),'window_ns':[int(start_ns),int(end_ns)]}
        locked_at_ns=time.monotonic_ns()
        for device,data in proposed.items():
            data['locked_at_ns']=locked_at_ns
            estimator=self.engine._tilt_filters[device]
            if estimator.neutral_up is None:estimator.neutral_up=np.array(data['up_world'])
        self.events.append({'time_ns':time.monotonic_ns(),'type':'gravity_lock','devices':proposed})
        return proposed

    def install(self,report):
        if report.get('orientation_source')!='live_runtime_pose':raise ValueError('旧离线标定没有实时参考系，需重新引导录制')
        proposed={}
        for part,data in report['parts'].items():
            if part not in ARM_SEGMENTS or not data.get('accepted'):continue
            device=data['device_id'];epoch=data.get('reference_epoch')
            if any(entry[0]['device_id']==device for entry in proposed.values()):raise ValueError('多个部位不能使用同一设备进行纠偏')
            if self.engine.assignments.get(part)!=device or self.engine.epochs.get(device)!=epoch:raise ValueError('设备绑定/积分参考系已变化，请重新标定')
            if self.engine.algorithm_modes[part]==GYRO_RGBD:raise ValueError('请先切回陀螺＋加计，再应用新标定')
            up=unit(data['up_world']);estimator=self.engine._tilt_filters.get(device)
            if estimator is None:raise ValueError('没有实时积分器')
            if estimator.neutral_up is not None and angle_deg(up,estimator.neutral_up)>5:raise ValueError('实时重力参考与标定不一致')
            q=np.asarray(data['Q_camera_from_imu_world'],float)
            if q.shape!=(3,3) or not np.isfinite(q).all() or not np.allclose(q.T@q,np.eye(3),atol=1e-6) or np.linalg.det(q)<.99:raise ValueError('标定Q无效')
            if angle_deg(q@up,report['camera_up'])>.5:raise ValueError('Q没有对齐两系重力，拒绝改变倾斜')
            bone=np.asarray(data['bone_axis_sensor'],float)
            if bone.shape!=(3,) or not np.isfinite(bone).all() or not np.isclose(np.linalg.norm(bone),1,atol=1e-5):raise ValueError('标定骨轴无效')
            proposed[part]=(data,HeadingCorrector(report['camera_up'],min_innovation_deg=1.,max_offset_deg=self.max_offset_deg))
        if not proposed:raise ValueError('没有通过的部位标定')
        for part,data in report['parts'].items():
            if part in ARM_SEGMENTS and not data.get('accepted') and part in self.bindings:
                old_device=self.bindings[part][0]['device_id']
                self.engine.fusion_ready.discard(old_device)
                self.states[part]={'reason':'本轮标定未通过，原纠偏已停用','offset_deg':0.}
        for part,(data,corrector) in proposed.items():
            device=data['device_id'];estimator=self.engine._tilt_filters[device]
            if estimator.neutral_up is None:estimator.neutral_up=unit(data['up_world'])
            self.bindings[part]=(data,corrector);self.last_stamp.pop(part,None)
            self.engine.fusion_ready.add(device);self.engine.set_fusion_correction(device,np.eye(3))
            neutral=self.engine.neutral_for(device,GYRO_TILT)
            if neutral is not None:self.engine.set_neutral_for_mode(device,GYRO_RGBD,neutral)
            self.states[part]={'reason':'已应用，待选择融合算法','offset_deg':0.}
        self.events.append({'time_ns':time.monotonic_ns(),'type':'install','report':report,'max_offset_deg':self.max_offset_deg})

    def at(self,device,stamp):
        rows=self.history.get(device,())
        times=[r[0] for r in rows];i=int(np.searchsorted(times,stamp))
        if i<len(rows) and rows[i][0]==stamp:return rows[i][1]
        if i==0 or i==len(rows) or times[i]-times[i-1]>100_000_000:return None
        return Slerp([0.,float(times[i]-times[i-1])],Rotation.from_matrix([rows[i-1][1],rows[i][1]]))([float(stamp-times[i-1])]).as_matrix()[0]

    def observe(self,estimate,*,allow=True,now_ns=None):
        now=time.monotonic_ns() if now_ns is None else now_ns
        row=estimate.to_record();stamp=estimate.captured_mono_ns
        for part,(data,corrector) in self.bindings.items():
            device=data['device_id'];reason=None;innovation=None;applied=False
            if self.engine.assignments.get(part)!=device or self.engine.epochs.get(device)!=data['reference_epoch']:
                self.engine.fusion_ready.discard(device);reason='绑定/参考系已失效，需重标定'
            elif device not in self.engine.fusion_ready:reason='标定已停用：保留修正，需重新标定'
            elif not allow:reason='标定/相机停用：冻结纠偏'
            elif now-stamp>500_000_000 or stamp>now:reason='源帧过期/未来时间，冻结纠偏'
            elif stamp<=self.last_stamp.get(part,-1):reason='重复或乱序观测'
            elif self.engine.algorithm_modes[part]!=GYRO_RGBD:reason='诊断待机，尚未选择融合算法'
            else:
                self.last_stamp[part]=stamp
                base=self.at(device,stamp);direction,reason=visual_direction(row,part,data['bone_length_m'])
                if base is None:reason='没有同期IMU姿态/间隔过大'
                elif direction is not None:
                    q=np.asarray(data['Q_camera_from_imu_world'])
                    predicted=q@base@np.asarray(data['bone_axis_sensor'])
                    applied,reason,innovation=corrector.update(stamp/1e9,predicted,direction)
                    if applied:self.engine.set_fusion_correction(device,q.T@corrector.rotation()@q)
            state={'reason':reason,'offset_deg':float(np.degrees(corrector.offset_rad)), 'innovation_deg':innovation,'applied':applied,'age_ms':(now-stamp)/1e6,
                   'max_offset_deg':self.max_offset_deg,'limit_status':corrector.limit_status()}
            self.states[part]=state
            self.events.append({'time_ns':now,'type':'observation','source_ns':stamp,'part':part,**state})

    def drain_events(self):
        rows=list(self.events);self.events.clear();return rows

    def invalidate(self,reason):
        for part,(data,corrector) in self.bindings.items():
            self.engine.fusion_ready.discard(data['device_id'])
            self.states[part]={'reason':reason,'offset_deg':float(np.degrees(corrector.offset_rad))}
        self.events.append({'time_ns':time.monotonic_ns(),'type':'invalidate','reason':reason})
