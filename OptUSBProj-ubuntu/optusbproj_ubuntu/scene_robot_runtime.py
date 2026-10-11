"""Qt-free runtime orchestration with injected callbacks and clock.

The GUI owns PoseEngine; this runtime reads copied snapshots only.
"""
import time
import numpy as np
from .pose import PoseEngine
from .scene_robot_state import dispatch_reason, cycle_wait_ms
from .scene_robot_solver import SceneSolver
from .scene_robot_session import SceneSession
from .scene_robot_log import SceneLog, SceneDiagnostics
from .wrist_diagnostics import WristDiagnostics
from .x1_live import MAX_JOINT_STEP_DEG, MAX_JOINT_SPEED_DEG_S


class SceneRuntime:
    def __init__(self, port, mailbox, speed, angle, logging, wrist_speed,
                 hand_mailbox, report, failed, interrupted, sleep_ms, *,
                 bridge_factory=None, clock=time.monotonic, log_folder=None):
        self.port, self.mailbox, self.speed, self.angle = port, mailbox, speed, angle
        self.logging, self.wrist_speed, self.hand_mailbox = logging, wrist_speed, hand_mailbox
        self.report, self.failed = report, failed
        self.interrupted, self.sleep_ms = interrupted, sleep_ms
        self.bridge_factory, self.clock, self.log_folder = bridge_factory, clock, log_folder

    def run(self):
        log = None
        session = SceneSession(
            self.port, self.mailbox, self.hand_mailbox, self.report,
            self.interrupted, self.sleep_ms, bridge_factory=self.bridge_factory,
            clock=self.clock)
        try:
            C, previous = session.connect(self.angle)
            models = session.models
            human_rest = PoseEngine().transforms(include_natural=False)
            solver = SceneSolver(models, C, human_rest, clock=self.clock)
            config={'scene':session.bridge.data,'axes':C.tolist(),'neutral':{s:m.neutral.tolist() for s,m in models.items()},
                    'preview_limits':{s:[m.lower.tolist(),m.upper.tolist()] for s,m in models.items()},
                    'speed_deg_s':self.speed,'wrist_speed_deg_s':self.wrist_speed,
                    'joint_speed_limits_deg_s':[self.speed]*4+[self.wrist_speed]*2,
                    'joint_step_caps_deg':[MAX_JOINT_STEP_DEG]*4+
                        [MAX_JOINT_STEP_DEG*self.wrist_speed/MAX_JOINT_SPEED_DEG_S]*2,'version':2}
            log = SceneLog(self.logging, config, folder=self.log_folder)
            session.log = log
            self.report({'connected':True,'angles':previous,'axes':C,'config':config,
                         'log':str(log.path) if log.path else None})
            last=self.clock(); last_read=last
            wrist_tracker=WristDiagnostics()
            diagnose = SceneDiagnostics(self.report, log, clock=self.clock).emit
            while not self.interrupted():
                now=self.clock(); dt=max(.001,now-last);last=now
                enabled,sample,generation=self.mailbox.get()
                if enabled and sample and now-sample['at']<.25:
                    solve_start=self.clock()
                    result = solver.solve(sample, previous, self.speed, self.wrist_speed, dt)
                    targets, errors = result.targets, result.errors
                    segment_errors, side_ms = result.segment_errors, result.side_ms
                    palm_targets, relative_targets = result.palm_targets, result.relative_targets
                    step = result.step
                    # Re-check stop/staleness AFTER solving and before sending.
                    still_on,current,current_generation=self.mailbox.get()
                    finished=self.clock()
                    timing={'ik_ms':(finished-solve_start)*1000,'side_ik_ms':side_ms,
                            'snapshot_age_ms':(finished-sample['at'])*1000,
                            'latest_snapshot_age_ms':None if current is None else (finished-current['at'])*1000,
                            'device_ages_ms':sample.get('device_ages_ms',{}),'gui_interval_ms':sample.get('gui_interval_ms')}
                    reason=dispatch_reason(still_on,generation,current_generation,current,sample,finished)
                    if reason is None:
                        rpc_start=self.clock()
                        actual=session.exchange(targets)
                        timing['rpc_ms']=(self.clock()-rpc_start)*1000
                        if actual is None:break
                        if self.mailbox.get()[2]!=generation:
                            previous=actual
                            wrist_tracker.reset()
                            continue
                        if any(np.max(np.abs(actual[s]-q))>1e-5 for s,q in targets.items()):
                            raise ValueError('场景未应用目标角度，请检查其他控制器')
                        readback_at=self.clock()
                        wrist={side:wrist_tracker.measure(side,sample['at'],readback_at,palm_targets[side],
                            models[side].tool_frame(actual[side])[:3,:3],actual[side],previous[side],step,
                            models[side].lower,models[side].upper,relative_targets[side]) for side in targets}
                        previous=actual
                        log.write({'mono_s':readback_at,'input_mono_s':sample['at'],
                            'target':{s:q.tolist() for s,q in targets.items()},
                            'actual':{s:q.tolist() for s,q in actual.items()},'errors':errors,
                            'task':sample.get('task','positions'),'scale':sample.get('scale','uniform'),
                            'segment_orientation':sample.get('segment_orientation',False),
                            'segment_weight':sample.get('segment_weight',.15),
                            'segment_errors_deg':segment_errors,'timing':timing,
                            'palm_offsets':sample.get('palm_offsets',{}),
                            'solver_mode':sample.get('solver_mode','staged'),
                            'wrist_diagnostics':wrist})
                        diagnose('sent','正在发送：IK %.0f ms；通信 %.0f ms；快照 %.0f ms'%(timing['ik_ms'],timing['rpc_ms'],timing['snapshot_age_ms']),**timing)
                        self.report({'angles':actual,'errors':errors,'following':True,'task':sample.get('task','positions'),
                                          'segment_errors':segment_errors,'solver_mode':sample.get('solver_mode','staged'),
                                          'wrist_diagnostics':wrist})
                    else:
                        # Hand authorization is independent of a rejected arm snapshot.
                        if self.hand_mailbox.snapshot()[0]:
                            actual=session.exchange()
                            if actual is None:break
                            previous=actual
                        messages={'sending_stopped':'发送已停止：'+self.mailbox.stop_reason,
                                  'generation_changed':'授权已改变：丢弃旧求解结果',
                                  'missing_snapshot':'没有 GUI 快照：本轮不发送',
                                  'solve_snapshot_expired':'求解快照过期：%.0f ms ≥ 250 ms；IK %.0f ms；最新快照 %s ms，本轮不发送'%(
                                      timing['snapshot_age_ms'],timing['ik_ms'],
                                      '无' if current is None else '%.0f'%timing['latest_snapshot_age_ms'])}
                        diagnose(reason,messages[reason],**timing)
                else:
                    wrist_tracker.reset()
                    if now-last_read>.2 or self.hand_mailbox.snapshot()[0]:
                        actual=session.exchange()
                        if actual is None:break
                        previous=actual;last_read=now
                        self.report({'angles':previous,'following':False})
                    if not enabled:
                        diagnose('sending_stopped','发送已停止：'+self.mailbox.stop_reason)
                    elif sample is None:
                        diagnose('missing_snapshot','没有 GUI 快照：保持当前角度')
                    else:
                        diagnose('gui_snapshot_expired','GUI 快照超时：%.0f ms ≥ 250 ms；等待新快照，保持当前角度'%((now-sample['at'])*1000),snapshot_age_ms=(now-sample['at'])*1000)
                wait=cycle_wait_ms(now,self.clock())
                if wait:self.sleep_ms(wait)
        except Exception as exc:
            if log is not None:
                log.write({'event':'failure','error':str(exc),'mono_s':self.clock()},
                          ensure_ascii=False)
            self.mailbox.set(False)
            self.hand_mailbox.stop()
            self.failed(str(exc))
        finally:
            try:
                if log is not None:
                    log.write({'event':'worker_closed','mono_s':self.clock()})
            finally:
                try:
                    if log is not None:
                        log.close()
                finally:
                    session.close()
