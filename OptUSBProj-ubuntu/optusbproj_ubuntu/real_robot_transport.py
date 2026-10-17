"""Single UART writer for explicitly armed following and manual ordered return."""
import threading
import time
import math

from PySide6.QtCore import QThread, Signal
from .real_robot_control import ElbowControl
from .real_robot_stop_gesture import GESTURE_KINDS
from .real_robot_hand import HandBank


class ElbowSerialWorker(QThread):
    ready = Signal()
    failed = Signal(str)
    progress = Signal(object)
    sent = Signal(bytes)

    def __init__(self, port, recorder, parent=None, serial_factory=None):
        super().__init__(parent)
        self.port, self.recorder, self.factory = port, recorder, serial_factory
        self.lock = threading.Lock()
        self.quit_requested = False
        self.pending_start = None
        self.latest = None
        self.pause_reason = None
        self.active = False
        self.follow_requested = False
        self.task = 'follow'
        self.last_written = None
        self.last_state = None
        self.authorized_limits = None
        self.heartbeat_at = 0.
        self.next_step_requested = False
        self.home_pending = False
        self.task_id = 0
        self.target_timing = None
        self.target_received_at = None
        self.gap_deadline = None
        self.hands = HandBank()
        self.authorized_hands = {}

    def observe_hand(self, side, data):
        with self.lock:
            self.hands.observe(side, data, time.monotonic())
            if self.active:
                self.recorder.record('hand_observation', side=side, **data)

    def enable_hand(self, side, config, *, allow_residual=False):
        with self.lock:
            if (not self.active or self.task != 'follow' or not self.last_state
                    or self.last_state.get('phase') != 'following' or self.last_state['mode'] != 'active'
                    or self.pause_reason or self.quit_requested):
                raise ValueError('须先进入健康的同源跟随，再明确启用手指')
            if side=='right' and len(self.authorized_limits.indices)!=12:
                raise ValueError('左臂模式不许可右手')
            self.hands.enable(side, config, self.authorized_limits.frequency, time.monotonic(),allow_residual=allow_residual)
            previous=self.authorized_hands.get(side)
            if previous is None:self.authorized_hands[side]=config
            else:
                from dataclasses import replace
                merged={name:tuple(new if enabled else old for new,old,enabled in
                                  zip(getattr(config,name),getattr(previous,name),config.enabled))
                        for name in ('opened','closed','lower','upper','speed','acceleration')}
                self.authorized_hands[side]=replace(config,**merged,
                    enabled=tuple(a or b for a,b in zip(config.enabled,previous.enabled)))
            self.recorder.record('hand_enabled', side=side, config=vars(config), feedback=False,residual_bypass=allow_residual)

    def stop_hand(self, side):
        with self.lock:
            self.hands.stop('用户停用该手',side)
            self.recorder.record('hand_stop', side=side)

    def hand_snapshot(self):
        with self.lock:
            return self.hands.status()

    def heartbeat(self):
        with self.lock:
            self.heartbeat_at = time.monotonic()

    def return_snapshot(self):
        with self.lock:
            return self.last_written, None if self.last_state is None else dict(self.last_state), self.home_pending

    def advance_return(self):
        with self.lock:
            if not self.active or self.task != 'return' or not self.last_state or self.last_state.get('phase') != 'await_step':
                raise ValueError('当前没有等待确认的回位步骤')
            self.next_step_requested = True

    def confirm_home(self):
        from .real_robot_protocol import decode
        with self.lock:
            if (self.active or not self.home_pending or not self.last_state
                    or self.last_state.get('phase') != 'complete' or self.quit_requested
                    or self.last_written is None or any(q != 90. for q in decode(self.last_written)[0])):
                raise ValueError('没有本次连接内成功完成的回位指令')
            if any(decode(self.last_written)[1]):
                raise ValueError('仅手臂回位；手指非零。请先确认并执行手指平滑回零，再确认全机上电构型。')
            self.home_pending = False
            self.recorder.record('home_manually_confirmed', frame_hex=self.last_written.hex(), measured=False)
            return self.last_written

    def begin_return(self, limits, config):
        from .real_robot_return import ReturnControl
        with self.lock:
            if (self.active or self.pending_start is not None or self.quit_requested
                    or self.last_written is None or self.last_state is None
                    or self.last_state['mode'] != 'paused' or self.authorized_limits is None):
                raise ValueError('须先完成本次随动的指令侧暂停，且串口连接仍有效')
            previous = self.authorized_limits
            if (tuple(previous.indices) != tuple(limits.indices)
                    or len(previous.enabled) not in (6,12)):
                raise ValueError('回位须保持本次随动侧别和许可维数')
            if any(new and not old for new,old in zip(limits.enabled,previous.enabled)):
                raise ValueError('回位不能新增本次随动未许可的关节')
            if limits.signs != previous.signs:
                raise ValueError('回位期间不得更改场景→协议方向')
            control = ReturnControl(self.last_written, limits, config)
            self._reserve(control, None, time.monotonic()+1.)
            self.task = 'return'
            self.latest = None
            self.heartbeat_at = time.monotonic()
            self.home_pending = False

    def _reserve(self, control, target, valid_until):
        if getattr(control,'task',None)!='return' and not getattr(control,'controls_fingers',False):
            self.authorized_hands.clear()
        self.hands.reset(control.baseline)
        self.pending_start = (control, valid_until)
        self.arm_target_size = getattr(control, 'size', 6)
        self.latest = (target, valid_until)
        self.pause_reason = None
        self.follow_requested = False
        self.next_step_requested = False
        self.active = True
        self.task_id += 1
        self.target_timing = None
        self.target_received_at = None
        self.gap_deadline = None

    def begin_finger_return(self):
        from .real_robot_finger_return import FingerReturnControl
        with self.lock:
            if (self.active or self.pending_start is not None or self.quit_requested
                    or not self.home_pending or self.last_written is None or self.last_state is None
                    or self.last_state.get('phase')!='complete' or self.last_state.get('mode')!='paused'
                    or self.authorized_limits is None):
                raise ValueError('须先完成本次手臂回位，保持连接并人工核对路径')
            control=FingerReturnControl(self.last_written,self.authorized_limits,self.authorized_hands)
            self._reserve(control,None,time.monotonic()+1.)
            self.task='return';self.latest=None;self.heartbeat_at=time.monotonic();self.home_pending=False
            self.recorder.record('finger_return_requested',baseline=control.baseline.hex(),feedback=False,
                                 configs={s:vars(c) for s,c in self.authorized_hands.items()})

    def _input_guard(self, control, now, deadline, reason):
        """250 ms freshness, then bounded deceleration; never extend on old data.

        Runs under the writer lock. Only a fresh solved target may resume a gap.
        Explicit faults and hard timeout always latch the existing brake state.
        """
        if reason:
            control.brake(reason)
            return
        if self.gap_deadline is not None and now > self.gap_deadline:
            if control.mode == 'gap':
                self.recorder.record('input_timeout', task_id=self.task_id,
                    deadline=self.gap_deadline-.25, expired_ms=(now-self.gap_deadline+.25)*1000,
                    target_age_ms=None if self.target_received_at is None else
                                  (now-self.target_received_at)*1000,
                    timing=self.target_timing, grace_ms=250)
                control.brake('IMU / IK 目标超过 500 ms 未及时恢复；制动后需重新接管')
            return
        if now > deadline:
            if control.mode == 'active':
                self.gap_deadline = deadline + .25
                self.recorder.record('input_gap_started', task_id=self.task_id,
                    deadline=deadline, hard_deadline=self.gap_deadline, timing=self.target_timing)
                control.brake('输入短暂中断 / 等待新 IK；减速等待，最多额外 250 ms')
                control.mode = 'gap'
                # A stalled writer must not gain an extra window on waking.
                if now > self.gap_deadline:
                    self._input_guard(control, now, deadline, None)
            return
        if control.mode == 'gap':
            control.mode, control.reason = 'active', ''
            if getattr(control, 'phase', None) == 'transition':
                control.target = control.initial_target.copy()
            self.recorder.record('input_gap_recovered', task_id=self.task_id,
                remaining_ms=(self.gap_deadline-now)*1000, timing=self.target_timing)
            self.gap_deadline = None
        if control.mode == 'active':
            control.set_target(self.latest[0])

    def confirm_follow(self):
        with self.lock:
            if not self.active or self.task != 'follow':
                raise ValueError('没有正在运行的同源控制任务')
            if self.gap_deadline is not None:
                raise ValueError('输入恢复后再确认开始跟随')
            self.follow_requested = True

    def begin(self, baseline, theta, limits, valid_until):
        self.begin_control(ElbowControl(baseline, theta, limits), theta, valid_until)

    def begin_control(self, control, target, valid_until):
        """Accept an already validated command planner; never open a port here."""
        with self.lock:
            if not math.isfinite(valid_until):
                raise ValueError('输入有效期无效')
            if self.active or self.pending_start is not None or self.quit_requested:
                raise ValueError('已有控制任务或串口正在关闭')
            # Validate/planning dependency before reserving a physical operation.
            if self.home_pending:
                raise ValueError('回位后须人工确认真实到位，再重新接管')
            if getattr(control,'task','follow') != 'follow':
                raise ValueError('回位须使用独立入口')
            self._reserve(control, target, valid_until)
            self.task = 'follow'
            self.authorized_limits = control.limits

    def update_target(self, theta, valid_until):
        with self.lock:
            if self.task != 'follow' or not self.active:
                return
            if not math.isfinite(theta) or not 0 <= theta <= 180 or not math.isfinite(valid_until):
                self.pause_reason = '输入角度或有效期无效'
                return
            self.latest = (theta, valid_until)

    def update_arm_target(self, target, valid_until, diagnostic=None):
        requested_at = time.monotonic()
        with self.lock:
            if self.task != 'follow' or not self.active:
                return
            if (len(target) != getattr(self, 'arm_target_size', 6) or not all(math.isfinite(x) for x in target)
                    or not math.isfinite(valid_until)):
                self.pause_reason = '六轴目标或有效期无效'
                return
            self.latest = (list(target), valid_until)
            now = time.monotonic()
            self.target_timing = dict(diagnostic or {})
            self.target_timing.update(transport_received_at=now,
                transport_lock_wait_ms=(now-requested_at)*1000,
                target_update_gap_ms=None if self.target_received_at is None else
                                     (now-self.target_received_at)*1000)
            self.target_received_at = now
            self.recorder.record('arm_target_received', deadline=valid_until, timing=self.target_timing)

    def pause(self, reason='用户暂停'):
        with self.lock:
            self.pause_reason = reason

    def shutdown(self):
        with self.lock:
            self.quit_requested = True
            self.pause_reason = '关闭窗口 / 断开串口'

    def run(self):
        serial = None
        control = None
        next_tick = 0.
        previous_write = None
        try:
            factory = self.factory
            if factory is None:
                from serial import Serial
                factory = Serial
            serial = factory(port=None, baudrate=115200, bytesize=8, parity='N',
                             stopbits=1, timeout=0, write_timeout=.2,
                             xonxoff=False, rtscts=False, dsrdtr=False, exclusive=True)
            serial.dtr = False
            serial.rts = False
            serial.port = self.port
            serial.open()
            self.recorder.record('connected', port=self.port, feedback=False)
            self.ready.emit()
            while True:
                with self.lock:
                    now = time.monotonic()
                    if self.pending_start is not None:
                        control, deadline = self.pending_start
                        self.pending_start = None
                        if now > deadline:
                            if self.task == 'return':
                                control.brake('回位启动请求已过期；未发送')
                                state = {**control.status(), 'mode':'paused'}
                                self.last_state = state
                                self.progress.emit(state)
                            else:
                                self.progress.emit(dict(mode='paused', reason='启动前输入已过期；未发送'))
                            control = None
                            self.active = False
                        else:
                            next_tick = now
                            self.recorder.record('engage', baseline=control.baseline.hex(),
                                                 limits=control.limits.as_dict() if hasattr(control.limits,'as_dict') else vars(control.limits),
                                                 **control.status())
                    if control is not None:
                        if self.next_step_requested:
                            self.next_step_requested = False
                            control.advance()
                            self.recorder.record('return_step_confirmed', **control.status())
                        if self.follow_requested:
                            self.follow_requested = False
                            control.request_follow()
                            self.recorder.record('shared_follow_confirmed', **control.status())
                        if self.task == 'follow' and self.latest is not None:
                            theta, deadline = self.latest
                            age = max(0., now - (deadline - .25))
                        else:
                            deadline, age = 0., 0.
                        reason = self.pause_reason
                        if self.recorder.error:
                            reason = self.recorder.error
                        if self.task == 'follow':
                            self._input_guard(control, now, deadline, reason)
                        if self.task == 'return' and now-self.heartbeat_at > 1.:
                            reason = reason or '回位界面心跳超过 1 秒；减速暂停，不自动恢复'
                        if reason:
                            control.brake(reason)
                        if now >= next_tick:
                            frame = control.step()
                            if getattr(control,'controls_fingers',False):
                                from .real_robot_protocol import decode
                                self.hands.values=decode(frame)[1]
                            else:
                                frame = self.hands.merge(frame, now, arm_mode=control.mode,
                                    following=self.task=='follow' and getattr(control,'phase',None)=='following')
                            before = time.monotonic()
                            count = serial.write(frame)
                            if count != len(frame):
                                raise IOError(f'串口部分写入 {count}/54；指令状态不确定，不重试')
                            after = time.monotonic()
                            self.last_written = frame
                            gesture = getattr(control,'gesture',None)
                            if gesture is not None:
                                triggered = gesture.update(frame, after,
                                    following=getattr(control,'phase',None)=='following' and control.mode=='active',
                                    fresh=after <= deadline,
                                    enabled=tuple(dict(zip(control.indices,control.limits.enabled)).get(i,False)
                                                  for i in range(15)),
                                    max_gap=max(.25,2/control.limits.frequency))
                                self.recorder.record('end_gesture_status', elapsed=gesture.elapsed,
                                                     triggered=gesture.triggered, config=vars(gesture.config))
                                if triggered:
                                    self.pause_reason = f'{GESTURE_KINDS[gesture.config.kind]}结束手势保持满 {gesture.config.seconds:g} 秒'
                                    control.brake(self.pause_reason)
                                    self.recorder.record('end_gesture_triggered', frame_hex=frame.hex(),
                                                         kind=gesture.config.kind, indices=gesture.config.indices,
                                                         reason=self.pause_reason)
                            self.last_state = control.status()
                            if control.mode == 'paused' and not self.hands.stopped:
                                self.last_state = {**self.last_state, 'mode':'braking',
                                                   'reason':'臂已暂停，等待手指指令侧减速结束'}
                            self.recorder.record('write', frame_hex=frame.hex(), count=count,
                                                 input_age_ms=age * 1000,
                                                 gap_ms=None if previous_write is None else (before-previous_write)*1000,
                                                 write_ms=(after-before)*1000, hands=self.hands.status(), **control.status())
                            previous_write = before
                            self.sent.emit(frame)
                            self.progress.emit(self.last_state)
                            # Never catch up with a burst after an OS/USB scheduling delay.
                            next_tick = after + 1 / control.limits.frequency
                            if control.mode == 'paused' and self.hands.stopped:
                                if self.task == 'return' and control.phase == 'complete':
                                    self.home_pending = True
                                    self.recorder.record('return_commands_complete', frame_hex=frame.hex(), feedback=False)
                                self.recorder.record('paused', **control.status())
                                control = None
                                self.active = False
                    if self.quit_requested and control is None:
                        break
                self.msleep(2)
        except Exception as exc:
            self.recorder.record('error', message=str(exc), command_origin_valid=False)
            self.failed.emit(str(exc))
        finally:
            with self.lock:
                self.active = False
                self.pending_start = None
                self.last_written = self.last_state = None
                self.home_pending = False
            try:
                if serial is not None:
                    serial.close()
            finally:
                self.recorder.close()
