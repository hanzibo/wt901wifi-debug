"""GUI-thread, one-use delayed takeover. No serial access or position feedback."""
from copy import deepcopy
import math
import time
from PySide6 import QtCore, QtWidgets
from .real_robot_start_settings import DEFAULT_START, START_RANGES, START_LABELS
from .real_robot_protocol import decode


class AutoStartSession:
    def __init__(self, owner):
        self.owner = owner
        self.phase = 'idle'
        self.authorized = False
        self.deadline = self.last_tick = self.wait_until = None
        self.signature = None
        self.panel = QtWidgets.QGroupBox('同源接管设置 / 倒计时自动启动（每次单独确认）')
        layout = QtWidgets.QVBoxLayout(self.panel)
        form = QtWidgets.QFormLayout()
        self.widgets = {}
        for key, value in DEFAULT_START.items():
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(*START_RANGES[key]); widget.setDecimals(2); widget.setValue(value)
            self.widgets[key] = widget
            form.addRow(START_LABELS[key], widget)
            widget.valueChanged.connect(owner._invalidate_capture)
        layout.addLayout(form)
        self.start = QtWidgets.QPushButton('倒计时锁定目标 → 平滑接管 → 自动跟随…')
        self.safety_confirm = QtWidgets.QCheckBox('我已确认安全空间和独立断电手段')
        self.safety_confirm.setToolTip('允许本次勾选轴输出；与下方安全确认框同步。启动仍需二次确认。')
        self.safety_confirm.toggled.connect(owner.safety.setChecked)
        owner.safety.toggled.connect(lambda _: owner._controls())
        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(self.safety_confirm)
        start_row.addWidget(self.start, 1)
        self.cancel_button = QtWidgets.QPushButton('取消自动流程 / 运动中减速暂停')
        self.label = QtWidgets.QLabel('默认未授权。原逐步启动流程仍可使用；自动等待不是真机到位反馈。')
        self.label.setWordWrap(True); self.label.setTextFormat(QtCore.Qt.PlainText)
        self.start.clicked.connect(self.arm)
        self.cancel_button.clicked.connect(lambda: self.cancel('用户取消自动接管', stop=True))
        layout.addLayout(start_row); layout.addWidget(self.cancel_button); layout.addWidget(self.label)

    @property
    def pending(self):
        return self.phase != 'idle'

    def values(self):
        return {k: w.value() for k, w in self.widgets.items()}

    def eligible(self):
        d = self.owner
        return (d._shared_mode() and not self.pending and not d.busy and d.ik_worker is None
                and d.connected and d.worker is not None and d.worker.isRunning()
                and d.origin is not None and d.safety.isChecked()
                and not d.end_session.await_home and not d.end_session.returning
                and not d.worker.return_snapshot()[2])

    def arm(self):
        d = self.owner
        if not self.eligible(): return
        try:
            d._poll()
            if d.arm_sample is None or time.monotonic() > d.arm_sample[1]:
                raise ValueError('当前输入无效，不能授权倒计时')
            limits = d._limits()
            if not any(limits.enabled): raise ValueError('须明确勾选输出关节')
            if any(q != 90. for q in decode(d.origin)[0]):
                raise ValueError('必须先确认全 90° 上电起点')
            d.end_session.gesture(limits)
            settings = self.values()
            signature, origin = deepcopy(d.arm_sample[2]), d.origin
            rows = '\n'.join(f'{name}: {lo:g}–{hi:g}°，速度 {v:g}°/s，加速度 {a:g}°/s²'
                             for name, lo, hi, v, a, enabled in zip(
                                 [('R' if i < 6 else 'L')+str(i%6+1) for i in limits.indices],
                                 limits.lower, limits.upper, limits.speed, limits.acceleration, limits.enabled) if enabled)
            answer = QtWidgets.QMessageBox.warning(d, '二次确认：本次允许自动运动并跟随',
                f'确认真机仍在已确认的上电姿态，空间及整个过渡路径安全，并有独立断电手段？\n'
                f'倒计时 {settings["countdown"]:g} 秒后锁定人体目标，须稳定 ≥0.5 秒。\n'
                f'过渡容差 {settings["tolerance"]:g}°；指令完成后等待 {settings["settle"]:g} 秒并自动跟随。\n'
                '无位置反馈：等待不证明真机到位，之后不会再弹出到位确认。\n'
                '倒计时期间不运动；之后保持人体姿态不变。暂停不是硬件急停。\n'+rows,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
            if answer != QtWidgets.QMessageBox.Yes: return
            d._poll()
            if (not self.eligible() or d.arm_sample is None or d.arm_sample[2] != signature
                    or time.monotonic() > d.arm_sample[1]
                    or d.origin != origin or d._limits() != limits or self.values() != settings):
                raise ValueError('确认期间配置、连接或输入变化，请重新授权')
            d.arm_capture = None; d.arm_history.clear()
            self.signature = signature
            self.authorized = True; self.phase = 'countdown'
            self.last_tick = time.monotonic(); self.deadline = self.last_tick + settings['countdown']
            self.owner._arm_event('auto_start_authorized', settings=settings, feedback=False)
            self.label.setText(f'倒计时 {settings["countdown"]:g} 秒：请离开鼠标、摆好姿态；尚未发送。')
            d.start_notice.setText('本次自动接管已授权；倒计时尚未发送，原接管参考已清除。')
        except Exception as exc:
            self.label.setText('未启动自动流程：'+str(exc))
        d._controls()

    def cancel(self, reason, *, stop=False):
        if not self.pending: return
        d = self.owner
        self.phase = 'idle'; self.authorized = False
        self.deadline = self.last_tick = self.wait_until = None
        d.arm_capture = None
        self.label.setText('自动流程已取消：'+reason+'；不会自动重试。')
        d.start_notice.setText(self.label.text())
        d._arm_event('auto_start_cancelled', reason=reason)
        if stop and d.busy: d._pause(reason)
        d._controls()

    def tick(self, now=None):
        if not self.pending: return
        d = self.owner
        now = time.monotonic() if now is None else now
        if (not d.connected or d.worker is None or not d.safety.isChecked()
                or not d.worker.isRunning()
                or d.arm_sample is None or now > d.arm_sample[1]
                or d.arm_sample[2] != self.signature):
            self.cancel('连接、安全许可或新鲜输入无效', stop=True); return
        if self.last_tick is not None and now-self.last_tick > .5:
            self.cancel('界面调度中断超过 0.5 秒', stop=True); return
        self.last_tick = now
        if self.phase == 'countdown':
            if now < self.deadline:
                self.label.setText(f'倒计时 {math.ceil(self.deadline-now)} 秒：请摆好姿态；尚未发送。')
                return
            self.phase = 'locking'
            history = d.arm_history
            if (len(history) < 5 or history[-1][0]-history[0][0] < .5
                    or now-history[-1][0] > .25
                    or any(b[0]-a[0] > .25 for a, b in zip(history, list(history)[1:]))):
                self.cancel('倒计时结束时没有连续稳定的 0.5 秒输入'); return
            d._capture()
            if d.arm_capture is None:
                self.cancel('倒计时结束时人体未稳定 ≥0.5 秒或目标锁定失败'); return
            d._start(automatic=True)
            if not d.busy:
                self.cancel('接管启动失败，请检查启动诊断'); return
            self.phase = 'transition'
            self.label.setText('自动平滑过渡中：保持人体姿态；等待不代表真机到位。')
        elif self.phase == 'settling':
            if now >= self.wait_until:
                self.phase = 'confirming'
                d._confirm_follow(automatic=True)
            else:
                self.label.setText(f'过渡指令完成；等待 {self.wait_until-now:.1f} 秒后自动跟随（非到位反馈）。')

    def progress(self, state):
        if not self.pending: return
        if state['mode'] != 'active':
            self.cancel('控制进入 '+state['mode']+'：'+state.get('reason',''), stop=True)
        elif self.phase == 'transition' and state.get('phase') == 'await_confirmation':
            self.phase = 'settling'; self.wait_until = time.monotonic() + self.values()['settle']
            self.owner._arm_event('auto_start_settling', seconds=self.values()['settle'], feedback=False)
        elif self.phase == 'confirming' and state.get('phase') == 'following':
            self.phase = 'idle'; self.authorized = False
            self.label.setText('本次自动接管完成，正在跟随；授权已消耗。')
            self.owner._arm_event('auto_start_completed', feedback=False)

    def controls(self):
        d = self.owner
        # Checkpoint loading clears safety with signals blocked. Reconcile here
        # as well as on toggles, keeping a single authorization source of truth.
        with QtCore.QSignalBlocker(self.safety_confirm):
            self.safety_confirm.setChecked(d.safety.isChecked())
        self.safety_confirm.setEnabled(d.safety.isEnabled())
        self.panel.setVisible(d._shared_mode())
        self.start.setEnabled(self.eligible())
        self.cancel_button.setEnabled(self.pending)
        for w in self.widgets.values(): w.setEnabled(not self.pending and not d.busy and d.ik_worker is None)
        if self.pending:
            for w in (d.mode_select, d.frequency, d.manual, d.capture_button, d.start_button,
                      d.follow_button, d.origin_button, *d.end_session.configuration_widgets,
                      d.end_session.return_button, d.end_session.continuous_button,
                      d.shared_palm, d.shared_offset, d.shared_offset_axis, d.shared_offset_angle,
                      d.shared_offset_copy, d.mirror_check, *d.right_options,
                      *(w for row in d.axis_rows+d.right_axis_rows for w in row)):
                w.setEnabled(False)
