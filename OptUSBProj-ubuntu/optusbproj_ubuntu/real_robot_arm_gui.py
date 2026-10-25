"""Six-axis option added to the existing single-elbow physical dialog."""
from collections import deque
from copy import deepcopy
import time
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtCore, QtWidgets
from .real_robot_gui import RealRobotDialog
from .real_robot_arm import ArmLimits, ArmControl, RelativeArmIK, load_profile
from .real_robot_arm_worker import ArmIKWorker
from .real_robot_protocol import ARM_LIMITS, decode
from .scene_robot_state import pose_sample, StaleImuInput
from .real_robot_start_diagnostics import snapshot as start_snapshot, delta_text
from .real_robot_timing import observe as observe_input_timing
from .real_robot_shared import SharedArmIK, SharedArmControl
from .real_robot_mirror import CommandMirror
from .real_robot_bilateral import BilateralLimits, BilateralIK
from .real_robot_right_panel import build_right_panel
from .real_robot_end_panel import EndSession
from .real_robot_checkpoint_gui import CheckpointSession
from .real_robot_auto_start import AutoStartSession
from .real_robot_start_diagnostics import reference_deltas
from .real_robot_hand_gui import HandSession


def pose_distance(a, b):
    """Largest orientation displacement, including both segments and the palm."""
    if 'rotations' not in a:
        if set(a) != set(b) or not a:
            raise ValueError('参与手臂配置变化')
        return max(pose_distance(a[s],b[s]) for s in a)
    return max(np.rad2deg(Rotation.from_matrix(x.T@y).magnitude()) for x,y in
               [(a['hand'], b['hand']), *[(a['rotations'][k], b['rotations'][k])
                                         for k in ('upper_arm', 'forearm')]])


class RealRobotModesDialog(RealRobotDialog):
    def __init__(self, viewer):
        # Base construction calls overridden _controls: initialize guards first.
        self.mode_select = None
        self.end_session = None
        self.checkpoints = None
        self.auto_start = None
        self.hand_session = None
        self.ik_worker = None
        self.arm_history = deque()
        self.arm_sample = None
        self.arm_capture = None
        self.arm_configuration = None
        self.arm_ages = None
        self.arm_input_error = None
        self.start_attempt = 0
        self.start_gate = None
        self.shared_phase = None
        self.shared_control_mode = None
        self.mirror_worker = None
        self.mirror_ready = False
        super().__init__(viewer)
        self.setWindowTitle('真机受限 IMU 控制 · 单肘 / 左臂六轴 · 无位置反馈')
        layout = self.layout()
        self.mode_select = QtWidgets.QComboBox()
        self.mode_select.addItems(['单肘 L4（原方案）', '左臂六轴相对 IK（原方案）',
                                  '左臂仿真同源 IK（上电姿态平滑接管）',
                                  '双臂仿真同源 IK（可逐轴启用左 / 右臂）'])
        layout.insertWidget(1, self.mode_select)
        layout.itemAt(0).widget().setText(
            '真实硬件，无位置反馈！头腰保持 90°；手指默认不动，须单独授权；右臂仅双臂模式可动。\n'
            '只有明确勾选的关节可动；显示均为指令值，暂停不是物理急停。\n'
            '预留安全空间及独立断电手段；禁止同时打开手动串口 demo。')
        self.arm_panel = QtWidgets.QGroupBox('左臂六轴：默认勾选六轴，仍须安全确认与明确启动')
        grid = QtWidgets.QGridLayout(self.arm_panel)
        for col, text in enumerate(('输出', '最小 °', '最大 °', '速度 °/s', '加速度 °/s²', '场景→协议方向')):
            grid.addWidget(QtWidgets.QLabel(text), 0, col)
        self.axis_rows = []
        defaults = ArmLimits()
        for i in range(6):
            enabled = QtWidgets.QCheckBox(f'L{i+1}')
            enabled.setChecked(True)
            enabled.toggled.connect(self._invalidate_capture)
            lo, hi = ARM_LIMITS[6+i]
            lower = self._spin(lo, hi, defaults.lower[i], '')
            upper = self._spin(lo, hi, defaults.upper[i], '')
            speed = self._spin(.1, 250, defaults.speed[i], '')
            accel = self._spin(.1, 250, defaults.acceleration[i], '')
            sign = QtWidgets.QComboBox()
            sign.addItems(['+1（待逐轴验证）', '−1（反向）'])
            sign.currentIndexChanged.connect(self._invalidate_capture)
            row = (enabled, lower, upper, speed, accel, sign)
            self.axis_rows.append(row)
            for col, widget in enumerate(row):
                grid.addWidget(widget, i+1, col)
        notice = QtWidgets.QLabel('默认采用用户指定范围；L4 为 0–119°。上下界受原 PC 软件范围约束，\n'
                                  '不是机械限位认证。±1 为映射候选，须小幅逐轴验证。部分输出不保证满足完整姿态任务。')
        notice.setWordWrap(True)
        grid.addWidget(notice, 7, 0, 1, 6)
        _, _, provenance = load_profile()
        grid.addWidget(QtWidgets.QLabel('运动学：已验证的场景导出几何；无需启动 CoppeliaSim。'), 8, 0, 1, 6)
        self.arm_panel.setToolTip(provenance)
        self.ik_label = QtWidgets.QLabel('等待左上臂、左前臂、左手背三个不同设备；先自然下垂归零。')
        self.ik_label.setWordWrap(True)
        grid.addWidget(self.ik_label, 9, 0, 1, 6)
        self.shared_options = QtWidgets.QWidget()
        options = QtWidgets.QVBoxLayout(self.shared_options)
        self.shared_palm = QtWidgets.QCheckBox('同源目标使用手掌朝向（需要左手背 IMU）')
        self.shared_palm.setChecked(True)
        self.shared_offset = QtWidgets.QCheckBox('手掌目标固定转向（局部坐标，默认关闭）')
        self.shared_offset_axis = QtWidgets.QComboBox()
        self.shared_offset_axis.addItems(['x', 'y', 'z'])
        self.shared_offset_axis.setCurrentText('z')
        self.shared_offset_angle = self._spin(-180, 180, 90, ' °')
        self.shared_offset_copy = QtWidgets.QPushButton('复制仿真窗口左手转向配置')
        self.shared_offset_copy.clicked.connect(self._copy_scene_offset)
        self.shared_offset_axis.currentIndexChanged.connect(self._invalidate_capture)
        offset_row = QtWidgets.QHBoxLayout()
        for widget in (self.shared_offset, self.shared_offset_axis,
                       self.shared_offset_angle, self.shared_offset_copy):
            offset_row.addWidget(widget)
        self.mirror_check = QtWidgets.QCheckBox('镜像最终发送量到 CoppeliaSim（手指为名义近似；停止的原场景；23000）')
        self.mirror_check.toggled.connect(self._mirror_changed)
        self.follow_button = QtWidgets.QPushButton('确认真机已稳定到位，开始同源跟随')
        self.follow_button.clicked.connect(self._confirm_follow)
        for widget in (self.shared_palm, self.shared_offset):
            widget.toggled.connect(self._invalidate_capture)
        options.addWidget(self.shared_palm)
        options.addLayout(offset_row)
        for widget in (self.mirror_check, self.follow_button):
            options.addWidget(widget)
        grid.addWidget(self.shared_options, 10, 0, 1, 6)
        self.shared_options.hide()
        layout.insertWidget(6, self.arm_panel)
        self.arm_panel.hide()
        self.right_panel, self.right_axis_rows, self.right_options = build_right_panel(self)
        layout.insertWidget(7, self.right_panel)
        self.right_panel.hide()
        self.end_session = EndSession(self)
        layout.insertWidget(8,self.end_session.panel)
        self.end_session.panel.hide()
        self.setWindowTitle('真机受限 IMU 控制 · 单肘 / 单臂 / 双臂 · 无位置反馈')
        self.start_notice = QtWidgets.QLabel('六轴待启动：请采集参考并确认安全许可。')
        self.start_notice.setWordWrap(True)
        self.start_notice.setTextFormat(QtCore.Qt.PlainText)
        self.start_notice.hide()
        layout.insertWidget(layout.indexOf(self.start_button)+1, self.start_notice)
        # Keep the six-axis status accessible on smaller desktop screens too.
        self.content_layout = QtWidgets.QVBoxLayout()
        while layout.count():
            self.content_layout.addItem(layout.takeAt(0))
        content = QtWidgets.QWidget()
        content.setLayout(self.content_layout)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.mode_select.currentIndexChanged.connect(self._mode_changed)
        self.resize(1100, 850)
        # Reuse an already configured preview, including its disabled state.
        self._copy_scene_offset(report=False)
        self._copy_right_offset(report=False)
        self.auto_start = AutoStartSession(self)
        self.content_layout.insertWidget(2, self.auto_start.panel)
        self.checkpoints = CheckpointSession(self)
        self.content_layout.insertWidget(2, self.checkpoints.panel)
        self.hand_session = HandSession(self)
        self.content_layout.insertWidget(4, self.hand_session.panel)
        self.content_layout.removeWidget(self.runtime_recording.panel)
        self.content_layout.insertWidget(2, self.runtime_recording.panel)
        self._controls()

    def _arm_mode(self):
        return self.mode_select is not None and self.mode_select.currentIndex() in (1, 2, 3)

    def _shared_mode(self):
        return self.mode_select is not None and self.mode_select.currentIndex() in (2, 3)

    def _bilateral_mode(self):
        return self.mode_select is not None and self.mode_select.currentIndex() == 3

    def _right_offset(self):
        _, enabled, axis, angle, _ = self.right_options
        return dict(enabled=enabled.isChecked(), axis=axis.currentText(), degrees=angle.value())

    def _copy_right_offset(self, checked=False, *, report=True):
        if self.busy or self.ik_worker is not None:
            return
        scene = getattr(self.viewer, 'scene_robot_dialog', None)
        if scene is None:
            return
        offset = scene.palm_offsets().get('right')
        if offset is None:
            return
        _,enabled,axis,angle,_ = self.right_options
        enabled.setChecked(offset['enabled'])
        axis.setCurrentText(offset['axis'])
        angle.setValue(offset['degrees'])
        if report:
            self._invalidate_capture()
            self.status.setText('已复制右手转向配置；请重新锁定目标。')

    def _selected_sides(self):
        return self._limits().sides if self._bilateral_mode() else ('left',)

    def _read_arm_sample(self):
        sides = self._selected_sides()
        if not sides:
            raise ValueError('至少勾选一个关节输出')
        sample, signature = pose_sample(self.viewer.engine, sides, sides, self.manual.isChecked())
        payload = sample['sides'] if self._bilateral_mode() else sample['sides']['left']
        return sample, signature, payload

    def _palm_offset(self):
        return dict(enabled=self.shared_offset.isChecked(),
                    axis=self.shared_offset_axis.currentText(),
                    degrees=self.shared_offset_angle.value())

    def _copy_scene_offset(self, checked=False, *, report=True):
        if self.busy or self.ik_worker is not None:
            return
        scene = getattr(self.viewer, 'scene_robot_dialog', None)
        if scene is None:
            if report:
                self.status.setText('仿真配置窗口尚未打开；保留当前转向配置。')
            return
        offset = scene.palm_offsets().get('left')
        if offset is None:
            return
        self.shared_offset.setChecked(offset['enabled'])
        self.shared_offset_axis.setCurrentText(offset['axis'])
        self.shared_offset_angle.setValue(offset['degrees'])
        if report:
            self._invalidate_capture()
            self.status.setText('已复制左手转向配置：'+
                ('启用' if offset['enabled'] else '关闭')+
                f'，局部 {offset["axis"].upper()}，{offset["degrees"]:+g}°；请重新锁定目标。')

    def _mirror_changed(self, checked):
        self._invalidate_capture()
        if checked:
            scene = getattr(self.viewer, 'scene_robot_dialog', None)
            if scene is not None and scene.worker is not None:
                self.mirror_check.setChecked(False)
                self.status.setText('请先断开原仿真预览连接，避免两个写入者。')
                return
            if self.mirror_worker is not None:
                self.mirror_check.setChecked(False)
                return
            self.mirror_worker = CommandMirror(parent=self, sides=('left','right') if self._bilateral_mode() else ('left',))
            self.mirror_worker.ready.connect(self._mirror_ready)
            self.mirror_worker.failed.connect(self._mirror_failed)
            self.mirror_worker.finished.connect(self._mirror_finished)
            self.mirror_worker.start()
        elif self.mirror_worker is not None:
            self.mirror_ready = False
            self.mirror_worker.shutdown()

    def _mirror_ready(self):
        self.mirror_ready = True
        self.status.setText('场景几何已核对；镜像待命，未发送关节角。')

    def _mirror_failed(self, reason):
        self.mirror_ready = False
        self._arm_event('mirror_failed', reason=reason)
        if self.busy:
            self._pause('仿真镜像失败：'+reason)
        self.status.setText('仿真镜像失败：'+reason)

    def _mirror_finished(self):
        worker = self.sender()
        if worker is self.mirror_worker:
            self.mirror_worker = None
            self.mirror_ready = False
            with QtCore.QSignalBlocker(self.mirror_check):
                self.mirror_check.setChecked(False)
        worker.deleteLater()
        if self.closing and self.worker is None and self.ik_worker is None:
            self.close()

    def _sent(self, frame):
        super()._sent(frame)
        if self._shared_mode() and self.mirror_ready and self.arm_configuration is not None:
            signs = (self.arm_configuration.mirror_signs if self._bilateral_mode()
                     else self.arm_configuration.signs)
            self.mirror_worker.submit(frame, signs)

    def _transition_tolerance(self):
        return self.auto_start.widgets['tolerance'].value() if self.auto_start is not None and self._shared_mode() else 2.

    def _pose_guard(self, reference, current, context):
        deltas = reference_deltas((current,), (reference,))
        key = max(deltas, key=deltas.get)
        limit = self._transition_tolerance()
        if deltas[key] > limit:
            label = key.replace('left.', '左臂 ').replace('right.', '右臂 ')
            label = label.replace('upper_arm', '上臂').replace('forearm', '前臂').replace('hand', '手掌')
            raise ValueError(f'{context}：{label}变化 {deltas[key]:.2f}° > 容差 {limit:g}°；请重新接管')

    def _confirm_follow(self, *, automatic=False):
        if not automatic and self.auto_start is not None and self.auto_start.pending:
            return
        if automatic and not (self.auto_start.authorized and self.auto_start.phase == 'confirming'):
            return
        if self.shared_phase != 'await_confirmation' or not self.busy or self.arm_sample is None:
            if automatic: self.auto_start.cancel('未处于等待跟随确认状态', stop=True)
            return
        try:
            if self.arm_capture is None or time.monotonic() > self.arm_sample[1]:
                raise ValueError('过渡后参考或输入已失效')
            self._pose_guard(self.arm_capture[0], self.arm_sample[0], '过渡后人体超差')
        except ValueError as exc:
            self._pause(str(exc))
            return
        answer = QtWidgets.QMessageBox.Yes if automatic else QtWidgets.QMessageBox.warning(self, '人工确认，不是位置反馈',
            '确认真机已稳定，人体仍保持过渡时姿态，且路径与周边无碰撞？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        if answer == QtWidgets.QMessageBox.Yes:
            self._poll()
            if (not self.busy or self.arm_capture is None or self.arm_sample is None
                    or time.monotonic() > self.arm_sample[1]
                    or self.arm_sample[2] != self.arm_capture[2]
                    or (automatic and not self.auto_start.authorized)):
                self._pause('确认期间输入变化，未开始跟随')
                return
            try:
                self._pose_guard(self.arm_capture[0], self.arm_sample[0], '确认期间人体超差')
                self.worker.confirm_follow()
            except Exception as exc:
                self._pause('无法开始跟随：'+str(exc))

    def _mode_changed(self):
        self._invalidate_capture()
        self.history.clear()
        self.arm_history.clear()
        self.arm_sample = self.sample = None
        arm = self._arm_mode()
        self.shared_phase = None
        if self.mirror_check.isChecked():
            self.mirror_check.setChecked(False)
        self.shared_options.setVisible(self._shared_mode())
        self.arm_panel.setVisible(arm)
        self.right_panel.setVisible(self._bilateral_mode())
        self.start_notice.setVisible(arm)
        for widget in (self.speed, self.acceleration, self.gain, self.sign):
            widget.setVisible(not arm)
            for i in range(self.content_layout.count()):
                form = self.content_layout.itemAt(i).layout()
                if isinstance(form, QtWidgets.QFormLayout):
                    label = form.labelForField(widget)
                    if label is not None:
                        label.setVisible(not arm)
        self.safety.setText('我已确认安全空间和独立断电手段，允许本次勾选轴输出' if arm else
                            '我已确认安全空间和物理断电手段，允许本次仅 L4 控制')
        self.safety.setChecked(False)
        self.capture_button.setText('采集当前人体三段相对参考（稳定 ≥0.5 秒）' if arm else
                                    '采集当前人体屈肘参考（稳定保持 ≥0.5 秒）')
        if self._shared_mode():
            self.capture_button.setText('锁定当前同源目标（稳定 ≥0.5 秒；过渡时保持不动）')
        self.start_button.setText('明确启动：勾选的左臂关节' if arm else '明确启动：仅左肘 L4')
        if self._bilateral_mode():
            self.start_button.setText('明确启动：勾选的左 / 右臂关节')
        self.range_label.setText('六轴独立限速 / 限加速度；250 只是设置上限，不是已验证的安全值。' if arm else
                                f'L4 指令范围 {self.range_text}；首帧保持起点。\n'
                                '250 仅为可设置上限，不是已验证的真机安全值；请逐步调高。')
        self._controls()

    def _invalidate_capture(self, *_):
        if self.auto_start is not None:
            self.auto_start.cancel('配置、设备或安全许可变化', stop=True)
        if self.arm_capture is not None:
            self._arm_event('arm_reference_invalidated', reason='配置、设备签名或安全许可变化',
                            **start_snapshot(self))
            self.start_notice.setText('六轴参考已失效：配置、设备或安全许可变化；请重新采集。')
        self.arm_capture = None
        super()._invalidate_capture()

    def _motion_controls_ready(self, connected):
        if self.mode_select is None or not self._arm_mode():
            return super()._motion_controls_ready(connected)
        allowed = connected and not self.busy and self.ik_worker is None and self.safety.isChecked()
        return allowed and self.origin is not None, allowed and self.arm_capture is not None

    def _controls(self):
        super()._controls()
        if self.mode_select is None:
            return
        for widget in (self.shared_palm, self.shared_offset, self.shared_offset_axis,
                       self.shared_offset_angle, self.shared_offset_copy, self.mirror_check):
            widget.setEnabled(not self.busy and self.ik_worker is None)
        self.follow_button.setEnabled(self.busy and self.shared_phase == 'await_confirmation'
                                      and self.shared_control_mode == 'active'
                                      and self.arm_capture is not None and self.safety.isChecked())
        self.follow_button.setToolTip('仅过渡指令完成且未暂停时可用；暂停原因见启动诊断。无位置反馈，须人工确认真机。')
        self.mode_select.setEnabled(not self.busy and self.ik_worker is None)
        for row in self.axis_rows:
            for widget in row:
                widget.setEnabled(not self.busy and self.ik_worker is None)
        for widget in (*getattr(self, 'right_options', ()),
                       *(w for row in getattr(self, 'right_axis_rows', ()) for w in row)):
            widget.setEnabled(not self.busy and self.ik_worker is None)
        if self._arm_mode():
            gate = (self.connected, self.busy, self.ik_worker is not None,
                    self.arm_capture is not None, self.safety.isChecked(), self.start_button.isEnabled())
            if gate != self.start_gate:
                self.start_gate = gate
                self._arm_event('arm_start_gate', button_enabled=self.start_button.isEnabled(),
                                **start_snapshot(self))
        if self.end_session is not None:
            self.end_session.controls()
        if self.checkpoints is not None:
            self.checkpoints.controls()
        if self.auto_start is not None:
            self.auto_start.controls()
        if self.hand_session is not None:
            self.hand_session.panel.setVisible(self._shared_mode())

    def _arm_event(self, event, **data):
        if self.worker is not None:
            self.worker.recorder.record(event, **data)

    def _ready(self):
        super()._ready()
        self._detach_acquisition_trace()
        source = getattr(self.viewer,'source',None)
        if source is not None and hasattr(source,'add_diagnostic_sink'):
            recorder = self.worker.recorder
            self._acquisition_sink = lambda row: recorder.record('acquisition', source_event=row['event'],
                                                **{k:v for k,v in row.items() if k != 'event'})
            self._acquisition_source = source
            source.add_diagnostic_sink(self._acquisition_sink)

    def _detach_acquisition_trace(self):
        source = getattr(self,'_acquisition_source',None)
        if source is not None:
            source.remove_diagnostic_sink(self._acquisition_sink)
            self._acquisition_source = None

    def _start_rejected(self, code, message, data, clear_reference=False):
        if self.auto_start is not None: self.auto_start.cancel(message)
        self._arm_event('arm_start_rejected', attempt=self.start_attempt, reason_code=code,
                        reason=message, **data)
        self.start_notice.setText('六轴未启动：'+message+'\n'+delta_text(data))
        self.status.setText(self.start_notice.text())
        if clear_reference:
            self.arm_capture = None
        self._controls()

    def _limits(self):
        def limits(rows, side):
            return ArmLimits(frequency=self.frequency.value(), side=side,
                speed=tuple(r[3].value() for r in rows),
                acceleration=tuple(r[4].value() for r in rows),
                lower=tuple(r[1].value() for r in rows), upper=tuple(r[2].value() for r in rows),
                enabled=tuple(r[0].isChecked() for r in rows),
                signs=tuple(1 if r[5].currentIndex() == 0 else -1 for r in rows))
        left = limits(self.axis_rows, 'left')
        return BilateralLimits(limits(self.right_axis_rows,'right'),left) if self._bilateral_mode() else left

    def _poll(self):
        if self.end_session is not None and self.end_session.tick():
            return
        if not self._arm_mode():
            return super()._poll()
        self.input_timing = observe_input_timing(self)
        try:
            sample, signature, side = self._read_arm_sample()
            deadline = sample['at']+.25-max(sample['device_ages_ms'].values())/1000
            if self.signature != signature:
                self.arm_history.clear()
                if self.arm_capture is not None:
                    self._invalidate_capture()
            self.signature = signature
            self.arm_sample = (side, deadline, signature)
            self.arm_ages = {k:float(v) for k,v in sample['device_ages_ms'].items()}
            if self.arm_input_error is not None:
                self._arm_event('arm_input_recovered', device_ages_ms=self.arm_ages)
            self.arm_input_error = None
            now = time.monotonic()
            self.arm_history.append((now, side))
            while self.arm_history and now-self.arm_history[0][0] > .7:
                self.arm_history.popleft()
            self.input_label.setText(f'{len(sample["device_ages_ms"])} 设备有效，最大年龄 {max(sample["device_ages_ms"].values()):.0f} ms；手掌只分配给各侧 5/6')
            if self.busy and self.arm_capture is not None and signature == self.arm_capture[2] and self.ik_worker is not None:
                if self._shared_mode() and self.shared_phase != 'following':
                    self._pose_guard(self.arm_capture[0], side, '过渡期间人体超差')
                self.ik_worker.submit(side, deadline, self.input_timing)
            self._arm_event('arm_input_timing', valid=True, deadline=deadline, **self.input_timing)
        except Exception as exc:
            self._arm_event('arm_input_timing', valid=False, reason=str(exc), **self.input_timing)
            if self.arm_input_error != str(exc):
                self._arm_event('arm_input_invalid', reason=str(exc), **start_snapshot(self))
            self.arm_input_error = str(exc)
            self.arm_sample = None
            self.arm_history.clear()
            self.input_label.setText(f'三设备输入无效：{exc}')
            if self.busy and not (isinstance(exc, StaleImuInput) and .25 < exc.age):
                self._pause(str(exc))
        if self.auto_start is not None:
            self.auto_start.tick()
        self._controls()

    def _capture(self):
        if self.end_session is not None and self.end_session.await_home:
            self.status.setText('先人工确认真机回位，再重新锁定目标')
            return
        if not self._arm_mode():
            return super()._capture()
        if self.arm_sample is None or self.origin is None or self.busy or not self.safety.isChecked():
            return
        history = self.arm_history
        if (len(history) < 5 or history[-1][0]-history[0][0] < .5 or
                max(pose_distance(history[0][1], side) for _, side in history) > 2.):
            self.status.setText('请上臂、前臂、手掌一起稳定保持 ≥0.5 秒，朝向波动不超过 2°')
            return
        try:
            limits = self._limits()
            ArmControl(self.origin, limits)  # Check whitelist/ranges/explicit output permissions before capture.
            self.arm_capture = deepcopy(self.arm_sample)
            self.start_notice.setText('六轴参考已采集：请保持三段姿态不变，再点击明确启动。')
            self.status.setText('三段相对参考已采集；机器人起点是最后确认指令，不要求人体与机器人同姿态。')
            if self._shared_mode():
                if any(q != 90 for q in decode(self.origin)[0]):
                    raise ValueError('同源模式须重新上电并确认起点，不支持从暂停位置重新接管')
                self.status.setText('同源目标已锁定；启动后先平滑过渡，请保持人体不动。')
            self.worker.recorder.record('arm_reference', baseline=self.origin.hex(), signature=repr(self.signature),
                                        limits=limits.as_dict() if self._bilateral_mode() else vars(limits), feedback=False,
                                        **self._reference_log(self.arm_capture[0]))
        except Exception as exc:
            self.arm_capture = None
            self.status.setText(f'无法采集：{exc}')
        self._controls()

    def _reference_log(self, payload):
        if self._bilateral_mode():
            return dict(human_sides={s:dict(points=np.asarray(p['points']).tolist(),
                        hand=p['hand'].tolist(), segments={k:v.tolist() for k,v in p['rotations'].items()})
                        for s,p in payload.items()})
        return dict(human_points=np.asarray(payload['points']).tolist(), human_hand=payload['hand'].tolist(),
                    human_segments={k:v.tolist() for k,v in payload['rotations'].items()})

    def _start(self, *, automatic=False):
        if not automatic and self.auto_start is not None and self.auto_start.pending:
            return
        if automatic and not (self.auto_start.authorized and self.auto_start.phase == 'locking'):
            return
        if self.end_session is not None and self.end_session.await_home:
            self.status.setText('回位后尚未人工确认，不允许重新随动')
            return
        if not self._arm_mode():
            return super()._start()
        self.start_attempt += 1
        data = start_snapshot(self)
        self._arm_event('arm_start_attempt', attempt=self.start_attempt, **data)
        guards = [(self.busy, 'already_active', '已有随动任务'),
                  (self.arm_capture is None, 'missing_reference', '尚无有效六轴参考，请重新采集'),
                  (self.worker is None, 'missing_serial', '串口未连接'),
                  (self.ik_worker is not None, 'ik_worker_present', '上一轮 IK 尚未退出'),
                  (not self.safety.isChecked(), 'safety_not_confirmed', '未勾选本次安全许可'),
                  (self.arm_sample is None, 'missing_input', '当前三设备输入无效：'+str(self.arm_input_error or '无样本')),
                  (data['signature_matches'] is False, 'signature_changed', '设备分配、归零或穿戴映射变化'),
                  (data['input_remaining_ms'] is not None and data['input_remaining_ms'] < 0,
                   'input_expired', '输入已过期（年龄 %.1f ms，上限 250 ms）' % (data['input_age_ms'] or 0)),
                  (data['orientation_check_error'] is not None, 'invalid_orientation', '参考朝向差计算失败'),
                  (data['orientation_delta_deg'] is not None and max(data['orientation_delta_deg'].values()) > self._transition_tolerance(),
                   'reference_moved', '采集后姿态超差，请重新采集：'+delta_text(data))]
        for rejected, code, message in guards:
            if rejected:
                self._start_rejected(code, message, data, clear_reference=code in
                                     ('missing_input', 'signature_changed', 'input_expired', 'invalid_orientation', 'reference_moved'))
                return
        try:
            limits = self._limits()
            if self._shared_mode():
                scene = getattr(self.viewer, 'scene_robot_dialog', None)
                if scene is not None and scene.worker is not None:
                    raise ValueError('请先断开独立仿真预览，避免两个写入者')
                if self.mirror_check.isChecked() and not self.mirror_ready:
                    raise ValueError('仿真镜像尚未就绪')
                solver = (BilateralIK(self.origin, limits,
                            palms={'left':self.shared_palm.isChecked(), 'right':self.right_options[0].isChecked()},
                            offsets={'left':self._palm_offset(),'right':self._right_offset()})
                          if self._bilateral_mode() else
                          SharedArmIK(self.origin, limits, palm=self.shared_palm.isChecked(), offset=self._palm_offset()))
                initial = solver.solve(self.arm_capture[0])
                control = SharedArmControl(self.origin, limits, initial['target'])
                if self._shared_mode():
                    control.gesture = self.end_session.gesture(limits)
                    if control.gesture is not None:
                        self._arm_event('end_gesture_config', **vars(control.gesture.config))
                reply = QtWidgets.QMessageBox.Yes if automatic else QtWidgets.QMessageBox.warning(self, '确认平滑过渡路径',
                    '从确认的全 90° 上电姿态过渡到勾选轴目标：\n'+
                    ', '.join(f'{name}={q:.2f}°' for name,q in zip(control.names,control.initial_target))+
                    '\n请保持人体不动，确认路径安全。限速不等于防碰撞。',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
                if reply != QtWidgets.QMessageBox.Yes:
                    self._start_rejected('transition_cancelled', '用户取消过渡；未发送', data)
                    return
                # Modal confirmation can take arbitrarily long: take a fresh sample
                # and recheck it before reserving any physical output.
                self._poll()
                if (self.arm_sample is None or self.arm_capture is None
                        or self.arm_sample[2] != self.arm_capture[2]
                        or time.monotonic() > self.arm_sample[1]
                        or not self.safety.isChecked() or not self.connected
                        or (automatic and not self.auto_start.authorized)):
                    raise ValueError('确认期间输入变化，请重新锁定目标')
                self._pose_guard(self.arm_capture[0], self.arm_sample[0], '启动前人体超差')
                self.shared_phase = 'transition'
                self._arm_event('shared_transition', target=initial['target'],
                                permitted_target=control.initial_target.tolist())
            else:
                control = ArmControl(self.origin, limits)
                solver = RelativeArmIK(self.arm_capture[0], self.origin, limits)
            worker = ArmIKWorker(solver, self, recorder=self.worker.recorder)
            worker.result.connect(self._ik_result)
            worker.failed.connect(self._ik_failed)
            worker.finished.connect(self._ik_finished)
            self.arm_configuration = limits
            self.worker.recorder.record('arm_model', provenance=solver.provenance,
                                        mapping='protocol = 90 + sign * degrees(scene_q)', measured=False,
                                        task_mode='simulation_shared' if self._shared_mode() else 'relative',
                                        palm= self.shared_palm.isChecked() if self._shared_mode() else True,
                                        palm_offset=({'left':self._palm_offset(),'right':self._right_offset()}
                                                     if self._bilateral_mode() else self._palm_offset()) if self._shared_mode() else None)
            self.worker.begin_control(control, control.q0.tolist(), self.arm_sample[1])
            self.ik_worker = worker
            self.busy = True
            worker.start()
            worker.submit(self.arm_sample[0], self.arm_sample[1], getattr(self,'input_timing',None))
            self._arm_event('arm_start_accepted', attempt=self.start_attempt,
                            **{**data, 'busy':self.busy, 'ik_worker_present':True})
            self.start_notice.setText('六轴启动已提交；等待串口 engage / write 和 IK 日志。')
            self.status.setText('首帧保持确认起点；仅勾选轴发送。全六轴目标不代表部分输出时的实际构型。')
        except Exception as exc:
            self._start_rejected('initialization_failed', f'启动初始化失败：{exc}', data)
        self._controls()

    def _ik_result(self, result):
        if self.sender() is not self.ik_worker or not self.busy or self.arm_capture is None:
            return
        if time.monotonic() > result['deadline']:
            self._arm_event('arm_ik_gui_discard', reason='result_expired', deadline=result['deadline'],
                            timing=result.get('timing'))
            return
        # Verify current assignment/zero/calibration before publishing a queued result.
        try:
            _, signature, _ = self._read_arm_sample()
            if signature != self.arm_capture[2]:
                raise ValueError('设备分配 / 归零 / 穿戴映射已变化')
        except Exception as exc:
            if isinstance(exc, StaleImuInput) and .25 < exc.age:
                self._arm_event('arm_ik_gui_discard', reason='current_input_stale',
                                deadline=result['deadline'], timing=result.get('timing'))
                return  # Writer owns the bounded gap, never publish stale input.
            self._pause(str(exc))
            return
        timing = dict(result.get('timing') or {})
        now = time.monotonic()
        timing.update(gui_received_at=now,
                      gui_delivery_ms=None if timing.get('solve_finished_at') is None else
                                      (now-timing['solve_finished_at'])*1000)
        result['timing'] = timing
        self.worker.update_arm_target(result['target'], result['deadline'], timing)
        self.worker.recorder.record('arm_ik', **result)
        orientation = result['orientation_error_deg']
        orientation_text = '未启用' if orientation is None else f'{orientation:.1f}°'
        names = (tuple(f'R{i+1}' for i in range(6))+tuple(f'L{i+1}' for i in range(6))
                 if self._bilateral_mode() else tuple(f'L{i+1}' for i in range(6)))
        self.ik_label.setText(('双臂 IK 指令目标：' if self._bilateral_mode() else '全六轴 IK 目标：')+
                             ', '.join(f'{name}={q:.2f}°' for name,q in zip(names,result['target']))+
                             f'\n位置残差 {result["position_error_mm"]:.1f} mm；掌向残差 {orientation_text}；'
                             f'IK {result["ik_ms"]:.1f} ms（残差不代表真机测量）')

    def _ik_failed(self, reason):
        if self.sender() is self.ik_worker:
            self._pause('IK 失败：'+reason)

    def _stop_ik(self):
        if self.ik_worker is not None:
            self.ik_worker.shutdown()

    def _ik_finished(self):
        worker = self.sender()
        if worker is self.ik_worker:
            self.ik_worker = None
        worker.deleteLater()
        self._controls()
        if self.closing and self.worker is None:
            self.close()

    def _pause(self, reason):
        if self.auto_start is not None:
            self.auto_start.cancel(reason)
        self.start_notice.setText('已暂停，不能开始跟随：'+reason)
        self.arm_capture = None
        self._stop_ik()
        super()._pause(reason)

    def _progress(self, state):
        if self.auto_start is not None:
            self.auto_start.progress(state)
        if self.end_session is not None:
            self.end_session.progress(state)
        if state['mode'] == 'paused':
            self.start_notice.setText('已暂停，不能开始跟随：'+state.get('reason', '控制已停止'))
            self.arm_capture = None
            self._stop_ik()
        if state.get('kind') not in ('left_arm','bilateral_arm'):
            return super()._progress(state)
        self.shared_phase = state.get('phase')
        self.shared_control_mode = state['mode']
        if state['mode'] == 'paused':
            self._stop_ik()
            self.busy = False
            self.origin = self.last_frame
            self.arm_capture = None
            self.safety.setChecked(False)
        self.status.setText({'active':'勾选轴随动（指令，不是实际角度）', 'braking':'参与轴统一指令侧减速',
                             'gap':'输入短暂中断：减速等待有效新解（最多额外 250 ms）',
                             'paused':'已暂停；重新确认安全并采集参考'}[state['mode']]+ '\n' +
            '\n'.join(f'{name}: {state["command"][i]:.2f}° → {state["target"][i]:.2f}°；'
                      f'v={state["velocity"][i]:.1f}，a={state["acceleration"][i]:.1f}；'
                      + ('允许输出' if state['enabled'][i] else '冻结') for i,name in enumerate(state.get('names', ('L1','L2','L3','L4','L5','L6')))) + '\n'+state.get('reason',''))
        if self.shared_phase in ('transition', 'await_confirmation') and state['mode'] == 'active':
            self.status.setText(('固定目标平滑过渡中，请保持人体不动' if self.shared_phase == 'transition' else
                                 '过渡指令已完成（不代表真机到位）；请确认真机稳定后开始跟随')+
                                '\n'+self.status.text())
        self._controls()

    def _failed(self, message):
        if self.auto_start is not None: self.auto_start.cancel(message)
        if self.end_session is not None:
            self.end_session.reset()
        self.arm_capture = None
        self._stop_ik()
        super()._failed(message)

    def _disconnect(self):
        if self.auto_start is not None: self.auto_start.cancel('断开串口', stop=True)
        if self.end_session is not None:
            self.end_session.reset()
        self.arm_capture = None
        self._stop_ik()
        super()._disconnect()

    def _finished(self):
        if self.auto_start is not None: self.auto_start.cancel('串口线程退出')
        self._detach_acquisition_trace()
        if self.end_session is not None:
            self.end_session.reset()
        super()._finished()

    def closeEvent(self, event):
        if self.hand_session is not None:
            self.hand_session.dialog.close()
        if self.auto_start is not None: self.auto_start.cancel('窗口关闭', stop=True)
        if self.mirror_worker is not None:
            self.closing = True
            self.mirror_worker.shutdown()
            self._disconnect()
            event.ignore()
            return
        self._stop_ik()
        if self.ik_worker is not None:
            self.closing = True
            self._disconnect()
            event.ignore()
            return
        super().closeEvent(event)
