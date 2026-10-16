"""Restricted physical-robot experiment, separate from all preview pipelines."""
from collections import deque
from datetime import datetime
from pathlib import Path
import time
import uuid

from PySide6 import QtCore, QtWidgets
from serial.tools.list_ports import comports
from .real_robot_control import (Limits, elbow_angle, MAX_COMMAND_SPEED,
                                 MAX_COMMAND_ACCELERATION, L4_MIN_ANGLE, L4_MAX_ANGLE)
from .real_robot_protocol import encode, decode
from .real_robot_recording import Recorder
from .real_robot_transport import ElbowSerialWorker
from .scene_robot_state import pose_sample, StaleImuInput


class RealRobotDialog(QtWidgets.QDialog):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.worker = None
        self.connected = False
        self.origin = None
        self.capture = None
        self.sample = None
        self.signature = None
        self.history = deque()
        self.busy = False
        self.closing = False
        self.last_frame = None
        self.setWindowTitle('真机受限 IMU 控制 · 仅左肘 L4 · 无反馈')
        self.resize(720, 650)
        layout = QtWidgets.QVBoxLayout(self)
        warning = QtWidgets.QLabel('真实硬件！仅 L4 可动，其他 14 个角度保持 90°，10 个手指量保持 0。\n'
                                  '电脑只有发送接口：显示的是指令，不是实际角度。串口暂停不是急停。\n'
                                  '机器人周围必须留空，并准备独立物理断电手段。勿同时运行手动发送 demo。')
        warning.setWordWrap(True)
        layout.addWidget(warning)
        row = QtWidgets.QHBoxLayout()
        self.port = QtWidgets.QComboBox()
        self.port.setEditable(True)
        self.port.setCurrentIndex(-1)
        self.port.setEditText('')
        row.addWidget(self.port, 1)
        self.refresh_button = QtWidgets.QPushButton('刷新串口')
        self.refresh_button.setToolTip('仅重新扫描端口；不会打开串口或发送指令。连接期间不可刷新。')
        self.refresh_button.clicked.connect(lambda: self._refresh_ports())
        row.addWidget(self.refresh_button)
        self.connect_button = QtWidgets.QPushButton('连接串口（不发送）')
        self.connect_button.clicked.connect(self._connect)
        row.addWidget(self.connect_button)
        self.disconnect_button = QtWidgets.QPushButton('制动并断开')
        self.disconnect_button.clicked.connect(self._disconnect)
        row.addWidget(self.disconnect_button)
        layout.addLayout(row)
        self.origin_button = QtWidgets.QPushButton('确认：刚重新上电，全部角度 90° / 手指 0')
        self.origin_button.clicked.connect(self._confirm_origin)
        layout.addWidget(self.origin_button)
        self.manual = QtWidgets.QCheckBox('使用手动轴映射（仍须设备在线、自然下垂归零）')
        self.manual.toggled.connect(self._invalidate_capture)
        layout.addWidget(self.manual)
        form = QtWidgets.QFormLayout()
        self.frequency = self._spin(1, 100, 20, ' Hz')
        self.speed = self._spin(.1, MAX_COMMAND_SPEED, 10, ' °/s')
        self.acceleration = self._spin(.1, MAX_COMMAND_ACCELERATION, 20, ' °/s²')
        self.gain = self._spin(.1, 3, 1, '')
        self.sign = QtWidgets.QComboBox()
        self.sign.addItems(['同向 +1（伸肘降低目标角）', '反向 −1'])
        self.sign.currentIndexChanged.connect(self._invalidate_capture)
        for label, widget in [('发送频率', self.frequency), ('角速度上限', self.speed),
                              ('角加速度上限', self.acceleration), ('增益', self.gain),
                              ('方向', self.sign)]:
            form.addRow(label, widget)
        layout.addLayout(form)
        self.range_text = f'{L4_MIN_ANGLE:g}–{L4_MAX_ANGLE:g}°'
        range_label = QtWidgets.QLabel(f'L4 指令范围 {self.range_text}；首帧保持起点。\n'
                                      '250 仅为可设置上限，不是已验证的真机安全值；请逐步调高。')
        range_label.setWordWrap(True)
        self.range_label = range_label
        layout.addWidget(range_label)
        self.safety = QtWidgets.QCheckBox('我已确认安全空间和物理断电手段，允许本次仅 L4 控制')
        self.safety.toggled.connect(self._safety_changed)
        layout.addWidget(self.safety)
        self.capture_button = QtWidgets.QPushButton('采集当前人体屈肘参考（稳定保持 ≥0.5 秒）')
        self.capture_button.clicked.connect(self._capture)
        layout.addWidget(self.capture_button)
        self.start_button = QtWidgets.QPushButton('明确启动：仅左肘 L4')
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton('减速暂停（保持最后指令 / 不自动恢复）')
        self.stop_button.clicked.connect(lambda: self._pause('用户减速暂停'))
        layout.addWidget(self.stop_button)
        self.input_label = QtWidgets.QLabel('等待左上臂和左前臂姿态')
        self.input_label.setWordWrap(True)
        layout.addWidget(self.input_label)
        self.status = QtWidgets.QLabel('未连接；没有发送任何指令')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.log_label = QtWidgets.QLabel('每次连接自动记录完整指令和控制状态')
        self.log_label.setWordWrap(True)
        layout.addWidget(self.log_label)
        self._refresh_ports(report=False)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._controls()

    def _spin(self, lower, upper, default, suffix):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(lower, upper)
        widget.setValue(default)
        widget.setSuffix(suffix)
        widget.valueChanged.connect(self._invalidate_capture)
        return widget

    def _refresh_ports(self, report=True):
        if self.worker is not None:
            return
        current = self.port.currentText()
        try:
            ports = sorted(comports(), key=lambda port: port.device)
            with QtCore.QSignalBlocker(self.port):
                self.port.clear()
                for port in ports:
                    self.port.addItem(port.device)
                    self.port.setItemData(self.port.count()-1, port.description, QtCore.Qt.ToolTipRole)
                # Preserve explicit selection/manual paths, but never select an
                # arbitrary newly discovered UART as the robot automatically.
                self.port.setCurrentIndex(self.port.findText(current) if current else -1)
                self.port.setEditText(current)
            if report:
                self.status.setText(f'串口列表已刷新：发现 {len(ports)} 个端口；未连接、未发送。'
                                    '请选择机器人串口后手动连接。')
        except Exception as exc:
            self.status.setText(f'串口扫描失败：{exc}；保留原列表，未连接、未发送。')

    def _invalidate_capture(self, *_):
        self.capture = None
        if self.busy:
            self._pause('配置变化；需重新采集参考')

    def _safety_changed(self, checked):
        if not checked:
            self._invalidate_capture()

    def _motion_controls_ready(self, connected):
        allowed = connected and not self.busy and self.safety.isChecked()
        return allowed and self.origin is not None, allowed and self.capture is not None

    def _controls(self):
        connected = self.connected and self.worker is not None and self.worker.isRunning()
        for widget in (self.frequency, self.speed, self.acceleration, self.gain, self.sign, self.manual):
            widget.setEnabled(not self.busy)
        self.port.setEnabled(self.worker is None)
        self.refresh_button.setEnabled(self.worker is None)
        self.connect_button.setEnabled(self.worker is None and not self.viewer.replay_mode)
        self.disconnect_button.setEnabled(self.worker is not None)
        self.origin_button.setEnabled(connected and not self.busy and self.origin is None)
        # Apply the active mode's final state once: disabling a pressed Qt button
        # clears its down state and prevents the subsequent release from clicking.
        capture_ready, start_ready = self._motion_controls_ready(connected)
        self.capture_button.setEnabled(capture_ready)
        self.start_button.setEnabled(start_ready)
        self.stop_button.setEnabled(self.busy)

    def _connect(self):
        if self.viewer.replay_mode or self.worker is not None:
            return
        port = self.port.currentText().strip()
        if not port:
            self.status.setText('请选择串口；连接时不会发送角度帧')
            return
        try:
            path = Path(__file__).resolve().parents[1] / 'recordings' / 'real_robot_imu' / (
                datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]) / 'control.jsonl'
            recorder = Recorder(path)
            self.worker = ElbowSerialWorker(port, recorder, self)
            self.worker.ready.connect(self._ready)
            self.worker.failed.connect(self._failed)
            self.worker.progress.connect(self._progress)
            self.worker.sent.connect(self._sent)
            self.worker.finished.connect(self._finished)
            self.worker.start()
            self.log_label.setText(f'日志：{path}')
            self.status.setText('正在连接（不发送）；DTR/RTS 设置为关闭，但电气瞬态仍不可排除')
        except Exception as exc:
            self.status.setText(str(exc))
        self._controls()

    def _ready(self):
        self.connected = True
        self.connect_button.setText('已连接')
        self.status.setText('串口已打开，尚未发送；请确认机器人确实刚上电')
        self._controls()

    def _confirm_origin(self):
        if self.busy or self.worker is None:
            return
        reply = QtWidgets.QMessageBox.warning(self, '不是实际位置反馈',
            '请确认已重新上电，且机器人仍处于已验证的全部 90° / 手指 0 起始姿态。\n'
            '若此前已有运动、断电、写入异常或姿态不确定，请勿确认。确认本身不会发送。',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.origin = encode([90.] * 15, [0] * 10)
        self.last_frame = self.origin
        self.capture = None
        self.worker.recorder.record('origin_confirmed', baseline=self.origin.hex(), measured=False)
        self.status.setText('已人工确认上电指令起点；等待人体稳定并采集相对参考')
        self._controls()

    def _poll(self):
        try:
            sample, signature = pose_sample(self.viewer.engine, ['left'], [], self.manual.isChecked())
            theta = elbow_angle(sample['sides']['left']['directions'])
            deadline = sample['at'] + .25 - max(sample['device_ages_ms'].values()) / 1000
            if self.signature != signature:
                self.history.clear()
                if self.capture is not None:
                    self._invalidate_capture()
            self.signature = signature
            self.sample = (theta, deadline, signature)
            now = time.monotonic()
            self.history.append((now, theta))
            while self.history and now - self.history[0][0] > .7:
                self.history.popleft()
            self.input_label.setText(f'左肘人体角 θ = {theta:.2f}°；最大设备年龄 {max(sample["device_ages_ms"].values()):.0f} ms')
            if self.busy and self.capture is not None and signature == self.capture[2]:
                self.worker.update_target(theta, deadline)
        except Exception as exc:
            self.sample = None
            self.history.clear()
            self.input_label.setText(f'输入无效：{exc}')
            if self.busy and not (isinstance(exc, StaleImuInput) and .25 < exc.age):
                self._pause(str(exc))
        self._controls()

    def _capture(self):
        if self.sample is None or self.origin is None or self.busy or not self.safety.isChecked():
            return
        if (len(self.history) < 5 or self.history[-1][0] - self.history[0][0] < .5 or
                max(t for _, t in self.history) - min(t for _, t in self.history) > 2.):
            self.status.setText('请稳定保持当前左肘姿态 ≥0.5 秒，波动不超过 2°')
            return
        self.capture = self.sample
        self.status.setText(f'参考 θ₀ = {self.capture[0]:.2f}°；q₀ = {decode(self.origin)[0][9]:.2f}°。启动首帧保持 q₀。')
        self.worker.recorder.record('reference', theta0=self.capture[0], baseline=self.origin.hex(),
                                    signature=repr(self.capture[2]))
        self._controls()

    def _start(self):
        if self.busy or self.capture is None or self.worker is None or not self.safety.isChecked():
            return
        if self.sample is None or self.sample[2] != self.capture[2] or time.monotonic() > self.sample[1]:
            self.status.setText('启动失败：输入过期或配置变化，重新采集参考')
            self.capture = None
            return
        if abs(self.sample[0] - self.capture[0]) > 2.:
            self.capture = None
            self.status.setText('采集后人体已移动超过 2°；请重新采集，避免启动突变')
            return
        try:
            limits = Limits(self.frequency.value(), self.speed.value(), self.acceleration.value(),
                            self.gain.value(), 1 if self.sign.currentIndex() == 0 else -1)
            self.worker.begin(self.origin, self.capture[0], limits, self.sample[1])
            self.busy = True
            self.status.setText('已明确启动，仅 L4 输出；其他通道冻结')
        except Exception as exc:
            self.status.setText(f'无法启动：{exc}')
        self._controls()

    def _pause(self, reason):
        if self.worker is not None and self.busy:
            self.worker.pause(reason)
            self.status.setText(f'正在指令侧减速：{reason}；这不是物理急停')

    def _sent(self, frame):
        self.last_frame = frame

    def _progress(self, state):
        mode = state['mode']
        if mode == 'paused':
            self.busy = False
            self.origin = self.last_frame
            self.capture = None
            self.safety.setChecked(False)
        details = '' if 'command' not in state else (
            f'\nL4 指令 {state["command"]:.2f}° → 目标 {state["target"]:.2f}°；'
            f'指令速度 {state["velocity"]:.2f}°/s；加速度 {state["acceleration"]:.2f}°/s²'
            f'\nθ₀ {state["theta0"]:.2f}°，θ {state["theta"]:.2f}°，q₀ {state["q0"]:.2f}°'
            + (f'；已达到 {self.range_text} 范围边界' if state['saturated'] else ''))
        self.status.setText({'active': '仅 L4 随动（指令值，不是测量值）',
                             'gap': '输入短暂中断：减速等待有效新目标（最多额外 250 ms）',
                             'braking': '指令侧正在制动', 'paused': '已暂停；须重新确认安全并采集参考'}[mode]
                            + details + '\n' + state.get('reason', ''))
        self._controls()

    def _failed(self, message):
        self.busy = False
        self.origin = self.capture = self.last_frame = None
        self.safety.setChecked(False)
        self.status.setText(f'通信 / 控制异常：{message}\n起点不可信；不重试，不自动恢复。请物理确认并重新上电。')

    def _disconnect(self):
        if self.worker is not None:
            self.worker.shutdown()
            self.status.setText('正在制动并断开；通信故障时无法保证制动送达，必要时物理断电')

    def _finished(self):
        worker = self.worker
        self.worker = None
        self.connected = False
        if worker is not None:
            worker.deleteLater()
        self.busy = False
        self.origin = self.capture = self.last_frame = None
        self.connect_button.setText('连接串口（不发送）')
        self.safety.setChecked(False)
        self.status.setText(self.status.text() + '\n串口已关闭；再次连接必须重新上电确认。')
        self._controls()
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.closing = True
            self._disconnect()
            event.ignore()
            return
        self.closing = False
        event.accept()

    def reject(self):
        # QDialog's Escape key otherwise bypasses closeEvent and hides a sender.
        self.close()

    def accept(self):
        self.close()
