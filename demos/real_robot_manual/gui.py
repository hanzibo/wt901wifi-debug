"""Explicitly armed manual control; optional single-axis live slider mode."""
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox,
    QSpinBox, QPushButton, QComboBox, QCheckBox, QPlainTextEdit, QMessageBox,
    QScrollArea)
from .protocol import JOINT_NAMES, FINGER_NAMES, ARM_LIMITS, encode, decode
from .transport import SerialWorker
from .trajectory import Trajectory
from .recording import Recorder
from .live import LiveTrajectory
from .pot_panel import PotPanel, same_port


class Window(QMainWindow):
    def __init__(self, worker_factory=SerialWorker):
        super().__init__()
        self.worker_factory, self.worker = worker_factory, None
        self.connected = False
        self.origin = None
        self.executing = False
        self.motion = None
        self.sent_steps = 0
        self.live_active = False
        self.pot_active = False
        self.pot_panel = None
        self.syncing = False
        self.setWindowTitle('五指机器人 · 手动串口控制 Demo（默认干运行）')
        self.resize(1180, 880)
        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        warning = QLabel('默认滑杆只改目标；逐轴实时模式须单独明确启动。每帧仍写全部通道。\n'
            '无位置／电流反馈：无法检测机械极限，禁止顶撞限位来测量。停止发送／关闭串口不是物理急停。')
        warning.setStyleSheet('color:#b43b20;font-weight:bold;')
        warning.setWordWrap(True); layout.addWidget(warning)
        bar = QHBoxLayout(); layout.addLayout(bar)
        self.ports = QComboBox(); self.ports.setEditable(True)
        self.ports.setMinimumWidth(230)
        self.refresh = QPushButton('枚举端口（不打开）')
        self.connect_button = QPushButton('连接串口（不发送）')
        self.disconnect_button = QPushButton('断开／撤销发送')
        for widget in (self.ports, self.refresh, self.connect_button, self.disconnect_button):bar.addWidget(widget)
        bar.addWidget(QLabel('115200 · 8N1'))
        self.confirm = QCheckBox('我已核对所有通道目标、零位／电平／机械范围，并备好物理急停')
        self.arm = QCheckBox('解锁一次平滑执行（默认关闭；不是滑杆实时随动）')
        layout.addWidget(self.confirm); layout.addWidget(self.arm)
        live_bar = QHBoxLayout(); layout.addLayout(live_bar)
        self.live_mode = QCheckBox('逐轴滑杆实时控制（默认关闭）')
        self.live_axis = QComboBox(); self.live_axis.addItems(JOINT_NAMES); self.live_axis.setCurrentIndex(9)
        self.live_lower = QDoubleSpinBox(); self.live_upper = QDoubleSpinBox()
        for spin in (self.live_lower, self.live_upper):
            spin.setRange(0, 327.67); spin.setDecimals(2); spin.setSuffix('°')
        self.live_lower.setValue(85); self.live_upper.setValue(95)
        self.live_button = QPushButton('明确启动逐轴实时控制')
        for widget in (self.live_mode, self.live_axis, QLabel('软件测试区间'),
                       self.live_lower, self.live_upper, self.live_button):live_bar.addWidget(widget)
        self.record_checkbox = QCheckBox('记录串口发送诊断（连接前勾选；不是实际关节反馈）')
        layout.addWidget(self.record_checkbox)
        self.record_label = QLabel('日志：未启用'); self.record_label.setWordWrap(True)
        layout.addWidget(self.record_label)
        motion_bar = QHBoxLayout(); layout.addLayout(motion_bar)
        self.origin_button = QPushButton('确认刚上电默认姿态（不发送）')
        motion_bar.addWidget(self.origin_button)
        self.origin_label = QLabel('起点未确认'); motion_bar.addWidget(self.origin_label)
        rates = QHBoxLayout(); layout.addLayout(rates)
        def rate(label, default, maximum):
            rates.addWidget(QLabel(label))
            spin = QDoubleSpinBox(); spin.setRange(.1, maximum); spin.setValue(default)
            rates.addWidget(spin); spin.valueChanged.connect(self.edited)
            return spin
        self.speed = rate('角速度 °/s', 10, 180)
        self.acceleration = rate('角加速度 °/s²', 20, 720)
        self.finger_speed = rate('手指单位/s', 200, 4095)
        self.finger_acceleration = rate('手指单位/s²', 400, 16380)
        frequency_bar = QHBoxLayout(); layout.addLayout(frequency_bar)
        frequency_bar.addWidget(QLabel('发送频率（Hz）'))
        self.frequency = QDoubleSpinBox(); self.frequency.setRange(1, 100)
        self.frequency.setDecimals(1); self.frequency.setValue(50)
        self.frequency.setMaximumWidth(140)
        self.frequency.valueChanged.connect(self.edited)
        frequency_bar.addWidget(self.frequency)
        frequency_bar.addWidget(QLabel('默认50；可对比25／10。保持轨迹速率，低频单帧增量会更大。'))
        frequency_bar.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); layout.addWidget(scroll, 1)
        content = QWidget(); grid = QGridLayout(content); scroll.setWidget(content)
        self.joints, self.fingers, self.sliders = [], [], []
        for side, start in (('右臂 · 协议目标角（度）', 0), ('左臂 · 协议目标角（度）', 6)):
            box = QGroupBox(side); rows = QGridLayout(box)
            grid.addWidget(box, 0, start // 6)
            for i in range(start, start + 6):
                spin = self.row(rows, i-start, JOINT_NAMES[i], *ARM_LIMITS[i], 90.)
                self.joints.append(spin)
        box = QGroupBox('头／腰 · 协议角（度）· 未知机械限位，0…327.67 仅是编码范围')
        rows = QGridLayout(box); grid.addWidget(box, 1, 0, 1, 2)
        for i, name in enumerate(JOINT_NAMES[12:]):
            self.joints.append(self.row(rows, i, name, 0, 327.67, 90.))
        for side, start in (('右手 · 0…4095（不是角度）', 0), ('左手 · 0…4095（不是角度）', 5)):
            box = QGroupBox(side); rows = QGridLayout(box); grid.addWidget(box, 2, start // 5)
            for i in range(start, start + 5):
                self.fingers.append(self.row(rows, i-start, FINGER_NAMES[i], 0, 4095, 0, integer=True))
        self.frame_view = QPlainTextEdit(); self.frame_view.setReadOnly(True); self.frame_view.setMaximumHeight(105)
        layout.addWidget(self.frame_view)
        buttons = QHBoxLayout(); layout.addLayout(buttons)
        self.preview_button = QPushButton('干运行检查／复制完整帧')
        self.send_button = QPushButton('平滑执行到目标（完整通道）')
        self.revoke = QPushButton('撤销发送权限（不是急停）')
        for widget in (self.preview_button, self.send_button, self.revoke):buttons.addWidget(widget)
        self.status = QLabel('未连接 · 干运行 · 尚未发送任何指令'); layout.addWidget(self.status)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(200); self.log.setMaximumHeight(115); layout.addWidget(self.log)
        self.refresh.clicked.connect(self.enumerate_ports)
        self.origin_button.clicked.connect(self.confirm_origin)
        self.connect_button.clicked.connect(self.connect_serial)
        self.disconnect_button.clicked.connect(self.disconnect_serial)
        self.confirm.toggled.connect(self.permissions)
        self.arm.toggled.connect(self.permissions)
        self.revoke.clicked.connect(lambda:self.arm.setChecked(False))
        self.preview_button.clicked.connect(self.preview)
        self.send_button.clicked.connect(self.send_once)
        self.live_button.clicked.connect(self.start_live)
        self.live_mode.toggled.connect(self.edited)
        self.live_axis.currentIndexChanged.connect(self.live_axis_changed)
        self.live_lower.valueChanged.connect(self.edited)
        self.live_upper.valueChanged.connect(self.edited)
        self.pot_panel = PotPanel(self)
        grid.addWidget(self.pot_panel,3,0,1,2)
        self.update_preview(); self.permissions()

    def row(self, layout, row, name, lower, upper, default, integer=False):
        factor = 1 if integer else 100
        spin = QSpinBox() if integer else QDoubleSpinBox()
        if not integer:spin.setDecimals(2); spin.setSuffix('°')
        spin.setRange(lower, upper); spin.setValue(default)
        slider = QSlider(Qt.Horizontal); slider.setRange(round(lower*factor), round(upper*factor))
        slider.setValue(round(default*factor))
        slider.valueChanged.connect(lambda value:spin.setValue(value/factor if not integer else value))
        spin.valueChanged.connect(lambda value:slider.setValue(round(value*factor)))
        spin.valueChanged.connect(self.edited)
        layout.addWidget(QLabel(name), row, 0); layout.addWidget(slider, row, 1); layout.addWidget(spin, row, 2)
        self.sliders.append(slider)
        return spin

    def frame(self):
        return encode([s.value() for s in self.joints], [s.value() for s in self.fingers])

    def edited(self):
        if self.syncing:
            return
        if self.live_active and self.sender() is self.joints[self.motion.axis]:
            spin = self.joints[self.motion.axis]
            target = min(self.motion.upper, max(self.motion.lower, spin.value()))
            self.syncing = True
            try:spin.setValue(target)
            finally:self.syncing = False
            try:self.worker.retarget_live(target)
            except Exception as exc:self.on_failed(str(exc))
            self.update_preview()
            return
        # Any edit invalidates the previous full-body-target confirmation.
        self.arm.setChecked(False); self.confirm.setChecked(False)
        self.update_preview()
        self.permissions()

    def live_axis_changed(self):
        axis = self.live_axis.currentIndex()
        spin = self.joints[axis]
        q = decode(self.origin)[0][axis] if self.origin else 90.
        self.live_lower.setValue(max(spin.minimum(), q-5))
        self.live_upper.setValue(min(spin.maximum(), q+5))
        self.edited()

    def update_preview(self):
        frame = self.frame()
        values, fingers = decode(frame)
        self.frame_view.setPlainText('仅待发送目标；不是实际角度\n' +
            ' '.join(f'{n}={v:.2f}°' for n,v in zip(JOINT_NAMES,values)) + '\n' +
            ' '.join(f'{n}={v}' for n,v in zip(FINGER_NAMES,fingers)) + '\n54 bytes: ' + frame.hex(' '))

    def preview(self):
        self.update_preview(); QApplication.clipboard().setText(self.frame().hex(' '))
        self.log.appendPlainText('干运行通过：54 字节，校验和正确。已复制 HEX；未发送。')

    def enumerate_ports(self):
        try:
            from serial.tools import list_ports
            current = self.ports.currentText(); self.ports.clear()
            for port in list_ports.comports():
                self.ports.addItem(port.device)
                self.log.appendPlainText(f'{port.device} · {port.description} · {port.hwid}')
            if current:self.ports.setEditText(current)
        except Exception as exc:self.log.appendPlainText(str(exc))

    def permissions(self):
        self.arm.setEnabled(self.connected and self.confirm.isChecked())
        enabled = self.connected and self.confirm.isChecked() and self.arm.isChecked()
        if self.executing and not enabled:
            self.executing = False
            self.live_active = False
            self.pot_active = False
            self.origin = None
            self.origin_label.setText('执行被中断；起点不确定，需重新确认')
        ready = enabled and self.origin is not None and not self.executing
        self.send_button.setEnabled(ready and not self.live_mode.isChecked())
        self.live_button.setEnabled(ready and self.live_mode.isChecked())
        self.arm.setText('解锁逐轴实时控制（仍须点击明确启动）' if self.live_mode.isChecked()
                         else '解锁一次平滑执行（默认关闭；不是滑杆实时随动）')
        for widget in (self.live_mode, self.live_axis, self.live_lower, self.live_upper):
            widget.setEnabled(not self.executing)
        for widget in (self.speed, self.acceleration, self.frequency, self.finger_speed, self.finger_acceleration):
            widget.setEnabled(not (self.live_active or self.pot_active))
        for i, spin in enumerate(self.joints+self.fingers):
            allowed = not self.pot_active and (not self.live_active or i == self.motion.axis)
            spin.setEnabled(allowed); self.sliders[i].setEnabled(allowed)
        self.origin_button.setEnabled(self.connected and not self.executing)
        self.connect_button.setEnabled(self.worker is None)
        self.disconnect_button.setEnabled(self.worker is not None)
        self.ports.setEnabled(self.worker is None); self.refresh.setEnabled(self.worker is None)
        self.record_checkbox.setEnabled(self.worker is None)
        if self.worker:self.worker.enable(enabled)
        if self.pot_panel:self.pot_panel.update_controls()

    def start_live(self):
        if not self.live_button.isEnabled() or not self.worker:return
        axis = self.live_axis.currentIndex()
        spin = self.joints[axis]
        lower, upper = self.live_lower.value(), self.live_upper.value()
        try:
            if lower < spin.minimum() or upper > spin.maximum():
                raise ValueError('测试区间不能超出该轴原软件范围（原范围也不是机械限位认证）')
            motion = LiveTrajectory(self.origin, axis, lower, upper, self.speed.value(),
                                    self.acceleration.value(), self.frequency.value())
        except Exception as exc:
            self.status.setText(str(exc)); return
        answer = QMessageBox.warning(self, '逐轴实时控制确认',
            f'仅 {JOINT_NAMES[axis]} 滑杆可动，软件区间 {lower:g}–{upper:g}°。\n'
            f'{motion.frequency_hz:g}Hz，速度 {self.speed.value():g}°/s，加速度 {self.acceleration.value():g}°/s²。\n'
            '所有其他角度／手指将保持最后确认的指令起点，不采用未发送的界面目标。\n'
            '启动会将滑杆同步到起点；随后拖动或修改该轴数字框即可更新目标。\n'
            '禁止靠顶撞来寻找机械限位。撤销只停止后续发送，不保证机器人立即停止。是否启动？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:return
        self.syncing = True
        try:
            angles, fingers = decode(self.origin)
            for widget, value in zip(self.joints+self.fingers, angles+fingers):widget.setValue(value)
        finally:self.syncing = False
        self.update_preview()
        self.motion = motion; self.sent_steps = 0
        self.live_active = self.executing = True
        self.permissions()
        try:
            self.worker.submit_motion(motion)
            self.status.setText(f'{JOINT_NAMES[axis]} 实时控制已启动；等待滑杆目标，无位置反馈')
        except Exception as exc:self.on_failed(str(exc))

    def confirm_origin(self):
        if not self.connected or self.executing:return
        answer = QMessageBox.warning(self, '确认物理起点（不发送）',
            '仅当机器人刚上电且静止于已验证默认构型时确认：\n'
            '十五个协议角均为90°，十个手指量均为0。\n'
            '打开demo或重新连接不代表机器人已复位。是否确认？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:return
        self.origin = encode([90.]*15, [0]*10)
        if self.worker:self.worker.record('origin_confirmed', source='user_confirmed_power_on_default',
                                          angles_deg=[90.]*15, fingers=[0]*10)
        self.origin_label.setText('起点：人工确认上电默认姿态（无反馈）')
        self.permissions()

    def connect_serial(self):
        if self.worker:return
        port = self.ports.currentText().strip()
        if not port.startswith('/dev/') or '..' in port:
            QMessageBox.warning(self,'端口错误','请输入真实 /dev/ttyUSB… 或 /dev/serial/by-id/… 端口。'); return
        from serial.tools import list_ports
        if ((self.pot_panel.connection and same_port(port,self.pot_panel.port.currentText())) or
                any(same_port(port,p.device) and p.vid==0x1a86 for p in list_ports.comports())):
            QMessageBox.warning(self,'端口错误','这是 Nano 输入串口，请选择机器人 CP2102 串口。');return
        answer = QMessageBox.warning(self,'连接确认',
            '打开串口可能改变引脚电平，不能视为电气上无影响。\n'
            '请先确认 TTL 电平兼容和物理安全条件。连接本身不发送控制帧。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:return
        self.arm.setChecked(False); self.confirm.setChecked(False)
        self.worker = self.worker_factory(port, self)
        if self.record_checkbox.isChecked():
            path = (Path(__file__).resolve().parents[2] / 'OptUSBProj-ubuntu' / 'recordings' /
                    'real_robot_manual' / (datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6]) / 'commands.jsonl')
            try:
                self.worker.recorder = Recorder(path)
                self.worker.record('session_start', schema_version=1, port=port,
                    joint_names=list(JOINT_NAMES), finger_names=list(FINGER_NAMES),
                    warning='Host command/write timings only; no physical position feedback')
                self.record_label.setText('日志：'+str(path))
                self.log.appendPlainText('诊断记录：'+str(path))
            except Exception as exc:
                self.worker.deleteLater(); self.worker = None
                self.record_label.setText('日志创建失败：'+str(exc))
                self.status.setText('未连接：请处理日志错误或取消记录后重试')
                self.permissions(); return
        self.worker.ready.connect(self.on_ready)
        self.worker.sent.connect(self.on_sent)
        self.worker.motion_done.connect(self.on_motion_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.recording_failed.connect(self.on_recording_failed)
        self.worker.received.connect(lambda data:self.log.appendPlainText('RX原始字节（不是位置反馈）：'+data.hex(' ')))
        self.worker.finished.connect(self.on_finished)
        self.status.setText('正在打开串口 · 禁止发送'); self.permissions(); self.worker.start()

    def on_ready(self):
        if self.worker and not self.worker.stopping:
            self.connected = True
            self.status.setText('串口已打开 · 未解锁 · RXD 未接时无机器人反馈')
            self.permissions()

    def on_recording_failed(self, message):
        self.record_label.setText('⚠ '+message)
        self.log.appendPlainText('记录异常（不改变发送权限）：'+message)

    def send_once(self):
        if not self.send_button.isEnabled() or not self.worker:return
        self.update_preview()
        motion = Trajectory(self.origin, self.frame(), self.speed.value(),
            self.acceleration.value(), self.finger_speed.value(), self.finger_acceleration.value(),
            self.frequency.value())
        answer = QMessageBox.warning(self,'全身平滑执行确认',
            '这不是单轴命令，而是所有 25 个通道的完整目标。\n'
            f'预计至少 {motion.duration:.2f} 秒；目标频率{motion.frequency_hz:g}Hz，共{motion.steps}帧。\n'
            '通信延迟只会延长执行；低频单帧变化更大，并非一定更安全。\n'
            '起点来自人工确认或上次写出目标，不是位置反馈。\n'
            '请核对全部目标及物理安全条件。是否开始？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:return
        try:
            self.motion = motion; self.sent_steps = 0; self.executing = True
            self.send_button.setEnabled(False); self.origin_button.setEnabled(False)
            self.worker.submit_motion(motion)
            self.status.setText('平滑执行已排队 · 不代表实际到位')
        except Exception as exc:self.on_failed(str(exc))

    def on_sent(self, frame):
        if self.executing:
            self.origin = frame
            self.sent_steps += 1
            if self.pot_active:
                values=decode(frame)[1]
                self.status.setText('五指随动 · 最后写出 '+str(values)+'（指令量，不是反馈）')
                self.pot_panel.output.setText('最后写出十指量：'+str(values)+'；其余十五个角冻结，无位置反馈。')
                return
            if self.live_active:
                q = decode(frame)[0][self.motion.axis]
                self.status.setText(f'{JOINT_NAMES[self.motion.axis]} 实时控制 · 最后写出 {q:.2f}°'
                                    '（不是实际角度）· 撤销权限停止后续发送')
                return
            self.status.setText(f'平滑执行 {100*self.sent_steps/self.motion.steps:.1f}% · 仅写出目标，无位置反馈')
            return
        self.log.appendPlainText('TX 已写入操作系统（未确认到位）：'+frame.hex(' '))
        self.status.setText('54 字节已写入 · 无到位证据 · 再发送需重新解锁')
        self.arm.setChecked(False)

    def on_motion_done(self):
        if not self.executing:return
        self.executing = False
        was_pot=self.pot_active
        self.pot_active = False
        self.origin_label.setText('起点：最后写出目标估计（请确认已到位）')
        self.arm.setChecked(False)
        self.status.setText('轨迹全部写出 · 未确认实际到位 · 再执行需重新解锁')
        if was_pot:self.status.setText(self.motion.reason+'；保持最后指令，重新启动需解锁')
        self.log.appendPlainText('平滑轨迹完成；末帧：'+self.origin.hex(' '))
        self.permissions()

    def on_failed(self, message):
        self.origin = None
        self.origin_label.setText('通信异常；起点不确定')
        self.connected = False; self.arm.setChecked(False)
        self.status.setText('通信错误：'+message); self.log.appendPlainText(message); self.permissions()

    def disconnect_serial(self):
        if self.worker:self.worker.record('disconnect_requested', executing=self.executing)
        self.origin = None
        self.origin_label.setText('断连后需重新确认起点')
        self.connected = False; self.arm.setChecked(False); self.confirm.setChecked(False)
        if self.worker:self.worker.stop()
        self.status.setText('正在关闭串口；未发送归零／停止命令，不等于电机急停')
        self.permissions()

    def on_finished(self):
        self.origin = None
        old = self.worker; self.worker = None; self.connected = False
        self.arm.setChecked(False); self.confirm.setChecked(False)
        if old:old.deleteLater()
        self.permissions()

    def closeEvent(self, event):
        self.pot_panel.timer.stop();self.pot_panel.close_input()
        if self.worker and self.worker.isRunning():
            self.disconnect_serial(); event.ignore(); QTimer.singleShot(100,self.close)
        else:event.accept()


def main():
    app = QApplication(sys.argv); window = Window(); window.show()
    return app.exec()
