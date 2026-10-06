"""Per-device WT901 settings dialog, sharing the viewer's UDP socket."""

from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from wt901.registers import (
    REG_ACCRANGE, REG_AXIS6, REG_BANDWIDTH, REG_CALSW, REG_GYRORANGE,
    REG_ORIENT, REG_RATE, REG_SAVE, UNLOCK_SETTLE,
)


@dataclass(frozen=True)
class Setting:
    address: int
    value: int
    label: str


class DeviceSettingsDialog(QtWidgets.QDialog):
    """One selected online device at a time; all replies come through the main poll loop."""

    neutral_required = QtCore.Signal(str)
    READ_ADDRESSES = (REG_CALSW, REG_BANDWIDTH, REG_ORIENT, REG_RATE)
    MAG_MIN_SECONDS = 20.0
    MAG_MAX_SECONDS = 90.0

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.setWindowTitle("WT901 · 设备参数与校准")
        self.resize(620, 620)
        self.device_id = ""
        self.registers: dict[int, int] = {}
        self.pending_reads: list[int] = []
        self.waiting_address: int | None = None
        self.read_deadline = 0.0
        self.setting: Setting | None = None
        self.calibration: str | None = None
        self.calibration_started = 0.0
        self.mag_min = [float("inf")] * 3
        self.mag_max = [float("-inf")] * 3
        self._last_packet = None
        self._last_live_at = 0.0

        root = QtWidgets.QVBoxLayout(self)
        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel("目标设备"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(230)
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        picker.addWidget(self.device_combo, 1)
        refresh = QtWidgets.QPushButton("刷新在线列表")
        refresh.clicked.connect(lambda: self.refresh_devices())
        picker.addWidget(refresh)
        root.addLayout(picker)
        self.endpoint = QtWidgets.QLabel("请先连接设备到电脑热点")
        self.endpoint.setWordWrap(True)
        root.addWidget(self.endpoint)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        live = QtWidgets.QWidget()
        live_layout = QtWidgets.QVBoxLayout(live)
        self.live_text = QtWidgets.QLabel("等待实时数据")
        self.live_text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.live_text.setWordWrap(True)
        live_layout.addWidget(self.live_text)
        live_layout.addStretch()
        self.tabs.addTab(live, "实时状态")

        settings = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings)
        read = QtWidgets.QPushButton("读取当前设备参数")
        read.clicked.connect(self.read_parameters)
        settings_layout.addWidget(read)
        self.parameters = QtWidgets.QLabel("尚未读取；设置前请先读取设备参数。")
        self.parameters.setWordWrap(True)
        self.parameters.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        settings_layout.addWidget(self.parameters)
        form = QtWidgets.QFormLayout()
        self.algorithm = self._choice(form, "融合算法", (("9 轴 · 磁场辅助航向", 0),
                                                      ("6 轴 · 相对航向", 1)), REG_AXIS6)
        self.orientation = self._choice(form, "安装方向", (("水平安装", 0),
                                                        ("垂直安装 · Y 轴向上", 1)), REG_ORIENT)
        self.bandwidth = self._choice(form, "内部带宽", tuple((f"{hz} Hz", i) for i, hz in
                                                        enumerate((256, 188, 98, 42, 20, 10, 5))), REG_BANDWIDTH)
        self.rate = QtWidgets.QDoubleSpinBox()
        self.rate.setRange(0.2, 200.0)
        self.rate.setDecimals(1)
        self.rate.setSuffix(" Hz")
        self.rate_button = QtWidgets.QPushButton("设置")
        self.rate_button.clicked.connect(self._set_rate)
        rate_row = QtWidgets.QHBoxLayout()
        rate_row.addWidget(self.rate, 1)
        rate_row.addWidget(self.rate_button)
        form.addRow("回传频率", rate_row)
        self.gyro = self._choice(form, "陀螺量程", tuple((f"±{v} °/s", i) for i, v in
                                                       enumerate((250, 500, 1000, 2000))), REG_GYRORANGE)
        self.acc = self._choice(form, "加计量程", tuple((f"±{v} g", i) for i, v in
                                                      enumerate((2, 4, 8, 16))), REG_ACCRANGE)
        settings_layout.addLayout(form)
        note = QtWidgets.QLabel("带宽影响滤波响应，并非 Wi-Fi 带宽；回传频率决定发包间隔。"
                                "设置将解锁、写入、保存并回读验证；安装方向改变后需重新校准。")
        note.setWordWrap(True)
        settings_layout.addWidget(note)
        settings_layout.addStretch()
        self.tabs.addTab(settings, "参数设置")

        calibration = QtWidgets.QWidget()
        cal_layout = QtWidgets.QVBoxLayout(calibration)
        cal_note = QtWidgets.QLabel("磁场校准：举离磁源，先绕三轴充分翻转完成预检；正式校准后继续翻转，"
                                    "最后手动点击结束。加计校准：设备正面朝上、水平静止。"
                                    "设备自身归零与模型中立位归零是不同操作。")
        cal_note.setWordWrap(True)
        cal_layout.addWidget(cal_note)
        self.mag_start = QtWidgets.QPushButton("开始磁场预检 / 校准")
        self.mag_start.clicked.connect(self._start_mag)
        cal_layout.addWidget(self.mag_start)
        self.mag_stop = QtWidgets.QPushButton("结束当前校准 / 操作并保存")
        self.mag_stop.clicked.connect(self._stop_calibration)
        cal_layout.addWidget(self.mag_stop)
        self.acc_button = QtWidgets.QPushButton("水平静止：加速度计校准")
        self.acc_button.clicked.connect(self._start_acc)
        cal_layout.addWidget(self.acc_button)
        self.heading_button = QtWidgets.QPushButton("Z 轴航向归零（仅 6 轴）")
        self.heading_button.clicked.connect(lambda: self._reference(0x04))
        cal_layout.addWidget(self.heading_button)
        self.angle_button = QtWidgets.QPushButton("X/Y 角度参考归零")
        self.angle_button.clicked.connect(lambda: self._reference(0x08))
        cal_layout.addWidget(self.angle_button)
        cal_layout.addStretch()
        self.tabs.addTab(calibration, "校准与归零")

        self.status = QtWidgets.QLabel("请选择在线设备")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.close)
        root.addWidget(close_button)
        self._apply_buttons = [self.algorithm_button, self.orientation_button,
                               self.bandwidth_button, self.rate_button,
                               self.gyro_button, self.acc_button_setting]
        self._update_controls()
        self.refresh_devices()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _choice(self, form, label, options, address):
        combo = QtWidgets.QComboBox()
        for title, value in options:
            combo.addItem(title, value)
        button = QtWidgets.QPushButton("设置")
        button.clicked.connect(lambda: self._apply(Setting(address, combo.currentData(), label)))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(combo, 1)
        row.addWidget(button)
        form.addRow(label, row)
        setattr(self, {REG_AXIS6: "algorithm_button", REG_ORIENT: "orientation_button",
                       REG_BANDWIDTH: "bandwidth_button", REG_GYRORANGE: "gyro_button",
                       REG_ACCRANGE: "acc_button_setting"}[address], button)
        return combo

    def refresh_devices(self, preferred: str = "") -> None:
        if self._busy():
            return
        seen = self.source.seen_devices(max_age=3.0)
        selected = preferred or self.device_id
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("— 选择在线设备 —", "")
        for device_id in sorted(seen):
            self.device_combo.addItem(device_id, device_id)
        index = self.device_combo.findData(selected)
        self.device_combo.setCurrentIndex(index if index >= 0 else 0)
        self.device_combo.blockSignals(False)
        self._device_changed()

    def _device_changed(self, _index: int = 0) -> None:
        selected = self.device_combo.currentData() or ""
        if selected == self.device_id:
            return
        if self._busy():
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentIndex(self.device_combo.findData(self.device_id))
            self.device_combo.blockSignals(False)
            return
        self.device_id = selected
        self.registers.clear()
        self.pending_reads.clear()
        self.waiting_address = None
        self._last_packet = None
        self._last_live_at = 0.0
        self.parameters.setText("尚未读取；设置前请先读取设备参数。")
        self.live_text.setText("等待实时数据" if selected else "请选择在线设备")
        self.status.setText("已选择设备；正在读取参数…" if selected else "请选择在线设备")
        self._update_controls()
        if selected:
            self.read_parameters()

    def _busy(self) -> bool:
        return bool(self.pending_reads or self.waiting_address is not None or
                    self.setting is not None or self.calibration is not None)

    def _update_controls(self) -> None:
        available = bool(self.device_id)
        busy = self._busy()
        for button, addr in zip(self._apply_buttons,
                                (REG_AXIS6, REG_ORIENT, REG_BANDWIDTH, REG_RATE,
                                 REG_GYRORANGE, REG_ACCRANGE)):
            button.setEnabled(available and not busy and addr in self.registers)
        self.mag_start.setEnabled(available and not busy)
        self.mag_stop.setEnabled(self.calibration in ("mag", "acc", "reference"))
        self.acc_button.setEnabled(available and not busy and self.registers.get(REG_ORIENT) == 0)
        self.heading_button.setEnabled(available and not busy and self.registers.get(REG_AXIS6) == 1)
        self.angle_button.setEnabled(available and not busy)
        self.device_combo.setEnabled(not busy)
        if self.device_id:
            try:
                ip = self.source.device_ip(self.device_id)
                self.endpoint.setText(f"{self.device_id} · 在线 · {ip}:9250（配置端口）")
            except RuntimeError:
                self.endpoint.setText(f"{self.device_id} · 当前离线；禁止新配置操作")
        else:
            self.endpoint.setText("请先连接设备到电脑热点")

    def read_parameters(self) -> None:
        if self._busy() or not self.device_id:
            return
        try:
            self.source.device_ip(self.device_id)
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        self.registers.clear()
        self.pending_reads = list(self.READ_ADDRESSES)
        self.status.setText("正在读取设备参数…")
        self._read_next()

    def _read_next(self) -> None:
        if not self.pending_reads:
            self._show_parameters()
            self.status.setText("设备参数读取完成")
            self._update_controls()
            return
        address = self.pending_reads.pop(0)
        try:
            self.source.read_registers(self.device_id, address)
        except (RuntimeError, OSError) as exc:
            self.pending_reads.clear()
            self.status.setText(f"读取失败：{exc}")
            self._update_controls()
            return
        self.waiting_address = address
        self.read_deadline = time.monotonic() + 2.0
        self._update_controls()

    def on_replies(self, replies) -> None:
        for device_id, values in replies:
            if device_id != self.device_id or self.waiting_address is None:
                continue
            address = self.waiting_address
            self.waiting_address = None
            for offset, value in enumerate(values):
                self.registers[address + offset] = value
            if self.setting is not None:
                setting = self.setting
                self.setting = None
                if values[0] == setting.value:
                    self.status.setText(f"{setting.label}已保存并回读确认"
                                        + ("；请重新校准安装方向" if address == REG_ORIENT else ""))
                    if address in (REG_ORIENT, REG_AXIS6):
                        self.neutral_required.emit(self.device_id)
                else:
                    self.status.setText(f"{setting.label}未确认：期望 {setting.value}，回读 {values[0]}")
                self._show_parameters()
                self._update_controls()
            else:
                self._read_next()

    def _show_parameters(self) -> None:
        v = self.registers
        bw = (256, 188, 98, 42, 20, 10, 5)
        gyro = (250, 500, 1000, 2000)
        acc = (2, 4, 8, 16)
        index_label = lambda value, options, unit: (f"{options[value]} {unit}"
                    if value is not None and 0 <= value < len(options) else "未知")
        period = v.get(REG_RATE)
        first = (f"融合算法：{ {0: '9 轴', 1: '6 轴'}.get(v.get(REG_AXIS6), '未知')}  ·  "
                 f"安装方向：{ {0: '水平', 1: '垂直'}.get(v.get(REG_ORIENT), '未知')}")
        second = (f"内部带宽：{index_label(v.get(REG_BANDWIDTH), bw, 'Hz')}  ·  "
                  + (f"回传频率：{1000 / period:.1f} Hz（{period} ms）" if period else "回传频率：未知"))
        self.parameters.setText("\n".join((first, second,
            f"陀螺量程：{index_label(v.get(REG_GYRORANGE), gyro, '°/s')}  ·  "
            f"加计量程：{index_label(v.get(REG_ACCRANGE), acc, 'g')}",
            f"校准模式：0x{v.get(REG_CALSW, 0):04X}")))
        for combo, address in ((self.algorithm, REG_AXIS6), (self.orientation, REG_ORIENT),
                               (self.bandwidth, REG_BANDWIDTH), (self.gyro, REG_GYRORANGE),
                               (self.acc, REG_ACCRANGE)):
            index = combo.findData(v.get(address))
            if index >= 0:
                combo.setCurrentIndex(index)
        if period and 0.2 <= 1000 / period <= 200:
            self.rate.setValue(1000 / period)

    def _set_rate(self) -> None:
        period = max(5, min(5000, round(1000 / self.rate.value())))
        self._apply(Setting(REG_RATE, period, f"回传频率约 {1000 / period:.1f} Hz"))

    def _apply(self, setting: Setting) -> None:
        if self._busy() or not self.device_id or setting.address not in self.registers:
            return
        try:
            self.source.device_ip(self.device_id)
        except (RuntimeError, OSError) as exc:
            self.status.setText(str(exc))
            return
        if self.registers[setting.address] == setting.value:
            self.status.setText("当前参数已是所选值，无需写入")
            return
        extra = "\n切换安装方向后需要重新校准。" if setting.address == REG_ORIENT else ""
        if QtWidgets.QMessageBox.question(self, "确认写入设备",
                f"将 {self.device_id} 的{setting.label}设为所选值？\n"
                f"会保存到设备并回读验证。{extra}") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.setting = setting
        self.status.setText(f"正在设置{setting.label}…")
        self._update_controls()
        try:
            self.source.unlock(self.device_id)
        except (RuntimeError, OSError) as exc:
            self.setting = None
            self.status.setText(f"解锁失败：{exc}")
            self._update_controls()
            return
        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50,
                                 lambda: self._write_setting(setting))

    def _write_setting(self, setting: Setting) -> None:
        if self.setting is not setting:
            return
        try:
            self.source.write_register(self.device_id, setting.address, setting.value)
        except (RuntimeError, OSError) as exc:
            self.setting = None
            self.status.setText(f"写入失败：{exc}")
            self._update_controls()
            return
        QtCore.QTimer.singleShot(350, lambda: self._save_setting(setting))

    def _save_setting(self, setting: Setting) -> None:
        if self.setting is not setting:
            return
        try:
            self.source.write_register(self.device_id, REG_SAVE, 0)
        except (RuntimeError, OSError) as exc:
            self.setting = None
            self.status.setText(f"设置可能生效，但保存失败：{exc}；请重新读取")
            self._update_controls()
            return
        QtCore.QTimer.singleShot(350, lambda: self._verify_setting(setting))

    def _verify_setting(self, setting: Setting) -> None:
        if self.setting is not setting:
            return
        try:
            self.source.read_registers(self.device_id, setting.address)
        except (RuntimeError, OSError) as exc:
            self.setting = None
            self.status.setText(f"已发送设置，但回读失败：{exc}")
            self._update_controls()
            return
        self.waiting_address = setting.address
        self.read_deadline = time.monotonic() + 2.0

    def on_packets(self, packets) -> None:
        for packet in packets:
            if packet.device_id != self.device_id:
                continue
            self._last_packet = packet
            if self.calibration in ("mag_precheck", "mag"):
                for i, value in enumerate((packet.mag_x, packet.mag_y, packet.mag_z)):
                    self.mag_min[i] = min(self.mag_min[i], value)
                    self.mag_max[i] = max(self.mag_max[i], value)
        now = time.monotonic()
        if self._last_packet is not None and now - self._last_live_at >= 0.2:
            self._show_live()
            self._last_live_at = now

    def _show_live(self) -> None:
        p = self._last_packet
        if p is None:
            return
        charge = f"约 {p.battery_percent}%" if p.battery_percent is not None else "未知"
        self.live_text.setText(
            f"电池 {p.battery_v:.2f} V（{charge}） · RSSI {p.rssi} dBm · "
            f"温度 {p.temperature:.1f} °C · 固件 {p.version}\n\n"
            f"原始角度 R/P/Y  {p.roll:+.1f} / {p.pitch:+.1f} / {p.yaw:+.1f} °\n"
            f"加速度 X/Y/Z  {p.acc_x:+.3f} / {p.acc_y:+.3f} / {p.acc_z:+.3f} g\n"
            f"角速度 X/Y/Z  {p.gyro_x:+.1f} / {p.gyro_y:+.1f} / {p.gyro_z:+.1f} °/s\n"
            f"磁场 X/Y/Z  {p.mag_x:+.1f} / {p.mag_y:+.1f} / {p.mag_z:+.1f} µT")

    def _start_mag(self) -> None:
        if self._busy() or not self.device_id:
            return
        try:
            self.source.device_ip(self.device_id)
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        if QtWidgets.QMessageBox.question(self, "确认磁场校准",
                f"对 {self.device_id} 做磁场预检？\n至少旋转 20 秒，"
                "覆盖三轴后自动进入校准；校准完成需手动结束并保存。") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.calibration = "mag_precheck"
        self.calibration_started = time.monotonic()
        self.mag_min = [float("inf")] * 3
        self.mag_max = [float("-inf")] * 3
        self.status.setText("磁场预检中：绕 X/Y/Z 充分翻转设备；达标后自动开始正式校准")
        self._update_controls()

    def _start_acc(self) -> None:
        if self._busy() or self.registers.get(REG_ORIENT) != 0:
            return
        if QtWidgets.QMessageBox.question(self, "确认加计校准",
                f"确认校准 {self.device_id}？设备须正面朝上、水平静止，"
                "接下来至少 5 秒不要移动。") == QtWidgets.QMessageBox.StandardButton.Yes:
            self._begin_calibration("acc")

    def _begin_calibration(self, kind: str) -> None:
        try:
            self.source.unlock(self.device_id)
        except (RuntimeError, OSError) as exc:
            self.calibration = None
            self.status.setText(f"无法开始校准：{exc}")
            self._update_controls()
            return
        self.calibration = f"{kind}_pending"
        self.status.setText("已解锁；等待进入校准模式…")
        self._update_controls()
        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50,
                                 lambda: self._enter_calibration(kind))

    def _enter_calibration(self, kind: str) -> None:
        if self.calibration != f"{kind}_pending":
            return
        try:
            mode = 0x07 if kind == "mag" else 0x01 if kind == "acc" else self._reference_value
            self.source.write_register(self.device_id, REG_CALSW, mode)
        except (RuntimeError, OSError) as exc:
            self.calibration = None
            self.status.setText(f"进入校准失败：{exc}")
            self._update_controls()
            return
        self.calibration = kind
        self.calibration_started = time.monotonic()
        self.status.setText("磁场校准中：继续绕三轴翻转，完成后点击结束并保存" if kind == "mag"
                            else "加速度计校准中：请保持水平静止 5 秒" if kind == "acc"
                            else "设备归零指令已发送，正在保存…")
        self._update_controls()
        if kind == "acc":
            QtCore.QTimer.singleShot(5000, self._stop_calibration)
        elif kind == "reference":
            QtCore.QTimer.singleShot(400, self._stop_calibration)

    def _reference(self, value: int) -> None:
        if self._busy() or (value == 0x04 and self.registers.get(REG_AXIS6) != 1):
            return
        try:
            self.source.device_ip(self.device_id)
        except RuntimeError as exc:
            self.status.setText(str(exc))
            return
        label = "Z 轴航向归零" if value == 0x04 else "X/Y 角度参考归零"
        if QtWidgets.QMessageBox.question(self, "确认设备归零",
                f"对 {self.device_id} 执行{label}？设备输出将改变，模型请再执行中立位归零。") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._reference_value = value
        self._begin_calibration("reference")

    def _stop_calibration(self) -> None:
        if self.calibration not in ("mag", "acc", "reference"):
            return
        previous = self.calibration
        self.calibration = "stopping"
        self.status.setText("正在退出校准并保存…")
        self._update_controls()
        try:
            self.source.unlock(self.device_id, max_age=None)
        except (RuntimeError, OSError) as exc:
            self.calibration = previous
            self.status.setText(f"退出失败：{exc}；请重试结束并保存")
            self._update_controls()
            return
        def finish():
            if self.calibration != "stopping":
                return
            try:
                self.source.write_register(self.device_id, REG_CALSW, 0, max_age=None)
            except (RuntimeError, OSError) as exc:
                self.calibration = previous
                self.status.setText(f"退出校准失败：{exc}；请重试")
                self._update_controls()
                return
            QtCore.QTimer.singleShot(350, save)
        def save():
            if self.calibration != "stopping":
                return
            try:
                self.source.write_register(self.device_id, REG_SAVE, 0, max_age=None)
            except (RuntimeError, OSError) as exc:
                self.calibration = previous
                self.status.setText(f"保存失败：{exc}；请重试")
                self._update_controls()
                return
            self.calibration = None
            self.registers[REG_CALSW] = 0
            self.status.setText("已发送退出与保存指令；建议重新读取设备参数确认")
            self.neutral_required.emit(self.device_id)
            self._update_controls()
        QtCore.QTimer.singleShot(int(UNLOCK_SETTLE * 1000) + 50, finish)

    def _tick(self) -> None:
        now = time.monotonic()
        if self.waiting_address is not None and now > self.read_deadline:
            self.waiting_address = None
            self.pending_reads.clear()
            if self.setting is not None:
                label = self.setting.label
                self.setting = None
                self.status.setText(f"{label}已发送，但回读超时；请重新读取确认")
            else:
                self.status.setText("参数读取超时；可重试")
            self._update_controls()
        if self.calibration == "mag_precheck":
            span = [max(0.0, hi - lo) if lo != float("inf") else 0.0
                    for lo, hi in zip(self.mag_min, self.mag_max)]
            elapsed = now - self.calibration_started
            covered = min(span) >= 30.0 and max(span) >= 50.0
            if elapsed >= self.MAG_MIN_SECONDS and covered:
                try:
                    self.source.device_ip(self.device_id)
                except RuntimeError:
                    self.calibration = None
                    self.status.setText("设备已离线，预检已取消；未写入校准模式")
                    self._update_controls()
                else:
                    self._begin_calibration("mag")
            elif elapsed >= self.MAG_MAX_SECONDS:
                self.calibration = None
                self.status.setText("磁场预检未达标，未写入设备；请远离磁源后重试")
                self._update_controls()
            else:
                self.status.setText(f"磁场预检 {elapsed:.0f}s；三轴跨度 "
                                    f"{span[0]:.1f} / {span[1]:.1f} / {span[2]:.1f} µT；"
                                    "要求各轴≥30 µT 且至少一轴≥50 µT")
        if self.device_id and not self._busy():
            self._update_controls()

    def closeEvent(self, event) -> None:
        if self.calibration not in (None, "mag_precheck") or self.setting is not None:
            self.status.setText("设备操作尚未结束；请先结束校准或等待设置确认")
            event.ignore()
            return
        if self.calibration == "mag_precheck":
            self.calibration = None
        self.timer.stop()
        super().closeEvent(event)
