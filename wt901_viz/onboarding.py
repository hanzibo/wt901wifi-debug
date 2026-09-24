"""GUI wizard for moving a WT901WIFI from its own AP to the PC hotspot.

NetworkManager commands are asynchronous so the wrist viewer remains usable.
The Wi-Fi password is never included in the on-screen log.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time

from PySide6 import QtCore, QtWidgets


DEVICE_SSID = re.compile(r"WT55\d{8}\Z")
DEVICE_AP_IP = "192.168.4.1"
DEVICE_AP_PORT = 9250


def build_sta_command(ssid: str, password: str, server_ip: str, port: int) -> bytes:
    """Build the vendor's combined AP->STA UDP command, without logging secrets."""
    if not 1 <= len(ssid.encode("utf-8")) <= 32:
        raise ValueError("热点名称必须为 1～32 字节")
    if not 8 <= len(password) <= 63:
        raise ValueError("热点密码必须为 8～63 个字符")
    for name, value in (("热点名称", ssid), ("热点密码", password)):
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"设备配置指令只支持 ASCII {name}") from exc
        if any(char in value for char in ('"', ',', ';', '\r', '\n')):
            raise ValueError(f"{name}不能包含双引号、逗号、分号或换行")
    ipaddress.IPv4Address(server_ip)
    if not 1 <= port <= 65535:
        raise ValueError("接收端口无效")
    return f'IPWIFI:"{ssid}","{password}";UDP{server_ip},{port}\r\n'.encode("ascii")


class DeviceSetupDialog(QtWidgets.QDialog):
    def __init__(self, source, parent=None, hotspot_profile: str = "WT901-LAN"):
        super().__init__(parent)
        self.source = source
        self.hotspot_profile = hotspot_profile
        self.interface = ""
        self.connection = ""
        self.mode = ""
        self.current_ssid = ""
        self.local_ip = ""
        self.profile_ssid = ""
        self.profile_mode = ""
        self.profile_band = ""
        self.profile_ipv4_method = ""
        self.profile_ip = ""
        self.profile_network = None
        self.expected_id: str | None = None
        self.sent_at = 0.0
        self.join_reported = False
        self._process: QtCore.QProcess | None = None

        self.setWindowTitle("接入新的 WT901 设备")
        self.resize(690, 610)
        layout = QtWidgets.QVBoxLayout(self)
        self.state_label = QtWidgets.QLabel("正在检查 Wi-Fi 状态…")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        top_actions = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("刷新状态")
        self.refresh_button.clicked.connect(self.refresh)
        self.hotspot_down_button = QtWidgets.QPushButton("关闭电脑热点")
        self.hotspot_down_button.clicked.connect(self.stop_hotspot)
        self.hotspot_up_button = QtWidgets.QPushButton("启动电脑热点")
        self.hotspot_up_button.clicked.connect(self.start_hotspot)
        for button in (self.refresh_button, self.hotspot_down_button, self.hotspot_up_button):
            top_actions.addWidget(button)
        layout.addLayout(top_actions)

        device_group = QtWidgets.QGroupBox("1. 连接新设备自带的 WT55… 热点")
        device_layout = QtWidgets.QVBoxLayout(device_group)
        device_layout.addWidget(QtWidgets.QLabel(
            "先关闭电脑热点，再扫描并连接设备 AP；也可以使用 Ubuntu 的 Wi-Fi 菜单连接后点“刷新状态”。"))
        device_row = QtWidgets.QHBoxLayout()
        self.scan_button = QtWidgets.QPushButton("扫描 WT 设备")
        self.scan_button.clicked.connect(self.scan_devices)
        self.device_combo = QtWidgets.QComboBox()
        self.connect_button = QtWidgets.QPushButton("连接选中设备")
        self.connect_button.clicked.connect(self.connect_device)
        device_row.addWidget(self.scan_button)
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.connect_button)
        device_layout.addLayout(device_row)
        self.ap_password = QtWidgets.QLineEdit()
        self.ap_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.ap_password.setPlaceholderText("设备 AP 密码（若为开放网络或已有保存配置，可留空）")
        device_layout.addWidget(self.ap_password)
        layout.addWidget(device_group)

        sta_group = QtWidgets.QGroupBox("2. 写入电脑热点与 UDP 接收地址")
        form = QtWidgets.QFormLayout(sta_group)
        self.hotspot_ssid = QtWidgets.QLineEdit()
        self.hotspot_ssid.setPlaceholderText("电脑热点名称")
        self.hotspot_password = QtWidgets.QLineEdit()
        self.hotspot_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.hotspot_password.setPlaceholderText("输入电脑热点密码；不会显示在日志中")
        self.confirm_password = QtWidgets.QLineEdit()
        self.confirm_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.confirm_password.setPlaceholderText("再次输入电脑热点密码，避免设备断连后无法加入")
        self.show_password = QtWidgets.QCheckBox("显示密码")
        self.show_password.toggled.connect(self._toggle_password_visibility)
        self.server_label = QtWidgets.QLabel("等待读取电脑热点地址…")
        form.addRow("电脑热点 SSID", self.hotspot_ssid)
        form.addRow("电脑热点密码", self.hotspot_password)
        form.addRow("确认热点密码", self.confirm_password)
        form.addRow("", self.show_password)
        form.addRow("电脑接收地址", self.server_label)
        self.send_button = QtWidgets.QPushButton("写入 STA + UDP 配置到当前 WT 设备")
        self.send_button.clicked.connect(self.send_sta_configuration)
        form.addRow(self.send_button)
        layout.addWidget(sta_group)

        self.result_label = QtWidgets.QLabel("写入后启动电脑热点，等待设备以 STA 模式接入。")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(100)
        layout.addWidget(self.log, 1)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(5000)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start()
        self.join_timer = QtCore.QTimer(self)
        self.join_timer.setInterval(1000)
        self.join_timer.timeout.connect(self.check_join)
        self.join_timer.start()
        self._render_buttons()
        QtCore.QTimer.singleShot(0, self.refresh)

    def _toggle_password_visibility(self, checked: bool):
        mode = (QtWidgets.QLineEdit.EchoMode.Normal if checked
                else QtWidgets.QLineEdit.EchoMode.Password)
        self.hotspot_password.setEchoMode(mode)
        self.confirm_password.setEchoMode(mode)

    def _log(self, message: str):
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {message}")

    def _run_nmcli(self, args: list[str], callback):
        if self._process is not None:
            return
        process = QtCore.QProcess(self)
        self._process = process
        self._render_buttons()

        def finish(exit_code: int, _exit_status=None):
            if self._process is not process:
                return
            out = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
            err = bytes(process.readAllStandardError()).decode("utf-8", "replace")
            self._process = None
            process.deleteLater()
            self._render_buttons()
            callback(exit_code == 0, out, err)

        process.finished.connect(finish)
        process.errorOccurred.connect(
            lambda error: finish(1) if error == QtCore.QProcess.ProcessError.FailedToStart else None)
        process.start("nmcli", args)

    def refresh(self):
        if self._process is not None:
            return
        self._run_nmcli(["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
                        self._on_device_status)

    def _on_device_status(self, ok: bool, out: str, err: str):
        if not ok:
            self.state_label.setText("无法读取 NetworkManager 状态；请确认 nmcli 可用。")
            self._log(f"读取网络状态失败：{err.strip()}")
            return
        self.interface = self.connection = self.mode = self.current_ssid = self.local_ip = ""
        for line in out.splitlines():
            parts = line.split(":", 3)
            if len(parts) == 4 and parts[1] == "wifi":
                self.interface = parts[0]
                self.connection = parts[3] if parts[2] == "connected" else ""
                break
        if not self.interface:
            self.state_label.setText("未检测到可用 Wi-Fi 网卡。")
            self._render_buttons()
            return
        if self.connection:
            self._run_nmcli(
                ["-g", "802-11-wireless.mode,802-11-wireless.ssid",
                 "connection", "show", self.connection], self._on_connection)
        else:
            self._load_hotspot_profile()

    def _on_connection(self, ok: bool, out: str, err: str):
        if ok:
            lines = out.splitlines()
            self.mode = lines[0].strip() if lines else ""
            self.current_ssid = lines[1].strip() if len(lines) > 1 else ""
        else:
            self._log(f"读取当前 Wi-Fi 连接失败：{err.strip()}")
        self._run_nmcli(["-g", "IP4.ADDRESS", "device", "show", self.interface],
                        self._on_local_ip)

    def _on_local_ip(self, ok: bool, out: str, _err: str):
        self.local_ip = out.splitlines()[0].split("/")[0].strip() if ok and out.strip() else ""
        self._load_hotspot_profile()

    def _load_hotspot_profile(self):
        self._run_nmcli(
            ["-g", "802-11-wireless.mode,802-11-wireless.ssid,802-11-wireless.band,"
             "ipv4.method,ipv4.addresses", "connection", "show",
             self.hotspot_profile], self._on_hotspot_profile)

    def _on_hotspot_profile(self, ok: bool, out: str, _err: str):
        old_ssid = self.profile_ssid
        self.profile_mode = self.profile_ssid = self.profile_band = ""
        self.profile_ipv4_method = self.profile_ip = ""
        self.profile_network = None
        if ok:
            lines = out.splitlines()
            self.profile_mode = lines[0].strip() if lines else ""
            self.profile_ssid = lines[1].strip() if len(lines) > 1 else ""
            self.profile_band = lines[2].strip() if len(lines) > 2 else ""
            self.profile_ipv4_method = lines[3].strip() if len(lines) > 3 else ""
            if len(lines) > 4 and lines[4].strip():
                try:
                    interface = ipaddress.IPv4Interface(lines[4].strip())
                    self.profile_ip = str(interface.ip)
                    self.profile_network = interface.network
                except ValueError:
                    pass
        if not self.hotspot_ssid.text() or self.hotspot_ssid.text() == old_ssid:
            self.hotspot_ssid.setText(self.profile_ssid)
        port = self.source.sock.getsockname()[1]
        self.server_label.setText(f"{self.profile_ip or '未知'}:{port}（由热点配置自动读取）")
        self._render_status()

    def _render_status(self):
        if not self.interface:
            text = "未检测到 Wi-Fi 网卡。"
        elif self.mode == "ap":
            text = f"当前：电脑热点 {self.current_ssid}（{self.interface}，{self.local_ip}）"
        elif DEVICE_SSID.fullmatch(self.current_ssid):
            text = f"当前：已连接 WT 设备 AP {self.current_ssid}（{self.interface}，{self.local_ip}）"
        elif self.connection:
            text = f"当前：Wi-Fi 已连接 {self.current_ssid or self.connection}，但不是 WT 设备 AP。"
        else:
            text = f"当前：Wi-Fi 网卡 {self.interface} 未连接。请扫描 WT 设备。"
        if not self.profile_ssid or not self.profile_ip:
            text += "\n未找到可用的电脑热点配置 WT901-LAN；请先在系统中创建热点。"
        elif self.profile_mode != "ap" or self.profile_ipv4_method != "shared":
            text += "\n已保存的 WT901-LAN 不是带地址共享的 AP 配置；不会向设备写入它。"
        elif self.profile_band != "bg":
            text += "\n电脑热点未固定为 2.4 GHz（bg）；WT901 可能无法连接，请先修改热点配置。"
        self.state_label.setText(text)
        self._render_buttons()

    def _render_buttons(self):
        idle = self._process is None
        hotspot_active = self.mode == "ap" and self.connection == self.hotspot_profile
        device_connected = bool(DEVICE_SSID.fullmatch(self.current_ssid)) and self.mode != "ap"
        self.refresh_button.setEnabled(idle)
        self.hotspot_down_button.setEnabled(idle and hotspot_active)
        profile_ready = (self.profile_mode == "ap" and self.profile_ipv4_method == "shared"
                         and self.profile_band == "bg" and bool(self.profile_ip))
        self.hotspot_up_button.setEnabled(idle and profile_ready and not hotspot_active)
        self.scan_button.setEnabled(idle and bool(self.interface) and self.mode != "ap")
        self.connect_button.setEnabled(idle and bool(self.device_combo.currentText())
                                       and self.mode != "ap")
        self.send_button.setEnabled(idle and device_connected and profile_ready)

    def stop_hotspot(self):
        if self.mode != "ap" or self.connection != self.hotspot_profile:
            self._log("当前并非本程序使用的电脑热点，不会关闭其他 Wi-Fi 连接。")
            return
        answer = QtWidgets.QMessageBox.question(
            self, "关闭电脑热点",
            "关闭电脑热点后，已经连接的 WT 设备会暂时离线。确认继续？")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run_nmcli(["connection", "down", self.hotspot_profile], self._after_hotspot_down)

    def _after_hotspot_down(self, ok: bool, _out: str, err: str):
        self._log("电脑热点已关闭，请扫描或连接新设备 AP。" if ok
                  else f"关闭电脑热点失败：{err.strip()}")
        self.refresh()

    def start_hotspot(self):
        if (self.profile_mode != "ap" or self.profile_ipv4_method != "shared"
                or self.profile_band != "bg" or not self.profile_ssid or not self.profile_ip):
            self._log("未找到可用的 2.4 GHz 电脑热点 AP 配置。")
            return
        self._run_nmcli(["connection", "up", self.hotspot_profile], self._after_hotspot_up)

    def _after_hotspot_up(self, ok: bool, _out: str, err: str):
        self._log("电脑热点已启动，等待新设备接入…" if ok
                  else f"启动电脑热点失败：{err.strip()}")
        self.refresh()

    def scan_devices(self):
        if not self.interface or self.mode == "ap":
            self._log("请先关闭电脑热点，再扫描设备 AP。")
            return
        self._run_nmcli(["-g", "SSID", "device", "wifi", "list",
                         "ifname", self.interface, "--rescan", "yes"], self._after_scan)

    def _after_scan(self, ok: bool, out: str, err: str):
        if not ok:
            self._log(f"扫描失败：{err.strip()}")
            return
        ids = sorted({line.strip() for line in out.splitlines()
                      if DEVICE_SSID.fullmatch(line.strip())})
        self.device_combo.clear()
        self.device_combo.addItems(ids)
        self._log(f"扫描到 {len(ids)} 台 WT 设备。" if ids else
                  "未扫描到 WT55… 设备；可尝试使用 Ubuntu Wi-Fi 菜单连接后刷新。")
        self._render_buttons()

    def connect_device(self):
        ssid = self.device_combo.currentText()
        if not DEVICE_SSID.fullmatch(ssid) or not self.interface or self.mode == "ap":
            return
        args = ["device", "wifi", "connect", ssid, "ifname", self.interface]
        password = self.ap_password.text()
        if password:
            args += ["password", password]

        def finished(ok: bool, _out: str, err: str):
            if ok:
                self.ap_password.clear()
                self._log(f"已连接 {ssid}；请填写电脑热点密码并发送 STA 配置。")
            else:
                safe_err = err.replace(password, "***") if password else err
                self._log(f"连接 {ssid} 失败：{safe_err.strip()}。也可用系统 Wi-Fi 菜单连接。")
            self.refresh()

        self._run_nmcli(args, finished)

    def send_sta_configuration(self):
        if not DEVICE_SSID.fullmatch(self.current_ssid) or self.mode == "ap":
            self._log("当前未连接 WT 设备 AP；不会发送指令。")
            return
        if self.hotspot_ssid.text() != self.profile_ssid or not self.profile_ip:
            self._log("输入的热点名称与电脑已保存的热点配置不一致；请先核对。")
            return
        if self.profile_mode != "ap" or self.profile_ipv4_method != "shared":
            self._log("电脑热点配置不是 AP + 地址共享模式；配置未发送。")
            return
        if self.profile_band != "bg":
            self._log("电脑热点未设为 2.4 GHz（bg）；配置未发送。")
            return
        if self.hotspot_password.text() != self.confirm_password.text():
            self._log("两次输入的电脑热点密码不一致；配置未发送。")
            return
        if not self.local_ip:
            self._log("电脑当前没有设备 AP 网段的 IP；请刷新网络状态。")
            return
        try:
            if ipaddress.IPv4Address(self.local_ip) not in ipaddress.IPv4Network("192.168.4.0/24"):
                raise ValueError("当前连接不在设备默认的 192.168.4.0/24 网段")
            listen_ip = self.source.sock.getsockname()[0]
            if listen_ip not in ("0.0.0.0", self.profile_ip):
                raise ValueError(
                    f"GUI 正监听 {listen_ip}；请用 --listen-ip 0.0.0.0 重启后再配置")
            port = self.source.sock.getsockname()[1]
            command = build_sta_command(self.profile_ssid, self.hotspot_password.text(),
                                        self.profile_ip, port)
        except ValueError as exc:
            self._log(f"配置未发送：{exc}")
            return
        answer = QtWidgets.QMessageBox.question(
            self, "确认写入 STA 配置",
            f"当前目标设备：{self.current_ssid}\n"
            f"它将连接电脑热点 {self.profile_ssid}，并以 UDP 发送到 {self.profile_ip}:{port}。\n"
            "发送后设备 AP 可能立刻消失；请再启动电脑热点。若密码错误无法入网，"
            "可能需要按设备说明恢复 AP 后重试。确认继续？")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((self.local_ip, 0))
                sock.sendto(command, (DEVICE_AP_IP, DEVICE_AP_PORT))
        except OSError as exc:
            self._log(f"发送失败：{exc}")
            return
        self.expected_id = self.current_ssid
        self.sent_at = time.monotonic()
        self.join_reported = False
        self.hotspot_password.clear()
        self.confirm_password.clear()
        self._log(f"已向 {self.expected_id} 发送 STA + UDP 配置；发送成功不等于设备已接入。")
        self.result_label.setText(
            f"已发送给 {self.expected_id}。请点击“启动电脑热点”，等待该 ID 的实时数据出现。")
        QtCore.QTimer.singleShot(1500, self.refresh)

    def check_join(self):
        if not self.expected_id or self.join_reported or not self.profile_network:
            return
        found = self.source.seen_devices(max_age=3.0).get(self.expected_id)
        if found and found[1] > self.sent_at and ipaddress.IPv4Address(found[0]) in self.profile_network:
            self.join_reported = True
            self.result_label.setText(f"接入成功：{self.expected_id} 已从 {found[0]} 发送实时数据。")
            self._log(f"接入成功：{self.expected_id}，地址 {found[0]}。")
        elif time.monotonic() - self.sent_at > 30 and self.mode == "ap":
            self.result_label.setText(
                f"暂未收到 {self.expected_id} 的数据。请检查热点密码、设备电量和热点是否为 2.4 GHz。")

    def closeEvent(self, event):
        if self._process is not None:
            self._log("网络操作尚未结束，请稍候再关闭窗口。")
            event.ignore()
            return
        self.refresh_timer.stop()
        self.join_timer.stop()
        self.hotspot_password.clear()
        self.confirm_password.clear()
        self.ap_password.clear()
        super().closeEvent(event)
