"""Move a WT901WIFI from its AP to a PC hotspot or external Wi-Fi.

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
        self.join_target = None
        self.local_network = None
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
        self.device_instruction = QtWidgets.QLabel()
        self.device_instruction.setWordWrap(True)
        device_layout.addWidget(self.device_instruction)
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

        sta_group = QtWidgets.QGroupBox("2. 写入目标 Wi-Fi 与 UDP 接收端地址")
        form = QtWidgets.QFormLayout(sta_group)
        self.network_mode = QtWidgets.QComboBox()
        self.network_mode.addItems(["电脑热点（原模式）", "外部 Wi-Fi / 手机热点（电脑接收）",
                                    "外部 Wi-Fi / 路由器（其他设备 / Jetson 接收）"])
        form.addRow("接入方式", self.network_mode)
        self.hotspot_ssid = QtWidgets.QLineEdit()
        self.hotspot_ssid.setPlaceholderText("电脑热点名称")
        self.external_ssid = QtWidgets.QLineEdit()
        self.external_ssid.setPlaceholderText("例如 Redmi K60；必须支持 2.4 GHz")
        self.external_ip = QtWidgets.QLineEdit()
        self.external_ip.setPlaceholderText("电脑连接该 Wi-Fi 时的 IPv4，不是板子或路由器地址")
        self.external_port = QtWidgets.QSpinBox()
        self.external_port.setRange(1, 65535)
        self.external_port.setValue(self.source.sock.getsockname()[1])
        self.external_port.setReadOnly(True)
        self.external_port.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.capture_address = QtWidgets.QPushButton("使用当前外部 Wi-Fi 的 SSID / IP")
        self.capture_address.clicked.connect(self._capture_external_address)
        self.external_hint = QtWidgets.QLabel(
            "先在电脑连接手机热点时记录 SSID / IPv4，再连接 WT 设备 AP 写入。\n"
            "写入后用系统 Wi-Fi 菜单重新连接同一热点；电脑和 IMU 须能互访。\n"
            "手机热点须为 2.4 GHz；单 Wi-Fi 网卡切入设备 AP 时上网会暂时中断。")
        self.external_hint.setWordWrap(True)
        self.hotspot_password = QtWidgets.QLineEdit()
        self.hotspot_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.hotspot_password.setPlaceholderText("输入目标 Wi-Fi 密码；不会显示在日志中")
        self.confirm_password = QtWidgets.QLineEdit()
        self.confirm_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.confirm_password.setPlaceholderText("再次输入目标 Wi-Fi 密码，避免设备断连后无法加入")
        self.show_password = QtWidgets.QCheckBox("显示密码")
        self.show_password.toggled.connect(self._toggle_password_visibility)
        self.server_label = QtWidgets.QLabel("等待读取电脑热点地址…")
        form.addRow("电脑热点 SSID", self.hotspot_ssid)
        form.addRow("外部 Wi-Fi SSID", self.external_ssid)
        form.addRow("UDP 目标 IPv4", self.external_ip)
        form.addRow("UDP 目标端口", self.external_port)
        form.addRow(self.capture_address)
        form.addRow(self.external_hint)
        form.addRow("目标 Wi-Fi 密码", self.hotspot_password)
        form.addRow("确认热点密码", self.confirm_password)
        form.addRow("", self.show_password)
        form.addRow("电脑接收地址", self.server_label)
        self.send_button = QtWidgets.QPushButton("写入 STA + UDP 配置到当前 WT 设备")
        self.send_button.clicked.connect(self.send_sta_configuration)
        form.addRow(self.send_button)
        layout.addWidget(sta_group)
        self.sta_form = form
        self.network_mode.currentIndexChanged.connect(self._network_mode_changed)

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
        self._network_mode_changed()
        QtCore.QTimer.singleShot(0, self.refresh)

    def _external_mode(self):
        return self.network_mode.currentIndex() in (1, 2)

    def _remote_mode(self):
        return self.network_mode.currentIndex() == 2

    def _network_mode_changed(self):
        external = self._external_mode()
        remote = self._remote_mode()
        if remote and self.external_ip.text() == self.local_ip:
            self.external_ip.clear()
        self.external_port.setReadOnly(not remote)
        self.external_port.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows if remote
                                           else QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        if not remote:
            self.external_port.setValue(self.source.sock.getsockname()[1])
        self.external_ip.setPlaceholderText('例如 192.168.3.8：板子有线 IPv4，不是路由器 .1' if remote
                                           else '本电脑在目标网络中的 IPv4')
        self.external_hint.setText(
            '由本电脑连接设备 AP 写入；IMU 加入目标路由器的 2.4 GHz Wi-Fi，向板子有线 IPv4 发送 UDP。\n'
            '请手动填写路由器 SSID（如 HUAWEI-509）和板子地址（如 192.168.3.8:1399）。\n'
            '本电脑不必与 IMU 保持同一 Wi-Fi；接入成功须到板子的 IMU 管理窗口查看，不能在本机验证。'
            if remote else
            '先记录电脑在目标外部 Wi-Fi 下的 IPv4，再连接设备 AP 写入。\n'
            '写入后回到目标网络，电脑和 IMU 须能互访；路由器可使用电脑有线 IPv4。')
        self.device_instruction.setText(
            "先记录目标外部 Wi-Fi 的 SSID / 电脑地址，再连接 WT 设备 AP；"
            "可用 Ubuntu Wi-Fi 菜单连接后点“刷新状态”。"
            if external else
            "先关闭电脑热点，再扫描并连接设备 AP；也可以使用 Ubuntu 的 Wi-Fi 菜单连接后点“刷新状态”。")
        for widget in (self.hotspot_ssid, self.server_label):
            self.sta_form.setRowVisible(widget, not external)
        for widget in (self.external_ssid, self.external_ip, self.external_port,
                       self.capture_address, self.external_hint):
            self.sta_form.setRowVisible(widget, external)
        self.hotspot_password.clear()
        self.confirm_password.clear()
        if not self.expected_id:
            self.result_label.setText(
                ("写入后请在目标板子上检查实时设备 ID / 收包率；本电脑不验证远端接入。" if remote else
                 "写入后请用系统 Wi-Fi 菜单回到目标外部热点，等待本电脑收到该设备数据。")
                if external else "写入后启动电脑热点，等待设备以 STA 模式接入。")
        self._render_status()

    def _external_connection(self):
        return (bool(self.current_ssid) and self.mode != "ap"
                and not DEVICE_SSID.fullmatch(self.current_ssid) and bool(self.local_ip))

    def _capture_external_address(self):
        if self._remote_mode():
            self._log('远端接收模式须手动填写板子有线 IPv4，不会覆盖为本电脑 Wi-Fi 地址。')
            return
        if not self._external_connection():
            self._log("请先连接目标外部 Wi-Fi 并刷新状态，再读取本电脑地址。")
            return
        self.external_ssid.setText(self.current_ssid)
        self.external_ip.setText(self.local_ip)
        self._log(f"已填写 {self.current_ssid} / {self.local_ip}；切入设备 AP 后会保留。")

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
        self.local_network = (ipaddress.IPv4Interface(out.splitlines()[0].strip()).network
                              if self.local_ip else None)
        if self._external_connection() and not self._remote_mode():
            if not self.external_ssid.text():
                self.external_ssid.setText(self.current_ssid)
            if not self.external_ip.text() and self.external_ssid.text() == self.current_ssid:
                self.external_ip.setText(self.local_ip)
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
        if self._external_mode():
            text += f"\n外部 Wi-Fi 模式；当前电脑 Wi-Fi IPv4：{self.local_ip or '未获取'}。"
            text += ("远端接收模式：填写板子有线 IPv4，接入结果在板子验证。" if self._remote_mode() else
                     "写入后须让电脑回到目标热点；不会自动启动电脑热点。")
        elif not self.profile_ssid or not self.profile_ip:
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
        self.hotspot_down_button.setEnabled(idle and hotspot_active and not self._external_mode())
        profile_ready = (self.profile_mode == "ap" and self.profile_ipv4_method == "shared"
                         and self.profile_band == "bg" and bool(self.profile_ip))
        self.hotspot_up_button.setEnabled(idle and profile_ready and not hotspot_active
                                          and not self._external_mode())
        self.capture_address.setEnabled(idle and self._external_connection() and not self._remote_mode())
        self.network_mode.setEnabled(idle)
        self.scan_button.setEnabled(idle and bool(self.interface) and self.mode != "ap")
        self.connect_button.setEnabled(idle and bool(self.device_combo.currentText())
                                       and self.mode != "ap")
        self.send_button.setEnabled(idle and device_connected
                                    and (self._external_mode() or profile_ready))

    def stop_hotspot(self):
        if self._external_mode():
            return
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
        if self._external_mode():
            return
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
                self._log(f"已连接 {ssid}；请核对目标 Wi-Fi 与接收地址，再发送 STA 配置。")
            else:
                safe_err = err.replace(password, "***") if password else err
                self._log(f"连接 {ssid} 失败：{safe_err.strip()}。也可用系统 Wi-Fi 菜单连接。")
            self.refresh()

        self._run_nmcli(args, finished)

    def send_sta_configuration(self):
        if not DEVICE_SSID.fullmatch(self.current_ssid) or self.mode == "ap":
            self._log("当前未连接 WT 设备 AP；不会发送指令。")
            return
        external = self._external_mode()
        if not external and (self.hotspot_ssid.text() != self.profile_ssid or not self.profile_ip):
            self._log("输入的热点名称与电脑已保存的热点配置不一致；请先核对。")
            return
        if not external and (self.profile_mode != "ap" or self.profile_ipv4_method != "shared"):
            self._log("电脑热点配置不是 AP + 地址共享模式；配置未发送。")
            return
        if not external and self.profile_band != "bg":
            self._log("电脑热点未设为 2.4 GHz（bg）；配置未发送。")
            return
        if self.hotspot_password.text() != self.confirm_password.text():
            self._log("两次输入的目标 Wi-Fi 密码不一致；配置未发送。")
            return
        if not self.local_ip:
            self._log("电脑当前没有设备 AP 网段的 IP；请刷新网络状态。")
            return
        try:
            ssid = self.external_ssid.text() if external else self.profile_ssid
            server_ip = self.external_ip.text().strip() if external else self.profile_ip
            address = ipaddress.IPv4Address(server_ip)
            if (address.is_unspecified or address.is_loopback or address.is_multicast
                    or address.is_link_local or address == ipaddress.IPv4Address('255.255.255.255')):
                raise ValueError("接收地址须为接收端在目标网络中的有效单播 IPv4")
            if external and (DEVICE_SSID.fullmatch(ssid) or
                             server_ip in (DEVICE_AP_IP, self.local_ip)):
                raise ValueError("请填写外部 Wi-Fi 与电脑地址，不要填写 WT 设备 AP 的 SSID / 地址")
            if ipaddress.IPv4Address(self.local_ip) not in ipaddress.IPv4Network("192.168.4.0/24"):
                raise ValueError("当前连接不在设备默认的 192.168.4.0/24 网段")
            listen_ip = self.source.sock.getsockname()[0]
            if not self._remote_mode() and listen_ip not in ("0.0.0.0", server_ip):
                raise ValueError(
                    f"GUI 正监听 {listen_ip}；请用 --listen-ip 0.0.0.0 重启后再配置")
            port = self.external_port.value() if self._remote_mode() else self.source.sock.getsockname()[1]
            command = build_sta_command(ssid, self.hotspot_password.text(), server_ip, port)
        except ValueError as exc:
            self._log(f"配置未发送：{exc}")
            return
        answer = QtWidgets.QMessageBox.question(
            self, "确认写入 STA 配置",
            f"当前目标设备：{self.current_ssid}\n"
            f"它将连接 Wi-Fi {ssid}，并以 UDP 发送到{'远端接收设备' if self._remote_mode() else '本电脑'} {server_ip}:{port}。\n"
            + ("请确认目标是板子的有线 IPv4，且板子已开启 IMU 管理或控制程序。接入结果须在板子验证。\n"
               if self._remote_mode() else
               "请确认该地址是电脑在目标 Wi-Fi 中的地址，不是路由器或板子地址。\n"
               "须启用 2.4 GHz 并允许热点终端互访。发送后请用系统菜单重新连接该 Wi-Fi。\n"
               if external else "发送后设备 AP 可能立刻消失；请再启动电脑热点。\n") +
            "若密码错误无法入网，"
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
        self.join_target = {'external': external, 'remote': self._remote_mode(), 'ssid': ssid, 'ip': server_ip,
                            'network': None if external else self.profile_network}
        self.sent_at = time.monotonic()
        self.join_reported = False
        self.hotspot_password.clear()
        self.confirm_password.clear()
        self._log(f"已向 {self.expected_id} 发送 STA + UDP 配置；发送成功不等于设备已接入。")
        self.result_label.setText(
            f"已发送给 {self.expected_id}。" +
            (f"设备将向远端 {server_ip}:{port} 发送。请在板子上确认 {self.expected_id} 的实时收包；本机不判定成功。"
             if self._remote_mode() else
             f"请用系统 Wi-Fi 菜单连接 {ssid}，保持电脑地址 {server_ip}，等待实时数据。"
             if external else "请点击“启动电脑热点”，等待该 ID 的实时数据出现。"))
        QtCore.QTimer.singleShot(1500, self.refresh)

    def check_join(self):
        if not self.expected_id or self.join_reported or not self.join_target:
            return
        target = self.join_target
        if target.get('remote'):
            self.result_label.setText(
                f"配置已发送，远端接入尚未在本机验证：{self.expected_id} → {target['ip']}。请查看板子 IMU 管理窗口。")
            return
        network = target['network']
        if target['external']:
            if self.current_ssid != target['ssid'] or not self._external_connection():
                self.result_label.setText(f"等待电脑重新连接 {target['ssid']}，再验证 {self.expected_id}。")
                return
            if self.local_ip != target['ip']:
                self.result_label.setText(
                    f"电脑当前 IP {self.local_ip} 与已写入目标 {target['ip']} 不一致；"
                    "请恢复原地址或重新配置 IMU 的 UDP 目标。")
                return
            network = self.local_network
        if network is None:
            return
        found = self.source.seen_devices(max_age=3.0).get(self.expected_id)
        if found and found[1] > self.sent_at and ipaddress.IPv4Address(found[0]) in network:
            self.join_reported = True
            self.result_label.setText(f"接入成功：{self.expected_id} 已从 {found[0]} 发送实时数据。")
            self._log(f"接入成功：{self.expected_id}，地址 {found[0]}。")
        elif time.monotonic() - self.sent_at > 30 and (target['external'] or self.mode == "ap"):
            self.result_label.setText(
                f"暂未收到 {self.expected_id} 的数据。请检查密码、2.4 GHz、设备电量、"
                "终端互访与电脑防火墙；发送成功不代表已接入。")

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
