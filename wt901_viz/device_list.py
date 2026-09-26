"""Read-only view of WT devices associated with the computer's Wi-Fi hotspot."""

from __future__ import annotations

import re
import time

from PySide6 import QtCore, QtWidgets


_STATION = re.compile(r"^Station ([0-9a-f:]{17})\b", re.IGNORECASE)
_NEIGHBOR = re.compile(
    r"^(\d+\.\d+\.\d+\.\d+)\s+.*\blladdr\s+([0-9a-f:]{17})\b",
    re.IGNORECASE,
)


def parse_stations(output: str) -> list[str]:
    """Only include currently associated and authorized Wi-Fi clients."""
    stations: list[str] = []
    mac = ""
    associated = authorized = False

    def save():
        if mac and associated and authorized:
            stations.append(mac)

    for line in output.splitlines():
        match = _STATION.match(line)
        if match:
            save()
            mac = match.group(1).lower()
            associated = authorized = False
        elif mac:
            field = line.strip().lower()
            if field.startswith("associated:"):
                associated = field.split(":", 1)[1].strip() == "yes"
            elif field.startswith("authorized:"):
                authorized = field.split(":", 1)[1].strip() == "yes"
    save()
    return stations


def parse_neighbors(output: str) -> dict[str, str]:
    """Map MAC to IPv4; stale ARP entries are harmless after station filtering."""
    neighbors: dict[str, str] = {}
    for line in output.splitlines():
        match = _NEIGHBOR.match(line)
        if match:
            neighbors[match.group(2).lower()] = match.group(1)
    return neighbors


class ConnectedDevicesDialog(QtWidgets.QDialog):
    def __init__(self, source, parent=None, can_switch=None):
        super().__init__(parent)
        self.source = source
        self.can_switch = can_switch
        self.interface = ""
        self.connection = ""
        self.hotspot_ip = ""
        self.hotspot_active = False
        self.stations: list[str] = []
        self.neighbors: dict[str, str] = {}
        self._process: QtCore.QProcess | None = None
        self._last_scan = 0.0
        self._last_ap_command: tuple[str, float] | None = None

        self.setWindowTitle("电脑热点中的 WT 设备")
        self.resize(920, 400)
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("正在检查电脑热点…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["设备唯一 ID", "设备端点（IP:源端口）", "电脑接收端点", "Wi-Fi MAC", "电量", "状态"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._update_switch_button)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(self.table)
        self.note = QtWidgets.QLabel(
            "唯一 ID 来自实时 WT 数据；仅连接热点但尚未发送数据的终端无法确认是否为 WT。"
            "设备端点的源端口可能变化；电脑接收端点是设备应发送到的地址。"
            "电量由最新数据包的电池电压按手册档位估算，并非精确电量计读数。"
            "切回 AP/UDP 后，设备会离开电脑热点并创建以其 ID 命名的 Wi-Fi。")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.action_status = QtWidgets.QLabel("")
        self.action_status.setWordWrap(True)
        layout.addWidget(self.action_status)
        actions = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("立即刷新")
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_button)
        self.switch_ap_button = QtWidgets.QPushButton("将选中设备切回 AP/UDP")
        self.switch_ap_button.setToolTip("向选中的在线 WT 设备发送 WITAPUDP 指令")
        self.switch_ap_button.clicked.connect(self.switch_selected_to_ap)
        self.switch_ap_button.setEnabled(False)
        actions.addWidget(self.switch_ap_button)
        actions.addStretch(1)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        QtCore.QTimer.singleShot(0, self.refresh)

    def _run(self, program: str, args: list[str], callback):
        if self._process is not None:
            return
        process = QtCore.QProcess(self)
        self._process = process
        self.refresh_button.setEnabled(False)
        self._update_switch_button()

        def finish(exit_code: int, _exit_status=None):
            if self._process is not process:
                return
            out = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
            err = bytes(process.readAllStandardError()).decode("utf-8", "replace")
            self._process = None
            process.deleteLater()
            self.refresh_button.setEnabled(True)
            self._update_switch_button()
            callback(exit_code == 0, out, err)

        process.finished.connect(finish)
        process.errorOccurred.connect(
            lambda error: finish(1) if error == QtCore.QProcess.ProcessError.FailedToStart else None)
        process.start(program, args)

    def _tick(self):
        self._render()
        self._update_switch_button()
        if time.monotonic() - self._last_scan >= 3.0:
            self.refresh()

    def refresh(self):
        if self._process is not None:
            return
        self._last_scan = time.monotonic()
        self._run("nmcli", ["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
                  self._on_status)

    def _on_status(self, ok: bool, out: str, err: str):
        if not ok:
            self._clear(f"无法读取 NetworkManager 状态：{err.strip() or 'nmcli 不可用'}")
            return
        self.interface = self.connection = ""
        for line in out.splitlines():
            parts = line.split(":", 3)
            if len(parts) == 4 and parts[1] == "wifi":
                self.interface = parts[0]
                self.connection = parts[3] if parts[2] == "connected" else ""
                break
        if not self.interface or not self.connection:
            self._clear("电脑 Wi-Fi 当前未连接或未开启热点。")
            return
        self._run("nmcli", ["-g", "802-11-wireless.mode", "connection", "show",
                            self.connection], self._on_mode)

    def _on_mode(self, ok: bool, out: str, err: str):
        if not ok:
            self._clear(f"无法确认 Wi-Fi 模式：{err.strip()}")
        elif out.strip() != "ap":
            self._clear(f"当前是 Wi-Fi 客户端连接（{self.connection}），不是电脑热点。")
        else:
            self.hotspot_active = True
            self._run("nmcli", ["-g", "IP4.ADDRESS", "device", "show", self.interface],
                      self._on_hotspot_ip)

    def _on_hotspot_ip(self, ok: bool, out: str, _err: str):
        self.hotspot_ip = out.splitlines()[0].split("/")[0].strip() if ok and out.strip() else ""
        self._run("iw", ["dev", self.interface, "station", "dump"], self._on_stations)

    def _on_stations(self, ok: bool, out: str, err: str):
        if not ok:
            self._clear(f"无法读取热点关联终端：{err.strip() or '请确认 iw 可用'}")
            return
        self.stations = parse_stations(out)
        self._run("ip", ["-4", "neigh", "show", "dev", self.interface], self._on_neighbors)

    def _on_neighbors(self, ok: bool, out: str, err: str):
        self.neighbors = parse_neighbors(out) if ok else {}
        self.status.setText(
            f"电脑热点 {self.connection}（{self.interface}）：{len(self.stations)} 个已关联终端"
            + (f"；IP/MAC 映射读取失败：{err.strip()}" if not ok else ""))
        self._render()

    def _clear(self, message: str):
        self.hotspot_active = False
        self.hotspot_ip = ""
        self.stations.clear()
        self.neighbors.clear()
        self.status.setText(message)
        self.table.setRowCount(0)
        self._update_switch_button()

    def _selected_device(self) -> tuple[str, str] | None:
        row = self.table.currentRow()
        if row < 0 or not self.table.selectionModel().isRowSelected(row, QtCore.QModelIndex()):
            return None
        id_item = self.table.item(row, 0)
        endpoint_item = self.table.item(row, 1)
        if id_item is None or endpoint_item is None or id_item.text() == "未识别":
            return None
        device_id = id_item.text()
        ip = endpoint_item.text().split(":", 1)[0]
        return device_id, ip

    def _eligible_device(self) -> tuple[str, str] | None:
        selected = self._selected_device()
        if (not selected or not self.hotspot_active or self._process is not None
                or time.monotonic() - self._last_scan > 5.0):
            return None
        device_id, ip = selected
        associated_ips = {self.neighbors[mac] for mac in self.stations if mac in self.neighbors}
        current = self.source.seen_endpoints(max_age=3.0).get(device_id)
        if not current or current[0] != ip or ip not in associated_ips:
            return None
        if (self._last_ap_command is not None and self._last_ap_command[0] == device_id
                and time.monotonic() - self._last_ap_command[1] < 5.0):
            return None
        return selected

    def _update_switch_button(self):
        self.switch_ap_button.setEnabled(self._eligible_device() is not None)

    def switch_selected_to_ap(self):
        selected = self._eligible_device()
        if selected is None:
            self.action_status.setText("所选设备已不在线或热点状态过期；请刷新并重新选择。")
            return
        device_id, ip = selected
        if self.can_switch is not None and not self.can_switch(device_id):
            return
        answer = QtWidgets.QMessageBox.question(
            self, "确认切回设备热点",
            f"只对 {device_id}（{ip}）发送切换到 AP/UDP 的指令。\n"
            "它将从电脑热点断开，并创建名称为该设备 ID 的 Wi-Fi；"
            "其他已连接设备不受影响。\n"
            "发送本身不能证明切换成功，需要之后用电脑连接该设备热点验证。确认继续？")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if self._eligible_device() != selected:
            self.action_status.setText("设备状态已变化，未发送；请刷新后重新选择。")
            return
        try:
            if self.source.device_ip(device_id) != ip:
                raise RuntimeError("设备 IP 已变化")
            self.source.switch_to_ap_udp(device_id)
        except (RuntimeError, OSError) as exc:
            self.action_status.setText(f"未能发送到 {device_id}：{exc}")
            return
        self._last_ap_command = (device_id, time.monotonic())
        self.action_status.setText(
            f"已向 {device_id} 发送 AP/UDP 切换指令。等待它从列表消失；"
            "可在另一台电脑的 Wi-Fi 列表中查找同名热点进行验证；本电脑热点无需关闭。"
            "UDP 无确认回执，列表消失也不能单独证明切换成功。")
        self._update_switch_button()

    def _render(self):
        if not self.hotspot_active:
            return
        selected = self._selected_device()
        endpoints = self.source.seen_endpoints(max_age=3.0)
        listen_port = self.source.sock.getsockname()[1]
        listen_ip = self.hotspot_ip or self.source.sock.getsockname()[0]
        rows = []
        for mac in self.stations:
            ip = self.neighbors.get(mac, "")
            matching = [(device_id, source_port) for device_id, (sender_ip, source_port, _seen)
                        in endpoints.items() if sender_ip == ip] if ip else []
            if matching:
                for device_id, source_port in matching:
                    packet = self.source.latest_packet(device_id)
                    if packet is None or packet.battery_percent is None:
                        charge = "未知"
                    else:
                        charge = f"{packet.battery_v:.2f} V（约 {packet.battery_percent}%）"
                    rows.append((device_id, f"{ip}:{source_port}",
                                 f"{listen_ip}:{listen_port}", mac, charge, "WT 数据在线"))
            else:
                rows.append(("未识别", f"{ip}:—" if ip else "IP 未知",
                             f"{listen_ip}:{listen_port}", mac, "未知", "已关联，未收到 WT 数据"))
        self.table.clearSelection()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(value)
                self.table.setItem(row_index, column, item)
            if selected is not None and row[0] == selected[0] and row[1].startswith(selected[1] + ":"):
                self.table.selectRow(row_index)
        self._update_switch_button()

    def closeEvent(self, event):
        if self._process is not None:
            self.status.setText("设备列表正在刷新，请稍候再关闭。")
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)
