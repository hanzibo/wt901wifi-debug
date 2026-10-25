"""Assign streaming WT devices on any reachable network to body-part slots."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtWidgets

from .device_list import parse_neighbors, parse_stations


class DeviceAssignmentDialog(QtWidgets.QDialog):
    def __init__(self, source, part_states, apply_assignments, parent=None):
        super().__init__(parent)
        self.source = source
        self.part_states = part_states
        self.apply_assignments = apply_assignments
        self.interface = ""
        self.connection = ""
        self.hotspot_active = False
        self.stations: list[str] = []
        self.neighbors: dict[str, str] = {}
        self.online_ids: set[str] = set()
        self._process: QtCore.QProcess | None = None
        self._last_scan = 0.0
        self._network_note = "正在检测网络"

        self.setWindowTitle("身体部位与 WT 设备分配")
        self.resize(560, 350)
        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("正在检测在线 WT 设备…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.form = QtWidgets.QFormLayout()
        self.combos: dict[str, QtWidgets.QComboBox] = {}
        for key, state in self.part_states.items():
            combo = QtWidgets.QComboBox()
            self.combos[key] = combo
            self.form.addRow(state.name, combo)
        layout.addLayout(self.form)
        self.hint = QtWidgets.QLabel(
            "候选列表包含最近 3 秒收到有效 WT 数据的设备 ID；支持电脑热点、外部 Wi-Fi 和局域网。"
            "同一台设备只能分配给一个部位；换绑后会重建该部位的中立位。")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        layout.addStretch(1)
        actions = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("刷新设备")
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_button)
        actions.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("应用分配")
        self.apply_button.clicked.connect(self._apply)
        actions.addWidget(self.apply_button)
        close_button = QtWidgets.QPushButton("取消")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._fill_choices()
        QtCore.QTimer.singleShot(0, self.refresh)

    def _run(self, program: str, args: list[str], callback):
        if self._process is not None:
            return
        process = QtCore.QProcess(self)
        self._process = process
        self.refresh_button.setEnabled(False)

        def finish(exit_code: int, _exit_status=None):
            if self._process is not process:
                return
            out = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
            err = bytes(process.readAllStandardError()).decode("utf-8", "replace")
            self._process = None
            process.deleteLater()
            self.refresh_button.setEnabled(True)
            callback(exit_code == 0, out, err)

        process.finished.connect(finish)
        process.errorOccurred.connect(
            lambda error: finish(1) if error == QtCore.QProcess.ProcessError.FailedToStart else None)
        process.start(program, args)

    def _tick(self):
        self._fill_choices()
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
            self._clear("电脑 Wi-Fi 未连接；仍可使用其他接口收到的 WT 数据。")
            return
        self._run("nmcli", ["-g", "802-11-wireless.mode", "connection", "show",
                            self.connection], self._on_mode)

    def _on_mode(self, ok: bool, out: str, err: str):
        if not ok:
            self._clear(f"无法确认 Wi-Fi 模式：{err.strip()}")
        elif out.strip() != "ap":
            self._clear(f"外部 Wi-Fi：{self.connection}。")
        else:
            self.hotspot_active = True
            self._network_note = f"电脑热点：{self.connection}"
            self._run("iw", ["dev", self.interface, "station", "dump"], self._on_stations)

    def _on_stations(self, ok: bool, out: str, err: str):
        if not ok:
            self._clear(f"无法读取热点关联终端：{err.strip() or '请确认 iw 可用'}")
            return
        self.stations = parse_stations(out)
        self._run("ip", ["-4", "neigh", "show", "dev", self.interface], self._on_neighbors)

    def _on_neighbors(self, ok: bool, out: str, err: str):
        if not ok:
            self._clear(f"无法匹配热点设备 IP：{err.strip() or '请确认 ip 可用'}")
            return
        self.neighbors = parse_neighbors(out)
        self._fill_choices()

    def _clear(self, message: str):
        self.hotspot_active = False
        self.stations.clear()
        self.neighbors.clear()
        self._network_note = message
        self._fill_choices()

    def _fill_choices(self):
        old_choices = {key: combo.currentData() for key, combo in self.combos.items()}
        self.online_ids = set(self.source.seen_endpoints(max_age=3.0))
        self.status.setText(f"{self._network_note}；{len(self.online_ids)} 台 WT 设备正在发送数据。")
        for key, combo in self.combos.items():
            chosen = old_choices.get(key)
            state = self.part_states[key]
            combo.blockSignals(True)
            combo.clear()
            current = state.device_id or "未分配"
            combo.addItem(f"保持当前：{current}", None)
            combo.addItem("不分配", "")
            for device_id in sorted(self.online_ids):
                combo.addItem(device_id, device_id)
            if chosen and chosen not in self.online_ids:
                # Keep the user's pending selection visible, but reject it at apply.
                combo.addItem(f"{chosen}（已离线）", chosen)
            index = combo.findData(chosen)
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)

    def _apply(self):
        if self._process is not None:
            self.status.setText("正在刷新网络状态，请稍候再应用分配。")
            return
        # Recheck at click time, rather than trusting the timer's cached candidates.
        self.online_ids = set(self.source.seen_endpoints(max_age=3.0))
        assignments = {}
        for key, combo in self.combos.items():
            selected = combo.currentData()
            if selected is None:
                selected = self.part_states[key].device_id
            elif selected and selected not in self.online_ids:
                self.status.setText(f"设备 {selected} 已不在线，请刷新后重新选择。")
                return
            assignments[key] = selected
        used = [device_id for device_id in assignments.values() if device_id]
        if len(used) != len(set(used)):
            self.status.setText("一台设备不能同时分配给多个身体部位；请调整选择。")
            return
        if self.apply_assignments(assignments):
            self.timer.stop()
            self.accept()

    def reset_choices(self):
        """Discard edits from a previous opening and show the current bindings."""
        for combo in self.combos.values():
            combo.setCurrentIndex(0)
        self._fill_choices()

    def closeEvent(self, event):
        if self._process is not None:
            self.status.setText("设备列表正在刷新，请稍候再关闭。")
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)
