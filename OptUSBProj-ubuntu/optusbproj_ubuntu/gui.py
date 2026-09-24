"""PySide6/PyVista upper-body viewer using the shared WT901 UDP source."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor

from wt901.gui_udp import GuiUdpSource
from wt901_viz.device_list import ConnectedDevicesDialog
from wt901_viz.onboarding import DeviceSetupDialog

from .pose import PoseEngine
from .recording import PacketRecorder, ReplaySource
from .settings import save_assignments
from .skeleton import PARTS, SENSOR_KEYS


MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models"


class BodyWindow(QtWidgets.QMainWindow):
    def __init__(self, source: GuiUdpSource | ReplaySource, settings_path: Path,
                 assignments: dict[str, str] | None = None) -> None:
        super().__init__()
        self.source = source
        self.settings_path = settings_path
        self.engine = PoseEngine(assignments)
        self.actors = []
        self.combos: dict[str, QtWidgets.QComboBox] = {}
        self.status_labels: dict[str, QtWidgets.QLabel] = {}
        self._render_dirty = True
        self._last_render = 0.0
        self._last_counts: dict[str, int] = {}
        self._last_stats_at = time.monotonic()
        self._choice_state: tuple[tuple[str, bool], ...] | None = None
        self._dialogs: list[QtWidgets.QDialog] = []
        self.recorder: PacketRecorder | None = None
        self.replay_mode = isinstance(source, ReplaySource)

        self.setWindowTitle("WT901 上身姿态映射 · Ubuntu")
        self.resize(1500, 920)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 4)
        self._load_model()

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(390)
        panel.setMaximumWidth(540)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(panel, 2)

        title = QtWidgets.QLabel("上身设备分配与姿态")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        panel_layout.addWidget(title)
        self.overview = QtWidgets.QLabel("等待 WT901 UDP 数据…")
        self.overview.setWordWrap(True)
        panel_layout.addWidget(self.overview)

        toolbar = QtWidgets.QHBoxLayout()
        for label, callback in (("热点设备", self._show_devices),
                                ("接入新设备", self._show_setup),
                                ("全部归零", self._zero_all)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            if self.replay_mode and label != "全部归零":
                button.setEnabled(False)
            toolbar.addWidget(button)
        panel_layout.addLayout(toolbar)
        self.record_button = QtWidgets.QPushButton("开始录制 CSV")
        self.record_button.clicked.connect(self._toggle_recording)
        self.record_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.record_button)

        group = QtWidgets.QGroupBox("设备 ID → 身体部位（可先只连接 3 台）")
        grid = QtWidgets.QGridLayout(group)
        grid.addWidget(QtWidgets.QLabel("部位"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("设备 ID"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("中立位"), 0, 2)
        for index, part in enumerate((p for p in PARTS if p.sensor_slot)):
            row = index * 2 + 1
            grid.addWidget(QtWidgets.QLabel(part.label), row, 0)
            combo = QtWidgets.QComboBox()
            combo.setMinimumWidth(185)
            combo.currentIndexChanged.connect(
                lambda _index, key=part.key: self._assignment_changed(key))
            grid.addWidget(combo, row, 1)
            self.combos[part.key] = combo
            zero = QtWidgets.QPushButton("归零")
            zero.clicked.connect(lambda _checked=False, key=part.key: self._zero(key))
            grid.addWidget(zero, row, 2)
            status = QtWidgets.QLabel("未分配")
            status.setWordWrap(True)
            grid.addWidget(status, row + 1, 0, 1, 3)
            self.status_labels[part.key] = status
        panel_layout.addWidget(group)
        refresh = QtWidgets.QPushButton("刷新在线设备列表")
        refresh.clicked.connect(self._refresh_choices)
        panel_layout.addWidget(refresh)

        hint = QtWidgets.QLabel(
            "模型共有 15 个部件，目前仅躯干和双臂 7 处可绑定 IMU；"
            "未绑定部位跟随父部位。每只设备首次收到数据时自动建立中立位。"
            "空格：全部归零；R：重置视角。\n"
            "本窗口占用 UDP 1399，运行时请关闭手腕控制台。")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)
        panel_layout.addStretch(1)
        self.statusBar().showMessage("就绪；等待设备数据")
        self._refresh_choices()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(20)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()
        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(500)
        self.stats_timer.timeout.connect(self._refresh_stats)
        self.stats_timer.start()

        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        shortcut.activated.connect(self._zero_all)
        reset_view = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        reset_view.activated.connect(self.plotter.reset_camera)

    def _load_model(self) -> None:
        missing = [part.mesh for part in PARTS if not (MODEL_DIR / part.mesh).is_file()]
        if missing:
            raise FileNotFoundError(f"缺少人体 STL 模型：{', '.join(missing)}")
        for part in PARTS:
            mesh = pv.read(MODEL_DIR / part.mesh)
            mesh.points *= 0.001  # Original mesh units are millimetres.
            actor = self.plotter.add_mesh(mesh, color=part.color,
                                          smooth_shading=False, show_edges=False)
            self.actors.append(actor)
        self.plotter.set_background("#1a1d23")
        self.plotter.show_axes()
        self._render_pose()
        self._front_view()

    def _front_view(self) -> None:
        self.plotter.camera_position = [(2.5, 0.0, 0.15), (0.0, 0.0, 0.15), (0.0, 0.0, 1.0)]
        self.plotter.reset_camera()

    def _render_pose(self) -> None:
        for actor, matrix in zip(self.actors, self.engine.transforms()):
            actor.user_matrix = matrix
        self.plotter.render()
        self._render_dirty = False
        self._last_render = time.monotonic()

    def _poll(self) -> None:
        try:
            packets = self.source.poll()
        except OSError as exc:
            self.poll_timer.stop()
            self.statusBar().showMessage(f"UDP 接收失败：{exc}")
            return
        for packet in packets:
            self.engine.feed(packet)
            if self.recorder is not None:
                try:
                    self.recorder.write(packet)
                except OSError as exc:
                    self._stop_recording()
                    self.statusBar().showMessage(f"录制中断：{exc}")
        if packets:
            self._render_dirty = True
        now = time.monotonic()
        if self._render_dirty and now - self._last_render >= 1 / 30:
            self._render_pose()

    def _refresh_choices(self, force: bool = False) -> None:
        seen = self.source.seen_devices(max_age=10.0)
        ids = sorted(set(seen) | {device_id for device_id in self.engine.assignments.values()
                                  if device_id})
        choice_state = tuple((device_id, device_id in seen) for device_id in ids)
        if not force and choice_state == self._choice_state:
            return
        self._choice_state = choice_state
        for key, combo in self.combos.items():
            selected = self.engine.assignments[key]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— 未分配 —", "")
            for device_id in ids:
                suffix = "" if device_id in seen else "（暂离线）"
                combo.addItem(device_id + suffix, device_id)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
            combo.blockSignals(False)

    def _assignment_changed(self, key: str) -> None:
        combo = self.combos[key]
        device_id = combo.currentData() or ""
        try:
            self.engine.assign(key, device_id)
            save_assignments(self.settings_path, self.engine.assignments)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "分配未保存", str(exc))
        self._refresh_choices(force=True)
        self._render_dirty = True

    def _zero(self, key: str) -> None:
        self.engine.zero(key)
        self._render_dirty = True

    def _zero_all(self) -> None:
        self.engine.zero()
        self._render_dirty = True
        self.statusBar().showMessage("已将在线设备当前姿态设为中立位", 3000)

    def _refresh_stats(self) -> None:
        now = time.monotonic()
        seen = self.source.seen_devices(max_age=3.0)
        endpoints = self.source.seen_endpoints(max_age=3.0)
        elapsed = max(0.001, now - self._last_stats_at)
        for key, label in self.status_labels.items():
            device_id = self.engine.assignments[key]
            if not device_id:
                label.setText("未分配；模型跟随父部位")
                continue
            packet = self.engine.packets.get(device_id)
            if packet is None:
                label.setText(f"{device_id}：等待数据")
                continue
            count = self.engine.counts.get(device_id, 0)
            rate = (count - self._last_counts.get(device_id, count)) / elapsed
            age = now - self.engine.received_at[device_id]
            state = "在线" if device_id in seen else f"断流 {age:.1f}s；保持末次姿态"
            ip = endpoints.get(device_id, ("?", 0, 0))[0]
            battery = packet.battery_percent
            charge = f"约 {battery}%" if battery is not None else "未知"
            label.setText(f"{device_id} · {state} · {rate:.0f} 包/s · {ip}\n"
                          f"角度 X {packet.roll:+.1f}° Y {packet.pitch:+.1f}° Z {packet.yaw:+.1f}°"
                          f" · 电池 {packet.battery_v:.2f}V/{charge} · RSSI {packet.rssi}dBm")
        self._last_counts = self.engine.counts.copy()
        self._last_stats_at = now
        self.overview.setText(
            f"当前收到 {len(seen)} 台 WT 设备；已绑定 "
            f"{sum(bool(value) for value in self.engine.assignments.values())}/7 个上身部位。"
            "支持少于 7 台先行测试。")
        self._refresh_choices()

    def _show_devices(self) -> None:
        dialog = ConnectedDevicesDialog(self.source, self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        self._dialogs.append(dialog)
        dialog.destroyed.connect(lambda: self._dialogs.remove(dialog)
                                 if dialog in self._dialogs else None)

    def _show_setup(self) -> None:
        dialog = DeviceSetupDialog(self.source, self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
        self._dialogs.append(dialog)
        dialog.destroyed.connect(lambda: self._dialogs.remove(dialog)
                                 if dialog in self._dialogs else None)

    def _toggle_recording(self) -> None:
        if self.recorder is not None:
            self._stop_recording()
            return
        default = (Path(__file__).resolve().parent.parent / "recordings" /
                   (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"))
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存 WT901 原始数据", str(default), "CSV 文件 (*.csv)")
        if not filename:
            return
        try:
            self.recorder = PacketRecorder(Path(filename))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "录制失败", str(exc))
            return
        self.record_button.setText("停止录制 CSV")
        self.statusBar().showMessage(f"正在录制：{filename}")

    def _stop_recording(self) -> None:
        if self.recorder is not None:
            count = self.recorder.count
            path = self.recorder.path
            self.recorder.close()
            self.recorder = None
            self.record_button.setText("开始录制 CSV")
            self.statusBar().showMessage(f"已保存 {count} 包：{path}", 5000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.poll_timer.stop()
        self.stats_timer.stop()
        self._stop_recording()
        self.source.close()
        self.plotter.close()
        super().closeEvent(event)
