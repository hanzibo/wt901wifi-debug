"""PySide6/PyVista full-body viewer using the shared WT901 UDP source."""

from __future__ import annotations

import time
import math
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor

from wt901.gui_udp import GuiUdpSource
from wt901_viz.device_list import ConnectedDevicesDialog
from wt901_viz.onboarding import DeviceSetupDialog

from .axis_mapping import DEFAULT_AXIS_MAPPING, mapping_matrix
from .device_settings import DeviceSettingsDialog
from .pose import GYRO_TILT, VENDOR_RPY, GYRO_RGBD, MODE_LABELS, PoseEngine
from .recording import PacketRecorder, ReplaySource
from .settings import save_config, load_prior_weight
from .skeleton import PARTS, SENSOR_KEYS


MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models"


def _axis_triad(length: float) -> pv.PolyData:
    """One RGB mesh for the three local axes, avoiding 45 extra VTK actors."""
    arrows = []
    for direction, rgb in (((1, 0, 0), (255, 50, 50)),
                           ((0, 1, 0), (50, 220, 50)),
                           ((0, 0, 1), (50, 120, 255))):
        arrow = pv.Arrow(direction=direction, scale=length,
                         tip_length=0.24, tip_radius=0.09,
                         shaft_radius=0.025)
        arrow.point_data["axis_rgb"] = np.tile(
            np.asarray(rgb, dtype=np.uint8), (arrow.n_points, 1))
        arrows.append(arrow)
    return arrows[0].merge(arrows[1], merge_points=False).merge(
        arrows[2], merge_points=False)


class AxisMappingDialog(QtWidgets.QDialog):
    """Choose a proper signed permutation for one neutral sensor frame."""

    CHOICES = (("+X", "模型 +X（前）"), ("-X", "模型 -X（后）"),
               ("+Y", "模型 +Y（左）"), ("-Y", "模型 -Y（右）"),
               ("+Z", "模型 +Z（上）"), ("-Z", "模型 -Z（下）"))

    def __init__(self, part_label: str, mapping: tuple[str, str, str],
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{part_label} · 设备轴映射")
        self.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(self)
        instruction = QtWidgets.QLabel(
            "设备固定后，先摆出模型默认姿势并归零。此时设备的每根正轴"
            "分别指向模型的哪个方向？负轴请选择带 - 的方向。")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        form = QtWidgets.QFormLayout()
        self.selectors: list[QtWidgets.QComboBox] = []
        for device_axis, initial in zip("XYZ", mapping):
            combo = QtWidgets.QComboBox()
            for value, label in self.CHOICES:
                combo.addItem(label, value)
            combo.setCurrentIndex(combo.findData(initial))
            combo.currentIndexChanged.connect(self._validate)
            form.addRow(f"设备 +{device_axis} 轴 →", combo)
            self.selectors.append(combo)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.ok_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("保存映射")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self._validate()

    def mapping(self) -> tuple[str, str, str]:
        return tuple(combo.currentData() for combo in self.selectors)

    def _validate(self, _index: int = 0) -> None:
        try:
            mapping_matrix(self.mapping())
        except ValueError as exc:
            self.message.setText(str(exc))
            self.ok_button.setEnabled(False)
        else:
            self.message.setText("映射有效。模型坐标轴将按此规则随设备转动。")
            self.ok_button.setEnabled(True)


class BodyWindow(QtWidgets.QMainWindow):
    def __init__(self, source: GuiUdpSource | ReplaySource, settings_path: Path,
                 assignments: dict[str, str] | None = None,
                 axis_mappings: dict[str, tuple[str, str, str]] | None = None,
                 algorithm_modes: dict[str, str] | None = None) -> None:
        super().__init__()
        self.source = source
        self.settings_path = settings_path
        self.wearing_prior_weight = load_prior_weight(settings_path)
        self._prior_spins = []
        self.engine = PoseEngine(assignments, axis_mappings, algorithm_modes)
        self.actors = []
        self.axis_actors = []
        self.combos: dict[str, QtWidgets.QComboBox] = {}
        self.algorithm_combos: dict[str, QtWidgets.QComboBox] = {}
        self.mapping_buttons: dict[str, QtWidgets.QPushButton] = {}
        self.status_labels: dict[str, QtWidgets.QLabel] = {}
        self._render_dirty = True
        self._last_render = 0.0
        self._last_counts: dict[str, int] = {}
        self._last_stats_at = time.monotonic()
        self._choice_state: tuple[tuple[str, bool], ...] | None = None
        self._dialogs: list[QtWidgets.QDialog] = []
        self.settings_dialog: DeviceSettingsDialog | None = None
        self.calibration_dialog = None
        self.natural_hand_dialog = None
        self.x1_preview_dialog = None
        self.scene_robot_dialog = None
        self.gripper_gesture_dialog = None
        self.recorder: PacketRecorder | None = None
        self.replay_mode = isinstance(source, ReplaySource) or getattr(source, "is_replay", False)

        self.setWindowTitle("WT901 全身姿态映射 · Ubuntu")
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

        title = QtWidgets.QLabel("全身设备分配与姿态")
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
                                ("设备参数", self._show_settings),
                                ("接入新设备", self._show_setup),
                                ("全部归零", self._zero_all)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            if self.replay_mode and label != "全部归零":
                button.setEnabled(False)
            toolbar.addWidget(button)
        panel_layout.addLayout(toolbar)
        timed_zero_row = QtWidgets.QHBoxLayout()
        timed_zero_row.addWidget(QtWidgets.QLabel("延时"))
        self.zero_delay = QtWidgets.QSpinBox()
        self.zero_delay.setRange(3, 120)
        self.zero_delay.setValue(10)
        self.zero_delay.setSuffix(" 秒")
        self.zero_delay.setToolTip("点击开始后，可在倒计时期间穿戴设备并摆好中立姿势")
        timed_zero_row.addWidget(self.zero_delay)
        self.timed_zero_button = QtWidgets.QPushButton("定时全部归零")
        self.timed_zero_button.clicked.connect(self._start_timed_zero)
        timed_zero_row.addWidget(self.timed_zero_button, 1)
        self.cancel_zero_button = QtWidgets.QPushButton("取消")
        self.cancel_zero_button.setEnabled(False)
        self.cancel_zero_button.clicked.connect(self._cancel_timed_zero)
        timed_zero_row.addWidget(self.cancel_zero_button)
        panel_layout.addLayout(timed_zero_row)
        self.zero_countdown = QtWidgets.QLabel("定时归零未启动")
        self.zero_countdown.setWordWrap(True)
        panel_layout.addWidget(self.zero_countdown)
        self.calibration_button = QtWidgets.QPushButton("穿戴标定…（躯干 / 双臂 / 双手）")
        self.calibration_button.clicked.connect(self._show_calibration)
        self.calibration_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.calibration_button)
        self.bilateral_calibration_button = QtWidgets.QPushButton("双臂联合标定…（左右上臂 / 前臂）")
        self.bilateral_calibration_button.clicked.connect(self._show_bilateral_calibration)
        self.bilateral_calibration_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.bilateral_calibration_button)
        self.wearing_options_button = QtWidgets.QPushButton('穿戴标定参数…（弱先验 λ）')
        self.wearing_options_button.clicked.connect(self._show_wearing_options)
        self.wearing_options_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.wearing_options_button)
        self.natural_hand_button=QtWidgets.QPushButton('左手自然掌面参考…（手动原型）')
        self.natural_hand_button.clicked.connect(self._show_natural_hand)
        self.natural_hand_button.setEnabled(not self.replay_mode);panel_layout.addWidget(self.natural_hand_button)
        self.x1_preview_button = QtWidgets.QPushButton("X1 双臂实时预览…（RViz）")
        self.x1_preview_button.clicked.connect(self._show_x1_preview)
        self.x1_preview_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.x1_preview_button)
        self.scene_robot_button = QtWidgets.QPushButton("新机器人双臂实时预览…（CoppeliaSim）")
        self.scene_robot_button.clicked.connect(self._show_scene_robot)
        self.scene_robot_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.scene_robot_button)
        self.gripper_gesture_button = QtWidgets.QPushButton('左右手 IMU 夹爪配置…（四设备 / RViz）')
        self.gripper_gesture_button.clicked.connect(self._show_gripper_gesture)
        self.gripper_gesture_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.gripper_gesture_button)
        self._timed_zero_deadline: float | None = None
        self.zero_countdown_timer = QtCore.QTimer(self)
        self.zero_countdown_timer.setInterval(100)
        self.zero_countdown_timer.timeout.connect(self._update_zero_countdown)
        self.axes_checkbox = QtWidgets.QCheckBox("显示部位坐标轴（红 X · 绿 Y · 蓝 Z）")
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.toggled.connect(self._toggle_axes)
        panel_layout.addWidget(self.axes_checkbox)
        self.record_button = QtWidgets.QPushButton("开始录制 CSV")
        self.record_button.clicked.connect(self._toggle_recording)
        self.record_button.setEnabled(not self.replay_mode)
        panel_layout.addWidget(self.record_button)

        group = QtWidgets.QGroupBox("设备 ID → 身体部位（支持部分设备先行测试）")
        grid = QtWidgets.QGridLayout(group)
        section_color = ("#76D7FF" if group.palette().color(
            QtGui.QPalette.ColorRole.Window).lightness() < 128 else "#006D9B")
        grid.addWidget(QtWidgets.QLabel("部位"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("轴映射"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("设备 ID"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("中立位"), 0, 3)
        row = 1
        section = ""
        for part in (p for p in PARTS if p.sensor_slot):
            next_section = ("骨盆 / 躯干 / 头部" if part.key in ("pelvis", "torso", "head")
                            else "左臂" if part.key.startswith("left_") and
                            part.key in ("left_upper_arm", "left_forearm", "left_hand")
                            else "右臂" if part.key.startswith("right_") and
                            part.key in ("right_upper_arm", "right_forearm", "right_hand")
                            else "左腿" if part.key.startswith("left_") else "右腿")
            if next_section != section:
                section = next_section
                heading = QtWidgets.QLabel(section)
                heading.setObjectName("bodySectionHeading")
                heading.setStyleSheet(
                    f"color: {section_color}; font-weight: bold; "
                    "padding-top: 10px; padding-bottom: 2px;")
                grid.addWidget(heading, row, 0, 1, 4)
                row += 1
            grid.addWidget(QtWidgets.QLabel(part.label), row, 0)
            mapping_button = QtWidgets.QPushButton("轴映射")
            mapping_button.clicked.connect(
                lambda _checked=False, key=part.key: self._show_axis_mapping(key))
            grid.addWidget(mapping_button, row, 1)
            self.mapping_buttons[part.key] = mapping_button
            self._update_mapping_button(part.key)
            combo = QtWidgets.QComboBox()
            combo.setMinimumWidth(165)
            combo.currentIndexChanged.connect(
                lambda _index, key=part.key: self._assignment_changed(key))
            grid.addWidget(combo, row, 2)
            self.combos[part.key] = combo
            zero = QtWidgets.QPushButton("归零")
            zero.clicked.connect(lambda _checked=False, key=part.key: self._zero(key))
            grid.addWidget(zero, row, 3)
            status = QtWidgets.QLabel("未分配")
            status.setWordWrap(True)
            algorithm = QtWidgets.QComboBox()
            algorithm.addItem("厂商 R/P/Y", VENDOR_RPY)
            algorithm.addItem("陀螺＋加计", GYRO_TILT)
            if part.key in ('left_upper_arm','left_forearm','right_upper_arm','right_forearm'):
                algorithm.addItem("陀螺＋加计＋RGB-D（需相机标定）",GYRO_RGBD)
            algorithm.setCurrentIndex(algorithm.findData(self.engine.algorithm_modes[part.key]))
            algorithm.setToolTip("仅切换本部位软件姿态估计；不改变设备的 6/9 轴寄存器设置")
            algorithm.currentIndexChanged.connect(
                lambda _index, key=part.key: self._algorithm_changed(key))
            grid.addWidget(algorithm, row + 1, 0, 1, 2)
            self.algorithm_combos[part.key] = algorithm
            grid.addWidget(status, row + 1, 2, 1, 2)
            self.status_labels[part.key] = status
            row += 2
        assignment_scroll = QtWidgets.QScrollArea()
        assignment_scroll.setWidgetResizable(True)
        assignment_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        assignment_scroll.setWidget(group)
        assignment_scroll.setMinimumHeight(280)
        panel_layout.addWidget(assignment_scroll, 1)
        refresh = QtWidgets.QPushButton("刷新在线设备列表")
        refresh.clicked.connect(self._refresh_choices)
        panel_layout.addWidget(refresh)

        hint = QtWidgets.QLabel(
            "模型 15 个部件均可绑定 IMU；不必一次接满。"
            "未绑定部位跟随父部位。每只设备首次收到数据时自动建立中立位。"
            "彩色箭头是模型各部位的局部轴，不是 WT901 外壳上的轴。"
            "归零只重设姿态基准，不改变设备轴映射；更换绑法后需重新设置或标定。"
            "空格：立即全部归零；定时归零可留出穿戴和摆姿势的时间；R：重置视角。\n"
            "本窗口占用 UDP 1399，运行时请关闭手腕控制台。")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)
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
        toggle_axes = QtGui.QShortcut(QtGui.QKeySequence("A"), self)
        toggle_axes.activated.connect(self.axes_checkbox.toggle)

    def _load_model(self) -> None:
        missing = [part.mesh for part in PARTS if not (MODEL_DIR / part.mesh).is_file()]
        if missing:
            raise FileNotFoundError(f"缺少人体 STL 模型：{', '.join(missing)}")
        triad = _axis_triad(0.105)
        for part in PARTS:
            mesh = pv.read(MODEL_DIR / part.mesh)
            mesh.points *= 0.001  # Original mesh units are millimetres.
            actor = self.plotter.add_mesh(mesh, color=part.color,
                                          smooth_shading=False, show_edges=False)
            self.actors.append(actor)
            axis_actor = self.plotter.add_mesh(
                triad, scalars="axis_rgb", rgb=True, lighting=False,
                show_scalar_bar=False, pickable=False)
            self.axis_actors.append(axis_actor)
        self.plotter.set_background("#1a1d23")
        self.plotter.show_axes()
        self._render_pose()
        self._front_view()

    def _front_view(self) -> None:
        self.plotter.camera_position = [(2.5, 0.0, 0.15), (0.0, 0.0, 0.15), (0.0, 0.0, 1.0)]
        self.plotter.reset_camera()

    def _render_pose(self) -> None:
        for actor, axis_actor, matrix in zip(
                self.actors, self.axis_actors, self.engine.transforms()):
            actor.user_matrix = matrix
            axis_actor.user_matrix = matrix
        self.plotter.render()
        self._render_dirty = False
        self._last_render = time.monotonic()

    def _toggle_axes(self, visible: bool) -> None:
        for actor in self.axis_actors:
            actor.SetVisibility(visible)
        self._render_dirty = True

    def _poll(self) -> None:
        poll_started=time.monotonic()
        try:
            timed_packets = (self.source.poll_timed() if hasattr(self.source, 'poll_timed') else
                             [(time.monotonic_ns(), packet) for packet in self.source.poll()])
        except OSError as exc:
            self.poll_timer.stop()
            self.statusBar().showMessage(f"UDP 接收失败：{exc}")
            return
        packets = [packet for _, packet in timed_packets]
        for received_ns, packet in timed_packets:
            self._check_calibration_reconnect(packet.device_id, time.monotonic())
            self.engine.feed(packet)
            self._feed_gripper_packet(received_ns, packet)
            if self.calibration_dialog is not None:
                self.calibration_dialog.on_packet(time.monotonic_ns(), packet)
            if self.recorder is not None:
                try:
                    self.recorder.write(packet)
                except OSError as exc:
                    self._stop_recording()
                    self.statusBar().showMessage(f"录制中断：{exc}")
        if self.settings_dialog is not None or self.calibration_dialog is not None:
            replies = self.source.take_replies()
        else:
            replies = []
        if self.settings_dialog is not None:
            self.settings_dialog.on_packets(packets)
            self.settings_dialog.on_replies(replies)
        if self.calibration_dialog is not None:
            self.calibration_dialog.on_replies(replies)
        if packets:
            self._render_dirty = True
        now = time.monotonic()
        if self._render_dirty and now - self._last_render >= 1 / 30:
            self._render_pose()

        if hasattr(self.source,'diagnostic_event'):
            self.source.diagnostic_event('gui_poll_end',duration_ms=(time.monotonic()-poll_started)*1000,
                                         packets=len(timed_packets))

    def _check_calibration_reconnect(self, device_id: str, received_at: float) -> None:
        if self.replay_mode:
            return
        previous = self.engine.received_at.get(device_id)
        if previous is None or received_at - previous <= 5.0:
            return
        self.engine.invalidate_neutral(device_id)
        for key, assigned in self.engine.assignments.items():
            if assigned == device_id and key in self.engine.continuous_basis:
                self.engine.clear_calibration(key)
                if self.calibration_dialog is not None:
                    self.calibration_dialog.invalidate_for(device_id)
                self.statusBar().showMessage(
                    f"{device_id} 断流超过 5 秒后恢复；穿戴标定已停用，请重新验证", 10000)

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
        previous = self.engine.assignments[key]
        try:
            proposed = dict(self.engine.assignments)
            proposed[key] = device_id
            save_config(self.settings_path, proposed, self.engine.axis_mappings,
                        self.engine.algorithm_modes)
            self.engine.assign(key, device_id)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "分配未保存", str(exc))
        else:
            if previous != device_id and self.calibration_dialog is not None:
                self.calibration_dialog.invalidate_for(previous)
        self._refresh_choices(force=True)
        self._render_dirty = True

    def _update_mapping_button(self, key: str) -> None:
        mapping = self.engine.axis_mappings[key]
        button = self.mapping_buttons[key]
        button.setText("轴映射*" if mapping != DEFAULT_AXIS_MAPPING else "轴映射")
        button.setToolTip("；".join(f"设备 {axis} → 模型 {target}"
                                for axis, target in zip("XYZ", mapping)))

    def _show_axis_mapping(self, key: str) -> None:
        part = next(part for part in PARTS if part.key == key)
        dialog = AxisMappingDialog(part.label, self.engine.axis_mappings[key], self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            proposed = dict(self.engine.axis_mappings)
            proposed[key] = dialog.mapping()
            save_config(self.settings_path, self.engine.assignments,
                        proposed, self.engine.algorithm_modes)
            self.engine.set_axis_mapping(key, dialog.mapping())
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "轴映射未保存", str(exc))
            return
        if self.calibration_dialog is not None:
            self.calibration_dialog.invalidate_for(self.engine.assignments.get(key, ""))
        self._update_mapping_button(key)
        self._render_dirty = True

    def _algorithm_changed(self, key: str) -> None:
        combo = self.algorithm_combos[key]
        mode = combo.currentData()
        previous = self.engine.algorithm_modes[key]
        if mode == previous:
            return
        if (self.recorder is not None or
                getattr(self, "session", None) is not None):
            error = "录制期间不可切换姿态算法；请先结束录制"
        elif self.calibration_dialog is not None and self.calibration_dialog.isVisible():
            error = "穿戴标定窗口打开时不可切换姿态算法；请先关闭标定窗口"
        else:
            error = ""
        try:
            if error:
                raise ValueError(error)
            self.engine.set_algorithm_mode(key, mode)
            save_config(self.settings_path, self.engine.assignments,
                        self.engine.axis_mappings, self.engine.algorithm_modes)
        except (ValueError, OSError) as exc:
            self.engine.set_algorithm_mode(key, previous)
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(previous))
            combo.blockSignals(False)
            QtWidgets.QMessageBox.warning(self, "姿态算法未切换", str(exc))
            return
        self._render_dirty = True
        self.statusBar().showMessage(
            f"{key} 已切换为 {MODE_LABELS[mode]}；"
            "设备融合寄存器未改变", 5000)

    def _zero(self, key: str) -> None:
        self.engine.zero(key)
        self._render_dirty = True

    def _show_calibration(self) -> None:
        self._open_calibration(bilateral=False)

    def set_wearing_prior_weight(self, value):
        from .calibration_options import validate_prior_weight
        value = validate_prior_weight(value)
        save_config(self.settings_path, self.engine.assignments, self.engine.axis_mappings,
                    self.engine.algorithm_modes, prior_weight=value)
        self.wearing_prior_weight = value
        for spin in self._prior_spins:
            if not spin.isEnabled():continue
            spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
            spin.update_prior_hint(value)

    def _show_wearing_options(self):
        from .calibration_options import add_prior_control
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('穿戴标定参数')
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel('所有穿戴窗口共用；本轮开始后参数冻结，不修改已经应用的映射。'))
        add_prior_control(dialog, self)
        close = QtWidgets.QPushButton('关闭'); close.clicked.connect(dialog.accept)
        layout.addWidget(close); dialog.exec(); dialog.deleteLater()

    def _show_natural_hand(self):
        if self.replay_mode:return
        from .natural_hand_gui import NaturalHandDialog
        if self.natural_hand_dialog is not None:
            self.natural_hand_dialog.close();self.natural_hand_dialog.deleteLater()
        self.natural_hand_dialog=NaturalHandDialog(self)
        self.natural_hand_dialog.show();self.natural_hand_dialog.raise_()

    def _feed_gripper_packet(self, received_ns, packet) -> None:
        if self.gripper_gesture_dialog is not None:
            self.gripper_gesture_dialog.feed_host_packet(received_ns, packet, self.engine)

    def _show_gripper_gesture(self) -> None:
        if self.replay_mode:
            return
        from .gripper_gesture_gui import BilateralGestureWindow, SharedGestureSource
        if self.gripper_gesture_dialog is None:
            self.gripper_gesture_dialog = BilateralGestureWindow(
                SharedGestureSource(), Path(__file__).resolve().parents[1]/'recordings/imu_gripper', self)
            self.gripper_gesture_dialog.hand_output.connect(self._route_scene_hand)
            for device, packet in self.engine.packets.items():
                self._feed_gripper_packet(round(self.engine.received_at[device]*1e9), packet)
        self.gripper_gesture_dialog.show()
        self.gripper_gesture_dialog.raise_()
        self.gripper_gesture_dialog.activateWindow()

    def _route_scene_hand(self,side,data):
        if self.scene_robot_dialog is not None:
            self.scene_robot_dialog.receive_hand(side,data)
        elif data.get('enabled') and self.gripper_gesture_dialog is not None:
            panel=self.gripper_gesture_dialog.panels[side]
            panel.scene_status.setText('请先打开并连接新机器人窗口，配置手型后重新启用')
            panel.send_scene.setChecked(False)

    def _show_scene_robot(self) -> None:
        if self.replay_mode:
            return
        from .scene_robot_gui import SceneRobotDialog
        if self.scene_robot_dialog is None:
            self.scene_robot_dialog = SceneRobotDialog(self)
        self.scene_robot_dialog.show()
        self.scene_robot_dialog.raise_()
        self.scene_robot_dialog.activateWindow()

    def _show_x1_preview(self) -> None:
        if self.replay_mode:
            return
        from .x1_live_gui import X1PreviewDialog
        if self.x1_preview_dialog is None:
            self.x1_preview_dialog = X1PreviewDialog(self)
        self.x1_preview_dialog.show()
        self.x1_preview_dialog.raise_()
        self.x1_preview_dialog.activateWindow()

    def _show_bilateral_calibration(self) -> None:
        self._open_calibration(bilateral=True)

    def _open_calibration(self, *, bilateral: bool) -> None:
        if self.replay_mode:
            return
        if self.settings_dialog is not None and self.settings_dialog.isVisible():
            QtWidgets.QMessageBox.information(
                self, "请先关闭设备参数", "设备参数窗口正在使用寄存器回读；请关闭后再进行穿戴标定。")
            return
        from .calibration_gui import BilateralCalibrationDialog, GuidedCalibrationDialog

        dialog_type = BilateralCalibrationDialog if bilateral else GuidedCalibrationDialog
        if self.calibration_dialog is not None and type(self.calibration_dialog) is not dialog_type:
            self.calibration_dialog.close()
            self.calibration_dialog.deleteLater()
            self.calibration_dialog = None
        if self.calibration_dialog is None:
            self.calibration_dialog = dialog_type(self)
        elif bilateral:
            self.calibration_dialog.refresh_devices()
        else:
            self.calibration_dialog.refresh_parts()
        self.calibration_dialog.show()
        self.calibration_dialog.raise_()
        self.calibration_dialog.activateWindow()

    def _zero_all(self) -> None:
        self._cancel_timed_zero(silent=True)
        self.engine.zero()
        self._render_dirty = True
        self.statusBar().showMessage("已立即将当前设备姿态设为中立位", 3000)

    def _start_timed_zero(self) -> None:
        if not any(self.engine.assignments.values()):
            self.zero_countdown.setText("尚未分配设备；请先绑定至少一个身体部位")
            return
        seconds = self.zero_delay.value()
        self._timed_zero_deadline = time.monotonic() + seconds
        self.zero_countdown_timer.start()
        self.cancel_zero_button.setEnabled(True)
        self._update_zero_countdown()

    def _cancel_timed_zero(self, _checked: bool = False, *, silent: bool = False) -> None:
        if self._timed_zero_deadline is None:
            return
        self._timed_zero_deadline = None
        self.zero_countdown_timer.stop()
        self.cancel_zero_button.setEnabled(False)
        self.zero_countdown.setText("定时归零已取消" if not silent else "定时归零未启动")

    def _update_zero_countdown(self) -> None:
        if self._timed_zero_deadline is None:
            return
        remaining = math.ceil(self._timed_zero_deadline - time.monotonic())
        if remaining > 0:
            self.zero_countdown.setText(
                f"还剩 {remaining} 秒：请穿戴好设备并保持中立姿势；到时自动归零")
            return
        self._timed_zero_deadline = None
        self.zero_countdown_timer.stop()
        self.cancel_zero_button.setEnabled(False)
        online = self.source.seen_devices(max_age=2.0)
        assigned = tuple((key, device_id) for key, device_id in
                         self.engine.assignments.items() if device_id)
        completed = 0
        for key, device_id in assigned:
            if device_id in online and device_id in self.engine.current:
                self.engine.zero(key)
                completed += 1
        if completed:
            self._render_dirty = True
        message = (f"定时归零完成：{completed}/{len(assigned)} 台在线设备已设为中立位"
                   if completed else "倒计时结束，但没有已绑定设备在接收新数据；未归零")
        if 0 < completed < len(assigned):
            message += "；离线设备已跳过"
        self.zero_countdown.setText(message)
        self.statusBar().showMessage(message, 5000)

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
            calibrated = " · 本次穿戴标定生效" if key in self.engine.continuous_basis else ""
            natural=self.engine.natural_hand_previews.get(key,self.engine.natural_hand_references.get(key))
            if natural:
                if key in self.engine.natural_hand_previews:calibrated+=' · 自然掌面临时预览（X1不变）'
                elif natural['enabled'] and self.engine.natural_hand_error(natural) is None:calibrated+=' · 自然掌面显示（X1仍标准化）'
                else:calibrated+=' · 自然参考已保留，当前标准化显示'
            estimator = self.engine._tilt_filters.get(device_id)
            mode_label = ("陀螺＋加计" +
                          ("（静止学习偏置中）" if estimator is not None
                           and not estimator.bias_ready else "")
                          if self.engine.algorithm_modes[key] == GYRO_TILT
                          else MODE_LABELS[self.engine.algorithm_modes[key]])
            label.setText(f"{device_id} · {state} · {rate:.0f} 包/s · {ip}\n"
                          f"设备原始 R/P/Y {packet.roll:+.1f}° / {packet.pitch:+.1f}° / {packet.yaw:+.1f}°"
                          f" · 电池 {packet.battery_v:.2f}V/{charge} · RSSI {packet.rssi}dBm"
                          f" · 模型算法 {mode_label}{calibrated}")
        self._last_counts = self.engine.counts.copy()
        self._last_stats_at = now
        self.overview.setText(
            f"当前收到 {len(seen)} 台 WT 设备；已绑定 "
            f"{sum(bool(value) for value in self.engine.assignments.values())}/{len(SENSOR_KEYS)} 个身体部位。"
            "未绑定部位跟随父部位；可按需逐步增加设备。")
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

    def _show_settings(self) -> None:
        if self.replay_mode:
            return
        if self.settings_dialog is None:
            dialog = DeviceSettingsDialog(self.source, self)
            dialog.neutral_required.connect(self._zero_device_after_setting)
            dialog.finished.connect(lambda: setattr(self, "settings_dialog", None))
            self.settings_dialog = dialog
        self.settings_dialog.refresh_devices()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _zero_device_after_setting(self, device_id: str) -> None:
        for key, assigned in self.engine.assignments.items():
            if assigned == device_id:
                was_calibrated = key in self.engine.continuous_basis
                self.engine.clear_calibration(key)
                # The firmware may change its reported reference frame after
                # a setting/zero command. Do not use the last *old-frame*
                # packet as the new GUI neutral; wait for the next packet.
                self.engine.invalidate_neutral(device_id)
                self._render_dirty = True
                if self.calibration_dialog is not None:
                    self.calibration_dialog.invalidate_for(device_id)
                self.statusBar().showMessage(
                    f"{device_id}：等待下一包数据建立新的模型中立位"
                    + ("；设备配置改变，穿戴标定已失效" if was_calibrated else ""), 5000)

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
            metadata = {
                "format": "wt901-body-csv-settings-v1",
                "assignments": dict(self.engine.assignments),
                "axis_mappings": {key: list(value) for key, value in
                                  self.engine.axis_mappings.items()},
                "algorithm_modes": dict(self.engine.algorithm_modes),
            }
            Path(filename + ".config.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError as exc:
            if self.recorder is not None:
                self.recorder.close()
                self.recorder = None
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
        if (self.calibration_dialog is not None and
                self.calibration_dialog.phase in ("prepare", "hold")):
            self.calibration_dialog.show()
            self.calibration_dialog.raise_()
            self.calibration_dialog.progress.setText(
                "请先取消或完成穿戴标定，再关闭主窗口")
            event.ignore()
            return
        if (self.settings_dialog is not None and
                self.settings_dialog.calibration not in (None, "mag_precheck")):
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.status.setText("请先结束设备校准并保存，再关闭主窗口")
            event.ignore()
            return
        if self.settings_dialog is not None and self.settings_dialog.setting is not None:
            self.settings_dialog.show()
            self.settings_dialog.raise_()
            self.settings_dialog.status.setText("设备设置正在保存或回读，请等待完成")
            event.ignore()
            return
        if self.scene_robot_dialog is not None and not self.scene_robot_dialog.close():
            event.ignore()
            return
        self.poll_timer.stop()
        self.stats_timer.stop()
        self.zero_countdown_timer.stop()
        if self.gripper_gesture_dialog is not None:
            self.gripper_gesture_dialog.shutdown()
            self.gripper_gesture_dialog.hide()
        if self.natural_hand_dialog is not None:self.natural_hand_dialog.close()
        if self.x1_preview_dialog is not None:
            self.x1_preview_dialog.close()
        self._stop_recording()
        self.source.close()
        self.plotter.close()
        super().closeEvent(event)
