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


from .body_panel import build_body_panel
from .body_view import BodyView
from .body_device_controller import BodyDeviceController


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
        self.real_robot_dialog = None
        self.gripper_gesture_dialog = None
        self.recorder: PacketRecorder | None = None
        self.replay_mode = isinstance(source, ReplaySource) or getattr(source, "is_replay", False)

        self._body_view = BodyView(self, model_dir=MODEL_DIR, triad_factory=_axis_triad)
        self._device_controller = BodyDeviceController(
            self, save=lambda *a, **k: save_config(*a, **k),
            mapping_dialog_factory=lambda *a, **k: AxisMappingDialog(*a, **k),
            warn=lambda title, message: QtWidgets.QMessageBox.warning(self, title, message),
            accepted_code=QtWidgets.QDialog.DialogCode.Accepted)
        self.setWindowTitle("WT901 全身姿态映射 · Ubuntu")
        self.resize(1500, 920)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 4)
        self._load_model()

        build_body_panel(self, layout)

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
        return self._body_view._load_model()

    def _front_view(self) -> None:
        return self._body_view._front_view()

    def _render_pose(self) -> None:
        return self._body_view._render_pose()

    def _toggle_axes(self, visible: bool) -> None:
        return self._body_view._toggle_axes(visible)

    def _poll(self) -> None:
        poll_started=time.monotonic()
        from .real_robot_timing import acquisition_poll, acquisition_feed
        acquisition_poll(self,poll_started)
        try:
            timed_packets = (self.source.poll_timed() if hasattr(self.source, 'poll_timed') else
                             [(time.monotonic_ns(), packet) for packet in self.source.poll()])
        except OSError as exc:
            self.poll_timer.stop()
            self.statusBar().showMessage(f"UDP 接收失败：{exc}")
            return
        packets = [packet for _, packet in timed_packets]
        acquisition_poll(self,poll_started,timed_packets)
        for received_ns, packet in timed_packets:
            self._check_calibration_reconnect(packet.device_id, time.monotonic())
            feed_started = time.monotonic()
            self.engine.feed(packet)
            acquisition_feed(self,packet,received_ns,feed_started)
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
        return self._device_controller._check_calibration_reconnect(device_id, received_at)

    def _refresh_choices(self, force: bool = False) -> None:
        return self._device_controller._refresh_choices(force)

    def _assignment_changed(self, key: str) -> None:
        return self._device_controller._assignment_changed(key)

    def _update_mapping_button(self, key: str) -> None:
        return self._device_controller._update_mapping_button(key)

    def _show_axis_mapping(self, key: str) -> None:
        return self._device_controller._show_axis_mapping(key)

    def _algorithm_changed(self, key: str) -> None:
        return self._device_controller._algorithm_changed(key)

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
            self.gripper_gesture_dialog.hand_observation.connect(self._route_real_hand)
            for device, packet in self.engine.packets.items():
                self._feed_gripper_packet(round(self.engine.received_at[device]*1e9), packet)
        if self.real_robot_dialog is not None and self.real_robot_dialog.hand_session is not None:
            for side in ('left','right'):self.real_robot_dialog.hand_session.sync_policy(side)
        self.gripper_gesture_dialog.show()
        self.gripper_gesture_dialog.raise_()
        self.gripper_gesture_dialog.activateWindow()

    def _route_real_hand(self, side, data):
        dialog = self.real_robot_dialog
        if dialog is not None and dialog.hand_session is not None:
            dialog.hand_session.receive(side,data)

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

    def _show_real_robot(self) -> None:
        if self.replay_mode:
            return
        from .real_robot_arm_gui import RealRobotModesDialog as RealRobotDialog
        if self.real_robot_dialog is None:
            self.real_robot_dialog = RealRobotDialog(self)
        self.real_robot_dialog.show()
        self.real_robot_dialog.raise_()
        self.real_robot_dialog.activateWindow()

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
        if self.real_robot_dialog is not None and not self.real_robot_dialog.close():
            event.ignore()
            return
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
