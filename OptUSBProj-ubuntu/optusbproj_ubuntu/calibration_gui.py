"""Guided, unattended IMU wearing calibration in a separate window."""

from __future__ import annotations

from pathlib import Path
import json
import time

from PySide6 import QtCore, QtWidgets

from wt901.registers import REG_ORIENT

from .fusion_recording import FusionSessionRecorder
from .guided_calibration import (
    ARM_PARTS, HAND_PARTS, CALIBRATABLE_PARTS, STEPS, fit_guided_arm, fit_guided_torso)
from .skeleton import PART_BY_KEY, PARTS
from .wearing_protocol import (_attach_orientation_streams, STEP_LABELS, TORSO_STEP_LABELS,
    LEFT_WRIST_STEP_LABELS, RIGHT_WRIST_STEP_LABELS, BILATERAL_STEP_LABELS, bilateral_quality_notes)
from .wearing_workflow import GuidedWearingWorkflow
from .bilateral_wearing_workflow import BilateralWearingWorkflow


class GuidedCalibrationDialog(QtWidgets.QDialog):
    """Samples the existing main-window source; never binds UDP itself."""

    _workflow_type = GuidedWearingWorkflow

    def __init__(self, host: QtWidgets.QMainWindow) -> None:
        super().__init__(host)
        self.host = host
        self._workflow = self._workflow_type(
            self, recorder_factory=lambda *a, **k: FusionSessionRecorder(*a, **k),
            fit_arm=lambda *a, **k: fit_guided_arm(*a, **k),
            fit_torso=lambda *a, **k: fit_guided_torso(*a, **k),
            beep=lambda: QtWidgets.QApplication.beep(),
            warn=lambda *a: QtWidgets.QMessageBox.warning(self, *a),
            question=lambda *a: QtWidgets.QMessageBox.question(self, *a),
            yes_button=QtWidgets.QMessageBox.StandardButton.Yes,
            cancel_button=QtWidgets.QMessageBox.StandardButton.Cancel,
            clock=lambda: time.monotonic(), clock_ns=lambda: time.monotonic_ns())
        self.setWindowTitle("穿戴标定 · 躯干 / 上臂 / 前臂 / 双手")
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.resize(630, 490)
        self.phase = "idle"
        self.index = 0
        self.deadline = 0.0
        self.device_id = ""
        self.part = ""
        self.orientation: int | None = None
        self.algorithm: int | None = None
        self._read_waiting = False
        self.samples: dict[str, list[dict]] = {}
        self.context_samples: dict[str, dict[str, list[dict]]] = {}
        self.events: list[dict] = []
        self.recorder: FusionSessionRecorder | None = None
        self.result = None
        self.result_path: Path | None = None
        self.preview_window = None
        self._applied = False
        self.reference_device_id = ""
        self.shared_calibrations: list[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "每次重新穿戴后标定一次；整轮动作中绑带不能相对身体部位滑动。"
            "用静止动作求连续轴映射；手动映射先验权重可调，λ=0时不参与拟合。"
            "标定只读取设备，不发送归零或参数写入指令。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self.part_combo = QtWidgets.QComboBox()
        self.part_combo.currentIndexChanged.connect(self._part_changed)
        form.addRow("标定部位", self.part_combo)
        self.device_status = QtWidgets.QLabel("等待选择设备")
        device_row = QtWidgets.QHBoxLayout()
        device_row.addWidget(self.device_status, 1)
        self.read_button = QtWidgets.QPushButton("重新读取")
        self.read_button.clicked.connect(self._read_device)
        device_row.addWidget(self.read_button)
        form.addRow("设备参数", device_row)
        self.prepare_seconds = QtWidgets.QSpinBox()
        self.prepare_seconds.setRange(3, 30)
        self.prepare_seconds.setValue(7)
        self.prepare_seconds.setSuffix(" 秒")
        form.addRow("每步准备时间", self.prepare_seconds)
        self.first_prepare_label = QtWidgets.QLabel("首步准备时间")
        self.first_prepare_seconds = QtWidgets.QSpinBox()
        self.first_prepare_seconds.setRange(3, 60)
        self.first_prepare_seconds.setValue(12)
        self.first_prepare_seconds.setSuffix(" 秒")
        form.addRow(self.first_prepare_label, self.first_prepare_seconds)
        self.hold_seconds = QtWidgets.QDoubleSpinBox()
        self.hold_seconds.setRange(1.0, 5.0)
        self.hold_seconds.setSingleStep(0.5)
        self.hold_seconds.setValue(2.0)
        self.hold_seconds.setSuffix(" 秒")
        form.addRow("每步静止采样", self.hold_seconds)
        layout.addLayout(form)
        from .calibration_options import add_prior_control
        add_prior_control(self, host)
        self.capture_prior_weight = self.prior_weight_spin.value()
        self.steps_label = QtWidgets.QLabel(
            "动作：中立 → 前举 → 侧举 → 再次前举验证 → 回到中立验证。"
            "每一步都先倒计时准备，然后自动采样；听到提示音后切换姿势。")
        self.steps_label.setWordWrap(True)
        layout.addWidget(self.steps_label)
        self.progress = QtWidgets.QLabel("尚未开始")
        self.progress.setWordWrap(True)
        self.progress.setMinimumHeight(85)
        self.progress.setStyleSheet("font-size: 20px; font-weight: bold; padding: 9px")
        layout.addWidget(self.progress)
        self.details = QtWidgets.QLabel("必须保持肩部、肘部和绑带稳定；前臂测试时肘部伸直。")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始引导标定")
        self.start_button.clicked.connect(self.start)
        row.addWidget(self.start_button)
        self.cancel_button = QtWidgets.QPushButton("取消本轮")
        self.cancel_button.clicked.connect(self.abort)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        result_row = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("双模型回放预览")
        self.preview_button.clicked.connect(self.preview)
        self.preview_button.setEnabled(False)
        result_row.addWidget(self.preview_button)
        self.apply_button = QtWidgets.QPushButton("确认应用到实时模型")
        self.apply_button.clicked.connect(self.apply)
        self.apply_button.setEnabled(False)
        result_row.addWidget(self.apply_button)
        self.restore_button = QtWidgets.QPushButton("恢复该部位手动映射")
        self.restore_button.clicked.connect(self.restore)
        self.restore_button.setEnabled(False)
        result_row.addWidget(self.restore_button)
        layout.addLayout(result_row)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.refresh_parts()

    def refresh_parts(self) -> None:
        """Reflect assignments made after this dialog was first opened."""
        if self.phase in ("prepare", "hold"):
            return
        previous_part = self.part
        self.part_combo.blockSignals(True)
        self.part_combo.clear()
        for key in CALIBRATABLE_PARTS:
            device = self.host.engine.assignments.get(key, "")
            if device:
                part = PARTS[PART_BY_KEY[key]]
                self.part_combo.addItem(f"{part.label} · {device}", key)
        index = self.part_combo.findData(previous_part)
        self.part_combo.setCurrentIndex(index if index >= 0 else 0)
        self.part_combo.blockSignals(False)
        self._part_changed()

    def _part_changed(self) -> None:
        selected = self.part_combo.currentData() or ""
        self.first_prepare_label.setVisible(selected in HAND_PARTS)
        self.first_prepare_seconds.setVisible(selected in HAND_PARTS)
        device_id = self.host.engine.assignments.get(selected, "")
        if selected != self.part or device_id != self.device_id:
            self.result = None
            self.result_path = None
            self._applied = False
            self.preview_button.setEnabled(False)
            self.apply_button.setEnabled(False)
        self.part = selected
        self.device_id = device_id
        self.steps = STEPS
        self.orientation = self.algorithm = None
        self._read_waiting = False
        self.device_status.setText("正在只读查询安装方向与算法…" if self.device_id else "无已分配设备")
        self.start_button.setEnabled(bool(self.device_id))
        self.restore_button.setEnabled(self.part in self.host.engine.continuous_basis)
        if self.part == "torso":
            self.steps_label.setText(
                "动作：直立 → 前屈 → 向右侧屈 → 再次前屈验证 → 回到直立验证。"
                "各步保持骨盆不动；每一步自动倒计时并采样。")
            self.details.setText("躯干标定使用运动转轴；请不要连同骨盆一起转动。")
        elif self.part in HAND_PARTS:
            side_label = "左" if self.part == "left_hand" else "右"
            self.steps_label.setText(
                f"动作：自然下垂 → {side_label}臂前平举 → {side_label}臂侧平举 → 再次前平举验证 → 回到下垂。"
                f"肘和手腕保持伸直，{side_label}手随整条手臂移动；每步先倒计时准备，再静止采样。")
            self.details.setText(
                f"仅需{side_label}手背设备；自然下垂时手背朝前，手指沿手臂长轴向下。"
                "平举时不要屈腕、侧偏或单独旋前/旋后，躯干保持不动。"
                f"此流程按伸直手掌长轴拟合；应用只更新{side_label}手，不改变前臂标定或归零。")
        else:
            self.steps_label.setText(
                "动作：中立 → 前举 → 侧举 → 再次前举验证 → 回到中立验证。"
                "每一步先倒计时准备，再自动采样。")
            self.details.setText("保持肩部、肘部和绑带稳定；前臂测试时肘部伸直。")
        self._read_device()

    def _read_device(self) -> None:
        if not self.device_id or self.host.replay_mode:
            return
        self.orientation = self.algorithm = None
        self._read_waiting = False
        self.device_status.setText("正在只读查询安装方向与算法…")
        if self.device_id and not self.host.replay_mode:
            try:
                self.host.source.read_registers(self.device_id, REG_ORIENT)
                self._read_waiting = True
            except (RuntimeError, OSError) as exc:
                self.device_status.setText(f"读取失败：{exc}")

    def on_replies(self, replies: list[tuple[str, tuple[int, int, int, int]]]) -> None:
        for device_id, values in replies:
            if self._read_waiting and device_id == self.device_id:
                self._read_waiting = False
                self.orientation, self.algorithm = values[:2]
                orientation = {0: "水平", 1: "垂直"}.get(self.orientation, "未知")
                algorithm = {0: "9 轴", 1: "6 轴"}.get(self.algorithm, "未知")
                self.device_status.setText(f"{device_id} · {orientation}安装 · {algorithm}算法")

    def start(self) -> None:
        return self._workflow.start()

    def on_packet(self, received_ns: int, packet: object) -> None:
        return self._workflow.on_packet(received_ns, packet)

    def _tick(self) -> None:
        return self._workflow._tick()

    def _finish(self) -> None:
        return self._workflow._finish()

    def _save_events(self, session_path: Path, status: str) -> None:
        return self._workflow._save_events(session_path, status)

    def preview(self) -> None:
        if self.result_path is None or self.result is None:
            return
        self._show_preview([self.result_path])

    def _show_preview(self, result_paths: list[Path]) -> None:
        try:
            from .fusion_gui import FusionWindow
            from .fusion_replay import FusionSessionReplay

            source = FusionSessionReplay(result_paths[0].parent)
            try:
                window = FusionWindow(
                    source, self.host.settings_path,
                    dict(self.host.engine.assignments),
                    dict(self.host.engine.axis_mappings),
                    algorithm_modes=dict(self.host.engine.algorithm_modes),
                    calibration_candidates=[json.loads(path.read_text(encoding="utf-8"))
                                            for path in result_paths],
                    comparison_shared_calibrations=self.shared_calibrations,
                    manual_mapping_source="标定时快照")
            except Exception:
                source.close()
                raise
            self.preview_window = window
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.host._dialogs.append(window)
            window.destroyed.connect(
                lambda: self.host._dialogs.remove(window)
                if window in self.host._dialogs else None)
            window.show()
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, "回放失败", str(exc))

    def apply(self) -> None:
        return self._workflow.apply()

    def restore(self) -> None:
        return self._workflow.restore()

    def invalidate_for(self, device_id: str) -> None:
        if self.result is None or self.result.device_id != device_id:
            return
        self._applied = False
        self.result = None
        self.apply_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        self.details.setText("设备参数或设备端归零已改变；旧标定不能再次应用，请重新录制。")

    def abort(self, reason: str = "已取消本轮标定") -> None:
        return self._workflow.abort(reason)

    def closeEvent(self, event) -> None:
        if self.phase in ("prepare", "hold"):
            self.abort("关闭窗口，标定取消")
        super().closeEvent(event)


class BilateralCalibrationDialog(GuidedCalibrationDialog):
    """One shared five-pose capture, with independent fits for each arm segment."""

    _workflow_type = BilateralWearingWorkflow

    def __init__(self, host: QtWidgets.QMainWindow) -> None:
        self.device_checks: dict[str, tuple[int, int] | None] = {}
        self.read_errors: dict[str, str] = {}
        self._participants: dict[str, str] = {}
        self.samples_by_part: dict[str, dict[str, list[dict]]] = {}
        self.results: dict[str, object] = {}
        self.result_paths: dict[str, Path] = {}
        self.failures: dict[str, str] = {}
        self.quality_notes: dict[str, list[str]] = {}
        self._applied_parts: set[str] = set()
        super().__init__(host)
        self.setWindowTitle("双臂联合穿戴标定 · 上臂 / 前臂")
        self.resize(740, 570)
        self.part_combo.blockSignals(True)
        self.part_combo.clear()
        self.part_combo.addItem("已绑定且在线的左右上臂 / 前臂", "both_arms")
        self.part_combo.setEnabled(False)
        self.part_combo.blockSignals(False)
        self.part = "both_arms"
        self.device_id = ""
        self.layout().itemAt(0).widget().setText(
            "一次完成双臂下垂、前平举和侧平举；左右上臂与前臂分别计算结果。"
            "未绑定或离线的部位会跳过；任一部位失败不影响其余部位。"
            "整个过程保持肘部伸直、躯干不动，绑带不得滑动。")
        self.device_status.setWordWrap(True)
        self.steps_label.setText(
            "动作：双臂下垂 → 双臂前平举 → 双臂侧平举 → 再次前平举验证 → 回到下垂验证。"
            "每步倒计时后自动同步采样。")
        self.details.setText("开始前确认各设备已绑定并在线、安装方向为垂直；只读检查不会修改设备。")
        self.start_button.setText("开始双臂联合标定")
        self.preview_button.setText("双模型回放全部合格结果")
        self.apply_button.setText("应用全部合格结果")
        self.restore_button.setText("恢复选中部位手动映射")
        self.result_select = QtWidgets.QComboBox()
        self.result_select.setEnabled(False)
        self.result_select.currentIndexChanged.connect(self._result_changed)
        self.layout().insertWidget(self.layout().count() - 2, self.result_select)
        self.refresh_devices()

    def _part_changed(self) -> None:
        # The base constructor calls this while constructing its widgets.
        pass

    def refresh_devices(self) -> None:
        if self.phase in ("prepare", "hold"):
            return
        self.device_checks = {}
        self.read_errors = {}
        for key in ARM_PARTS:
            device_id = self.host.engine.assignments.get(key, "")
            if not device_id or not self.host.engine.is_live(device_id):
                continue
            self.device_checks[device_id] = None
            try:
                self.host.source.read_registers(device_id, REG_ORIENT)
            except (RuntimeError, OSError) as exc:
                self.read_errors[device_id] = str(exc)
        self._show_device_status()
        self.start_button.setEnabled(bool(self.device_checks))

    def _read_device(self) -> None:
        self.refresh_devices()

    def _show_device_status(self) -> None:
        lines = []
        for key in ARM_PARTS:
            label = PARTS[PART_BY_KEY[key]].label
            device_id = self.host.engine.assignments.get(key, "")
            if not device_id:
                state = "未绑定，跳过"
            elif not self.host.engine.is_live(device_id):
                state = f"{device_id} · 离线，跳过"
            elif device_id in self.read_errors:
                state = f"{device_id} · 读取失败：{self.read_errors[device_id]}"
            else:
                check = self.device_checks.get(device_id)
                if check is None:
                    state = f"{device_id} · 等待读取安装方向"
                else:
                    orientation = {0: "水平", 1: "垂直"}.get(check[0], "未知")
                    algorithm = {0: "9 轴", 1: "6 轴"}.get(check[1], "未知")
                    state = f"{device_id} · {orientation}安装 · {algorithm}算法"
            lines.append(f"{label}：{state}")
        self.device_status.setText("\n".join(lines))

    def on_replies(self, replies: list[tuple[str, tuple[int, int, int, int]]]) -> None:
        changed = False
        for device_id, values in replies:
            if device_id in self.device_checks:
                self.device_checks[device_id] = values[:2]
                changed = True
        if changed:
            self._show_device_status()

    def start(self) -> None:
        return self._workflow.start()

    def on_packet(self, received_ns: int, packet: object) -> None:
        return self._workflow.on_packet(received_ns, packet)

    def _tick(self) -> None:
        return self._workflow._tick()

    def _save_events(self, session_path: Path, status: str) -> None:
        return self._workflow._save_events(session_path, status)

    def _finish(self) -> None:
        return self._workflow._finish()

    def _result_changed(self) -> None:
        key = self.result_select.currentData() or ""
        self.part = key
        self.device_id = self._participants.get(key, "")
        self.result = self.results.get(key)
        self.result_path = self.result_paths.get(key)
        result = self.result
        if result is None:
            self.details.setText(self.failures.get(key, "尚无结果"))
        else:
            fit, check = result.fit_error_deg, result.check_error_deg
            self.details.setText(
                f"{PARTS[PART_BY_KEY[key]].label}：前举 {fit['forward']:.1f}°，"
                f"侧举 {fit['lateral']:.1f}°；复测前举 {check['check_forward']:.1f}°，"
                f"回中立 {check['check_neutral']:.1f}°。\n" +
                ("；".join(result.warnings) if result.warnings else "质量检查通过") +
                ("\n质量提醒：" + "；".join(self.quality_notes[key])
                 if self.quality_notes.get(key) else "") +
                f"\n原始记录与结果：{self.result_path.parent}")
        passed = {part for part, item in self.results.items() if item.status == "通过"}
        self.preview_button.setEnabled(bool(passed))
        self.apply_button.setEnabled(
            bool(passed - self._applied_parts))
        self.restore_button.setEnabled(key in self.host.engine.continuous_basis)

    def preview(self) -> None:
        paths = [self.result_paths[key] for key in ARM_PARTS
                 if key in self.result_paths and self.results[key].status == "通过"]
        if paths:
            self._show_preview(paths)

    def apply(self) -> None:
        return self._workflow.apply()

    def restore(self) -> None:
        return self._workflow.restore()

    def invalidate_for(self, device_id: str) -> None:
        for key, result in list(self.results.items()):
            if result.device_id == device_id:
                self.results.pop(key)
                self.result_paths.pop(key, None)
                self.failures[key] = "设备设置或归零已改变；此部位旧结果失效，请重新录制"
                self.quality_notes.pop(key, None)
                self._applied_parts.discard(key)
        if hasattr(self, "result_select"):
            for index in range(self.result_select.count()):
                key = self.result_select.itemData(index)
                if key in self.failures:
                    label = PARTS[PART_BY_KEY[key]].label
                    self.result_select.setItemText(
                        index, f"{label} · {self._participants.get(key, '')} · 无效")
            self._result_changed()

    def abort(self, reason: str = "已取消本轮标定") -> None:
        return self._workflow.abort(reason)
