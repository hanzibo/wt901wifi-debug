"""Guided, unattended IMU wearing calibration in a separate window."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from wt901.registers import REG_ORIENT

from .fusion_recording import FusionSessionRecorder
from .guided_calibration import (ARM_PARTS, CALIBRATABLE_PARTS, STEPS, WRIST_STEPS, fit_guided_arm,
                                 fit_guided_left_wrist, fit_guided_torso,
                                 reference_hold_quality, reference_motion_deg,
                                 reference_neutral_matrix)
from .pose import PoseEngine
from .skeleton import PART_BY_KEY, PARTS


STEP_LABELS = {
    "neutral": "双臂自然下垂，躯干站直",
    "forward": "目标手臂伸直，向正前方举到肩高",
    "lateral": "目标手臂伸直，向身体侧方举到肩高",
    "check_forward": "再次向正前方举到肩高（独立验证）",
    "check_neutral": "放下手臂，回到起始中立姿势",
}
TORSO_STEP_LABELS = {
    "neutral": "站直，面向正前方，骨盆保持不动",
    "forward": "保持骨盆不动，躯干向前屈约 30–45°",
    "lateral": "保持骨盆不动，躯干向身体右侧侧屈约 30–45°",
    "check_forward": "再次向前屈约 30–45°（独立验证）",
    "check_neutral": "站直并回到最初的朝向",
}
LEFT_WRIST_STEP_LABELS = {
    "neutral": "站到测试位置，左臂自然下垂，手与前臂伸直；此步建立共同中立位",
    "forward": "只弯左手腕：掌屈约 25–55°，前臂尽量稳定",
    "lateral": "回到伸直后，只将左手向拇指侧偏转约 12–25°",
    "check_forward": "再次只做左手腕掌屈约 25–55°（独立验证）",
    "check_neutral": "左手回到与前臂伸直的中立位",
    "check_long_axis": "手腕保持伸直，左前臂带着左手共同绕手臂长轴旋转约 40–60°；不要单独拧手腕",
}
BILATERAL_STEP_LABELS = {
    "neutral": "双臂自然下垂，躯干保持直立",
    "forward": "双臂伸直，同时向正前方举到肩高",
    "lateral": "双臂伸直，同时向两侧举到肩高（T 字姿势）",
    "check_forward": "双臂再次同时前平举到肩高（独立验证）",
    "check_neutral": "双臂放下，回到起始中立姿势",
}


def bilateral_quality_notes(part: str, lateral_error: float,
                            torso_motion: dict[str, float]) -> list[str]:
    """Advisory checks stricter than the solver's hard rejection thresholds."""
    notes = []
    if part.endswith("forearm") and lateral_error >= 15.0:
        notes.append(
            f"前臂侧举误差 {lateral_error:.1f}°（建议 <15°）；"
            "请检查肘部伸直、前臂未旋前/旋后及绑带稳定性")
    if torso_motion and max(torso_motion.values()) >= 5.0:
        step = max(torso_motion, key=torso_motion.get)
        notes.append(
            f"躯干在{BILATERAL_STEP_LABELS[step]}时移动 "
            f"{torso_motion[step]:.1f}°（建议 <5°）；联合拟合可能受影响")
    return notes


class GuidedCalibrationDialog(QtWidgets.QDialog):
    """Samples the existing main-window source; never binds UDP itself."""

    def __init__(self, host: QtWidgets.QMainWindow) -> None:
        super().__init__(host)
        self.host = host
        self.setWindowTitle("穿戴标定 · 躯干 / 上臂 / 前臂 / 左手腕")
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
        self.reference_original_neutral: np.ndarray | None = None
        self.shared_calibrations: list[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "每次重新穿戴后标定一次；整轮动作中绑带不能相对身体部位滑动。"
            "先从你的手动轴映射出发，仅用静止姿势求精细修正。"
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
        self.first_prepare_label.setVisible(selected == "left_hand")
        self.first_prepare_seconds.setVisible(selected == "left_hand")
        device_id = self.host.engine.assignments.get(selected, "")
        if selected != self.part or device_id != self.device_id:
            self.result = None
            self.result_path = None
            self._applied = False
            self.preview_button.setEnabled(False)
            self.apply_button.setEnabled(False)
        self.part = selected
        self.device_id = device_id
        self.steps = WRIST_STEPS if selected == "left_hand" else STEPS
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
        elif self.part == "left_hand":
            self.steps_label.setText(
                "动作：手与前臂伸直 → 掌屈 → 向拇指侧偏转 → 再次掌屈验证 → 回到伸直"
                " → 前臂与手一起绕长轴转动验证。"
                "开始后可走到空旷处；首步静止采样才建立共同中立位。"
                "后续各步尽量保持前臂稳定，只弯手腕。")
            self.details.setText(
                "左前臂必须已绑定在线设备并完成穿戴标定；手背设备要绑牢。"
                "最后一步是独立验证前臂与手一起旋前/旋后，不是单独扭转腕关节。")
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
        if self.phase in ("prepare", "hold") or not self.device_id:
            return
        if self.host.engine.assignments.get(self.part) != self.device_id:
            self.progress.setText("设备分配已改变；请重新打开标定窗口")
            return
        if not self.host.engine.is_live(self.device_id):
            self.progress.setText("设备未在实时回传；请先检查热点连接")
            return
        if self.orientation not in (0, 1) or self.algorithm not in (0, 1):
            self.progress.setText("未读到安装方向与算法；请点击‘重新读取’")
            return
        if self.part != "left_hand" and self.orientation != 1:
            self.progress.setText("当前不是垂直安装；本轮穿戴协议要求外壳 Y 轴向上")
            return
        if self.part == "left_hand":
            parent_id = self.host.engine.assignments.get("left_forearm", "")
            if not parent_id or not self.host.engine.is_live(parent_id):
                self.progress.setText("左前臂设备未绑定或离线；手腕标定需要前臂参考")
                return
            if "left_forearm" not in self.host.engine.continuous_basis:
                self.progress.setText("请先完成并应用左前臂穿戴标定，再标定左手腕")
                return
        if getattr(self.host, "session", None) is not None or self.host.recorder is not None:
            self.progress.setText("请先结束现有录制，再开始穿戴标定")
            return
        self.samples = {step: [] for step in self.steps}
        self.context_samples = {step: {} for step in self.steps}
        self.events = []
        self.result = None
        self.result_path = None
        self._applied = False
        self.reference_device_id = (self.host.engine.assignments["left_forearm"]
                                    if self.part == "left_hand" else "")
        self.reference_original_neutral = (
            self.host.engine.neutral[self.reference_device_id].copy()
            if self.part == "left_hand" else None)
        self.shared_calibrations = ([{
            "part": "left_forearm", "device_id": self.reference_device_id,
            "basis_matrix": self.host.engine.continuous_basis["left_forearm"].tolist(),
            "neutral_matrix": self.reference_original_neutral.tolist(),
        }] if self.part == "left_hand" else [])
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.restore_button.setEnabled(self.part in self.host.engine.continuous_basis)
        try:
            parent = Path(__file__).resolve().parent.parent / "calibrations"
            self.recorder = FusionSessionRecorder(
                parent, tail_trim_seconds=0,
                assignments=self.host.engine.assignments,
                axis_mappings=self.host.engine.axis_mappings)
        except OSError as exc:
            self.progress.setText(f"无法录制：{exc}")
            return
        self.index = 0
        self.phase = "prepare"
        self.deadline = time.monotonic() + (
            self.first_prepare_seconds.value() if self.part == "left_hand"
            else self.prepare_seconds.value())
        self.part_combo.setEnabled(False)
        self.read_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.timer.start()
        self._tick()

    def on_packet(self, received_ns: int, packet: object) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.write_imu(received_ns, packet)
        except (OSError, RuntimeError) as exc:
            self.abort(f"录制中断：{exc}")
            return
        if self.phase == "hold":
            row = {
                "received_ns": received_ns,
                "roll": packet.roll, "pitch": packet.pitch, "yaw": packet.yaw,
                "gyro_x": packet.gyro_x, "gyro_y": packet.gyro_y,
                "gyro_z": packet.gyro_z,
                "acc_x": packet.acc_x, "acc_y": packet.acc_y,
                "acc_z": packet.acc_z}
            step = self.steps[self.index]
            self.context_samples[step].setdefault(packet.device_id, []).append(row)
            if packet.device_id == self.device_id:
                self.samples[step].append(row)

    def _tick(self) -> None:
        if self.phase not in ("prepare", "hold"):
            return
        if not self.host.engine.is_live(self.device_id):
            self.abort("设备断流；本轮标定已取消，请检查热点后重做")
            return
        if self.part == "left_hand":
            parent_id = self.host.engine.assignments.get("left_forearm", "")
            if (parent_id != self.reference_device_id or
                    not self.host.engine.is_live(parent_id)):
                self.abort("左前臂设备断流或分配改变；本轮手腕标定已取消")
                return
        step = self.steps[self.index]
        labels = (TORSO_STEP_LABELS if self.part == "torso" else
                  LEFT_WRIST_STEP_LABELS if self.part == "left_hand" else STEP_LABELS)
        step_label = labels[step]
        remaining = self.deadline - time.monotonic()
        if remaining > 0:
            verb = "摆好姿势" if self.phase == "prepare" else "保持静止"
            count = len(self.samples[step])
            self.progress.setText(
                f"第 {self.index + 1}/{len(self.steps)} 步：{step_label}\n"
                f"{verb} · {remaining:.1f} 秒" +
                (f" · 已收 {count} 包" if self.phase == "hold" else ""))
            return
        if self.phase == "prepare":
            self.phase = "hold"
            self.deadline = time.monotonic() + self.hold_seconds.value()
            self.events.append({"step": step, "hold_start_mono_ns": time.monotonic_ns()})
            QtWidgets.QApplication.beep()
            self._tick()
            return
        self.events[-1]["hold_end_mono_ns"] = time.monotonic_ns()
        if len(self.samples[step]) < 30:
            self.abort(f"{step_label} 仅收到 {len(self.samples[step])} 包；请重做")
            return
        if self.part == "left_hand":
            parent_id = self.host.engine.assignments["left_forearm"]
            parent_count = len(self.context_samples[step].get(parent_id, []))
            if parent_count < 30:
                self.abort(f"{step_label} 左前臂仅收到 {parent_count} 包；请重做")
                return
        self.index += 1
        QtWidgets.QApplication.beep()
        if self.index == len(self.steps):
            self._finish()
            return
        self.phase = "prepare"
        self.deadline = time.monotonic() + self.prepare_seconds.value()
        self._tick()

    def _finish(self) -> None:
        self.phase = "done"
        self.timer.stop()
        self.part_combo.setEnabled(True)
        self.read_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        try:
            recorder.close()
            self._save_events(recorder.path, "recorded")
            mapping = self.host.engine.axis_mappings[self.part]
            wrist_context = None
            if self.part == "left_hand":
                wrist_context = {
                    step: self.context_samples[step].get(self.reference_device_id, [])
                    for step in self.steps}
            result = (fit_guided_torso(self.device_id, self.samples, mapping)
                      if self.part == "torso" else
                      fit_guided_left_wrist(
                          self.device_id, self.samples, mapping, wrist_context,
                          np.asarray(self.shared_calibrations[0]["basis_matrix"]))
                      if self.part == "left_hand" else
                      fit_guided_arm(self.part, self.device_id, self.samples, mapping))
            if wrist_context is not None:
                self.shared_calibrations[0]["neutral_matrix"] = (
                    reference_neutral_matrix(wrist_context["neutral"]).tolist())
            reference_key = ("pelvis" if self.part == "torso" else
                             "left_forearm" if self.part == "left_hand" else "torso")
            reference_id = self.host.engine.assignments.get(reference_key, "")
            if reference_id and reference_id != self.device_id:
                context = {step: self.context_samples[step].get(reference_id, [])
                           for step in self.steps}
                try:
                    motions = reference_motion_deg(context)
                except ValueError:
                    if self.part == "left_hand":
                        raise
                    motions = {}
                limit = 40.0 if self.part == "left_hand" else 12.0
                monitored_motions = (motions if self.part != "left_hand" else
                                     {step: angle for step, angle in motions.items()
                                      if step != "check_long_axis"})
                if monitored_motions and max(monitored_motions.values()) > limit:
                    result = replace(
                        result, status="需重录",
                        warnings=result.warnings + [
                            f"{reference_key} 在规定动作中转动了"
                            f" {max(monitored_motions.values()):.1f}°；"
                            + ("手腕标定姿势变化过大，请重录" if self.part == "left_hand"
                               else "请固定身体基准后重录")])
                if self.part == "left_hand":
                    quality = reference_hold_quality(context)
                    unstable = [step for step, (spread, gyro) in quality.items()
                                if spread > 4.0 or gyro > 10.0]
                    if unstable:
                        result = replace(
                            result, status="需重录",
                            warnings=result.warnings + [
                                "左前臂在静止采样期间移动：" + ", ".join(unstable)])
            self.result = result
            payload = asdict(result)
            payload.update({"format": "wt901-guided-calibration-v1",
                            "method": ("torso_axis" if self.part == "torso" else
                                       "wrist_relative_axis_v2" if self.part == "left_hand"
                                       else "bone"),
                            "neutral_policy": ("first_static_hold_shared"
                                               if self.part == "left_hand" else "part_hold"),
                            "session": str(recorder.path.resolve()),
                            "installation_direction": self.orientation,
                            "fusion_algorithm": self.algorithm,
                            "manual_mapping": list(self.host.engine.axis_mappings[self.part]),
                            "reference_device_id": self.reference_device_id,
                            "shared_calibrations": self.shared_calibrations,
                            "events": self.events})
            self.result_path = recorder.path / "guided_result.json"
            self.result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except (OSError, ValueError) as exc:
            self.progress.setText(f"无法完成标定：{exc}；原始记录：{recorder.path}")
            return
        fit = result.fit_error_deg
        check = result.check_error_deg
        forward_name, lateral_name = (("前屈", "侧屈") if self.part == "torso"
                                      else ("掌屈", "拇指侧偏") if self.part == "left_hand"
                                      else ("前举", "侧举"))
        self.progress.setText(
            f"结果：{result.status} · {forward_name} {fit['forward']:.1f}° · "
            f"{lateral_name} {fit['lateral']:.1f}°\n"
            f"独立复测：{forward_name} {check['check_forward']:.1f}° · "
            f"回到中立 {check['check_neutral']:.1f}°" +
            (f" · 长轴相对误差 {check['check_long_axis_relative']:.1f}° · "
             f"偏离手动映射 {check['manual_mapping_offset']:.1f}°"
             if self.part == "left_hand" else ""))
        self.details.setText(
            ("；".join(result.warnings) if result.warnings else
             ("质量检查通过；腰椎真实运动轴仍需人体实验复核。" if self.part == "torso"
              else "质量检查通过；请在回放中确认掌屈、侧偏和前臂长轴转动方向。"
              if self.part == "left_hand" else
              "质量检查通过。前臂长轴扭转尚未经此协议验证。"))
            + ("\n骨盆未绑定设备：无法独立确认侧屈时骨盆保持固定。"
               if self.part == "torso" and not self.host.engine.assignments.get("pelvis")
               else "")
            + f"\n原始记录与结果：{recorder.path}")
        if self.part == "left_hand":
            self.details.setText(self.details.text() +
                                 "\n应用时会将左前臂与左手一起归零到首个静止姿势。")
        self.preview_button.setEnabled(True)
        self.apply_button.setEnabled(result.status == "通过")

    def _save_events(self, session_path: Path, status: str) -> None:
        payload = {"format": "wt901-guided-steps-v1", "status": status,
                   "part": self.part, "device_id": self.device_id,
                   "installation_direction": self.orientation,
                   "fusion_algorithm": self.algorithm,
                   "manual_mapping": list(self.host.engine.axis_mappings[self.part]),
                   "reference_device_id": self.reference_device_id,
                   "events": self.events}
        (session_path / "steps.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

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
                    calibration_candidates=[json.loads(path.read_text(encoding="utf-8"))
                                            for path in result_paths],
                    comparison_shared_calibrations=self.shared_calibrations,
                    manual_mapping_source=("手部手动映射（前臂保持已标定）"
                                           if self.part == "left_hand" else "标定时快照"))
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
        result = self.result
        if result is None or result.status != "通过":
            return
        if self.host.engine.assignments.get(result.part) != result.device_id:
            QtWidgets.QMessageBox.warning(self, "无法应用", "设备分配已改变，请重新标定")
            return
        if result.part == "left_hand" and (
                self.host.engine.assignments.get("left_forearm") != self.reference_device_id or
                "left_forearm" not in self.host.engine.continuous_basis):
            QtWidgets.QMessageBox.warning(self, "无法应用", "左前臂标定已失效，请重新标定")
            return
        if result.part == "left_hand":
            shared = self.shared_calibrations[0]
            if (not np.allclose(self.host.engine.continuous_basis["left_forearm"],
                                shared["basis_matrix"], atol=1e-5) or
                    self.reference_device_id not in self.host.engine.neutral or
                    not np.allclose(self.host.engine.neutral[self.reference_device_id],
                                    self.reference_original_neutral, atol=1e-5)):
                QtWidgets.QMessageBox.warning(
                    self, "无法应用", "左前臂标定或归零基准已改变，请重新录制左手腕标定")
                return
        try:
            self.host.engine.apply_calibration(
                result.part, result.device_id, np.asarray(result.basis_matrix),
                np.asarray(result.neutral_matrix))
            if result.part == "left_hand":
                self.host.engine.neutral[self.reference_device_id] = np.asarray(
                    shared["neutral_matrix"], dtype=float).copy()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "无法应用", str(exc))
            return
        self._applied = True
        self.apply_button.setEnabled(False)
        self.restore_button.setEnabled(True)
        self.host._render_dirty = True
        self.host.statusBar().showMessage(
            f"已临时应用 {result.device_id} / {result.part} 穿戴标定；重新启动后需重新验证", 8000)

    def restore(self) -> None:
        key = self.part
        if key not in self.host.engine.continuous_basis:
            return
        self.host.engine.clear_calibration(key)
        self._applied = False
        self.restore_button.setEnabled(False)
        self.apply_button.setEnabled(self.result is not None and self.result.status == "通过")
        self.host._render_dirty = True
        self.host.statusBar().showMessage("已停用该部位的穿戴标定，恢复手动轴映射", 5000)

    def invalidate_for(self, device_id: str) -> None:
        if self.result is None or self.result.device_id != device_id:
            return
        self._applied = False
        self.result = None
        self.apply_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        self.details.setText("设备参数或设备端归零已改变；旧标定不能再次应用，请重新录制。")

    def abort(self, reason: str = "已取消本轮标定") -> None:
        self.timer.stop()
        self.phase = "idle"
        self.part_combo.setEnabled(True)
        self.read_button.setEnabled(True)
        self.start_button.setEnabled(bool(self.device_id))
        self.cancel_button.setEnabled(False)
        recorder, self.recorder = self.recorder, None
        if recorder is not None:
            try:
                recorder.close()
                self._save_events(recorder.path, "cancelled")
            except OSError:
                pass
        self.progress.setText(reason)

    def closeEvent(self, event) -> None:
        if self.phase in ("prepare", "hold"):
            self.abort("关闭窗口，标定取消")
        super().closeEvent(event)


class BilateralCalibrationDialog(GuidedCalibrationDialog):
    """One shared five-pose capture, with independent fits for each arm segment."""

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
        if self.phase in ("prepare", "hold"):
            return
        if getattr(self.host, "session", None) is not None or self.host.recorder is not None:
            self.progress.setText("请先结束现有录制，再开始穿戴标定")
            return
        online = {key: device_id for key in ARM_PARTS
                  if (device_id := self.host.engine.assignments.get(key, ""))
                  and self.host.engine.is_live(device_id)}
        if not online:
            self.progress.setText("没有已绑定且在线的上臂或前臂设备")
            return
        unread = [key for key, device_id in online.items()
                  if self.device_checks.get(device_id) is None]
        wrong = [key for key, device_id in online.items()
                 if self.device_checks.get(device_id) is not None
                 and (self.device_checks[device_id][0] != 1 or
                      self.device_checks[device_id][1] not in (0, 1))]
        if unread or wrong:
            def labels(keys: list[str]) -> str:
                return "、".join(PARTS[PART_BY_KEY[key]].label for key in keys)

            self.progress.setText(
                (f"尚未读到设备参数：{labels(unread)}；请点‘重新读取’。" if unread else "") +
                (f"安装方向不是垂直或算法未知：{labels(wrong)}。" if wrong else ""))
            return
        self._participants = online
        self.samples_by_part = {key: {step: [] for step in STEPS} for key in online}
        self.context_samples = {step: {} for step in STEPS}
        self.events = []
        self.results = {}
        self.result_paths = {}
        self.failures = {}
        self.quality_notes = {}
        self._applied_parts = set()
        self.result = None
        self.result_path = None
        self.result_select.blockSignals(True)
        self.result_select.clear()
        self.result_select.blockSignals(False)
        self.result_select.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        try:
            parent = Path(__file__).resolve().parent.parent / "calibrations"
            self.recorder = FusionSessionRecorder(
                parent, tail_trim_seconds=0,
                assignments=self.host.engine.assignments,
                axis_mappings=self.host.engine.axis_mappings)
        except OSError as exc:
            self.progress.setText(f"无法录制：{exc}")
            return
        self.index = 0
        self.phase = "prepare"
        self.deadline = time.monotonic() + self.prepare_seconds.value()
        self.read_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.timer.start()
        self._tick()

    def on_packet(self, received_ns: int, packet: object) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.write_imu(received_ns, packet)
        except (OSError, RuntimeError) as exc:
            self.abort(f"录制中断：{exc}")
            return
        if self.phase != "hold":
            return
        row = {
            "received_ns": received_ns,
            "roll": packet.roll, "pitch": packet.pitch, "yaw": packet.yaw,
            "gyro_x": packet.gyro_x, "gyro_y": packet.gyro_y,
            "gyro_z": packet.gyro_z,
            "acc_x": packet.acc_x, "acc_y": packet.acc_y,
            "acc_z": packet.acc_z}
        step = STEPS[self.index]
        self.context_samples[step].setdefault(packet.device_id, []).append(row)
        for key, device_id in self._participants.items():
            if packet.device_id == device_id:
                self.samples_by_part[key][step].append(row)

    def _tick(self) -> None:
        if self.phase not in ("prepare", "hold"):
            return
        if not any(self.host.engine.is_live(device_id)
                   for device_id in self._participants.values()):
            self.abort("所有参与设备均已断流；本轮标定取消")
            return
        step = STEPS[self.index]
        remaining = self.deadline - time.monotonic()
        if remaining > 0:
            counts = ", ".join(
                f"{PARTS[PART_BY_KEY[key]].label} {len(self.samples_by_part[key][step])}"
                for key in self._participants)
            self.progress.setText(
                f"第 {self.index + 1}/{len(STEPS)} 步：{BILATERAL_STEP_LABELS[step]}\n"
                f"{'摆好姿势' if self.phase == 'prepare' else '保持静止'} · "
                f"{remaining:.1f} 秒" + (f" · {counts} 包" if self.phase == "hold" else ""))
            return
        if self.phase == "prepare":
            self.phase = "hold"
            self.deadline = time.monotonic() + self.hold_seconds.value()
            self.events.append({"step": step, "hold_start_mono_ns": time.monotonic_ns()})
            QtWidgets.QApplication.beep()
            self._tick()
            return
        self.events[-1]["hold_end_mono_ns"] = time.monotonic_ns()
        self.index += 1
        QtWidgets.QApplication.beep()
        if self.index == len(STEPS):
            self._finish()
            return
        self.phase = "prepare"
        self.deadline = time.monotonic() + self.prepare_seconds.value()
        self._tick()

    def _save_events(self, session_path: Path, status: str) -> None:
        payload = {"format": "wt901-guided-bilateral-steps-v1", "status": status,
                   "participants": self._participants, "events": self.events,
                   "installation": {key: self.device_checks.get(device_id)
                                    for key, device_id in self._participants.items()}}
        (session_path / "steps.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _finish(self) -> None:
        self.phase = "done"
        self.timer.stop()
        self.read_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        try:
            recorder.close()
            self._save_events(recorder.path, "recorded")
            for key, device_id in self._participants.items():
                try:
                    result = fit_guided_arm(
                        key, device_id, self.samples_by_part[key],
                        self.host.engine.axis_mappings[key])
                    torso_id = self.host.engine.assignments.get("torso", "")
                    reference_failed = False
                    if torso_id and torso_id != device_id:
                        context = {step: self.context_samples[step].get(torso_id, [])
                                   for step in STEPS}
                        try:
                            motions = reference_motion_deg(context)
                        except ValueError:
                            motions = {}
                            reference_failed = True
                        if motions and max(motions.values()) > 12.0:
                            result = replace(
                                result, status="需重录",
                                warnings=result.warnings + [
                                    f"躯干在动作中转动了 {max(motions.values()):.1f}°"])
                    notes = bilateral_quality_notes(
                        key, result.fit_error_deg["lateral"], motions
                        if torso_id and torso_id != device_id else {})
                    if reference_failed:
                        notes.append("躯干设备数据不足，无法核验躯干是否固定")
                    self.quality_notes[key] = notes
                    payload = asdict(result)
                    payload.update({
                        "format": "wt901-guided-calibration-v1",
                        "method": "bone", "session": str(recorder.path.resolve()),
                        "installation_direction": self.device_checks[device_id][0],
                        "fusion_algorithm": self.device_checks[device_id][1],
                        "manual_mapping": list(self.host.engine.axis_mappings[key]),
                        "events": self.events, "capture_mode": "bilateral",
                        "quality_notes": notes})
                    path = recorder.path / f"guided_result_{key}.json"
                    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
                    self.results[key] = result
                    self.result_paths[key] = path
                except (ValueError, OSError) as exc:
                    self.failures[key] = str(exc)
            (recorder.path / "guided_summary.json").write_text(
                json.dumps({"format": "wt901-guided-bilateral-summary-v1",
                            "participants": self._participants,
                            "status": {key: result.status for key, result in self.results.items()},
                            "quality_notes": self.quality_notes,
                            "failures": self.failures}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError as exc:
            self.progress.setText(f"无法完成联合标定：{exc}；原始记录：{recorder.path}")
            return
        self.result_select.blockSignals(True)
        self.result_select.clear()
        for key, device_id in self._participants.items():
            status = self.results[key].status if key in self.results else "无效"
            self.result_select.addItem(
                f"{PARTS[PART_BY_KEY[key]].label} · {device_id} · {status}", key)
        self.result_select.blockSignals(False)
        self.result_select.setEnabled(True)
        passed = sum(result.status == "通过" for result in self.results.values())
        cautioned = sum(bool(self.quality_notes.get(key)) for key, result in
                        self.results.items() if result.status == "通过")
        self.progress.setText(
            f"联合标定完成：{passed}/{len(self._participants)} 个部位通过。"
            f"其中 {cautioned} 个有质量提醒。回放和应用均处理全部合格结果；"
            "下方选择部位查看详情或单独恢复。" +
            ("未绑定在线躯干设备，无法核验标定时躯干是否固定。"
             if not self.host.engine.assignments.get("torso") else ""))
        self._result_changed()

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
        pending = [(key, self.results[key]) for key in ARM_PARTS
                   if key in self.results and self.results[key].status == "通过"
                   and key not in self._applied_parts]
        if not pending:
            return
        cautioned = [PARTS[PART_BY_KEY[key]].label for key, _ in pending
                     if self.quality_notes.get(key)]
        if cautioned:
            answer = QtWidgets.QMessageBox.question(
                self, "质量提醒",
                f"以下部位虽通过基本检查，但有更严格的质量提醒："
                f"{'、'.join(cautioned)}。\n"
                "建议先查看下拉框中的具体误差；仍要应用全部合格结果吗？",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        # Preflight every candidate, so an invalid late candidate cannot leave
        # only a subset of the requested group active.
        check_engine = PoseEngine(dict(self.host.engine.assignments),
                                  dict(self.host.engine.axis_mappings))
        try:
            for key, result in pending:
                check_engine.apply_calibration(
                    key, result.device_id, np.asarray(result.basis_matrix),
                    np.asarray(result.neutral_matrix))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "无法应用全部结果", str(exc))
            return
        for key, result in pending:
            self.host.engine.apply_calibration(
                key, result.device_id, np.asarray(result.basis_matrix),
                np.asarray(result.neutral_matrix))
            self._applied_parts.add(key)
        self.host._render_dirty = True
        self.host.statusBar().showMessage(
            f"已临时应用 {len(pending)} 个合格部位；重新启动后需重新验证", 8000)
        self._result_changed()

    def restore(self) -> None:
        key = self.part
        super().restore()
        self._applied_parts.discard(key)
        self._result_changed()

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
        super().abort(reason)
        self.part_combo.setEnabled(False)
        self.start_button.setEnabled(bool(self.device_checks))
