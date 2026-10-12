"""GuidedWearingWorkflow: sampling phases, fitting and application, delegated by Qt dialogs.

State remains on the dialog; factories, clock and user interaction are injected.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
import json
import numpy as np
from .guided_calibration import HAND_PARTS, reference_motion_deg, neutral_matrices_by_mode
from .wearing_protocol import (_attach_orientation_streams, STEP_LABELS, TORSO_STEP_LABELS,
    LEFT_WRIST_STEP_LABELS, RIGHT_WRIST_STEP_LABELS)


class GuidedWearingWorkflow:
    def __init__(self, dialog, *, recorder_factory, fit_arm, fit_torso, beep, warn, question, yes_button, cancel_button, clock, clock_ns):
        self.dialog = dialog
        self.recorder_factory = recorder_factory
        self.fit_arm = fit_arm
        self.fit_torso = fit_torso
        self.beep = beep
        self.warn = warn
        self.question = question
        self.yes_button = yes_button
        self.cancel_button = cancel_button
        self.clock = clock
        self.clock_ns = clock_ns

    def start(self) -> None:
        dialog = self.dialog
        if dialog.phase in ("prepare", "hold") or not dialog.device_id:
            return
        if dialog.host.engine.assignments.get(dialog.part) != dialog.device_id:
            dialog.progress.setText("设备分配已改变；请重新打开标定窗口")
            return
        if not dialog.host.engine.is_live(dialog.device_id):
            dialog.progress.setText("设备未在实时回传；请先检查热点连接")
            return
        if dialog.orientation not in (0, 1) or dialog.algorithm not in (0, 1):
            dialog.progress.setText("未读到安装方向与算法；请点击‘重新读取’")
            return
        if dialog.part not in HAND_PARTS and dialog.orientation != 1:
            dialog.progress.setText("当前不是垂直安装；本轮穿戴协议要求外壳 Y 轴向上")
            return
        if getattr(dialog.host, "session", None) is not None or dialog.host.recorder is not None:
            dialog.progress.setText("请先结束现有录制，再开始穿戴标定")
            return
        dialog.samples = {step: [] for step in dialog.steps}
        dialog.capture_prior_weight = dialog.prior_weight_spin.value()
        dialog.context_samples = {step: {} for step in dialog.steps}
        dialog.events = []
        dialog.result = None
        dialog.result_path = None
        dialog._applied = False
        dialog.reference_device_id = ""
        dialog.shared_calibrations = []
        dialog.preview_button.setEnabled(False)
        dialog.apply_button.setEnabled(False)
        dialog.restore_button.setEnabled(dialog.part in dialog.host.engine.continuous_basis)
        try:
            parent = Path(__file__).resolve().parent.parent / "calibrations"
            dialog.recorder = self.recorder_factory(
                parent, tail_trim_seconds=0,
                assignments=dialog.host.engine.assignments,
                axis_mappings=dialog.host.engine.axis_mappings,
                algorithm_modes=dialog.host.engine.algorithm_modes)
        except OSError as exc:
            dialog.progress.setText(f"无法录制：{exc}")
            return
        dialog.index = 0
        dialog.phase = "prepare"
        from .calibration_options import freeze_prior_control
        freeze_prior_control(dialog)
        dialog.deadline = self.clock() + (
            dialog.first_prepare_seconds.value() if dialog.part in HAND_PARTS
            else dialog.prepare_seconds.value())
        dialog.part_combo.setEnabled(False)
        dialog.read_button.setEnabled(False)
        dialog.start_button.setEnabled(False)
        dialog.cancel_button.setEnabled(True)
        dialog.timer.start()
        dialog._tick()


    def on_packet(self, received_ns: int, packet: object) -> None:
        dialog = self.dialog
        if dialog.recorder is None:
            return
        try:
            dialog.recorder.write_imu(received_ns, packet)
        except (OSError, RuntimeError) as exc:
            dialog.abort(f"录制中断：{exc}")
            return
        if dialog.phase == "hold":
            row = {
                "received_ns": received_ns,
                "roll": packet.roll, "pitch": packet.pitch, "yaw": packet.yaw,
                "gyro_x": packet.gyro_x, "gyro_y": packet.gyro_y,
                "gyro_z": packet.gyro_z,
                "acc_x": packet.acc_x, "acc_y": packet.acc_y,
                "acc_z": packet.acc_z}
            _attach_orientation_streams(dialog.host.engine, packet, row)
            step = dialog.steps[dialog.index]
            dialog.context_samples[step].setdefault(packet.device_id, []).append(row)
            if packet.device_id == dialog.device_id:
                dialog.samples[step].append(row)


    def _tick(self) -> None:
        dialog = self.dialog
        if dialog.phase not in ("prepare", "hold"):
            return
        if not dialog.host.engine.is_live(dialog.device_id):
            dialog.abort("设备断流；本轮标定已取消，请检查热点后重做")
            return
        step = dialog.steps[dialog.index]
        labels = (TORSO_STEP_LABELS if dialog.part == "torso" else
                  LEFT_WRIST_STEP_LABELS if dialog.part == "left_hand" else
                  RIGHT_WRIST_STEP_LABELS if dialog.part == "right_hand" else STEP_LABELS)
        step_label = labels[step]
        remaining = dialog.deadline - self.clock()
        if remaining > 0:
            verb = "摆好姿势" if dialog.phase == "prepare" else "保持静止"
            count = len(dialog.samples[step])
            dialog.progress.setText(
                f"第 {dialog.index + 1}/{len(dialog.steps)} 步：{step_label}\n"
                f"{verb} · {remaining:.1f} 秒" +
                (f" · 已收 {count} 包" if dialog.phase == "hold" else ""))
            return
        if dialog.phase == "prepare":
            dialog.phase = "hold"
            dialog.deadline = self.clock() + dialog.hold_seconds.value()
            dialog.events.append({"step": step, "hold_start_mono_ns": self.clock_ns()})
            self.beep()
            dialog._tick()
            return
        dialog.events[-1]["hold_end_mono_ns"] = self.clock_ns()
        if len(dialog.samples[step]) < 30:
            dialog.abort(f"{step_label} 仅收到 {len(dialog.samples[step])} 包；请重做")
            return
        dialog.index += 1
        self.beep()
        if dialog.index == len(dialog.steps):
            dialog._finish()
            return
        dialog.phase = "prepare"
        dialog.deadline = self.clock() + dialog.prepare_seconds.value()
        dialog._tick()


    def _finish(self) -> None:
        dialog = self.dialog
        dialog.phase = "done"
        from .calibration_options import set_prior_enabled
        set_prior_enabled(dialog, True)
        dialog.timer.stop()
        dialog.part_combo.setEnabled(True)
        dialog.read_button.setEnabled(True)
        dialog.start_button.setEnabled(True)
        dialog.cancel_button.setEnabled(False)
        recorder, dialog.recorder = dialog.recorder, None
        if recorder is None:
            return
        try:
            recorder.close()
            dialog._save_events(recorder.path, "recorded")
            mapping = dialog.host.engine.axis_mappings[dialog.part]
            result = (self.fit_torso(dialog.device_id, dialog.samples, mapping, prior_weight=dialog.capture_prior_weight)
                      if dialog.part == "torso" else
                      self.fit_arm(dialog.part, dialog.device_id, dialog.samples, mapping, prior_weight=dialog.capture_prior_weight))
            reference_key = "pelvis" if dialog.part == "torso" else "torso"
            reference_id = dialog.host.engine.assignments.get(reference_key, "")
            if reference_id and reference_id != dialog.device_id:
                context = {step: dialog.context_samples[step].get(reference_id, [])
                           for step in dialog.steps}
                try:
                    motions = reference_motion_deg(context)
                except ValueError:
                    motions = {}
                if motions and max(motions.values()) > 12.0:
                    result = replace(
                        result, status="需重录",
                        warnings=result.warnings + [
                            f"{reference_key} 在规定动作中转动了"
                            f" {max(motions.values()):.1f}°；请固定身体基准后重录"])
            dialog.result = result
            payload = asdict(result)
            payload["attitude_mode"] = dialog.host.engine.algorithm_modes[dialog.part]
            payload["neutral_matrices_by_mode"] = {
                mode: matrix.tolist() for mode, matrix in
                neutral_matrices_by_mode(dialog.samples["neutral"]).items()}
            payload.update({"format": "wt901-guided-calibration-v1",
                            "method": ("torso_axis" if dialog.part == "torso" else
                                       "hand_bone_v1" if dialog.part in HAND_PARTS
                                       else "bone"),
                            "neutral_policy": ("first_static_hold_hand_only"
                                               if dialog.part in HAND_PARTS else "part_hold"),
                            "session": str(recorder.path.resolve()),
                            "installation_direction": dialog.orientation,
                            "fusion_algorithm": dialog.algorithm,
                            "manual_mapping": list(dialog.host.engine.axis_mappings[dialog.part]),
                            "reference_device_id": dialog.reference_device_id,
                            "shared_calibrations": dialog.shared_calibrations,
                            "events": dialog.events})
            dialog.result_path = recorder.path / "guided_result.json"
            dialog.result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except (OSError, ValueError) as exc:
            dialog.progress.setText(f"无法完成标定：{exc}；原始记录：{recorder.path}")
            return
        fit = result.fit_error_deg
        check = result.check_error_deg
        forward_name, lateral_name = (("前屈", "侧屈") if dialog.part == "torso"
                                      else ("前举", "侧举"))
        dialog.progress.setText(
            f"结果：{result.status} · {forward_name} {fit['forward']:.1f}° · "
            f"{lateral_name} {fit['lateral']:.1f}°\n"
            f"独立复测：{forward_name} {check['check_forward']:.1f}° · "
            f"回到中立 {check['check_neutral']:.1f}°")
        dialog.details.setText(
            ("；".join(result.warnings) if result.warnings else
             ("质量检查通过；腰椎真实运动轴仍需人体实验复核。" if dialog.part == "torso"
              else "质量检查通过；请在回放中确认前举、侧举。屈腕、侧偏和长轴转动还需实测。"
              if dialog.part in HAND_PARTS else
              "质量检查通过。前臂长轴扭转尚未经此协议验证。"))
            + ("\n骨盆未绑定设备：无法独立确认侧屈时骨盆保持固定。"
               if dialog.part == "torso" and not dialog.host.engine.assignments.get("pelvis")
               else "")
            + f"\n原始记录与结果：{recorder.path}")
        dialog.preview_button.setEnabled(True)
        dialog.apply_button.setEnabled(result.status == "通过")


    def _save_events(self, session_path: Path, status: str) -> None:
        dialog = self.dialog
        payload = {"format": "wt901-guided-steps-v1", "status": status,
                   "part": dialog.part, "device_id": dialog.device_id,
                   "wearing_prior_weight": dialog.capture_prior_weight,
                   "installation_direction": dialog.orientation,
                   "fusion_algorithm": dialog.algorithm,
                   "manual_mapping": list(dialog.host.engine.axis_mappings[dialog.part]),
                   "reference_device_id": dialog.reference_device_id,
                   "events": dialog.events}
        (session_path / "steps.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")


    def apply(self) -> None:
        dialog = self.dialog
        result = dialog.result
        if result is None or result.status != "通过":
            return
        if dialog.host.engine.assignments.get(result.part) != result.device_id:
            self.warn( "无法应用", "设备分配已改变，请重新标定")
            return
        try:
            dialog.host.engine.apply_calibration(
                result.part, result.device_id, np.asarray(result.basis_matrix),
                np.asarray(result.neutral_matrix),
                paired_neutrals=neutral_matrices_by_mode(dialog.samples["neutral"]))
        except ValueError as exc:
            self.warn( "无法应用", str(exc))
            return
        dialog._applied = True
        dialog.apply_button.setEnabled(False)
        dialog.restore_button.setEnabled(True)
        dialog.host._render_dirty = True
        dialog.host.statusBar().showMessage(
            f"已临时应用 {result.device_id} / {result.part} 穿戴标定；重新启动后需重新验证", 8000)


    def restore(self) -> None:
        dialog = self.dialog
        key = dialog.part
        if key not in dialog.host.engine.continuous_basis:
            return
        dialog.host.engine.clear_calibration(key)
        dialog._applied = False
        dialog.restore_button.setEnabled(False)
        dialog.apply_button.setEnabled(dialog.result is not None and dialog.result.status == "通过")
        dialog.host._render_dirty = True
        dialog.host.statusBar().showMessage("已停用该部位的穿戴标定，恢复手动轴映射", 5000)


    def abort(self, reason: str = "已取消本轮标定") -> None:
        dialog = self.dialog
        from .calibration_options import set_prior_enabled
        set_prior_enabled(dialog, True)
        dialog.timer.stop()
        dialog.phase = "idle"
        dialog.part_combo.setEnabled(True)
        dialog.read_button.setEnabled(True)
        dialog.start_button.setEnabled(bool(dialog.device_id))
        dialog.cancel_button.setEnabled(False)
        recorder, dialog.recorder = dialog.recorder, None
        if recorder is not None:
            try:
                recorder.close()
                dialog._save_events(recorder.path, "cancelled")
            except OSError:
                pass
        dialog.progress.setText(reason)

