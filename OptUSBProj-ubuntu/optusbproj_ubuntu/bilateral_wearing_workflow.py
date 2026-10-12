"""BilateralWearingWorkflow: sampling phases, fitting and application, delegated by Qt dialogs.

State remains on the dialog; factories, clock and user interaction are injected.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
import json
import numpy as np
from .guided_calibration import ARM_PARTS, STEPS, reference_motion_deg, neutral_matrices_by_mode
from .pose import PoseEngine
from .skeleton import PARTS, PART_BY_KEY
from .wearing_protocol import (
    _attach_orientation_streams, BILATERAL_STEP_LABELS, bilateral_quality_notes)
from .wearing_workflow import GuidedWearingWorkflow


class BilateralWearingWorkflow(GuidedWearingWorkflow):
    def start(self) -> None:
        dialog = self.dialog
        if dialog.phase in ("prepare", "hold"):
            return
        if getattr(dialog.host, "session", None) is not None or dialog.host.recorder is not None:
            dialog.progress.setText("请先结束现有录制，再开始穿戴标定")
            return
        online = {key: device_id for key in ARM_PARTS
                  if (device_id := dialog.host.engine.assignments.get(key, ""))
                  and dialog.host.engine.is_live(device_id)}
        if not online:
            dialog.progress.setText("没有已绑定且在线的上臂或前臂设备")
            return
        unread = [key for key, device_id in online.items()
                  if dialog.device_checks.get(device_id) is None]
        wrong = [key for key, device_id in online.items()
                 if dialog.device_checks.get(device_id) is not None
                 and (dialog.device_checks[device_id][0] != 1 or
                      dialog.device_checks[device_id][1] not in (0, 1))]
        if unread or wrong:
            def labels(keys: list[str]) -> str:
                return "、".join(PARTS[PART_BY_KEY[key]].label for key in keys)

            dialog.progress.setText(
                (f"尚未读到设备参数：{labels(unread)}；请点‘重新读取’。" if unread else "") +
                (f"安装方向不是垂直或算法未知：{labels(wrong)}。" if wrong else ""))
            return
        dialog._participants = online
        dialog.capture_prior_weight = dialog.prior_weight_spin.value()
        dialog.samples_by_part = {key: {step: [] for step in STEPS} for key in online}
        dialog.context_samples = {step: {} for step in STEPS}
        dialog.events = []
        dialog.results = {}
        dialog.result_paths = {}
        dialog.failures = {}
        dialog.quality_notes = {}
        dialog._applied_parts = set()
        dialog.result = None
        dialog.result_path = None
        dialog.result_select.blockSignals(True)
        dialog.result_select.clear()
        dialog.result_select.blockSignals(False)
        dialog.result_select.setEnabled(False)
        dialog.preview_button.setEnabled(False)
        dialog.apply_button.setEnabled(False)
        dialog.restore_button.setEnabled(False)
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
        dialog.deadline = self.clock() + dialog.prepare_seconds.value()
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
        if dialog.phase != "hold":
            return
        row = {
            "received_ns": received_ns,
            "roll": packet.roll, "pitch": packet.pitch, "yaw": packet.yaw,
            "gyro_x": packet.gyro_x, "gyro_y": packet.gyro_y,
            "gyro_z": packet.gyro_z,
            "acc_x": packet.acc_x, "acc_y": packet.acc_y,
            "acc_z": packet.acc_z}
        _attach_orientation_streams(dialog.host.engine, packet, row)
        step = STEPS[dialog.index]
        dialog.context_samples[step].setdefault(packet.device_id, []).append(row)
        for key, device_id in dialog._participants.items():
            if packet.device_id == device_id:
                dialog.samples_by_part[key][step].append(row)


    def _tick(self) -> None:
        dialog = self.dialog
        if dialog.phase not in ("prepare", "hold"):
            return
        if not any(dialog.host.engine.is_live(device_id)
                   for device_id in dialog._participants.values()):
            dialog.abort("所有参与设备均已断流；本轮标定取消")
            return
        step = STEPS[dialog.index]
        remaining = dialog.deadline - self.clock()
        if remaining > 0:
            counts = ", ".join(
                f"{PARTS[PART_BY_KEY[key]].label} {len(dialog.samples_by_part[key][step])}"
                for key in dialog._participants)
            dialog.progress.setText(
                f"第 {dialog.index + 1}/{len(STEPS)} 步：{BILATERAL_STEP_LABELS[step]}\n"
                f"{'摆好姿势' if dialog.phase == 'prepare' else '保持静止'} · "
                f"{remaining:.1f} 秒" + (f" · {counts} 包" if dialog.phase == "hold" else ""))
            return
        if dialog.phase == "prepare":
            dialog.phase = "hold"
            dialog.deadline = self.clock() + dialog.hold_seconds.value()
            dialog.events.append({"step": step, "hold_start_mono_ns": self.clock_ns()})
            self.beep()
            dialog._tick()
            return
        dialog.events[-1]["hold_end_mono_ns"] = self.clock_ns()
        dialog.index += 1
        self.beep()
        if dialog.index == len(STEPS):
            dialog._finish()
            return
        dialog.phase = "prepare"
        dialog.deadline = self.clock() + dialog.prepare_seconds.value()
        dialog._tick()


    def _save_events(self, session_path: Path, status: str) -> None:
        dialog = self.dialog
        payload = {"format": "wt901-guided-bilateral-steps-v1", "status": status,
                   "wearing_prior_weight": dialog.capture_prior_weight,
                   "participants": dialog._participants, "events": dialog.events,
                   "installation": {key: dialog.device_checks.get(device_id)
                                    for key, device_id in dialog._participants.items()}}
        (session_path / "steps.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


    def _finish(self) -> None:
        dialog = self.dialog
        dialog.phase = "done"
        from .calibration_options import set_prior_enabled
        set_prior_enabled(dialog, True)
        dialog.timer.stop()
        dialog.read_button.setEnabled(True)
        dialog.start_button.setEnabled(True)
        dialog.cancel_button.setEnabled(False)
        recorder, dialog.recorder = dialog.recorder, None
        if recorder is None:
            return
        try:
            recorder.close()
            dialog._save_events(recorder.path, "recorded")
            for key, device_id in dialog._participants.items():
                try:
                    result = self.fit_arm(
                        key, device_id, dialog.samples_by_part[key],
                        dialog.host.engine.axis_mappings[key], prior_weight=dialog.capture_prior_weight)
                    torso_id = dialog.host.engine.assignments.get("torso", "")
                    reference_failed = False
                    if torso_id and torso_id != device_id:
                        context = {step: dialog.context_samples[step].get(torso_id, [])
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
                    dialog.quality_notes[key] = notes
                    payload = asdict(result)
                    payload["attitude_mode"] = dialog.host.engine.algorithm_modes[key]
                    payload["neutral_matrices_by_mode"] = {
                        mode: matrix.tolist() for mode, matrix in
                        neutral_matrices_by_mode(dialog.samples_by_part[key]["neutral"]).items()}
                    payload.update({
                        "format": "wt901-guided-calibration-v1",
                        "method": "bone", "session": str(recorder.path.resolve()),
                        "installation_direction": dialog.device_checks[device_id][0],
                        "fusion_algorithm": dialog.device_checks[device_id][1],
                        "manual_mapping": list(dialog.host.engine.axis_mappings[key]),
                        "events": dialog.events, "capture_mode": "bilateral",
                        "quality_notes": notes})
                    path = recorder.path / f"guided_result_{key}.json"
                    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
                    dialog.results[key] = result
                    dialog.result_paths[key] = path
                except (ValueError, OSError) as exc:
                    dialog.failures[key] = str(exc)
            (recorder.path / "guided_summary.json").write_text(
                json.dumps({"format": "wt901-guided-bilateral-summary-v1",
                            "participants": dialog._participants,
                            "status": {key: result.status for key, result in dialog.results.items()},
                            "quality_notes": dialog.quality_notes,
                            "failures": dialog.failures}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError as exc:
            dialog.progress.setText(f"无法完成联合标定：{exc}；原始记录：{recorder.path}")
            return
        dialog.result_select.blockSignals(True)
        dialog.result_select.clear()
        for key, device_id in dialog._participants.items():
            status = dialog.results[key].status if key in dialog.results else "无效"
            dialog.result_select.addItem(
                f"{PARTS[PART_BY_KEY[key]].label} · {device_id} · {status}", key)
        dialog.result_select.blockSignals(False)
        dialog.result_select.setEnabled(True)
        passed = sum(result.status == "通过" for result in dialog.results.values())
        cautioned = sum(bool(dialog.quality_notes.get(key)) for key, result in
                        dialog.results.items() if result.status == "通过")
        dialog.progress.setText(
            f"联合标定完成：{passed}/{len(dialog._participants)} 个部位通过。"
            f"其中 {cautioned} 个有质量提醒。回放和应用均处理全部合格结果；"
            "下方选择部位查看详情或单独恢复。" +
            ("未绑定在线躯干设备，无法核验标定时躯干是否固定。"
             if not dialog.host.engine.assignments.get("torso") else ""))
        dialog._result_changed()


    def apply(self) -> None:
        dialog = self.dialog
        pending = [(key, dialog.results[key]) for key in ARM_PARTS
                   if key in dialog.results and dialog.results[key].status == "通过"
                   and key not in dialog._applied_parts]
        if not pending:
            return
        cautioned = [PARTS[PART_BY_KEY[key]].label for key, _ in pending
                     if dialog.quality_notes.get(key)]
        if cautioned:
            answer = self.question(
                "质量提醒",
                f"以下部位虽通过基本检查，但有更严格的质量提醒："
                f"{'、'.join(cautioned)}。\n"
                "建议先查看下拉框中的具体误差；仍要应用全部合格结果吗？",
                self.yes_button |
                self.cancel_button,
                self.cancel_button)
            if answer != self.yes_button:
                return
        # Preflight every candidate, so an invalid late candidate cannot leave
        # only a subset of the requested group active.
        check_engine = PoseEngine(dict(dialog.host.engine.assignments),
                                  dict(dialog.host.engine.axis_mappings),
                                  dict(dialog.host.engine.algorithm_modes))
        try:
            for key, result in pending:
                check_engine.apply_calibration(
                    key, result.device_id, np.asarray(result.basis_matrix),
                    np.asarray(result.neutral_matrix),
                    paired_neutrals=neutral_matrices_by_mode(
                        dialog.samples_by_part[key]["neutral"]))
        except ValueError as exc:
            self.warn( "无法应用全部结果", str(exc))
            return
        for key, result in pending:
            dialog.host.engine.apply_calibration(
                key, result.device_id, np.asarray(result.basis_matrix),
                np.asarray(result.neutral_matrix),
                paired_neutrals=neutral_matrices_by_mode(
                    dialog.samples_by_part[key]["neutral"]))
            dialog._applied_parts.add(key)
        dialog.host._render_dirty = True
        dialog.host.statusBar().showMessage(
            f"已临时应用 {len(pending)} 个合格部位；重新启动后需重新验证", 8000)
        dialog._result_changed()


    def restore(self) -> None:
        dialog = self.dialog
        key = dialog.part
        super().restore()
        dialog._applied_parts.discard(key)
        dialog._result_changed()


    def abort(self, reason: str = "已取消本轮标定") -> None:
        dialog = self.dialog
        super().abort(reason)
        dialog.part_combo.setEnabled(False)
        dialog.start_button.setEnabled(bool(dialog.device_checks))

