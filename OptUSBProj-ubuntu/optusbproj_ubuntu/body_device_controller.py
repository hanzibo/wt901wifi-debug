"""BodyDeviceController extracted from the main window; compatibility state stays on the host."""
from __future__ import annotations

from .axis_mapping import DEFAULT_AXIS_MAPPING
from .skeleton import PARTS
from .pose import MODE_LABELS


class BodyDeviceController:
    def __init__(self, window, *, save, mapping_dialog_factory, warn, accepted_code):
        self.window = window
        self.save = save
        self.mapping_dialog_factory = mapping_dialog_factory
        self.warn = warn
        self.accepted_code = accepted_code

    def _check_calibration_reconnect(self, device_id: str, received_at: float) -> None:
        window = self.window
        if window.replay_mode:
            return
        previous = window.engine.received_at.get(device_id)
        if previous is None or received_at - previous <= 5.0:
            return
        window.engine.invalidate_neutral(device_id)
        for key, assigned in window.engine.assignments.items():
            if assigned == device_id and key in window.engine.continuous_basis:
                window.engine.clear_calibration(key)
                if window.calibration_dialog is not None:
                    window.calibration_dialog.invalidate_for(device_id)
                window.statusBar().showMessage(
                    f"{device_id} 断流超过 5 秒后恢复；穿戴标定已停用，请重新验证", 10000)


    def _refresh_choices(self, force: bool = False) -> None:
        window = self.window
        seen = window.source.seen_devices(max_age=10.0)
        ids = sorted(set(seen) | {device_id for device_id in window.engine.assignments.values()
                                  if device_id})
        choice_state = tuple((device_id, device_id in seen) for device_id in ids)
        if not force and choice_state == window._choice_state:
            return
        window._choice_state = choice_state
        for key, combo in window.combos.items():
            selected = window.engine.assignments[key]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— 未分配 —", "")
            for device_id in ids:
                suffix = "" if device_id in seen else "（暂离线）"
                combo.addItem(device_id + suffix, device_id)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
            combo.blockSignals(False)


    def _assignment_changed(self, key: str) -> None:
        window = self.window
        combo = window.combos[key]
        device_id = combo.currentData() or ""
        previous = window.engine.assignments[key]
        try:
            proposed = dict(window.engine.assignments)
            proposed[key] = device_id
            self.save(window.settings_path, proposed, window.engine.axis_mappings,
                        window.engine.algorithm_modes)
            window.engine.assign(key, device_id)
        except (ValueError, OSError) as exc:
            self.warn( "分配未保存", str(exc))
        else:
            if previous != device_id and window.calibration_dialog is not None:
                window.calibration_dialog.invalidate_for(previous)
        window._refresh_choices(force=True)
        window._render_dirty = True


    def _update_mapping_button(self, key: str) -> None:
        window = self.window
        mapping = window.engine.axis_mappings[key]
        button = window.mapping_buttons[key]
        button.setText("轴映射*" if mapping != DEFAULT_AXIS_MAPPING else "轴映射")
        button.setToolTip("；".join(f"设备 {axis} → 模型 {target}"
                                for axis, target in zip("XYZ", mapping)))


    def _show_axis_mapping(self, key: str) -> None:
        window = self.window
        part = next(part for part in PARTS if part.key == key)
        dialog = self.mapping_dialog_factory(part.label, window.engine.axis_mappings[key], window)
        if dialog.exec() != self.accepted_code:
            return
        try:
            proposed = dict(window.engine.axis_mappings)
            proposed[key] = dialog.mapping()
            self.save(window.settings_path, window.engine.assignments,
                        proposed, window.engine.algorithm_modes)
            window.engine.set_axis_mapping(key, dialog.mapping())
        except (ValueError, OSError) as exc:
            self.warn( "轴映射未保存", str(exc))
            return
        if window.calibration_dialog is not None:
            window.calibration_dialog.invalidate_for(window.engine.assignments.get(key, ""))
        window._update_mapping_button(key)
        window._render_dirty = True


    def _algorithm_changed(self, key: str) -> None:
        window = self.window
        combo = window.algorithm_combos[key]
        mode = combo.currentData()
        previous = window.engine.algorithm_modes[key]
        if mode == previous:
            return
        if (window.recorder is not None or
                getattr(window, "session", None) is not None):
            error = "录制期间不可切换姿态算法；请先结束录制"
        elif window.calibration_dialog is not None and window.calibration_dialog.isVisible():
            error = "穿戴标定窗口打开时不可切换姿态算法；请先关闭标定窗口"
        else:
            error = ""
        try:
            if error:
                raise ValueError(error)
            window.engine.set_algorithm_mode(key, mode)
            self.save(window.settings_path, window.engine.assignments,
                        window.engine.axis_mappings, window.engine.algorithm_modes)
        except (ValueError, OSError) as exc:
            window.engine.set_algorithm_mode(key, previous)
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(previous))
            combo.blockSignals(False)
            self.warn( "姿态算法未切换", str(exc))
            return
        window._render_dirty = True
        window.statusBar().showMessage(
            f"{key} 已切换为 {MODE_LABELS[mode]}；"
            "设备融合寄存器未改变", 5000)

