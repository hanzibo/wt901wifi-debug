"""Wearing-protocol labels, orientation capture and advisory checks (no Qt)."""
from __future__ import annotations
from .pose import PoseEngine

def _attach_orientation_streams(engine: PoseEngine, packet: object,
                                row: dict) -> None:
    device_id = packet.device_id
    if engine.packets.get(device_id) is not packet:
        return
    by_mode = {mode: matrix.tolist() for mode, matrix in
               engine.mode_current.get(device_id, {}).items()}
    row["orientation_by_mode"] = by_mode
    part = next((key for key, assigned in engine.assignments.items()
                 if assigned == device_id), None)
    mode = engine.algorithm_modes[part] if part is not None else "vendor_rpy"
    row["orientation_mode"] = mode
    if mode in by_mode:
        aligned = engine.calibration_orientation_for(device_id, mode)
        if aligned is not None:
            row["orientation_matrix"] = aligned.tolist()


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
    "neutral": "站到测试位置，左臂自然下垂，肘和手腕伸直，手背朝前",
    "forward": "左臂伸直，带着手向正前方平举到肩高；手腕保持伸直",
    "lateral": "左臂伸直，向身体左侧平举到肩高；手腕保持伸直，不单独转腕",
    "check_forward": "再次左臂前平举到肩高，肘和手腕伸直（独立验证）",
    "check_neutral": "放下左臂，回到最初自然下垂、手背朝前的姿势",
}


RIGHT_WRIST_STEP_LABELS = {
    step: label.replace("左", "右") for step, label in LEFT_WRIST_STEP_LABELS.items()
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

