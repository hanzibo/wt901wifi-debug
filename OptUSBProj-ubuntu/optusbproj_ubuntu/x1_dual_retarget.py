"""Synchronized, visualization-only bilateral X1 trajectory generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .x1_retarget import (
    ARM_JOINTS, ArmKinematics, _angle_degrees, _human_points, _packet_rows,
    fit_arm, load_calibrated_session, retarget_bone_points,
)


SIDES = ("left", "right")
POINT_FIELDS = tuple(
    f"{side}_{name}" for side in SIDES for name in (
        "elbow_target_x", "elbow_target_y", "elbow_target_z",
        "elbow_actual_x", "elbow_actual_y", "elbow_actual_z",
        "wrist_target_x", "wrist_target_y", "wrist_target_z",
        "wrist_actual_x", "wrist_actual_y", "wrist_actual_z",
        "human_bend_deg", "robot_bend_deg", "elbow_error_m",
        "wrist_error_m", "fit_success", "input_stale",
    )
)
DUAL_FIELDS = ("time_s",) + ARM_JOINTS["left"] + ARM_JOINTS["right"] + POINT_FIELDS


def generate_dual_trajectory(
    session: Path, urdf: Path, output: Path, *, rate_hz: float = 30.0,
    max_age_s: float = 0.15,
) -> dict:
    if rate_hz <= 0 or max_age_s <= 0:
        raise ValueError("rate_hz and max_age_s must be positive")
    session = session.resolve()
    engine, assignments = load_calibrated_session(session, SIDES)
    devices = {side: (assignments[f"{side}_upper_arm"],
                      assignments[f"{side}_forearm"]) for side in SIDES}
    needed = set(assignments.values())
    kin = {side: ArmKinematics(urdf, side) for side in SIDES}
    robot_zero = {side: kin[side].points(np.zeros(6)) for side in SIDES}
    human_zero = {side: _human_points(engine, side) for side in SIDES}
    previous = {side: np.zeros(6) for side in SIDES}
    last_seen: dict[str, int] = {}
    first_ns = None
    next_ns = None
    interval_ns = round(1e9 / rate_hz)
    rows: list[dict] = []
    held = {side: 0 for side in SIDES}
    stale_frames = {side: 0 for side in SIDES}

    for stamp_ns, packet in _packet_rows(session / "imu.csv"):
        if packet.device_id not in needed:
            continue
        if first_ns is None:
            first_ns = stamp_ns
            next_ns = stamp_ns
        # No future-sample lookahead: each frame uses the newest already fed
        # packet from each side. A gap holds only the affected arm.
        while next_ns is not None and next_ns < stamp_ns:
            if all(device in last_seen for device in needed):
                row = {"time_s": round((next_ns - first_ns) / 1e9, 6)}
                actual_wrists = {}
                for side in SIDES:
                    model = kin[side]
                    stale = any((next_ns - last_seen[device]) / 1e9 > max_age_s
                                for device in devices[side])
                    if stale:
                        stale_frames[side] += 1
                        q = previous[side].copy()
                        _, target_elbow, target_wrist = model.points(q)
                        human_bend = 0.0
                        success = False
                    else:
                        target_elbow, target_wrist, human_bend = retarget_bone_points(
                            human_zero[side], _human_points(engine, side),
                            robot_zero[side])
                        q, _, wrist_error, success = fit_arm(
                            model, target_elbow, target_wrist, previous[side],
                            human_bend_deg=human_bend)
                        if not success or wrist_error > 0.08:
                            q = previous[side].copy()
                            held[side] += 1
                        previous[side] = q
                    shoulder, elbow, wrist = model.points(q)
                    actual_wrists[side] = wrist
                    elbow_error = float(np.linalg.norm(elbow - target_elbow))
                    wrist_error = float(np.linalg.norm(wrist - target_wrist))
                    for name, value in zip(ARM_JOINTS[side], q):
                        row[name] = float(value)
                    fields = (
                        *target_elbow, *elbow, *target_wrist, *wrist,
                        human_bend, _angle_degrees(elbow - shoulder, wrist - elbow),
                        elbow_error, wrist_error,
                        int(success and wrist_error <= 0.08), int(stale),
                    )
                    for name, value in zip(
                            (field for field in POINT_FIELDS if field.startswith(side + "_")),
                            fields):
                        row[name] = (int(value) if name.endswith(("_fit_success", "_input_stale"))
                                     else float(value))
                row["wrist_separation_m"] = float(np.linalg.norm(
                    actual_wrists["left"] - actual_wrists["right"]))
                rows.append(row)
            next_ns += interval_ns
        engine.feed(packet)
        last_seen[packet.device_id] = stamp_ns

    if not rows:
        raise ValueError("Session has insufficient synchronized bilateral packets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DUAL_FIELDS + ("wrist_separation_m",))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_session": str(session), "output": str(output.resolve()),
        "imu_devices": assignments, "frames": len(rows),
        "duration_s": rows[-1]["time_s"], "sample_rate_hz": rate_hz,
        "arms": {},
        "min_wrist_separation_m": min(row["wrist_separation_m"] for row in rows),
        "note": "Wrist separation is not a complete self-collision check",
        "mode": "RViz preview only; no physical-joint safety limits",
    }
    for side in SIDES:
        summary.setdefault("calibration_quality_notes", {})[side] = {
            segment: json.loads((session / f"guided_result_{side}_{segment}.json")
                                .read_text(encoding="utf-8")).get("quality_notes", [])
            for segment in ("upper_arm", "forearm")
        }
        elbow_errors = [row[f"{side}_elbow_error_m"] for row in rows
                        if not row[f"{side}_input_stale"]]
        wrist_errors = [row[f"{side}_wrist_error_m"] for row in rows
                        if not row[f"{side}_input_stale"]]
        summary["arms"][side] = {
            "held_ik_frames": held[side], "stale_input_frames": stale_frames[side],
            "elbow_error_median_m": float(np.median(elbow_errors)),
            "wrist_error_median_m": float(np.median(wrist_errors)),
            "wrist_error_p95_m": float(np.percentile(wrist_errors, 95)),
        }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
