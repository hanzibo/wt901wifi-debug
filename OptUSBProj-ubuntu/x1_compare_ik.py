"""Replay one v2/v3 X1 diagnostic target trace through four IK combinations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from optusbproj_ubuntu.x1_kinematics import PoEArmKinematics
from optusbproj_ubuntu.x1_tasks import ArmTask
from optusbproj_ubuntu.x1_ik import solve
from optusbproj_ubuntu.x1_live import SIDES, MAX_JOINT_SPEED_DEG_S, MAX_JOINT_STEP_DEG, NOMINAL_TICK_S

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "urdf-model/pkg_robot_model/urdf/pkg_robot_model.urdf"


def matrix(row, prefix, size):
    values = [row.get(f"{prefix}_{i}{j}", "") for i in range(size) for j in range(size)]
    return None if any(v == "" for v in values) else np.array(values, float).reshape(size, size)


def compare(session, output, urdf=DEFAULT_URDF):
    manifest = json.loads((session / "manifest.json").read_text())
    if manifest.get("format") not in ("x1-live-diagnostic-v2", "x1-live-diagnostic-v3"):
        raise ValueError("需 v2/v3 诊断记录，旧记录缺少统一完整位姿目标；请启用双手后录制")
    if manifest["urdf_sha256"] != hashlib.sha256(urdf.read_bytes()).hexdigest():
        raise ValueError("URDF 与录制时不一致，不能公平比较")
    with (session / "frames.csv").open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("记录为空")
    models = {side: PoEArmKinematics(urdf, side) for side in SIDES}
    # Skip the same stale frames in every run. Do not invent missing hand data.
    valid = {side: [matrix(row, f"{side}_target_pose", 4) is not None and
                     matrix(row, f"{side}_relative_target", 3) is not None and
                     row.get(f"{side}_input_stale") == "False" and
                     row.get(f"{side}_hand_state") == "跟随" for row in rows]
             for side in SIDES}
    if any(not any(valid[side]) for side in SIDES):
        raise ValueError("记录需包含两侧手背及有效完整位姿目标，请启用双手重新录制")
    output.mkdir(parents=True, exist_ok=False)
    summary = {"source": str(session.resolve()), "urdf_sha256": manifest["urdf_sha256"],
               "purpose": "offline comparison only; no ROS/hardware writes", "runs": {}}
    for backend in ("least_squares", "jacobian"):
        for mode in ("mimic", "pose"):
            previous = {side: np.zeros(6) for side in SIDES}
            result_rows = []
            key = f"{backend}_{mode}"
            for index, row in enumerate(rows):
                dt = float(row["dt_s"]) if row.get("dt_s") else NOMINAL_TICK_S
                limit = np.deg2rad(min(MAX_JOINT_STEP_DEG, MAX_JOINT_SPEED_DEG_S * max(0., dt)))
                for side in SIDES:
                    if not valid[side][index]:
                        continue
                    point = lambda name: np.array([float(row[f"{side}_{name}_{a}"]) for a in "xyz"])
                    pose = matrix(row, f"{side}_target_pose", 4)
                    task = ArmTask(models[side], point("target_elbow"), point("target_wrist"),
                                   previous[side], float(row[f"{side}_human_bend_deg"]),
                                   matrix(row, f"{side}_relative_target", 3),
                                   pose if mode == "pose" else None)
                    solved = solve(task, backend, limit)
                    accepted = solved.success and (mode == "pose" or solved.position_error_m <= .08)
                    q = solved.q if accepted else previous[side]
                    actual = models[side].frames(q)[models[side].joint_names[-1]]
                    old = previous[side]
                    _, elbow, wrist = models[side].points(q)
                    result_rows.append(dict(frame=row["frame"], side=side, elapsed_s=row["elapsed_s"],
                        accepted=accepted, position_error_cm=100 * float(np.linalg.norm(wrist - pose[:3, 3])),
                        orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(
                            pose[:3, :3].T @ actual[:3, :3]).magnitude())),
                        elbow_error_cm=100 * float(np.linalg.norm(elbow - task.elbow)),
                        solver_ms=solved.elapsed_ms, iterations=solved.iterations,
                        reason=solved.reason, at_limit=solved.at_limit,
                        max_speed_deg_s=float(np.rad2deg(np.max(np.abs(q - old))) / dt) if dt > 0 else 0.,
                        **{f"q{i}": float(v) for i, v in enumerate(q)}))
                    previous[side] = q.copy()
            with (output / f"{key}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
                writer.writeheader(); writer.writerows(result_rows)
            stats = {}
            for side in SIDES:
                selected = [r for r in result_rows if r["side"] == side]
                stats[side] = {"valid_frames": len(selected), "skipped_frames": len(rows) - len(selected)}
                for metric in ("position_error_cm", "orientation_error_deg", "elbow_error_cm", "solver_ms", "max_speed_deg_s"):
                    values = [r[metric] for r in selected]
                    stats[side][metric] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
                stats[side]["rejected_frames"] = sum(not r["accepted"] for r in selected)
                stats[side]["at_limit_frames"] = sum(r["at_limit"] for r in selected)
            summary["runs"][key] = stats
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="新的输出目录，不覆盖已有数据")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    try:
        summary = compare(args.session, args.output, args.urdf)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"无法比较：{exc}\n")
    print(f"四种组合比较完成：{args.output.resolve()}", flush=True)
    for key, sides in summary["runs"].items():
        for side, stats in sides.items():
            print(f"{key} / {side}: 平均位置 {stats['position_error_cm']['mean']:.2f} cm，"
                  f"平均朝向 {stats['orientation_error_deg']['mean']:.2f}°，"
                  f"求解 P95 {stats['solver_ms']['p95']:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
