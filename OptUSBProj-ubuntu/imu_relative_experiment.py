#!/usr/bin/env python3
"""Read-only, guided forearm/hand IMU experiment on one UDP listener.

The full device streams are saved before any cross-device time alignment.
Relative orientation is R_forearm(t)^-1 R_hand(t), using interpolated forearm
orientation at each hand sample's *host receive* time. Device clocks are not
assumed to be synchronized. The neutral-relative delta is
R_relative(t) R_relative(neutral)^-1, in the forearm sensor coordinates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation, Slerp

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from wt901.gui_udp import GuiUdpSource
from wt901.protocol import AttitudePacket


HAND_ID = "WT5500011590"
FOREARM_ID = "WT5500011381"
OUTPUT_DIR = PROJECT_DIR / "recordings" / "imu_relative"
PREPARE_SECONDS = 5
MAX_BRACKET_MS = 40.0
STAGES = (
    ("neutral_before", "双臂自然下垂、腕部不动，保持中立位", 5),
    ("raise", "抬起前臂和手，手腕保持不动；末尾停稳", 8),
    ("twist_out", "前臂与手作为整体绕长轴旋转约 90°，腕部不动；末尾停稳", 8),
    ("twist_back", "前臂与手一起沿原路旋回，腕部仍不动；末尾停稳", 8),
    ("lower", "放下手臂，回到最初中立位，腕部不动", 8),
    ("neutral_after", "保持最终中立位，不要移动", 5),
)
TOTAL_SECONDS = sum(duration for _, _, duration in STAGES)
PACKET_FIELDS = tuple(item.name for item in fields(AttitudePacket))
RAW_FIELDS = ("sample", "role", "host_recv_mono_ns", "host_write_wall_ns",
              "elapsed_s", "phase", "device_time") + PACKET_FIELDS
PAIR_FIELDS = (
    "sample", "elapsed_s", "phase", "hand_host_recv_mono_ns",
    "forearm_before_host_recv_mono_ns", "forearm_after_host_recv_mono_ns",
    "nearest_forearm_offset_ms", "forearm_bracket_ms",
    "hand_roll_deg", "hand_pitch_deg", "hand_yaw_deg",
    "forearm_nearest_roll_deg", "forearm_nearest_pitch_deg",
    "forearm_nearest_yaw_deg",
    "relative_qx", "relative_qy", "relative_qz", "relative_qw",
    "relative_rx_deg", "relative_ry_deg", "relative_rz_deg",
    "neutral_delta_qx", "neutral_delta_qy", "neutral_delta_qz", "neutral_delta_qw",
    "neutral_delta_rx_deg", "neutral_delta_ry_deg", "neutral_delta_rz_deg",
    "neutral_delta_angle_deg",
    "hand_neutral_delta_angle_deg", "forearm_neutral_delta_angle_deg",
)


def stage_at(elapsed_s: float) -> tuple[str, str, float] | None:
    if elapsed_s < 0:
        raise ValueError("elapsed_s must be nonnegative")
    end = 0.0
    for name, prompt, duration in STAGES:
        end += duration
        if elapsed_s < end:
            return name, prompt, end - elapsed_s
    return None


def packet_rotation(packet: AttitudePacket) -> Rotation:
    """Use the project's ZYX convention: Rz(yaw) Ry(pitch) Rx(roll)."""
    return Rotation.from_euler("xyz", (packet.roll, packet.pitch, packet.yaw), degrees=True)


def make_relative_rows(
    hand_samples: list[tuple[int, AttitudePacket]],
    forearm_samples: list[tuple[int, AttitudePacket]],
    start_mono_ns: int,
    max_bracket_ms: float = MAX_BRACKET_MS,
) -> tuple[list[dict], dict]:
    """Interpolate forearm on host time; reject poorly bracketed hand samples."""
    if not hand_samples or len(forearm_samples) < 2:
        return [], {"valid_pairs": 0, "rejected_pairs": len(hand_samples),
                    "neutral_baseline_pairs": 0, "neutral_baseline_quat_xyzw": None}
    forearm_times = np.array([stamp for stamp, _ in forearm_samples], dtype=np.int64)
    forearm_rotations = [packet_rotation(packet) for _, packet in forearm_samples]
    rows: list[dict] = []
    rels: list[Rotation] = []
    paired_hands: list[Rotation] = []
    paired_forearms: list[Rotation] = []
    rejected = 0
    for hand_time, hand in hand_samples:
        right = int(np.searchsorted(forearm_times, hand_time, side="left"))
        if right == 0 or right == len(forearm_times):
            rejected += 1
            continue
        left = right - 1
        before_ms = (hand_time - forearm_times[left]) / 1e6
        after_ms = (forearm_times[right] - hand_time) / 1e6
        if before_ms > max_bracket_ms or after_ms > max_bracket_ms:
            rejected += 1
            continue
        interval_ns = int(forearm_times[right] - forearm_times[left])
        if interval_ns <= 0:
            rejected += 1
            continue
        fraction = (hand_time - forearm_times[left]) / interval_ns
        interpolator = Slerp([0.0, 1.0], Rotation.concatenate(
            [forearm_rotations[left], forearm_rotations[right]]))
        forearm = interpolator([fraction])[0]
        nearest_packet = forearm_samples[left if before_ms <= after_ms else right][1]
        hand_rotation = packet_rotation(hand)
        relative = forearm.inv() * hand_rotation
        relative_vector = np.rad2deg(relative.as_rotvec())
        relative_quat = relative.as_quat()
        elapsed = (hand_time - start_mono_ns) / 1e9
        stage = stage_at(elapsed)
        if stage is None:
            continue
        row = {
            "sample": len(rows), "elapsed_s": elapsed, "phase": stage[0],
            "hand_host_recv_mono_ns": hand_time,
            "forearm_before_host_recv_mono_ns": int(forearm_times[left]),
            "forearm_after_host_recv_mono_ns": int(forearm_times[right]),
            "nearest_forearm_offset_ms": min(before_ms, after_ms),
            "forearm_bracket_ms": interval_ns / 1e6,
            "hand_roll_deg": hand.roll, "hand_pitch_deg": hand.pitch,
            "hand_yaw_deg": hand.yaw,
            "forearm_nearest_roll_deg": nearest_packet.roll,
            "forearm_nearest_pitch_deg": nearest_packet.pitch,
            "forearm_nearest_yaw_deg": nearest_packet.yaw,
            **dict(zip(("relative_qx", "relative_qy", "relative_qz", "relative_qw"),
                       relative_quat)),
            **dict(zip(("relative_rx_deg", "relative_ry_deg", "relative_rz_deg"),
                       relative_vector)),
        }
        rows.append(row)
        rels.append(relative)
        paired_hands.append(hand_rotation)
        paired_forearms.append(forearm)

    # The first neutral stage is deliberately still. Exclude its first second
    # (settling after the prompt) and use all remaining 1..5 s matched samples.
    baseline_indices = [index for index, row in enumerate(rows)
                        if 1 <= row["elapsed_s"] < 5]
    baseline: Rotation | None = None
    hand_baseline: Rotation | None = None
    forearm_baseline: Rotation | None = None
    if len(baseline_indices) >= 20:
        baseline = Rotation.concatenate([rels[index] for index in baseline_indices]).mean()
        hand_baseline = Rotation.concatenate(
            [paired_hands[index] for index in baseline_indices]).mean()
        forearm_baseline = Rotation.concatenate(
            [paired_forearms[index] for index in baseline_indices]).mean()
        for row, relative, hand_rotation, forearm_rotation in zip(
                rows, rels, paired_hands, paired_forearms):
            delta = relative * baseline.inv()
            quat = delta.as_quat()
            vector = np.rad2deg(delta.as_rotvec())
            row.update(dict(zip(("neutral_delta_qx", "neutral_delta_qy",
                                 "neutral_delta_qz", "neutral_delta_qw"), quat)))
            row.update(dict(zip(("neutral_delta_rx_deg", "neutral_delta_ry_deg",
                                 "neutral_delta_rz_deg"), vector)))
            row["neutral_delta_angle_deg"] = float(np.linalg.norm(vector))
            row["hand_neutral_delta_angle_deg"] = float(np.rad2deg(
                (hand_rotation * hand_baseline.inv()).magnitude()))
            row["forearm_neutral_delta_angle_deg"] = float(np.rad2deg(
                (forearm_rotation * forearm_baseline.inv()).magnitude()))
    else:
        for row in rows:
            row.update({key: "" for key in PAIR_FIELDS
                        if key.startswith("neutral_delta_") or
                        key.endswith("_neutral_delta_angle_deg")})

    offsets = [row["nearest_forearm_offset_ms"] for row in rows]
    stats = {
        "valid_pairs": len(rows), "rejected_pairs": rejected,
        "neutral_baseline_pairs": len(baseline_indices),
        "neutral_baseline_quat_xyzw": baseline.as_quat().tolist() if baseline else None,
        "hand_baseline_quat_xyzw": hand_baseline.as_quat().tolist() if hand_baseline else None,
        "forearm_baseline_quat_xyzw": (
            forearm_baseline.as_quat().tolist() if forearm_baseline else None),
        "pair_offset_median_ms": float(np.median(offsets)) if offsets else None,
        "pair_offset_p95_ms": float(np.percentile(offsets, 95)) if offsets else None,
    }
    return rows, stats


class DualRecorder:
    def __init__(self, output_dir: Path, hand_id: str, forearm_id: str,
                 start_mono_ns: int) -> None:
        self.path = output_dir / (datetime.now().strftime("%Y%m%d-%H%M%S")
                                  + "-" + uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.hand_id = hand_id
        self.forearm_id = forearm_id
        self.start_mono_ns = start_mono_ns
        self.start_wall_ns = time.time_ns()
        self.samples: dict[str, list[tuple[int, AttitudePacket]]] = {
            "hand": [], "forearm": []}
        self.phase_counts = {name: {"hand": 0, "forearm": 0}
                             for name, _, _ in STAGES}
        self.closed = False
        self.file = (self.path / "samples.csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=RAW_FIELDS)
        self.writer.writeheader()
        self._write_manifest("recording", None)

    def _write_manifest(self, reason: str, pair_stats: dict | None) -> None:
        manifest = {
            "format": "wt901-dual-relative-experiment-v1",
            "hand_id": self.hand_id, "forearm_id": self.forearm_id,
            "start_mono_ns": self.start_mono_ns, "start_wall_ns": self.start_wall_ns,
            "prepare_seconds": PREPARE_SECONDS,
            "stages": [{"name": name, "prompt": prompt, "seconds": seconds}
                       for name, prompt, seconds in STAGES],
            "sample_counts": {role: len(items) for role, items in self.samples.items()},
            "phase_counts": self.phase_counts,
            "pairing": {"clock": "host receive monotonic time",
                        "method": "SO(3) forearm SLERP at each hand sample time",
                        "max_bracket_ms_each_side": MAX_BRACKET_MS,
                        "relative_rotation": "inverse(R_forearm) * R_hand",
                        "neutral_delta": "R_relative(t) * inverse(R_relative(neutral))",
                        "neutral_baseline_window_s": [1, 5],
                        "limitation": (
                            "Both 6-axis headings may have independent offsets. The "
                            "neutral baseline removes only their initial relative "
                            "pose; this is a diagnostic sensor-frame rotation, not "
                            "a calibrated anatomical wrist angle."),
                        **(pair_stats or {})},
            "stop_reason": reason,
        }
        (self.path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def append(self, packet: AttitudePacket, received_ns: int) -> None:
        if self.closed:
            raise RuntimeError("recording already closed")
        role = ("hand" if packet.device_id == self.hand_id else
                "forearm" if packet.device_id == self.forearm_id else None)
        if role is None:
            return
        elapsed = (received_ns - self.start_mono_ns) / 1e9
        if elapsed < 0:
            return
        stage = stage_at(elapsed)
        if stage is None:
            return
        self.samples[role].append((received_ns, packet))
        self.phase_counts[stage[0]][role] += 1
        self.writer.writerow({
            "sample": len(self.samples["hand"]) + len(self.samples["forearm"]) - 1,
            "role": role, "host_recv_mono_ns": received_ns,
            "host_write_wall_ns": time.time_ns(), "elapsed_s": elapsed,
            "phase": stage[0], "device_time": packet.time_str, **asdict(packet),
        })
        if (len(self.samples["hand"]) + len(self.samples["forearm"])) % 100 == 0:
            self.file.flush()

    def close(self, reason: str = "completed") -> dict:
        if self.closed:
            raise RuntimeError("recording already closed")
        self.closed = True
        try:
            self.file.flush()
        finally:
            self.file.close()
        rows, stats = make_relative_rows(self.samples["hand"],
                                         self.samples["forearm"], self.start_mono_ns)
        with (self.path / "relative.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=PAIR_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self._write_manifest(reason, stats)
        return stats


class DualExperimentWindow(QtWidgets.QWidget):
    def __init__(self, source: GuiUdpSource, hand_id: str, forearm_id: str,
                 output_dir: Path) -> None:
        super().__init__()
        self.source = source
        self.hand_id = hand_id
        self.forearm_id = forearm_id
        self.output_dir = output_dir
        self.last_seen: dict[str, int] = {}
        self.last_packets: dict[str, AttitudePacket] = {}
        self.ready_announced = False
        self.prepare_until_ns: int | None = None
        self.recorder: DualRecorder | None = None
        self.last_stage: str | None = None
        self.setWindowTitle("WT901 双设备前臂—手背相对旋转实验")
        self.setMinimumWidth(760)
        layout = QtWidgets.QVBoxLayout(self)
        self.device_label = QtWidgets.QLabel("等待两台设备的数据…")
        layout.addWidget(self.device_label)
        self.pose_label = QtWidgets.QLabel("前臂与手背的 R/P/Y：—")
        layout.addWidget(self.pose_label)
        self.prompt_label = QtWidgets.QLabel(
            "识别两台设备后再开始；5 秒准备，随后自动录制 42 秒。")
        self.prompt_label.setWordWrap(True)
        font = self.prompt_label.font()
        font.setPointSize(max(font.pointSize(), 15))
        font.setBold(True)
        self.prompt_label.setFont(font)
        layout.addWidget(self.prompt_label)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, TOTAL_SECONDS)
        layout.addWidget(self.progress)
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始实验（5 秒准备）")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        buttons.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("提前结束并保存")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.finish("user stopped early"))
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        self.file_label = QtWidgets.QLabel(f"录制目录：{output_dir}")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.file_label)
        note = QtWidgets.QLabel(
            "前臂和手背应固定牢固，整轮只做抬臂和前臂旋转，不主动屈腕/转腕。"
            "程序不向设备发送指令。保存原始双设备数据及按电脑接收时间插值得到的相对旋转；"
            "不假定两台设备时钟同步。六轴独立航向可能偏移，相对值用于诊断，不直接当作"
            "解剖学腕角。独占 UDP 1399，请先关闭人体或手腕 GUI。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def _both_fresh(self, now_ns: int) -> bool:
        return all(device_id in self.last_seen and
                   now_ns - self.last_seen[device_id] <= 2_000_000_000
                   for device_id in (self.hand_id, self.forearm_id))

    def start(self) -> None:
        if not self._both_fresh(time.monotonic_ns()):
            QtWidgets.QMessageBox.warning(self, "设备未就绪", "请等待两台设备恢复数据")
            return
        self.prepare_until_ns = time.monotonic_ns() + PREPARE_SECONDS * 1_000_000_000
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.prompt_label.setText("准备：保持手腕与前臂共同中立，5 秒后开始")

    def finish(self, reason: str = "completed") -> None:
        self.prepare_until_ns = None
        recorder = self.recorder
        self.recorder = None
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(self._both_fresh(time.monotonic_ns()))
        if recorder is None:
            self.prompt_label.setText("实验已取消，未生成录制")
            return
        try:
            stats = recorder.close(reason)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "相对旋转计算失败", f"原始数据已保存：\n{recorder.path}\n\n{exc}")
            return
        counts = recorder.samples
        message = (f"前臂 {len(counts['forearm'])} 包，手背 {len(counts['hand'])} 包；"
                   f"有效配对 {stats['valid_pairs']} 包。\n{recorder.path}")
        self.prompt_label.setText("录制结束；请不要再移动设备")
        self.file_label.setText(f"数据已保存：{recorder.path}")
        print(f"录制结束：{message}", flush=True)
        QtWidgets.QMessageBox.information(self, "实验完成", message)

    def tick(self) -> None:
        try:
            packets = self.source.poll_timed()
        except OSError as exc:
            self.timer.stop()
            QtWidgets.QMessageBox.critical(self, "接收失败", str(exc))
            return
        for received_ns, packet in packets:
            if packet.device_id not in (self.hand_id, self.forearm_id):
                continue
            self.last_seen[packet.device_id] = received_ns
            self.last_packets[packet.device_id] = packet
            if self.recorder is not None:
                self.recorder.append(packet, received_ns)
        now_ns = time.monotonic_ns()
        statuses = []
        for label, device_id in (("前臂", self.forearm_id), ("手背", self.hand_id)):
            age = (now_ns - self.last_seen[device_id]) / 1e9 if device_id in self.last_seen else None
            statuses.append(f"{label} {device_id}：" +
                            ("未识别" if age is None else
                             "在线" if age <= 2 else f"中断 {age:.1f} 秒"))
        self.device_label.setText("　｜　".join(statuses))
        if len(self.last_packets) == 2:
            forearm, hand = self.last_packets[self.forearm_id], self.last_packets[self.hand_id]
            self.pose_label.setText(
                f"前臂 R/P/Y {forearm.roll:+.1f}/{forearm.pitch:+.1f}/{forearm.yaw:+.1f}°"
                f"　手背 {hand.roll:+.1f}/{hand.pitch:+.1f}/{hand.yaw:+.1f}°")
        ready = self._both_fresh(now_ns)
        if ready and not self.ready_announced:
            self.ready_announced = True
            print(f"识别成功：前臂 {self.forearm_id}；手背 {self.hand_id}", flush=True)
            QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(
                self, "识别成功", "两台设备都在线。穿戴固定并摆好中立位后，点击“开始实验”。"))
        if self.prepare_until_ns is None and self.recorder is None:
            self.start_button.setEnabled(ready)
        if self.prepare_until_ns is not None:
            remaining = max(0.0, (self.prepare_until_ns - now_ns) / 1e9)
            self.prompt_label.setText(f"准备：保持共同中立位，{remaining:.1f} 秒后开始")
            if remaining == 0:
                if not ready:
                    self.prepare_until_ns = None
                    self.stop_button.setEnabled(False)
                    QtWidgets.QMessageBox.warning(self, "设备断开", "两台设备必须在线；请重新开始")
                    return
                try:
                    self.prepare_until_ns = None
                    self.recorder = DualRecorder(
                        self.output_dir, self.hand_id, self.forearm_id, time.monotonic_ns())
                    self.last_stage = None
                    self.progress.setValue(0)
                    self.file_label.setText(f"正在保存：{self.recorder.path}")
                    print(f"开始录制：{self.recorder.path}", flush=True)
                    return  # now_ns predates the recorder start time
                except OSError as exc:
                    self.stop_button.setEnabled(False)
                    QtWidgets.QMessageBox.critical(self, "无法保存录制", str(exc))
        if self.recorder is not None:
            elapsed = (now_ns - self.recorder.start_mono_ns) / 1e9
            stage = stage_at(elapsed)
            if stage is None:
                self.finish()
                return
            name, prompt, remaining = stage
            if name != self.last_stage:
                self.last_stage = name
                QtWidgets.QApplication.beep()
            self.prompt_label.setText(
                f"{prompt}　剩余 {remaining:.1f} 秒；总计 {elapsed:.1f}/{TOTAL_SECONDS} 秒")
            self.progress.setValue(min(TOTAL_SECONDS, int(elapsed)))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        if self.recorder is not None:
            self.recorder.close("window closed early")
            self.recorder = None
        self.source.close()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901 前臂和手背双设备相对旋转实验")
    parser.add_argument("--hand-id", default=HAND_ID)
    parser.add_argument("--forearm-id", default=FOREARM_ID)
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.hand_id == args.forearm_id:
        parser.error("前臂和手背必须使用两台不同设备")
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        source = GuiUdpSource(port=args.listen_port)
    except OSError as exc:
        parser.exit(1, f"无法监听 UDP {args.listen_port}：{exc}\n请先关闭其他占用 1399 的 GUI。\n")
    window = DualExperimentWindow(source, args.hand_id, args.forearm_id, args.output_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
