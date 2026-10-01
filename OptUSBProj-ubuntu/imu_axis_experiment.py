#!/usr/bin/env python3
"""Guided, read-only WT901 axis experiment for one device.

Run this instead of the body GUI: both programs would otherwise bind UDP 1399.
The experiment records the device's raw R/P/Y, acceleration, angular velocity,
magnetometer and host receive times. It never writes device registers.
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

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from PySide6 import QtCore, QtGui, QtWidgets

from wt901.gui_udp import GuiUdpSource
from wt901.protocol import AttitudePacket


DEFAULT_DEVICE_ID = "WT5500011433"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "recordings" / "imu_axis"
PREPARE_SECONDS = 5
STAGES = (
    ("neutral_before", "保持初始中立位，不要移动设备", 5),
    ("raise", "模拟抬臂：抬起设备，末尾保持不动", 8),
    ("twist_out", "模拟旋臂：绕前臂长轴转约 90°，末尾保持", 8),
    ("twist_back", "沿原路反向旋臂，回到抬臂未旋转的姿势", 8),
    ("lower", "模拟落臂：放回最初的中立姿势", 8),
    ("neutral_after", "保持最终中立位，不要移动设备", 5),
)
TOTAL_SECONDS = sum(item[2] for item in STAGES)
PACKET_FIELDS = tuple(field.name for field in fields(AttitudePacket))
CSV_FIELDS = ("sample", "host_recv_mono_ns", "host_write_wall_ns", "elapsed_s",
              "phase", "device_time") + PACKET_FIELDS


def stage_at(elapsed_s: float) -> tuple[str, str, float] | None:
    """Return phase, prompt and remaining seconds; None after the full plan."""
    if elapsed_s < 0:
        raise ValueError("elapsed_s must be nonnegative")
    boundary = 0.0
    for name, prompt, duration in STAGES:
        boundary += duration
        if elapsed_s < boundary:
            return name, prompt, boundary - elapsed_s
    return None


class AxisExperimentRecorder:
    """Incrementally persist every matching packet, not GUI-rate snapshots."""

    def __init__(self, output_dir: Path, device_id: str, start_mono_ns: int) -> None:
        self.path = output_dir / (datetime.now().strftime("%Y%m%d-%H%M%S")
                                  + "-" + uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.device_id = device_id
        self.start_mono_ns = start_mono_ns
        self.start_wall_ns = time.time_ns()
        self.count = 0
        self.phase_counts = {name: 0 for name, _, _ in STAGES}
        self.closed = False
        self.file = (self.path / "samples.csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self._write_manifest("recording")

    def _write_manifest(self, reason: str) -> None:
        data = {
            "format": "wt901-single-axis-experiment-v1",
            "device_id": self.device_id,
            "purpose": "Read-only 6-axis raise/twist/untwist/lower test",
            "fusion_mode": "6-axis (user confirmed; not read from register)",
            "start_mono_ns": self.start_mono_ns,
            "start_wall_ns": self.start_wall_ns,
            "prepare_seconds": PREPARE_SECONDS,
            "stages": [{"name": name, "prompt": prompt, "seconds": duration}
                       for name, prompt, duration in STAGES],
            "samples": self.count,
            "phase_counts": self.phase_counts,
            "stop_reason": reason,
        }
        (self.path / "manifest.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def append(self, packet: AttitudePacket, received_ns: int) -> None:
        if self.closed:
            raise RuntimeError("recording already closed")
        if packet.device_id != self.device_id:
            return
        elapsed_s = (received_ns - self.start_mono_ns) / 1e9
        # The receive queue can still contain packets from before recording began.
        if elapsed_s < 0:
            return
        stage = stage_at(elapsed_s)
        if stage is None:
            return
        name = stage[0]
        row = {
            "sample": self.count,
            "host_recv_mono_ns": received_ns,
            "host_write_wall_ns": time.time_ns(),
            "elapsed_s": elapsed_s,
            "phase": name,
            "device_time": packet.time_str,
            **asdict(packet),
        }
        self.writer.writerow(row)
        self.count += 1
        self.phase_counts[name] += 1
        if self.count % 50 == 0:
            self.file.flush()

    def close(self, reason: str = "completed") -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.file.flush()
        finally:
            self.file.close()
        self._write_manifest(reason)


class AxisExperimentWindow(QtWidgets.QWidget):
    def __init__(self, source: GuiUdpSource, device_id: str,
                 output_dir: Path) -> None:
        super().__init__()
        self.source = source
        self.device_id = device_id
        self.output_dir = output_dir
        self.detected = False
        self.last_seen_ns: int | None = None
        self.prepare_until_ns: int | None = None
        self.recorder: AxisExperimentRecorder | None = None
        self.last_stage: str | None = None
        self.setWindowTitle("WT901 单设备六轴姿态实验")
        self.setMinimumWidth(650)
        layout = QtWidgets.QVBoxLayout(self)

        self.device_label = QtWidgets.QLabel(f"等待 {device_id} 的 UDP 数据…")
        layout.addWidget(self.device_label)
        self.pose_label = QtWidgets.QLabel("原始角度 R/P/Y：—")
        layout.addWidget(self.pose_label)
        self.prompt_label = QtWidgets.QLabel(
            "识别设备后会弹窗提示。先摆好设备，再点击开始；5 秒后自动录制。")
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
            "只读取原始 WT901 数据，不归零、不改 6/9 轴模式、不发送任何寄存器命令。"
            "每一阶段会响铃并显示动作提示；阶段末尾尽量静止。"
            "本窗口独占 UDP 1399，请先关闭人体模型或手腕 GUI。")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def start(self) -> None:
        if self.last_seen_ns is None or time.monotonic_ns() - self.last_seen_ns > 2e9:
            QtWidgets.QMessageBox.warning(self, "设备离线", "请等待目标设备恢复数据")
            return
        self.prepare_until_ns = time.monotonic_ns() + PREPARE_SECONDS * 1_000_000_000
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.prompt_label.setText("准备：保持起始中立位，5 秒后自动开始录制")

    def _begin_recording(self) -> None:
        self.prepare_until_ns = None
        self.recorder = AxisExperimentRecorder(
            self.output_dir, self.device_id, time.monotonic_ns())
        self.last_stage = None
        self.progress.setValue(0)
        self.file_label.setText(f"正在保存：{self.recorder.path}")
        print(f"开始录制 {self.device_id}：{self.recorder.path}", flush=True)

    def finish(self, reason: str = "completed") -> None:
        self.prepare_until_ns = None
        recorder = self.recorder
        self.recorder = None
        self.stop_button.setEnabled(False)
        if recorder is None:
            self.prompt_label.setText("实验已取消，未生成录制")
            self.start_button.setEnabled(self.detected)
            return
        recorder.close(reason)
        self.prompt_label.setText(f"录制结束：{recorder.count} 包。请不要再移动设备。")
        self.file_label.setText(f"数据已保存：{recorder.path}")
        self.start_button.setEnabled(self.detected)
        print(f"录制结束：{recorder.count} 包；{recorder.path}", flush=True)
        QtWidgets.QMessageBox.information(
            self, "实验完成", f"已保存 {recorder.count} 包数据：\n{recorder.path}")

    def tick(self) -> None:
        try:
            packets = self.source.poll_timed()
        except OSError as exc:
            self.timer.stop()
            QtWidgets.QMessageBox.critical(self, "接收失败", str(exc))
            return
        for received_ns, packet in packets:
            if packet.device_id != self.device_id:
                continue
            self.last_seen_ns = received_ns
            self.pose_label.setText(
                f"原始角度 R/P/Y：{packet.roll:+.1f}° / "
                f"{packet.pitch:+.1f}° / {packet.yaw:+.1f}°  "
                f"角速度 X/Y/Z：{packet.gyro_x:+.1f} / "
                f"{packet.gyro_y:+.1f} / {packet.gyro_z:+.1f}°/s")
            if not self.detected:
                self.detected = True
                endpoint = self.source.seen_endpoints().get(self.device_id)
                where = f"（{endpoint[0]}:{endpoint[1]}）" if endpoint else ""
                self.device_label.setText(f"已识别：{self.device_id} {where}")
                self.start_button.setEnabled(True)
                print(f"识别成功：{self.device_id} {where}", flush=True)
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(
                    self, "识别成功", f"已识别 {self.device_id} {where}\n"
                    "摆好起始中立位后，点击“开始实验”。"))
            if self.recorder is not None:
                self.recorder.append(packet, received_ns)

        now_ns = time.monotonic_ns()
        if self.last_seen_ns is not None:
            age_s = (now_ns - self.last_seen_ns) / 1e9
            if age_s > 2:
                self.device_label.setText(f"{self.device_id} 数据中断 {age_s:.1f} 秒")
                if self.recorder is None and self.prepare_until_ns is None:
                    self.start_button.setEnabled(False)
            elif self.recorder is None and self.prepare_until_ns is None:
                self.start_button.setEnabled(True)
        if self.prepare_until_ns is not None:
            remaining = max(0, (self.prepare_until_ns - now_ns) / 1e9)
            self.prompt_label.setText(f"准备：保持起始中立位，{remaining:.1f} 秒后开始")
            if remaining == 0:
                try:
                    self._begin_recording()
                    # now_ns was sampled before the recorder's start time.
                    # The next timer tick will display and record the first phase.
                    return
                except OSError as exc:
                    self.prepare_until_ns = None
                    self.stop_button.setEnabled(False)
                    QtWidgets.QMessageBox.critical(self, "无法保存录制", str(exc))
        if self.recorder is not None:
            elapsed_s = (now_ns - self.recorder.start_mono_ns) / 1e9
            stage = stage_at(elapsed_s)
            if stage is None:
                self.finish()
                return
            name, prompt, remaining = stage
            if name != self.last_stage:
                self.last_stage = name
                QtWidgets.QApplication.beep()
            self.prompt_label.setText(
                f"{prompt}　剩余 {remaining:.1f} 秒；总计 {elapsed_s:.1f}/{TOTAL_SECONDS} 秒")
            self.progress.setValue(min(TOTAL_SECONDS, int(elapsed_s)))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        if self.recorder is not None:
            self.recorder.close("window closed early")
            self.recorder = None
        self.source.close()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="WT901 单设备六轴姿态实验")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        source = GuiUdpSource(port=args.listen_port)
    except OSError as exc:
        parser.exit(1, f"无法监听 UDP {args.listen_port}：{exc}\n"
                    "请先关闭人体模型或手腕 GUI，不能同时占用这个端口。\n")
    window = AxisExperimentWindow(source, args.device_id, args.output_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
