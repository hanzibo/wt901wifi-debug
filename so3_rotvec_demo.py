#!/usr/bin/env python3
"""Live SO(3) logarithm of one WT901 relative to a clicked GUI neutral pose.

The 3D plot is *rotation-vector space*, not the sensor's position in metres.
Its yellow arrow is log(B @ R0.T @ Rt @ B.T), in radians. The cyan line traces
its endpoint. The principal logarithm jumps at 180 degrees by construction.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import sys
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

from wt901.gui_udp import GuiUdpSource
from wt901_viz.axis_mapping import mapping_matrix
from wt901_viz.orientation import rotation_matrix
from wt901_viz.so3 import rotation_vector


def arrow_segments(vector: np.ndarray) -> np.ndarray:
    """Three line segments, all starting/ending at a fixed spatial origin."""
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return np.zeros((6, 3), dtype=np.float32)
    direction = vector / length
    sideways = np.cross(direction, [0.0, 0.0, 1.0])
    if np.linalg.norm(sideways) < 1e-6:
        sideways = np.cross(direction, [0.0, 1.0, 0.0])
    sideways /= np.linalg.norm(sideways)
    head = vector - direction * min(0.18, length * 0.20)
    wing = sideways * min(0.07, length * 0.08)
    return np.asarray([[0, 0, 0], vector,
                       vector, head + wing,
                       vector, head - wing], dtype=np.float32)


class RotvecWindow(QtWidgets.QMainWindow):
    def __init__(self, source: GuiUdpSource, device_id: str, basis: np.ndarray):
        super().__init__()
        self.source = source
        self.device_id = device_id
        self.basis = basis
        self.current: np.ndarray | None = None
        self.neutral: np.ndarray | None = None
        self.last_packet_at = 0.0
        self.path: deque[np.ndarray] = deque(maxlen=900)
        self.branch_jumps = 0
        self.last_vector: np.ndarray | None = None
        self.setWindowTitle(f"SO(3) 旋转向量 · {device_id}")
        self.resize(1060, 710)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(root)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((19, 21, 28))
        self.view.setCameraPosition(distance=11.5, elevation=24, azimuth=-55)
        grid = gl.GLGridItem()
        grid.setSize(7, 7)
        grid.setSpacing(0.5, 0.5)
        self.view.addItem(grid)
        for axis, color in enumerate(((1.0, 0.25, 0.25, 1.0),
                                      (0.25, 1.0, 0.35, 1.0),
                                      (0.25, 0.48, 1.0, 1.0))):
            points = np.zeros((2, 3), dtype=np.float32)
            points[0, axis], points[1, axis] = -3.4, 3.4
            self.view.addItem(gl.GLLinePlotItem(
                pos=points, color=color, width=2, antialias=True,
                glOptions="additive"))
        # Three great circles outline the radius-pi principal-log ball.
        angles = np.linspace(0, 2 * np.pi, 145)
        for fixed_axis in range(3):
            circle = np.zeros((len(angles), 3), dtype=np.float32)
            free = [axis for axis in range(3) if axis != fixed_axis]
            circle[:, free[0]] = np.pi * np.cos(angles)
            circle[:, free[1]] = np.pi * np.sin(angles)
            self.view.addItem(gl.GLLinePlotItem(
                pos=circle, color=(0.55, 0.60, 0.68, 0.28), width=1,
                antialias=True, glOptions="additive"))
        self.path_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.28, 0.94, 0.96, 0.85), width=2, antialias=True,
            glOptions="additive")
        self.view.addItem(self.path_item)
        self.vector_item = gl.GLLinePlotItem(
            pos=arrow_segments(np.zeros(3)),
            color=(1.0, 0.88, 0.20, 1.0), width=4, mode="lines",
            antialias=True, glOptions="additive")
        self.view.addItem(self.vector_item)
        self.point_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.88, 0.20, 1.0), size=10, pxMode=True,
            glOptions="additive")
        self.view.addItem(self.point_item)
        layout.addWidget(self.view, 1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(360)
        controls = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("设备姿态 → 模型旋转向量")
        title.setStyleSheet("font-size: 17px; font-weight: bold")
        controls.addWidget(title)
        self.status = QtWidgets.QLabel("等待设备数据…")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        self.zero_button = QtWidgets.QPushButton("以当前姿态作为模型零位")
        self.zero_button.setEnabled(False)
        self.zero_button.clicked.connect(self.set_neutral)
        controls.addWidget(self.zero_button)
        self.clear_button = QtWidgets.QPushButton("清空轨迹（不归零）")
        self.clear_button.clicked.connect(self.clear_path)
        controls.addWidget(self.clear_button)
        self.values = QtWidgets.QLabel("请先点击归零")
        self.values.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.values.setStyleSheet("font-family: monospace; font-size: 14px")
        controls.addWidget(self.values)
        self.matrix = QtWidgets.QLabel("")
        self.matrix.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.matrix.setStyleSheet("font-family: monospace")
        controls.addWidget(self.matrix)
        note = QtWidgets.QLabel(
            "图中坐标是旋转向量分量（弧度），不是设备位置。\n"
            "红/绿/蓝：模型 X/Y/Z；黄箭头：旋转向量；青线：端点轨迹；"
            "灰色球面轮廓：转角 180°。\n\n"
            "旋转向量 = 单位旋转轴 × 转角。这里显示 SO(3) 对数的主值；"
            "转过 180° 后箭头可能跳到球的对侧，模型姿态本身仍连续。"
        )
        note.setWordWrap(True)
        controls.addWidget(note)
        controls.addStretch(1)
        layout.addWidget(panel)
        self.setCentralWidget(root)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def clear_path(self) -> None:
        self.path.clear()
        self.last_vector = None
        self.branch_jumps = 0
        self.path_item.setData(pos=np.zeros((2, 3), dtype=np.float32))

    def set_neutral(self) -> None:
        if self.current is None or time.monotonic() - self.last_packet_at > 2.0:
            self.status.setText("设备不在线，不能归零")
            return
        self.neutral = self.current.copy()
        self.clear_path()
        self._render(np.eye(3))
        self.status.setText(f"{self.device_id}：GUI 零位已建立；开始运动即可观察轨迹")

    def poll(self) -> None:
        packet = None
        for received in self.source.poll():
            if received.device_id == self.device_id:
                packet = received
        if packet is None:
            if self.last_packet_at and time.monotonic() - self.last_packet_at > 2.0:
                self.status.setText(f"{self.device_id}：数据中断，保持最后画面")
            return
        self.last_packet_at = time.monotonic()
        self.zero_button.setEnabled(True)
        self.current = rotation_matrix(packet.roll, packet.pitch, packet.yaw)
        if self.neutral is None:
            self.status.setText(
                f"已收到 {self.device_id}；R/P/Y = "
                f"{packet.roll:+.1f}/{packet.pitch:+.1f}/{packet.yaw:+.1f}°。"
                "摆好起始姿势后点击归零。")
            return
        relative = self.neutral.T @ self.current
        model_rotation = self.basis @ relative @ self.basis.T
        self._render(model_rotation)
        self.status.setText(f"{self.device_id} 在线 · 仅 GUI 归零，不写设备寄存器")

    def _render(self, model_rotation: np.ndarray) -> None:
        vector = rotation_vector(model_rotation)
        angle = float(np.linalg.norm(vector))
        if self.last_vector is not None and np.linalg.norm(vector - self.last_vector) > 4.0:
            self.branch_jumps += 1
        self.last_vector = vector.copy()
        self.path.append(vector.copy())
        if len(self.path) >= 2:
            self.path_item.setData(pos=np.asarray(self.path, dtype=np.float32))
        self.vector_item.setData(pos=arrow_segments(vector))
        self.point_item.setData(pos=np.asarray([vector], dtype=np.float32))
        axis = vector / angle if angle > 1e-8 else np.zeros(3)
        self.values.setText(
            f"转角 θ = {np.degrees(angle):7.2f}°\n"
            f"单位轴 s = ({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})\n"
            f"旋转向量 sθ（rad）：\n"
            f"  X {vector[0]:+8.4f}\n  Y {vector[1]:+8.4f}\n"
            f"  Z {vector[2]:+8.4f}\n"
            f"180° 分支跨越：{self.branch_jumps} 次")
        self.matrix.setText(
            "模型相对零位旋转矩阵：\n" +
            "\n".join("  " + " ".join(f"{value:+.3f}" for value in row)
                      for row in model_rotation))


def main() -> int:
    parser = argparse.ArgumentParser(description="单台 WT901 的 SO(3) 旋转向量可视化")
    parser.add_argument("--device-id", default="WT5500011361")
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--mapping", default="+X,+Y,+Z",
                        help="设备三轴到模型三轴，例如 +X,-Z,+Y；默认单位映射")
    args = parser.parse_args()
    try:
        basis = mapping_matrix(tuple(item.strip() for item in args.mapping.split(",")))
    except ValueError as exc:
        parser.error(str(exc))
    if (os.environ.get("WAYLAND_DISPLAY") and
            os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb")):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        source = GuiUdpSource(args.listen_port, host=args.listen_ip)
    except OSError as exc:
        parser.exit(1, f"无法监听 UDP {args.listen_ip}:{args.listen_port}: {exc}\n"
                       "请关闭其他占用 UDP 1399 的程序（例如手腕控制台）。\n")
    with source:
        window = RotvecWindow(source, args.device_id, basis)
        window.show()
        return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
