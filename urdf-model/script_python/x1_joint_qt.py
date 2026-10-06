#!/usr/bin/env python3
"""PySide6 control panel for the X1 robot model shown in RViz."""

import math
import signal
import sys
import time

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from x1_common import JOINT_NAMES, JOINT_SPECS, POSES_DEG, clamp_degrees


class JointStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__('x1_python_joint_qt')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)

    def publish_degrees(self, positions_deg: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [math.radians(positions_deg[name]) for name in JOINT_NAMES]
        self.publisher.publish(message)


class X1JointWindow(QWidget):
    PUBLISH_INTERVAL_MS = 33

    def __init__(self, node: JointStatePublisher) -> None:
        super().__init__()
        self.node = node
        self.spins: dict[str, QDoubleSpinBox] = {}
        self.demo_enabled = False
        self.demo_started_at = time.monotonic()

        self.setWindowTitle('X1 机器人关节控制（仅 RViz）')
        self.resize(1200, 680)
        self.setMinimumSize(960, 550)
        self._build_ui()
        self.apply_pose('zero')

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._publish_cycle)
        self.timer.start(self.PUBLISH_INTERVAL_MS)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        title = QLabel('X1 机器人关节控制')
        title_font = QFont('Noto Sans CJK SC', 19)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        note = QLabel('角度单位：度　|　发布 /joint_states 供 RViz 显示　|　启动 RViz 时使用 use_gui:=false')
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet('color: #9b3b28;')
        outer.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        groups = QVBoxLayout(content)
        groups.setSpacing(12)
        groups.addWidget(self._joint_group('头部', JOINT_SPECS[:2]))

        arms = QHBoxLayout()
        arms.setSpacing(14)
        arms.addWidget(self._joint_group('右臂与右夹爪', JOINT_SPECS[2:9]), 1)
        arms.addWidget(self._joint_group('左臂与左夹爪', JOINT_SPECS[9:]), 1)
        groups.addLayout(arms)
        groups.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        controls = QHBoxLayout()
        for pose_name, label in (
            ('zero', '全部归零'),
            ('arms_up', '双臂抬起'),
            ('greeting', '招手姿态'),
            ('grippers_open', '打开夹爪'),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, name=pose_name: self.apply_pose(name))
            controls.addWidget(button)

        self.demo_button = QPushButton('开始演示')
        self.demo_button.clicked.connect(self.toggle_demo)
        controls.addWidget(self.demo_button)
        controls.addStretch()
        self.status = QLabel('等待发布……')
        controls.addWidget(self.status)
        outer.addLayout(controls)

    def _joint_group(self, title: str, specs: tuple) -> QGroupBox:
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        grid.setColumnStretch(1, 1)

        for row, spec in enumerate(specs):
            label = QLabel(f'{spec.name}  {spec.label}')
            label.setMinimumWidth(125)
            grid.addWidget(label, row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(round(spec.minimum_deg * 10), round(spec.maximum_deg * 10))
            slider.setSingleStep(10)
            slider.setPageStep(50)
            slider.setMinimumWidth(130)
            slider.setMinimumHeight(32)
            grid.addWidget(slider, row, 1)

            spin = QDoubleSpinBox()
            spin.setRange(spec.minimum_deg, spec.maximum_deg)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix('°')
            spin.setMinimumWidth(100)
            grid.addWidget(spin, row, 2)
            self.spins[spec.name] = spin

            slider.valueChanged.connect(
                lambda value, target=spin: self._slider_changed(target, value)
            )
            spin.valueChanged.connect(
                lambda value, target=slider: self._spin_changed(target, value)
            )

        return box

    @staticmethod
    def _slider_changed(spin: QDoubleSpinBox, value: int) -> None:
        with QSignalBlocker(spin):
            spin.setValue(value / 10.0)

    @staticmethod
    def _spin_changed(slider: QSlider, value: float) -> None:
        with QSignalBlocker(slider):
            slider.setValue(round(value * 10))

    def current_positions(self) -> dict[str, float]:
        return {
            name: clamp_degrees(name, spin.value())
            for name, spin in self.spins.items()
        }

    def apply_pose(self, pose_name: str) -> None:
        self.demo_enabled = False
        self.demo_button.setText('开始演示')
        base = POSES_DEG['zero'] if pose_name != 'grippers_open' else self.current_positions()
        target = dict(base)
        target.update(POSES_DEG[pose_name])
        for name, value in target.items():
            self.spins[name].setValue(clamp_degrees(name, value))
        self.node.publish_degrees(self.current_positions())

    def toggle_demo(self) -> None:
        self.demo_enabled = not self.demo_enabled
        self.demo_started_at = time.monotonic()
        self.demo_button.setText('停止演示' if self.demo_enabled else '开始演示')

    def _update_demo(self) -> None:
        elapsed = time.monotonic() - self.demo_started_at
        slow = math.sin(elapsed * 0.8)
        fast = math.sin(elapsed * 1.4)
        demo_degrees = {
            '101': 30.0 * slow, '102': 12.0 * fast,
            '51': -45.0 * slow, '52': -20.0 * fast,
            '54': 55.0 * slow, '56': 25.0 * fast,
            '57': 28.0 + 25.0 * slow,
            '1': 45.0 * slow, '2': 20.0 * fast,
            '4': -55.0 * slow, '6': -25.0 * fast,
            '7': 28.0 - 25.0 * slow,
        }
        for name, value in demo_degrees.items():
            self.spins[name].setValue(clamp_degrees(name, value))

    def _publish_cycle(self) -> None:
        if not rclpy.ok():
            self.timer.stop()
            QApplication.quit()
            return
        if self.demo_enabled:
            self._update_demo()
        self.node.publish_degrees(self.current_positions())
        rclpy.spin_once(self.node, timeout_sec=0.0)
        subscribers = self.node.publisher.get_subscription_count()
        self.status.setText(f'30 Hz 发布中 · 订阅者：{subscribers}')

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Noto Sans CJK SC', 13))

    rclpy.init()
    signal.signal(signal.SIGINT, lambda _signum, _frame: app.quit())
    node = JointStatePublisher()
    try:
        window = X1JointWindow(node)
        window.show()
        app.exec()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
