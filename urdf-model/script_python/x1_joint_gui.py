#!/usr/bin/env python3
"""Tk GUI that publishes all X1 joint positions for RViz visualization only."""

import math
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from x1_common import JOINT_NAMES, JOINT_SPECS, POSES_DEG, clamp_degrees


class JointStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__("x1_python_joint_gui")
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)

    def publish_degrees(self, positions_deg: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [math.radians(positions_deg[name]) for name in JOINT_NAMES]
        self.publisher.publish(message)


class X1JointGui:
    PUBLISH_INTERVAL_MS = 33

    def __init__(self, node: JointStatePublisher) -> None:
        self.node = node
        self.root = tk.Tk()
        self.root.title("X1 机器人关节控制（仅 RViz）")
        self.root.geometry("1000x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.variables: dict[str, tk.DoubleVar] = {}
        self.demo_enabled = False
        self.demo_started_at = time.monotonic()
        self.closed = False

        self._build_ui()
        self.apply_pose("zero")
        self.root.after(self.PUBLISH_INTERVAL_MS, self._publish_cycle)

    def _build_ui(self) -> None:
        ttk.Label(
            self.root,
            text="X1 关节状态发布器（角度单位：度）",
            font=("Sans", 16, "bold"),
        ).pack(pady=(12, 4))

        ttk.Label(
            self.root,
            text=(
                "仅发布 /joint_states 供 RViz 显示，不连接串口、不控制真机。"
                "启动 RViz 时必须使用 use_gui:=false。"
            ),
            foreground="#a03020",
        ).pack(pady=(0, 10))

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)

        for index, spec in enumerate(JOINT_SPECS):
            row = index % 8
            base_column = (index // 8) * 3
            variable = tk.DoubleVar(value=0.0)
            self.variables[spec.name] = variable

            ttk.Label(
                body, text=f"{spec.name:>3}  {spec.label}", width=16
            ).grid(row=row, column=base_column, sticky="w", padx=(4, 2), pady=7)

            ttk.Scale(
                body,
                from_=spec.minimum_deg,
                to=spec.maximum_deg,
                variable=variable,
                orient="horizontal",
                length=260,
            ).grid(row=row, column=base_column + 1, sticky="ew", padx=4)

            value_label = ttk.Label(body, width=8, anchor="e")
            value_label.grid(row=row, column=base_column + 2, padx=(2, 14))

            def update_label(*_args, var=variable, label=value_label) -> None:
                label.configure(text=f"{var.get():6.1f}°")

            variable.trace_add("write", update_label)
            update_label()

        body.columnconfigure(1, weight=1)
        body.columnconfigure(4, weight=1)

        controls = ttk.Frame(self.root, padding=(10, 6, 10, 12))
        controls.pack(fill="x")
        for pose_name, text in (
            ("zero", "全部归零"),
            ("arms_up", "双臂抬起"),
            ("greeting", "招手姿态"),
            ("grippers_open", "打开夹爪"),
        ):
            ttk.Button(
                controls,
                text=text,
                command=lambda name=pose_name: self.apply_pose(name),
            ).pack(side="left", padx=4)

        self.demo_button = ttk.Button(
            controls, text="开始演示", command=self.toggle_demo
        )
        self.demo_button.pack(side="left", padx=12)
        self.status = ttk.Label(controls, text="等待发布……")
        self.status.pack(side="right", padx=6)

    def current_positions(self) -> dict[str, float]:
        return {
            name: clamp_degrees(name, variable.get())
            for name, variable in self.variables.items()
        }

    def apply_pose(self, pose_name: str) -> None:
        self.demo_enabled = False
        if hasattr(self, "demo_button"):
            self.demo_button.configure(text="开始演示")

        base = POSES_DEG["zero"] if pose_name != "grippers_open" else self.current_positions()
        target = dict(base)
        target.update(POSES_DEG[pose_name])
        for name, value in target.items():
            self.variables[name].set(clamp_degrees(name, value))

    def toggle_demo(self) -> None:
        self.demo_enabled = not self.demo_enabled
        self.demo_started_at = time.monotonic()
        self.demo_button.configure(text="停止演示" if self.demo_enabled else "开始演示")

    def _update_demo(self) -> None:
        elapsed = time.monotonic() - self.demo_started_at
        slow = math.sin(elapsed * 0.8)
        fast = math.sin(elapsed * 1.4)
        demo_degrees = {
            "101": 30.0 * slow, "102": 12.0 * fast,
            "51": -45.0 * slow, "52": -20.0 * fast,
            "54": 55.0 * slow, "56": 25.0 * fast,
            "57": 28.0 + 25.0 * slow,
            "1": 45.0 * slow, "2": 20.0 * fast,
            "4": -55.0 * slow, "6": -25.0 * fast,
            "7": 28.0 - 25.0 * slow,
        }
        for name, value in demo_degrees.items():
            self.variables[name].set(clamp_degrees(name, value))

    def _publish_cycle(self) -> None:
        if self.closed:
            return
        if self.demo_enabled:
            self._update_demo()

        self.node.publish_degrees(self.current_positions())
        rclpy.spin_once(self.node, timeout_sec=0.0)
        subscribers = self.node.publisher.get_subscription_count()
        self.status.configure(text=f"30 Hz 发布中；订阅者：{subscribers}")
        self.root.after(self.PUBLISH_INTERVAL_MS, self._publish_cycle)

    def close(self) -> None:
        self.closed = True
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    rclpy.init()
    node = JointStatePublisher()
    gui = X1JointGui(node)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
