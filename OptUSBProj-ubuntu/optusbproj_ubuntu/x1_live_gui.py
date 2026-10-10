"""Compact X1 RViz preview dialog embedded in the existing body viewer."""

from __future__ import annotations

import errno
import json
import math
import os
import shlex
import signal
import socket
import sys
import time
from uuid import uuid4
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .x1_live import PREVIEW_PORT, LiveRetargeter, PreviewSender, readiness
from .x1_live_recording import X1DiagnosticRecorder


URDF = (Path(__file__).resolve().parents[2] / "urdf-model" / "pkg_robot_model"
        / "urdf" / "pkg_robot_model.urdf")
RECORDINGS = Path(__file__).resolve().parents[1] / "recordings" / "x1_live"
ROS_WORKSPACE = Path(__file__).resolve().parents[2] / "urdf-model"
ROS_SETUP = Path("/opt/ros/lyrical/setup.bash")
ROS_WORKSPACE_SETUP = ROS_WORKSPACE / "install" / "setup.bash"


class X1PreviewDialog(QtWidgets.QDialog):
    def __init__(self, host: QtWidgets.QMainWindow) -> None:
        super().__init__(host)
        self.host = host
        self.setWindowTitle("X1 · 双臂实时 RViz 预览")
        self.setMinimumWidth(490)
        self.retargeter: LiveRetargeter | None = None
        self.sender: PreviewSender | None = None
        self.recorder: X1DiagnosticRecorder | None = None
        self.ros_process: QtCore.QProcess | None = None
        self.ros_output = ""
        self.rviz_process: QtCore.QProcess | None = None
        self.rviz_output = ""
        self.started_at = 0.0
        self.first_send_at: float | None = None
        self.gripper_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.gripper_socket.bind(('127.0.0.1', 0))
        self.gripper_socket.setblocking(False)
        self.gripper_feedback_at = 0.
        self._policy_id = None
        self._policy_ack = False
        self._policy_started = 0.
        self._policy_error = None
        self._safety = None
        self._bridge_has_safety = None
        self._planning_initialized = False
        self._angle_log_id = None
        self._angle_log_ack = False
        self._angle_log_started = 0.
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QtWidgets.QVBoxLayout(content)
        self.resize(680, 780)
        introduction = QtWidgets.QLabel(
            "复用主窗口的设备分配、穿戴标定与姿态；只预览 RViz，"
            "不会发送机器人硬件命令。双臂需四台已标定设备；"
            "每侧手掌跟随另需一台手背设备，可使用手动映射或已应用的平举标定。")
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        self.readiness_label = QtWidgets.QLabel()
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label)
        self.left_status = QtWidgets.QLabel("左臂：未启动")
        self.right_status = QtWidgets.QLabel("右臂：未启动")
        self.hand_status = QtWidgets.QLabel("左手掌朝向：未启用")
        self.right_hand_status = QtWidgets.QLabel("右手掌朝向：未启用")
        self.hand_status_labels = {"left": self.hand_status,
                                   "right": self.right_hand_status}
        self.ros_status = QtWidgets.QLabel("ROS 预览进程：未连接")
        for label in (self.left_status, self.right_status,
                      self.hand_status, self.right_hand_status, self.ros_status):
            label.setWordWrap(True)
            layout.addWidget(label)
        rviz_row = QtWidgets.QHBoxLayout()
        self.launch_rviz_button = QtWidgets.QPushButton("启动 RViz 模型")
        self.launch_rviz_button.clicked.connect(self.start_rviz)
        rviz_row.addWidget(self.launch_rviz_button)
        self.close_rviz_button = QtWidgets.QPushButton("关闭 RViz 模型")
        self.close_rviz_button.clicked.connect(self.stop_rviz)
        self.close_rviz_button.setEnabled(False)
        rviz_row.addWidget(self.close_rviz_button)
        layout.addLayout(rviz_row)
        self.rviz_status = QtWidgets.QLabel("RViz 模型：未由本窗口启动")
        self.rviz_status.setWordWrap(True)
        layout.addWidget(self.rviz_status)
        self.gripper_ros_button = QtWidgets.QPushButton("启动 ROS 接收（手机夹爪可单独预览）")
        self.gripper_ros_button.clicked.connect(self._start_gripper_receiver)
        layout.addWidget(self.gripper_ros_button)
        gripper_note = QtWidgets.QLabel(
            "手机网页勾选“发送至 RViz”，选择左 7 / 右 57。夹爪与双臂由同一 ROS 桥合并；"
            "也可不启动双臂随动，仅测试夹爪。手机断线或停用后保持末次角度。"
            "请勿同时运行关节滑杆发布器。若复用的是旧 ROS 桥，请先退出并重启。")
        gripper_note.setWordWrap(True)
        layout.addWidget(gripper_note)
        self.gripper_status_label = QtWidgets.QLabel("手机夹爪：尚无 ROS 回报（不是经人体 GUI 转发）")
        self.gripper_status_label.setWordWrap(True)
        layout.addWidget(self.gripper_status_label)
        self.follow_left_hand = QtWidgets.QCheckBox("左手掌朝向跟随")
        self.follow_right_hand = QtWidgets.QCheckBox("右手掌朝向跟随")
        self.hand_toggles = {"left": self.follow_left_hand,
                             "right": self.follow_right_hand}
        hand_row = QtWidgets.QHBoxLayout()
        for toggle in self.hand_toggles.values():
            toggle.toggled.connect(self._refresh_readiness)
            hand_row.addWidget(toggle)
        layout.addLayout(hand_row)
        solver_row = QtWidgets.QHBoxLayout()
        solver_row.addWidget(QtWidgets.QLabel("求解后端"))
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItem("数值最小二乘（现有）", "least_squares")
        self.backend_combo.addItem("PoE / 解析雅可比迭代", "jacobian")
        solver_row.addWidget(self.backend_combo)
        layout.addLayout(solver_row)
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("长度换算"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItem("统一比例（肩到腕，默认）", "uniform")
        self.scale_combo.addItem("独立比例（上臂 / 前臂）", "segments")
        self.scale_combo.setToolTip(
            "按各侧中立位参考点距离自动计算；独立比例不强制固定骨长，"
            "不保证目标更准确或可达。停止随动后切换，再用同一动作对比。")
        scale_row.addWidget(self.scale_combo)
        layout.addLayout(scale_row)
        self.precise_pose = QtWidgets.QCheckBox("优先精确末端位姿（不约束肘部模仿）")
        self.precise_pose.toggled.connect(self._refresh_readiness)
        layout.addWidget(self.precise_pose)
        self.manual_arms = QtWidgets.QCheckBox('允许未标定臂段使用手动轴映射（快速验证）')
        self.manual_arms.setToolTip('默认关闭。缺少穿戴标定的臂段使用主界面手动轴映射；'
                                   '已有标定仍优先使用。须先摆好中立位并归零，设备在线；不更改主界面标定。')
        self.manual_arms.toggled.connect(self._refresh_readiness)
        layout.addWidget(self.manual_arms)
        self.collision_protection = QtWidgets.QCheckBox('启用自碰撞检测与拦截（实验）')
        self.collision_protection.setToolTip(
            '默认关闭。启用后在 ROS 最终合并双臂和夹爪之后检查完整构型及运动区间；'
            '危险、超时或未知构型保持最后安全角度。仅 RViz 自碰撞实验，不是硬件安全保证。'
            '停止双臂和夹爪输入后切换；其他 JointState 发布器会绕过本桥。')
        layout.addWidget(self.collision_protection)
        self.collision_status = QtWidgets.QLabel('防碰撞：关闭（保留原预览限位和限速）')
        self.collision_status.setWordWrap(True)
        layout.addWidget(self.collision_status)
        self.angle_log_checkbox = QtWidgets.QCheckBox('记录 ROS 接收 / 发布关节角日志（诊断抖动）')
        self.angle_log_checkbox.setToolTip('默认关闭，可运行中切换。记录目标角、碰撞检查及 /joint_states 发布角；'
                                         '不是机器人硬件回读。取消勾选或停止 ROS 桥后结束记录。')
        self.angle_log_checkbox.toggled.connect(self._configure_angle_log)
        layout.addWidget(self.angle_log_checkbox)
        self.angle_log_status = QtWidgets.QLabel('角度日志：关闭')
        self.angle_log_status.setWordWrap(True)
        self.angle_log_status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.angle_log_status)
        mode_note = QtWidgets.QLabel(
            "默认保留动作模仿。完整位姿模式需要两侧手背在线；末端定义为 L6/R56，"
            "不是夹爪尖端。不直接指定腕关节角，允许肘部与人体不同；"
            "限速、限位或不可达目标会留下误差，不代表一定精确达到。切换前请停止随动。")
        mode_note.setWordWrap(True)
        layout.addWidget(mode_note)
        hand_note = QtWidgets.QLabel(
            "启用前让所选手背设备在线，手腕伸直，与前臂一起摆好中立位并归零。"
            "动作模仿模式：左手长轴转动驱动关节 5，侧偏驱动关节 6；右手对应 55、56。"
            "掌屈不能由这两个腕关节独立复现。手背断流时保持腕部，双臂位置继续随动。"
            "完整位姿模式：所有臂关节共同求解朝向；手背断流时保持该侧整臂。"
            "长轴腕关节预览范围 ±90°，侧偏 ±45°，并遵守 URDF 数值限位；"
            "若相对角很小，则前臂与手背的共同转动被抵消。")
        hand_note.setWordWrap(True)
        layout.addWidget(hand_note)
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始 RViz 随动")
        self.start_button.clicked.connect(self.start)
        row.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(lambda: self.stop())
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        self.recording_status = QtWidgets.QLabel(
            "诊断记录：开始随动时自动录制；含四个臂段及已启用的手背 IMU、目标点、IK 与关节角。")
        self.recording_status.setWordWrap(True)
        self.recording_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.recording_status)
        note = QtWidgets.QLabel(
            "点击开始后自动启动 x1_ros2_py 中的 ROS 接收进程；若本机端口已有"
            "接收进程，则复用它。关闭此弹窗只会隐藏窗口；随动和 RViz 仍会运行。"
            "请用“停止”结束随动，用“关闭 RViz 模型”结束模型进程。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        # Closing this child dialog only hides it. Clean up owned processes
        # when the whole GUI exits, before Qt destroys the QProcess objects.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)
        self._refresh_readiness()

    def _update_host_button(self) -> None:
        button = getattr(self.host, "x1_preview_button", None)
        if button is None:
            return
        suffix = ("后台随动" if self.retargeter is not None and not self.isVisible()
                  else "RViz 已打开" if self.rviz_process is not None and not self.isVisible()
                  else "RViz")
        button.setText(f"X1 双臂实时预览…（{suffix}）")

    def _shutdown(self) -> None:
        self.timer.stop()
        self.stop()
        self.stop_rviz()
        self.gripper_socket.close()

    def _poll_gripper_status(self) -> None:
        try:
            if self._angle_log_id is not None and not self._angle_log_ack:
                if time.monotonic()-self._angle_log_started < 5:
                    config = dict(version=1, type='angle_log_config', request_id=self._angle_log_id,
                        enabled=self.angle_log_checkbox.isChecked(), sent_mono_ns=time.monotonic_ns())
                    self.gripper_socket.sendto(json.dumps(config).encode(), ('127.0.0.1', 13991))
                else:
                    self.angle_log_status.setText('角度日志未确认：ROS 桥未启动或版本较旧；启动 / 重启接收后重试')
            if self._policy_id is not None and not self._policy_ack:
                config = dict(version=1, type='collision_config', request_id=self._policy_id,
                    enabled=self.collision_protection.isChecked(), sent_mono_ns=time.monotonic_ns())
                self.gripper_socket.sendto(json.dumps(config).encode(), ('127.0.0.1', 13991))
            query = dict(version=1, type='gripper_status_query', sent_mono_ns=time.monotonic_ns())
            self.gripper_socket.sendto(json.dumps(query).encode(), ('127.0.0.1', 13991))
            while True:
                try:
                    raw, peer = self.gripper_socket.recvfrom(4096)
                except BlockingIOError:
                    break
                data = json.loads(raw)
                if peer != ('127.0.0.1', 13991):
                    continue
                if data.get('type') == 'angle_log_config_ack':
                    if data.get('request_id') == self._angle_log_id:
                        self._angle_log_ack = True
                        self._show_angle_log(data.get('angle_log', {}))
                    continue
                if data.get('type') == 'collision_config_ack':
                    if data.get('request_id') == self._policy_id:
                        self._policy_ack = True
                        self._policy_error = data.get('error')
                        self._safety = data.get('safety')
                    continue
                if data.get('type') != 'gripper_status':
                    continue
                if self._angle_log_ack and data.get('angle_log'):
                    self._show_angle_log(data['angle_log'])
                self._bridge_has_safety = bool(data.get('safety', {}).get('supported'))
                if data.get('safety', {}).get('configuration_id') == self._policy_id:
                    self._safety = data['safety']
                self.gripper_feedback_at = time.monotonic()
                rows = []
                for joint, label in (('7', '左'), ('57', '右')):
                    state = data['joints'][joint]
                    age = state['age_s']
                    freshness = ('未收到手机数据' if age is None else
                                 f'跟随 · 延迟 {age*1000:.0f} ms' if age <= .3 else
                                 f'冻结 · 已断流 {age:.1f} s')
                    rows.append(f"{label}夹爪 {joint}：接收目标 {state['target']:.3f} rad · "
                                f"发布角度 {state['position']:.3f} rad · {freshness}")
                rows.append(f"ROS 发布计数 {data['published_count']} · /joint_states 订阅者 "
                            f"{data['subscribers']}（不代表 RViz 已绘制）")
                self.gripper_status_label.setText('\n'.join(rows))
            if time.monotonic()-self.gripper_feedback_at > 1.:
                self.gripper_status_label.setText("手机夹爪：无 ROS 回报；接收进程未运行或版本较旧，请停止后重新启动")
        except (OSError, ValueError, KeyError, TypeError):
            self.gripper_status_label.setText("手机夹爪：ROS 状态读取失败")
        self._show_collision_status()

    def _configure_angle_log(self, checked=False) -> None:
        self._angle_log_id = uuid4().hex
        self._angle_log_ack = False
        self._angle_log_started = time.monotonic()
        self.angle_log_status.setText('角度日志：等待 ROS 确认（未启动时将在下次启动应用）')

    def _show_angle_log(self, state) -> None:
        if state.get('error'):
            text = '角度日志失败：'+state['error']
        else:
            text = '角度日志：'+('记录中' if state.get('enabled') else '已停止')
            if state.get('path'):
                text += ' · '+state['path']
            if state.get('dropped'):
                text += f" · 队列丢失 {state['dropped']} 条（分析时需注意）"
        self.angle_log_status.setText(text)

    def _begin_collision_policy(self) -> None:
        self._planning_initialized = False
        self._configure_angle_log()
        self._policy_id = uuid4().hex
        self._policy_ack = False
        self._policy_error = None
        self._policy_started = time.monotonic()
        self._safety = None
        self._bridge_has_safety = None
        self.collision_protection.setEnabled(False)

    def _show_collision_status(self) -> None:
        if self._policy_error:
            self.collision_status.setText('防碰撞设置失败：'+self._policy_error)
            return
        if self._safety is None:
            self.collision_status.setText('防碰撞：等待 ROS 能力与配置回报' if self._policy_id else
                                         '防碰撞：尚未配置；下次启动时应用复选框')
            return
        data = self._safety
        result = data.get('result', {})
        detail = ''
        if result.get('pair'):
            detail += ' · '+' ↔ '.join(result['pair'])
        if result.get('distance_m') is not None:
            distance = result['distance_m']
            detail += (f" · 最近采样距离 {distance*1000:.2f} mm" if distance >= 0
                       else ' · 检测几何相交（此值不是穿透深度）')
        categories = dict(collision='几何相交', clearance='安全余量不足',
                          timeout='超时未确认', unverified='路径未认证', invalid='无效配置')
        if result.get('category') in categories:
            detail += ' · 判定：'+categories[result['category']]
        original = result.get('requested_result', {})
        if original.get('category') and original.get('category') != result.get('category'):
            detail += ' · 原目标：'+categories.get(original['category'], original.get('reason', ''))
        if result.get('total_elapsed_ms', result.get('elapsed_ms')) is not None:
            elapsed = result.get('total_elapsed_ms', result.get('elapsed_ms'))
            detail += f" · 检查 {elapsed:.1f} ms"
        if result.get('alpha', 1) < 1:
            detail += f" · 本次步长 {result['alpha']*100:.1f}%"
        self.collision_status.setText('防碰撞：'+data.get('state', '未知')+
            (' · '+data['reason'] if data.get('reason') else '')+detail)

    def _collision_ready(self) -> bool:
        if self._policy_error:
            raise ValueError(self._policy_error)
        elapsed = time.monotonic()-self._policy_started
        if not self._policy_ack:
            if (not self.collision_protection.isChecked() and elapsed > 1 and
                    self._bridge_has_safety is False):
                # A legacy bridge has no safety feature; OFF remains compatible.
                self._policy_ack = True
                self.collision_status.setText('防碰撞：关闭（旧 ROS 桥无安全能力）')
                return True
            if elapsed > 18:
                raise ValueError('ROS 桥未确认防碰撞能力；停止并升级/重启接收进程')
            return False
        if self._safety is None:
            return not self.collision_protection.isChecked()
        if self._safety.get('enabled') != self.collision_protection.isChecked():
            raise ValueError('GUI 与 ROS 防碰撞开关不一致')
        if self.collision_protection.isChecked() and self._safety.get('output_speed_limit') is not True:
            raise ValueError('旧 ROS 桥未支持末级限速；请停止并重启 ROS 接收进程')
        if self._safety.get('state') == '故障':
            raise ValueError('防碰撞故障：'+self._safety.get('reason', '未知'))
        return self._safety.get('state') != '初始化'

    def _preconditions(self) -> list[str]:
        reasons = readiness(self.host.engine,
                            follow_left_hand=self.follow_left_hand.isChecked(),
                            follow_right_hand=self.follow_right_hand.isChecked(),
                            task_mode="pose" if self.precise_pose.isChecked() else "mimic",
                            allow_manual_arms=self.manual_arms.isChecked())
        if self.host.calibration_dialog is not None and self.host.calibration_dialog.isVisible():
            reasons.append("请先关闭穿戴标定窗口")
        if self.host.settings_dialog is not None and self.host.settings_dialog.isVisible():
            reasons.append("请先关闭设备参数窗口")
        return reasons

    def _refresh_readiness(self) -> None:
        if self.retargeter is not None:
            self.readiness_label.setText("臂段及已启用手掌的标定快照已锁定；归零或改绑会自动停止预览。")
            return
        reasons = self._preconditions()
        self.readiness_label.setText(
            ("准备就绪：快速验证，未标定臂段使用手动映射；不代表映射精度已验证"
             if self.manual_arms.isChecked() else
             "准备就绪：四个臂段已标定，所选手背设备在线且有中立位") if not reasons
            else "尚不能开始：\n" + "\n".join("• " + reason for reason in reasons))
        self.start_button.setEnabled(not reasons and URDF.is_file())

    def start(self) -> None:
        if self.retargeter is not None:
            return
        reasons = self._preconditions()
        if reasons:
            self._refresh_readiness()
            return
        try:
            self.retargeter = LiveRetargeter(
                self.host.engine, URDF,
                follow_left_hand=self.follow_left_hand.isChecked(),
                follow_right_hand=self.follow_right_hand.isChecked(),
                backend=self.backend_combo.currentData(),
                task_mode="pose" if self.precise_pose.isChecked() else "mimic",
                scale_mode=self.scale_combo.currentData(),
                allow_manual_arms=self.manual_arms.isChecked())
            self.sender = PreviewSender()
            self._start_ros_receiver()
            self._begin_collision_policy()
            self.recorder = X1DiagnosticRecorder(RECORDINGS, self.retargeter, URDF)
        except (OSError, ValueError, KeyError) as exc:
            self._stop_ros_receiver()
            if self.sender is not None:
                self.sender.close()
                self.sender = None
            self.retargeter = None
            self.ros_status.setText(f"启动失败：{exc}")
            return
        self.started_at = time.monotonic()
        self.first_send_at = None
        self.start_button.setEnabled(False)
        self.follow_left_hand.setEnabled(False)
        self.follow_right_hand.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.scale_combo.setEnabled(False)
        self.precise_pose.setEnabled(False)
        self.manual_arms.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.ros_status.setText("正在等待 ROS 预览进程的应答…")
        self.recording_status.setText(f"诊断记录进行中：{self.recorder.path}")
        self._update_host_button()
        self._tick()

    def stop(self, message: str = "已停止；RViz 保持最后一帧") -> None:
        if self.recorder is not None:
            recorder = self.recorder
            self.recorder = None
            try:
                recorder.close(message)
                self.recording_status.setText(
                    f"诊断记录已保存 {recorder.count} 帧：{recorder.path}")
            except (OSError, ValueError) as exc:
                self.recording_status.setText(
                    f"诊断记录写入失败：{exc}；请检查 {recorder.path}")
                message += "；诊断记录未完整保存"
        if self.sender is not None:
            self.sender.close()
            self.sender = None
        self._stop_ros_receiver()
        self.retargeter = None
        self.first_send_at = None
        self.stop_button.setEnabled(False)
        self.follow_left_hand.setEnabled(True)
        self.follow_right_hand.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.scale_combo.setEnabled(True)
        self.precise_pose.setEnabled(True)
        self.manual_arms.setEnabled(True)
        self.collision_protection.setEnabled(True)
        self._policy_id = None
        self._policy_ack = False
        self.left_status.setText("左臂：未启动")
        self.right_status.setText("右臂：未启动")
        self.hand_status.setText("左手掌朝向：未启用")
        self.right_hand_status.setText("右手掌朝向：未启用")
        self.ros_status.setText(message)
        self._refresh_readiness()
        self._update_host_button()

    def _start_gripper_receiver(self) -> None:
        if self._policy_id is not None:
            return
        try:
            self._start_ros_receiver()
            self._begin_collision_policy()
            self.ros_status.setText("ROS 接收已启动或复用；手机夹爪监听本机 UDP 13991")
            self.stop_button.setEnabled(True)
        except (OSError, ValueError) as exc:
            self.ros_status.setText(f"夹爪 ROS 启动失败：{exc}")

    def _start_ros_receiver(self) -> None:
        if (self.ros_process is not None and
                self.ros_process.state() != QtCore.QProcess.ProcessState.NotRunning):
            return
        # A separately started listener belongs to the user, not this dialog.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("127.0.0.1", PREVIEW_PORT))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return
            raise
        finally:
            probe.close()

        conda_exe = os.environ.get("CONDA_EXE")
        conda_base = (Path(conda_exe).resolve().parent.parent if conda_exe else
                      Path(sys.prefix).parents[1])
        conda_sh = conda_base / "etc" / "profile.d" / "conda.sh"
        if not conda_sh.is_file():
            raise ValueError(f"找不到 Conda 初始化脚本：{conda_sh}")
        if not ROS_SETUP.is_file() or not ROS_WORKSPACE_SETUP.is_file():
            raise ValueError("找不到 ROS 2 或 X1 工作空间的 setup.bash")
        command = (f"source {shlex.quote(str(conda_sh))} && "
                   "conda activate x1_ros2_py && "
                   f"source {shlex.quote(str(ROS_SETUP))} && "
                   "source install/setup.bash && "
                   "exec python -u script_python/x1_imu_live.py --collision-python " +
                   shlex.quote(str(conda_base / 'envs/OptUSBProj-yolo/bin/python')) +
                   (' --collision-start-enabled' if self.collision_protection.isChecked() else ''))
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(ROS_WORKSPACE))
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_ros_output)
        process.finished.connect(lambda _code, _status, p=process: self._ros_finished(p))
        process.start("/bin/bash", ["-c", command])
        if not process.waitForStarted(3000):
            error = process.errorString()
            process.deleteLater()
            raise OSError(f"ROS 接收进程启动失败：{error}")
        self.ros_output = ""
        self.ros_process = process

    def _read_ros_output(self) -> None:
        if self.ros_process is not None:
            self.ros_output = (self.ros_output + bytes(
                self.ros_process.readAllStandardOutput()).decode("utf-8", "replace"))[-2000:]

    def _ros_finished(self, process: QtCore.QProcess) -> None:
        if process is not self.ros_process:
            return
        self._read_ros_output()
        detail = self.ros_output.strip().splitlines()
        self.ros_process = None
        process.deleteLater()
        if self.retargeter is not None:
            self.stop("ROS 接收进程已退出" + (f"：{detail[-1]}" if detail else ""))

    @staticmethod
    def _external_display_launch_running() -> bool:
        """Avoid starting a second model publisher for the same ROS graph."""
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                args = (entry / "cmdline").read_bytes().split(b"\0")
            except (OSError, PermissionError):
                continue
            if (b"pkg_robot_model" in args and b"display.launch.py" in args and
                    b"use_gui:=false" in args):
                return True
        return False

    def start_rviz(self) -> None:
        if self.rviz_process is not None:
            return
        if self._external_display_launch_running():
            self.rviz_status.setText(
                "检测到已有 X1 display.launch.py；请使用现有 RViz，或先关闭它再从此处启动。")
            return
        if not ROS_SETUP.is_file() or not ROS_WORKSPACE_SETUP.is_file():
            self.rviz_status.setText("启动失败：找不到 ROS 2 或 X1 工作空间的 setup.bash")
            return
        command = (f"source {shlex.quote(str(ROS_SETUP))} && "
                   "source install/setup.bash && "
                   "exec ros2 launch pkg_robot_model display.launch.py use_gui:=false")
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(ROS_WORKSPACE))
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        # ros2 launch owns RViz and robot_state_publisher. A private process
        # group lets the button stop its whole tree without touching user jobs.
        unix = QtCore.QProcess.UnixProcessParameters()
        unix.flags = QtCore.QProcess.UnixProcessFlag.CreateNewSession
        process.setUnixProcessParameters(unix)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        for name in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONPATH",
                     "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH",
                     "QT_QPA_PLATFORM_PLUGIN_PATH", "LD_LIBRARY_PATH"):
            environment.remove(name)
        environment.insert("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_rviz_output)
        process.finished.connect(lambda _code, _status, p=process: self._rviz_finished(p))
        process.start("/bin/bash", ["-c", command])
        if not process.waitForStarted(3000):
            self.rviz_status.setText(f"RViz 模型启动失败：{process.errorString()}")
            process.deleteLater()
            return
        self.rviz_process = process
        self.rviz_output = ""
        self.launch_rviz_button.setEnabled(False)
        self.close_rviz_button.setEnabled(True)
        self.rviz_status.setText("RViz 模型：正在启动…")
        self._update_host_button()

    def _read_rviz_output(self) -> None:
        if self.rviz_process is not None:
            self.rviz_output = (self.rviz_output + bytes(
                self.rviz_process.readAllStandardOutput()).decode("utf-8", "replace"))[-3000:]
            if "[rviz2" in self.rviz_output:
                self.rviz_status.setText("RViz 模型：已启动；等待窗口显示")

    def _rviz_finished(self, process: QtCore.QProcess) -> None:
        if process is not self.rviz_process:
            return
        self._read_rviz_output()
        detail = self.rviz_output.strip().splitlines()
        self.rviz_process = None
        self.launch_rviz_button.setEnabled(True)
        self.close_rviz_button.setEnabled(False)
        self.rviz_status.setText("RViz 模型：进程已退出" +
                                 (f"；{detail[-1]}" if detail else ""))
        process.deleteLater()
        self._update_host_button()

    def stop_rviz(self) -> None:
        process = self.rviz_process
        self.rviz_process = None
        if process is None:
            return
        if process.state() != QtCore.QProcess.ProcessState.NotRunning:
            group_id = int(process.processId())
            try:
                if group_id > 0:
                    os.killpg(group_id, signal.SIGINT)
            except ProcessLookupError:
                pass
            if not process.waitForFinished(4000):
                try:
                    if group_id > 0:
                        os.killpg(group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if not process.waitForFinished(1500):
                    try:
                        if group_id > 0:
                            os.killpg(group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.waitForFinished(500)
        process.deleteLater()
        self.launch_rviz_button.setEnabled(True)
        self.close_rviz_button.setEnabled(False)
        self.rviz_status.setText("RViz 模型：已关闭")
        self._update_host_button()

    def _stop_ros_receiver(self) -> None:
        process = self.ros_process
        self.ros_process = None
        if process is None:
            return
        if process.state() != QtCore.QProcess.ProcessState.NotRunning:
            process.terminate()
            if not process.waitForFinished(1500):
                process.kill()
                process.waitForFinished(500)
        process.deleteLater()

    def _tick(self) -> None:
        self._poll_gripper_status()
        if self.retargeter is None or self.sender is None:
            self._refresh_readiness()
            return
        if (self.host.calibration_dialog is not None and
                self.host.calibration_dialog.isVisible()) or (
                self.host.settings_dialog is not None and
                self.host.settings_dialog.isVisible()):
            self.stop("标定或设备参数窗口已打开，X1 预览已停止")
            return
        try:
            if not self._collision_ready():
                return
            tick_ns = time.monotonic_ns()
            self.sender.poll_ack()
            if self.collision_protection.isChecked():
                choices = [s for s in (self.sender.safety, self._safety) if s and
                           s.get('configuration_id') == self._policy_id]
                safety = max(choices, key=lambda s:s.get('revision', -1)) if choices else None
                if (safety and safety.get('configuration_id') == self._policy_id and
                        safety.get('accepted') and (not self._planning_initialized or
                            safety.get('state') != '正常')):
                    self.retargeter.accept_published(tuple(safety['accepted'][name]
                        for name in ('1','2','3','4','5','6','51','52','53','54','55','56')))
                    self._planning_initialized = True
                # Continue the candidate trajectory during normal tracking.
                # Re-anchoring EVERY tick to delayed feedback creates alternating
                # trajectories. The ROS bridge independently limits/checks output.
            q, status = self.retargeter.step(now=tick_ns / 1e9,
                                           commit=True)
            sent_at = time.monotonic()
            self.sender.send(q)
            if self.first_send_at is None:
                # Geometry initialization has its own watchdog. Start the
                # first-ACK deadline only when a frame was actually sent.
                self.first_send_at = sent_at
            self.sender.poll_ack()
            if self.recorder is not None:
                self.recorder.write_frame(
                    self.retargeter, q, status, tick_mono_ns=tick_ns,
                    send_seq=self.sender.sequence,
                    ack_seq=self.sender.last_ack_seq,
                    ack_recv_mono_ns=self.sender.last_ros_recv_mono_ns,
                    ros_subscribers=self.sender.subscribers,
                    safety=safety if self.collision_protection.isChecked() else self._safety)
        except (OSError, ValueError, RuntimeError) as exc:
            self.stop(f"X1 预览已停止：{exc}")
            return
        for side, label in (("left", self.left_status), ("right", self.right_status)):
            state = status[side]
            candidate_note = '（IK 候选，实际输出见防碰撞状态）' if self.collision_protection.isChecked() else ''
            label.setText(f"{'左' if side == 'left' else '右'}臂{candidate_note}：{state['state']}"
                          f" · 数据延迟 {1000 * state['age_s']:.0f} ms"
                          f" · 腕点误差 {state['wrist_error_m'] * 100:.1f} cm")
            if state["world_orientation_error_deg"] is not None:
                label.setText(label.text() +
                    f" · 世界朝向误差 {state['world_orientation_error_deg']:.1f}°")
            if state["solver_ms"] is not None:
                label.setText(label.text() + f" · 求解 {state['solver_ms']:.1f} ms"
                              f" / {state['solver_iterations']} 次 · {state['solver_reason']}"
                              + (" · 触限" if state["solver_at_limit"] else ""))
        for side, label in self.hand_status_labels.items():
            hand = status[side]
            side_label = "左" if side == "left" else "右"
            if hand["hand_age_s"] is None:
                label.setText(f"{side_label}手掌朝向：未启用")
                continue
            axial = hand["hand_axial_deg"]
            axial_text = ("相对长轴角：无新数据" if axial is None else
                          f"手掌相对前臂长轴 {axial:+.1f}°")
            offset = 0 if side == "left" else 6
            joint_names = ("5", "6") if side == "left" else ("55", "56")
            joint_axial_deg = math.degrees(q[offset + 4])
            at_limit = "（达到预览 ±90° 限幅）" if abs(joint_axial_deg) >= 89.5 else ""
            label.setText(
                f"{side_label}手掌朝向：{hand['hand_state']} · {axial_text} · "
                f"X1 关节 {joint_names[0]} {joint_axial_deg:+.1f}°{at_limit} · "
                f"关节 {joint_names[1]} {math.degrees(q[offset + 5]):+.1f}° · "
                f"朝向误差 {hand['hand_orientation_error_deg']:.1f}° · "
                f"数据延迟 {1000 * hand['hand_age_s']:.0f} ms")
        if self.sender.last_ack_at is None:
            if self.first_send_at is not None and time.monotonic() - self.first_send_at > 8.0:
                detail = self.ros_output.strip().splitlines()
                self.stop("ROS 预览进程未应答" +
                          (f"：{detail[-1]}" if detail else "；请检查 ROS 环境"))
            return
        if time.monotonic() - self.sender.last_ack_at > 2.0:
            self.stop("ROS 预览进程应答中断，已停止发送")
            return
        self.ros_status.setText(
            "ROS 进程已连接；" + ("/joint_states 有订阅者" if self.sender.subscribers
                                 else "/joint_states 暂无订阅者，请检查 RViz 启动链路"))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Keep the live loop and RViz running while the user observes the
        # human model and robot model without this control window in the way.
        super().closeEvent(event)
        self._update_host_button()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.timer.start()
        self._update_host_button()
