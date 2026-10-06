"""Compact X1 RViz preview dialog embedded in the existing body viewer."""

from __future__ import annotations

import errno
import math
import os
import shlex
import signal
import socket
import sys
import time
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

    def _preconditions(self) -> list[str]:
        reasons = readiness(self.host.engine,
                            follow_left_hand=self.follow_left_hand.isChecked(),
                            follow_right_hand=self.follow_right_hand.isChecked(),
                            task_mode="pose" if self.precise_pose.isChecked() else "mimic")
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
            "准备就绪：四个臂段已标定，所选手背设备在线且有中立位" if not reasons
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
                scale_mode=self.scale_combo.currentData())
            self.sender = PreviewSender()
            self._start_ros_receiver()
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
        self.start_button.setEnabled(False)
        self.follow_left_hand.setEnabled(False)
        self.follow_right_hand.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.scale_combo.setEnabled(False)
        self.precise_pose.setEnabled(False)
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
        self.stop_button.setEnabled(False)
        self.follow_left_hand.setEnabled(True)
        self.follow_right_hand.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.scale_combo.setEnabled(True)
        self.precise_pose.setEnabled(True)
        self.left_status.setText("左臂：未启动")
        self.right_status.setText("右臂：未启动")
        self.hand_status.setText("左手掌朝向：未启用")
        self.right_hand_status.setText("右手掌朝向：未启用")
        self.ros_status.setText(message)
        self._refresh_readiness()
        self._update_host_button()

    def _start_ros_receiver(self) -> None:
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
                   "exec python -u script_python/x1_imu_live.py")
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
            tick_ns = time.monotonic_ns()
            q, status = self.retargeter.step(now=tick_ns / 1e9)
            self.sender.send(q)
            self.sender.poll_ack()
            if self.recorder is not None:
                self.recorder.write_frame(
                    self.retargeter, q, status, tick_mono_ns=tick_ns,
                    send_seq=self.sender.sequence,
                    ack_seq=self.sender.last_ack_seq,
                    ack_recv_mono_ns=self.sender.last_ros_recv_mono_ns,
                    ros_subscribers=self.sender.subscribers)
        except (OSError, ValueError, RuntimeError) as exc:
            self.stop(f"X1 预览已停止：{exc}")
            return
        for side, label in (("left", self.left_status), ("right", self.right_status)):
            state = status[side]
            label.setText(f"{'左' if side == 'left' else '右'}臂：{state['state']}"
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
            if time.monotonic() - self.started_at > 8.0:
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
