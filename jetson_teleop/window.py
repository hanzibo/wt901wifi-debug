"""Small Qt shell around the validated PC control state machine; no VTK/vision."""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT
from optusbproj_ubuntu.skeleton import PARTS, PART_BY_KEY
from optusbproj_ubuntu.real_robot_arm_gui import RealRobotModesDialog
from optusbproj_ubuntu.real_robot_arm import load_profile
from optusbproj_ubuntu.real_robot_protocol import decode
from optusbproj_ubuntu.scene_robot_model import SceneArm
from optusbproj_ubuntu.real_robot_checkpoints import CheckpointStore


class ControlDialog(RealRobotModesDialog):
    def __init__(self, viewer):
        super().__init__(viewer, include_fingers=False)
        self.setWindowTitle('Jetson · 左臂 / 双臂真机控制（无反馈）')
        for index in (0, 1):
            self.mode_select.model().item(index).setEnabled(False)
        self.mode_select.setCurrentIndex(3 if viewer.config['mode'] == 'bilateral_shared_ik' else 2)
        self.frequency.setValue(viewer.config['frequency'])
        for side, rows, axis, angle in (
                ('left', self.axis_rows, self.shared_offset_axis, self.shared_offset_angle),
                ('right', self.right_axis_rows, self.right_options[2], self.right_options[3])):
            for row, sign in zip(rows, viewer.config['signs'][side]):
                row[5].setCurrentIndex(0 if sign == 1 else 1)
            axis.setCurrentText(viewer.config['palm_offsets'][side]['axis'])
            angle.setValue(viewer.config['palm_offsets'][side]['degrees'])
        self.mirror_check.setChecked(False)
        self.mirror_check.hide()
        self.shared_offset_copy.hide()
        self.right_options[4].hide()
        self.checkpoints.store = CheckpointStore(Path(__file__).parent/'checkpoints')
        self.mode_select.currentIndexChanged.connect(self.load_preset)
        self.load_preset()
        self.resize(1000, 760)

    def load_preset(self):
        self.viewer.zero_confirmed = False
        self.viewer.review.setChecked(False)
        self.viewer.message.setText('模式改变：请重新核对设备、自然下垂归零。')
        try:
            mode = self.checkpoints.mode()
            data = json.loads(self.viewer.config['checkpoints'][mode].read_text(encoding='utf-8'))
            if not data.get('input_settings', {}).get('manual_mapping'):
                raise ValueError('请选择启用手动轴映射的 checkpoint，不使用穿戴标定预设')
            self.checkpoints.apply(data)
            self.viewer.preset_error = ''
        except (OSError, ValueError, KeyError) as exc:
            self.viewer.preset_error = str(exc)
            self.viewer.message.setText(f'预设未加载，禁止连接：{exc}')
        self._controls()

    def _controls(self):
        super()._controls()
        if not getattr(self.viewer, 'zero_confirmed', False) or getattr(self.viewer, 'preset_error', ''):
            self.connect_button.setEnabled(False)
        if self.mode_select is not None:
            self.manual.setEnabled(False)  # This entry deliberately has no wearing calibration.
            self.mirror_check.setEnabled(False)

    def _connect(self):
        if not self.viewer.zero_confirmed or self.viewer.preset_error or not self.manual.isChecked():
            self.status.setText('须成功加载手动映射预设、核对配置并完成自然下垂归零')
            return
        super()._connect()

    def _motion_controls_ready(self, connected):
        ready = super()._motion_controls_ready(connected)
        return ready if self.manual.isChecked() else (False, False)


class Preview(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.models = {s: SceneArm(load_profile(s)[0], [-np.pi]*6, [np.pi]*6)
                       for s in ('left', 'right')}
        self.setMinimumSize(600, 330)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#171d29'))
        transforms = self.window.engine.transforms(include_natural=False)
        human = [[transforms[p.parent][:3, 3], transforms[i][:3, 3]]
                 for i, p in enumerate(PARTS) if p.parent is not None]
        robot = []
        dialog = self.window.control
        frame = dialog.last_frame
        if frame is not None:
            angles, _ = decode(frame)
            for side, rows, start in (('left', dialog.axis_rows, 6), ('right', dialog.right_axis_rows, 0)):
                signs = np.array([1 if row[5].currentIndex() == 0 else -1 for row in rows])
                q = np.deg2rad((np.array(angles[start:start+6])-90) / signs)
                points = self.models[side].points(q)
                robot.extend([[points[0], points[1]], [points[1], points[2]]])
        for column, (title, segments, color) in enumerate((
                ('人体姿态（IMU / 重力纠倾斜）', human, '#76d9f5'),
                ('机器人：末次发送指令 FK（非实测）', robot, '#ffce75'))):
            width = self.width()/2
            painter.setPen(QtGui.QColor('white'))
            painter.drawText(QtCore.QPointF(column*width+15, 28), title)
            if not segments:
                painter.drawText(QtCore.QPointF(column*width+15, 65), '未发送；不会自动连接机器人')
                continue
            points = np.array(segments)
            projected = np.stack((-points[..., 1]+.35*points[..., 0], -points[..., 2]), axis=-1)
            lo, hi = projected.reshape(-1, 2).min(0), projected.reshape(-1, 2).max(0)
            scale = min((width-50)/max(hi[0]-lo[0], .2), (self.height()-90)/max(hi[1]-lo[1], .2))
            projected = (projected-(lo+hi)/2)*scale + [column*width+width/2, self.height()/2+20]
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 5))
            for a, b in projected:
                painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
        painter.end()


class Window(QtWidgets.QWidget):
    replay_mode = False

    def __init__(self, source, config):
        super().__init__()
        self.source, self.config = source, config
        self.zero_confirmed = False
        self.preset_error = ''
        self.engine = PoseEngine(config['assignments'], config['axis_mappings'],
                                 {key: GYRO_TILT for key in config['assignments']})
        self.setWindowTitle('WT901 · Jetson 轻量双臂控制')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('无视觉 / 无穿戴标定 / 无手指控制；航向仍会漂移。暂停不是物理急停。'))
        self.devices = QtWidgets.QLabel()
        self.devices.setTextFormat(QtCore.Qt.PlainText)
        self.devices.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.devices)
        self.review = QtWidgets.QCheckBox('已核对配置文件中的设备部位、手动轴映射、掌向偏置及六轴协议方向')
        layout.addWidget(self.review)
        self.zero_button = QtWidgets.QPushButton('双臂自然下垂：采集当前中立参考（连接串口前）')
        self.zero_button.clicked.connect(self.zero)
        layout.addWidget(self.zero_button)
        self.message = QtWidgets.QLabel('等待 IMU；配置不会自动开启机器人串口。')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.control = ControlDialog(self)
        self.preview = Preview(self)
        layout.addWidget(self.preview, 1)
        button = QtWidgets.QPushButton('打开真机控制 / Checkpoint / 停止手势 / 回位')
        button.clicked.connect(self.show_control)
        layout.addWidget(button)
        self.review.toggled.connect(self.invalidate_zero)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.next_display = 0.
        self.resize(900, 710)

    def invalidate_zero(self):
        self.zero_confirmed = False
        self.control._controls()

    def active_parts(self):
        return [key for key in self.config['assignments']
                if self.control._bilateral_mode() or key.startswith('left_')]

    def zero(self):
        if self.control.worker is not None or not self.review.isChecked() or self.preset_error:
            self.message.setText('请先断开串口、成功加载预设并核对配置。')
            return
        if not all(self.engine.is_live(self.engine.assignments[key], time.monotonic(), .25)
                   for key in self.active_parts()):
            self.message.setText('所选手臂的 IMU 必须全部在线且距末包 ≤250 ms，不能用旧数据归零。')
            return
        self.engine.zero()
        self.zero_confirmed = True
        self.control._controls()
        self.message.setText('自然下垂参考已采集。可打开控制面板；仍须上电确认、安全许可及明确启动。')

    def show_control(self):
        self.control.show()
        self.control.raise_()

    def poll(self):
        for stamp, packet in self.source.poll_timed():
            self.engine.feed(packet, stamp/1e9)
        now = time.monotonic()
        if now >= self.next_display:
            lines = []
            for key, device in self.config['assignments'].items():
                stamp = self.engine.received_at.get(device)
                age = '未收到' if stamp is None else f'{(now-stamp)*1000:.0f} ms'
                mapping = '/'.join(self.config['axis_mappings'][key])
                lines.append(f'{PARTS[PART_BY_KEY[key]].label}: {device} · {mapping} · 距末包 {age}')
            self.devices.setText('\n'.join(lines))
            self.zero_button.setEnabled(self.control.worker is None and self.review.isChecked())
            self.review.setEnabled(self.control.worker is None)
            self.preview.update()
            self.next_display = now+.1

    def closeEvent(self, event):
        if not self.control.close():
            event.ignore()
            QtCore.QTimer.singleShot(100, self.close)
            return
        if getattr(self, 'imu_manager', None) is not None and self.imu_manager.isVisible():
            # The separate manager still needs the existing acquisition timer.
            self.hide()
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)
