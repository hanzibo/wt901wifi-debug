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
        self.viewer.invalidate_zero()
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
        if hasattr(self.viewer, 'mode_choice'):
            self.viewer.sync_mode()

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

    @staticmethod
    def project(points, mirror=False):
        points = np.asarray(points)
        horizontal = points[..., 1]+.35*points[..., 0]
        return np.stack((-horizontal if mirror else horizontal, -points[..., 2]), axis=-1)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#171d29'))
        transforms = self.window.engine.transforms(include_natural=False)
        human = [(transforms[p.parent][:3, 3], transforms[i][:3, 3],
                  'left' if p.key.startswith('left_') else 'right' if p.key.startswith('right_') else '')
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
                robot.extend([(points[0], points[1], side), (points[1], points[2], side)])
        for column, (title, segments, color) in enumerate((
                ('人体姿态（IMU / 重力纠倾斜）', human, '#76d9f5'),
                ('机器人：末次发送指令 FK（非实测）', robot, '#ffce75'))):
            width = self.width()/2
            painter.setPen(QtGui.QColor('white'))
            painter.drawText(QtCore.QPointF(column*width+15, 28), title)
            if not segments:
                painter.drawText(QtCore.QPointF(column*width+15, 65), '未发送；不会自动连接机器人')
                continue
            points = np.array([[a, b] for a, b, side in segments])
            projected = self.project(points, self.window.preview_mirror.isChecked())
            lo, hi = projected.reshape(-1, 2).min(0), projected.reshape(-1, 2).max(0)
            scale = min((width-50)/max(hi[0]-lo[0], .2), (self.height()-90)/max(hi[1]-lo[1], .2))
            projected = (projected-(lo+hi)/2)*scale + [column*width+width/2, self.height()/2+20]
            labeled = set()
            for (a, b), (_, _, side) in zip(projected, segments):
                side_color = {'left':'#76d9f5', 'right':'#ffb86b'}.get(side, '#dbe3ed')
                painter.setPen(QtGui.QPen(QtGui.QColor(side_color), 5))
                painter.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
                if side and side not in labeled:
                    labeled.add(side)
                    painter.drawText(QtCore.QPointF(*(b+[8, -8])), '左 L' if side == 'left' else '右 R')
        painter.end()


class Window(QtWidgets.QWidget):
    replay_mode = False

    def __init__(self, source, config):
        super().__init__()
        self.source, self.config = source, config
        self.zero_confirmed = False
        self.zeroed_parts = set()
        self.zero_pending = None
        self.zero_timer = QtCore.QTimer(self)
        self.zero_timer.setInterval(100)
        self.zero_timer.timeout.connect(self.tick_zero)
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
        self.learning_summary = QtWidgets.QLabel()
        self.learning_summary.setWordWrap(True)
        layout.addWidget(self.learning_summary)
        self.review = QtWidgets.QCheckBox('已核对配置文件中的设备部位、手动轴映射、掌向偏置及六轴协议方向')
        layout.addWidget(self.review)
        self.zero_delays = {}
        self.zero_buttons = {}
        for group, text in (('arms', '上臂 / 前臂自然下垂：定时采集中立位（不含手背）'),
                            ('hands', '手背保持参考姿态：单独定时采集中立位')):
            row = QtWidgets.QHBoxLayout()
            delay = QtWidgets.QDoubleSpinBox()
            delay.setRange(0, 60)
            delay.setValue(3)
            delay.setSuffix(' 秒')
            delay.setToolTip('点击后等待此时间，再采集当时的最新姿态；0 秒为立即采集。')
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, group=group: self.zero(group))
            self.zero_delays[group] = delay
            self.zero_buttons[group] = button
            row.addWidget(button, 1)
            row.addWidget(delay)
            layout.addLayout(row)
        self.zero_button = self.zero_buttons['arms']
        self.zero_cancel = QtWidgets.QPushButton('取消中立位采集倒计时')
        self.zero_cancel.clicked.connect(self.cancel_zero)
        self.zero_cancel.setEnabled(False)
        layout.addWidget(self.zero_cancel)
        self.message = QtWidgets.QLabel('等待 IMU；配置不会自动开启机器人串口。')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.control = ControlDialog(self)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel('当前控制 / 采集范围'))
        self.mode_choice = QtWidgets.QComboBox()
        self.mode_choice.addItems(['仅左臂（臂段 2 台 + 手背 1 台）', '双臂（臂段 4 台 + 手背 2 台）'])
        self.mode_choice.currentIndexChanged.connect(self.choose_mode)
        mode_row.addWidget(self.mode_choice, 1)
        layout.insertLayout(1, mode_row)
        self.mode_summary = QtWidgets.QLabel()
        self.mode_summary.setWordWrap(True)
        layout.insertWidget(2, self.mode_summary)
        self.preview_mirror = QtWidgets.QCheckBox('镜像预览（仅改变画面，不交换设备或控制左右）')
        self.preview_mirror.toggled.connect(self.update)
        layout.addWidget(self.preview_mirror)
        layout.addWidget(QtWidgets.QLabel('默认非镜像正视图：人体左侧在画面右侧。蓝色 = 左，橙色 = 右。'))
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
        self.sync_mode()
        self.resize(900, 710)

    def choose_mode(self, index):
        d = self.control
        if d.worker is not None or d.busy or d.ik_worker is not None or d.auto_start.pending:
            self.sync_mode()
            return
        d.mode_select.setCurrentIndex(3 if index else 2)
        self.sync_mode()

    def sync_mode(self):
        bilateral = self.control._bilateral_mode()
        with QtCore.QSignalBlocker(self.mode_choice):
            self.mode_choice.setCurrentIndex(1 if bilateral else 0)
        d = self.control
        self.mode_choice.setEnabled(d.worker is None and not d.busy and d.ik_worker is None
                                    and not d.auto_start.pending)
        arms = [k for k in self.active_parts() if not k.endswith('_hand')]
        hands = [k for k in self.active_parts() if k.endswith('_hand')]
        scope = '双臂' if bilateral else '仅左臂'
        self.zero_buttons['arms'].setText(f'{scope}上臂 / 前臂自然下垂：定时采集 {len(arms)} 台（不含手背）')
        self.zero_buttons['hands'].setText(f'{scope}手背保持参考姿态：单独定时采集 {len(hands)} 台')
        self.mode_summary.setText(
            f'中立参考完成：臂段 {len(self.zeroed_parts.intersection(arms))}/{len(arms)}，'
            f'手背 {len(self.zeroed_parts.intersection(hands))}/{len(hands)}。'+
            ('两侧均参与采集。' if bilateral else '右臂不参与本次采集 / 真机输出；右侧预览未归零不能作为姿态依据。'))

    def invalidate_zero(self):
        self.zero_confirmed = False
        self.zeroed_parts.clear()
        self.cancel_zero()
        if hasattr(self, 'control'):
            self.control._controls()

    def cancel_zero(self):
        pending = self.zero_pending is not None
        self.zero_pending = None
        self.zero_timer.stop()
        if hasattr(self, 'zero_cancel'):
            self.zero_cancel.setEnabled(False)
        if pending:
            self.message.setText('采集已取消；请重新采集未完成的组。')

    def active_parts(self):
        return [key for key in self.config['assignments']
                if self.control._bilateral_mode() or key.startswith('left_')]

    def zero_blockers(self, keys, require_static=False):
        problems = []
        for key in keys:
            device = self.engine.assignments[key]
            state = self.engine.tilt_status(device)
            label = PARTS[PART_BY_KEY[key]].label
            if not self.engine.is_live(device, time.monotonic(), .25):
                problems.append(label+'数据未到 / 超过250 ms')
            elif not state['gravity_ready'] or not state['bias_ready']:
                problems.append(label+'重力参考 / 零偏学习未完成')
            elif require_static and not state['static']:
                problems.append(label+'尚未静止')
        return problems

    def zero(self, group='arms'):
        if group not in self.zero_buttons or self.zero_pending is not None:
            return
        if self.control.worker is not None or not self.review.isChecked() or self.preset_error:
            self.message.setText('请先断开串口、成功加载预设并核对配置。')
            return
        keys = [key for key in self.active_parts() if key.endswith('_hand') == (group == 'hands')]
        problems = self.zero_blockers(keys)
        if problems:
            self.message.setText('不能开始采集：'+'；'.join(problems)+'。请保持本组设备静止，等待学习完成。')
            return
        self.zeroed_parts.difference_update(keys)
        self.zero_confirmed = False
        self.control._controls()
        self.zero_pending = (group, keys, time.monotonic()+self.zero_delays[group].value())
        self.zero_cancel.setEnabled(True)
        self.zero_timer.start()
        self.tick_zero()

    def tick_zero(self):
        if self.zero_pending is None:
            return
        group, keys, deadline = self.zero_pending
        if self.control.worker is not None or not self.review.isChecked() or self.preset_error:
            self.cancel_zero()
            return
        remaining = deadline-time.monotonic()
        label = '手背' if group == 'hands' else '上臂 / 前臂'
        if remaining > 0:
            self.message.setText(f'{label}：{remaining:.1f} 秒后采集，请保持对应参考姿态。')
            return
        self.zero_pending = None
        self.zero_timer.stop()
        self.zero_cancel.setEnabled(False)
        problems = self.zero_blockers(keys, require_static=True)
        if problems:
            self.message.setText(f'{label}采集失败：'+'；'.join(problems)+'。请保持静止后重新采集。')
            return
        for key in keys:
            self.engine.zero(key)
        self.zeroed_parts.update(keys)
        self.zero_confirmed = set(self.active_parts()).issubset(self.zeroed_parts)
        self.control._controls()
        self.message.setText(f'{label} {len(keys)} 台中立参考已采集。'+
                             ('两组均已完成，可打开控制面板；仍须上电确认、安全许可及明确启动。'
                              if self.zero_confirmed else '请继续采集另一组；未完成两组前禁止连接串口。'))

    def show_control(self):
        self.control.show()
        self.control.raise_()

    def poll(self):
        for stamp, packet in self.source.poll_timed():
            self.engine.feed(packet, stamp/1e9)
        now = time.monotonic()
        if now >= self.next_display:
            lines = []
            ready_count = 0
            for key, device in self.config['assignments'].items():
                stamp = self.engine.received_at.get(device)
                age = '未收到' if stamp is None else f'{(now-stamp)*1000:.0f} ms'
                mapping = '/'.join(self.config['axis_mappings'][key])
                state = self.engine.tilt_status(device)
                live = self.engine.is_live(device, now, .25)
                ready = live and state['gravity_ready'] and state['bias_ready']
                if key in self.active_parts() and ready:
                    ready_count += 1
                gravity = '已建立' if state['gravity_ready'] else '等待静止'
                bias = ('已完成' if state['bias_ready'] else
                        f"学习 {min(state['static_seconds'], state['bias_hold_seconds']):.1f}/{state['bias_hold_seconds']:.1f}s")
                static = ('静止' if state['static'] else '运动 / 未满足静止门槛') if live else '数据超时 / 未到'
                lines.append(f'{PARTS[PART_BY_KEY[key]].label}: {device} · 距末包 {age} · '
                             f'重力：{gravity} · 零偏：{bias} · {static} · {mapping}')
            self.devices.setText('\n'.join(lines))
            self.learning_summary.setText(
                f'陀螺＋加计纠倾斜已启用；本模式初始化就绪 {ready_count}/{len(self.active_parts())} 台。'
                '先保持静止至重力与零偏均就绪，再定时采集中立位；结束瞬间本组须静止。'
                '学习就绪不表示停止纠倾斜，快速运动时加计权重会自动降低。')
            for group, button in self.zero_buttons.items():
                keys = [key for key in self.active_parts() if key.endswith('_hand') == (group == 'hands')]
                problems = self.zero_blockers(keys)
                button.setEnabled(self.control.worker is None and self.review.isChecked()
                                  and not self.preset_error and self.zero_pending is None and not problems)
                button.setToolTip('；'.join(problems) if problems else '本组初始化已就绪，倒计时结束仍须保持静止。')
                self.zero_delays[group].setEnabled(self.zero_pending is None)
            self.review.setEnabled(self.control.worker is None)
            self.preview.update()
            self.sync_mode()
            self.next_display = now+.1

    def closeEvent(self, event):
        self.cancel_zero()
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
