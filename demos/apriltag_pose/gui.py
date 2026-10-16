"""Qt preview using the existing isolated Orbbec SDK subprocess."""
import time
from collections import Counter
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from optusbproj_ubuntu.orbbec_worker import OrbbecWorker
from .core import DEFAULT_SIZES_MM, Detector, camera_model, estimate_pose
from .diagnostics import PoseWindow, quality_reasons


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Orbbec · AprilTag 姿态验证（不连接机器人）')
        self.resize(1120, 850)
        self.worker = None
        self.detector = Detector()
        self.last_frame_at = None
        self.active_error = None
        self.pose_window = PoseWindow()
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('tag36h11；尺寸指黑框，不含白边。A4 请展平固定。红 X / 绿 Y / 蓝 Z。'
                               '\n显示 T_camera_tag：p_camera = R × p_tag + t。相机 X 向右、Y 向下、Z 向前。'
                               '\n标签 X 向右、Y 向上、Z 朝印刷正面外侧（按图案标准方向）；不是机器人关节角。')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.sizes = {}
        for tag_id, size in DEFAULT_SIZES_MM.items():
            row.addWidget(QtWidgets.QLabel(f'ID {tag_id} 黑框'))
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(1, 300)
            box.setDecimals(2)
            box.setSuffix(' mm')
            box.setValue(size)
            self.sizes[tag_id] = box
            row.addWidget(box)
        self.start = QtWidgets.QPushButton('启动相机')
        self.start.clicked.connect(self.start_camera)
        self.stop = QtWidgets.QPushButton('停止')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_camera)
        row.addWidget(self.start)
        row.addWidget(self.stop)
        layout.addLayout(row)
        diagnostics = QtWidgets.QHBoxLayout()
        self.measure = QtWidgets.QCheckBox('测量静止波动（请固定相机和纸张）')
        self.measure.toggled.connect(lambda _: self.pose_window.clear())
        diagnostics.addWidget(self.measure)
        self.thresholds = []
        for title, default, low, high, suffix in (
                ('最短边 ≥', 32., 16., 300., ' px'),
                ('重投影 ≤', 1., .1, 3., ' px'),
                ('位置波动 ≤', 2., .1, 100., ' mm'),
                ('朝向波动 ≤', 2., .1, 90., ' °')):
            diagnostics.addWidget(QtWidgets.QLabel(title))
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(low, high)
            box.setDecimals(1)
            box.setValue(default)
            box.setSuffix(suffix)
            diagnostics.addWidget(box)
            self.thresholds.append(box)
        layout.addLayout(diagnostics)
        self.image = QtWidgets.QLabel('先关闭主项目相机 / OrbbecViewer，再启动。')
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setMinimumSize(640, 360)
        self.image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.image, 1)
        self.table = QtWidgets.QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(['ID / 尺寸', 't X/Y/Z (mm)', '旋转向量 (°)',
                                             '重投影 RMS (px)', '最短/平均边 (px)',
                                             '距离 ‖t‖ / Z (mm)', '位置波动 RMS (mm)',
                                             '朝向波动 RMS (°)', '窗口 样本/秒', '状态'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(170)
        layout.addWidget(self.table)
        explanation = QtWidgets.QLabel('波动为最近 2 秒相对均值的 RMS，至少 10 帧且跨度 1 秒后判定；'
                                      '运动也会计入波动，不是绝对测量误差。阈值为可调经验筛选，不保证精度。')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.status = QtWidgets.QLabel('尚未启动；无真机、串口或 ROS 输出。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

    def start_camera(self):
        if self.worker is not None:
            return
        self.last_frame_at = None
        self.active_error = None
        self.pose_window.clear()
        self.worker = OrbbecWorker(fps=15., warmup=2.)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self.status.setText('相机启动/预热中……')

    def on_error(self, message):
        self.pose_window.clear()
        self.active_error = message
        self.status.setText(message)
        self.table.setRowCount(0)
        self.image.clear()
        self.image.setText('采集/姿态错误；不保留旧姿态。')

    def stop_camera(self):
        self.pose_window.clear()
        if self.worker is not None:
            self.worker.stop()
        self.stop.setEnabled(False)
        self.table.setRowCount(0)
        self.image.clear()
        self.image.setText('已请求停止；不保留旧姿态。')
        self.status.setText('正在停止相机……')

    def on_finished(self):
        self.pose_window.clear()
        self.worker = None
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.table.setRowCount(0)
        self.image.clear()
        self.image.setText('相机已停止。')
        self.status.setText(self.active_error or '相机已停止；未保留旧姿态。')

    def tick(self):
        if self.worker is None or not self.stop.isEnabled() or self.active_error:
            return
        observation = self.worker.take_latest()
        if observation is None:
            if self.last_frame_at and time.monotonic()-self.last_frame_at > 1:
                self.pose_window.clear()
                self.table.setRowCount(0)
                self.image.clear()
                self.image.setText('相机帧过期；已清空旧姿态。')
                self.status.setText('超过 1 秒没有新帧，等待恢复。')
            return
        self.last_frame_at = time.monotonic()
        try:
            self.show_observation(observation)
        except Exception as exc:
            self.on_error(f'姿态处理失败：{exc}')
            self.worker.stop()

    def show_observation(self, obs):
        started = time.monotonic()
        rgb = obs.frame_rgb
        k, d = camera_model(obs.rgbd.calibration, rgb.shape)
        detections = self.detector.detect(rgb)
        counts = Counter(i for i, _ in detections)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.table.setRowCount(len(detections))
        valid_ids = set()
        good_count = 0
        for row, (tag_id, corners) in enumerate(detections):
            cv2.polylines(bgr, [corners.astype(np.int32)], True, (0, 220, 220), 2)
            cv2.putText(bgr, f'ID {tag_id}', tuple(corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 120, 255), 2)
            edges = np.linalg.norm(corners-np.roll(corners, 1, axis=0), axis=1)
            edge = float(np.min(edges))
            values = [str(tag_id), '-', '-', '-', f'{edge:.1f} / {np.mean(edges):.1f}',
                      '-', '-', '-', '-', '未知 ID：仅检测，不估计尺寸']
            color = QtGui.QColor('#fff0cc')
            if counts[tag_id] > 1:
                values[9] = '重复 ID：拒绝姿态，请勿同时展示同 ID 副本'
            elif tag_id in self.sizes:
                size_mm = self.sizes[tag_id].value()
                values[0] = f'{tag_id} / {size_mm:g} mm'
                pose = estimate_pose(corners, size_mm/1000, k, d) if edge >= 16 else None
                if pose is None:
                    values[9] = '仅检测：角点过小或求解失败'
                else:
                    values[3] = f'{pose.error_px:.3f}'
                    if pose.error_px > 3:
                        values[9] = '仅检测：重投影误差过大，拒绝姿态'
                    else:
                        valid_ids.add(tag_id)
                        values[1] = ', '.join(f'{v:.1f}' for v in pose.tvec.flatten()*1000)
                        values[2] = ', '.join(f'{v:.1f}' for v in np.rad2deg(pose.rvec.flatten()))
                        values[5] = f'{np.linalg.norm(pose.tvec)*1000:.1f} / {pose.tvec.flatten()[2]*1000:.1f}'
                        variation = None
                        if self.measure.isChecked():
                            variation = self.pose_window.update(tag_id, obs.captured_mono_ns/1e9,
                                                                size_mm, pose)
                            values[6] = f'{variation.position_mm:.3f}'
                            values[7] = f'{variation.rotation_deg:.3f}'
                            values[8] = f'{variation.count} / {variation.span_s:.1f}'
                        reasons = quality_reasons(edge, pose, variation,
                                                  *(box.value() for box in self.thresholds))
                        if reasons:
                            values[9] = '姿态质量不足：' + '；'.join(reasons)
                        elif variation is not None and not variation.ready:
                            values[9] = '几何筛选通过；波动采样中'
                        else:
                            good_count += 1
                            values[9] = '经验筛选通过（非精度认证）' if variation else '几何筛选通过（未测波动）'
                            color = QtGui.QColor('#d6efda')
                        cv2.drawFrameAxes(bgr, k, d, pose.rvec, pose.tvec, size_mm/1000*.6, 2)
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setBackground(color)
                item.setForeground(QtGui.QColor('#17202a'))
                self.table.setItem(row, col, item)
        self.pose_window.retain(valid_ids if self.measure.isChecked() else set())
        shown = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = shown.shape[:2]
        qimage = QtGui.QImage(shown.data, w, h, shown.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.image.setPixmap(QtGui.QPixmap.fromImage(qimage).scaled(
            self.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        age_ms = (time.monotonic_ns()-obs.captured_mono_ns)/1e6
        self.status.setText(f'本帧识别 {len(detections)} 个标签 · 筛选通过 {good_count} 个 · 图像 {w}×{h}'
                            f' · 处理 {(time.monotonic()-started)*1000:.0f} ms'
                            f' · 接收至显示 {age_ms:.0f} ms（不是曝光延迟）\n'
                            '位置来自 RGB PnP，不使用深度验证；纸张弯曲、尺寸错误和正视双解都会影响结果。')

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            if not self.worker.wait(5000):
                self.status.setText('采集仍在退出，请稍后再关闭。')
                event.ignore()
                return
        event.accept()
