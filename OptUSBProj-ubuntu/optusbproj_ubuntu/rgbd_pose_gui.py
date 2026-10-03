"""Independent metric skeleton diagnostics; QPainter avoids a second GL scene."""
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .rgbd_pose import NAMES, INDICES, EDGES

LABELS = ("左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左髋", "右髋")


def draw_pose_rgb(estimate):
    import cv2
    image = estimate.frame_rgb.copy()
    xy = estimate.keypoints_xy[INDICES]
    for a, b in EDGES:
        if estimate.valid[a] and estimate.valid[b]:
            cv2.line(image, tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int)), (70, 230, 120), 3)
    for i, point in enumerate(xy):
        if np.isfinite(point).all() and estimate.confidence[INDICES[i]] >= .5:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 5,
                       (70, 230, 120) if estimate.valid[i] else (255, 160, 40), -1)
    return image


class MetricSkeletonWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(230)
        self.estimate = None
        self.yaw, self.tilt = -.5, .2
        self.zoom = 1.
        self._drag = None
        self.view_center = None
        self.view_span = None
        self.title_text = "真实深度三维观测 · 拖动旋转 / 滚轮缩放 · 自动居中"

    def mousePressEvent(self, event):
        self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position()-self._drag
            self.yaw += delta.x()*.01
            self.tilt = np.clip(self.tilt+delta.y()*.01, -1.4, 1.4)
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * math.exp(event.angleDelta().y()/1200), .3, 3.))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#191b20"))
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QColor("white"))
        painter.drawText(12, 22, self.title_text)
        estimate = self.estimate
        if estimate is None or not estimate.valid.any():
            painter.drawText(12, 50, "没有有效三维点；不会保持旧点冒充当前观测")
            return
        points = estimate.points_camera_m
        valid_points = points[estimate.valid]
        center = valid_points.mean(axis=0) if self.view_center is None else np.asarray(self.view_center)
        span = max(.5, float(np.ptp(valid_points, axis=0).max())) if self.view_span is None else self.view_span
        scale = .65 * min(self.width(), self.height()) / span * self.zoom
        cy, sy, cx, sx = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.tilt), math.sin(self.tilt)
        rotation = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        def project(point):
            p = rotation @ (point-center)
            return QtCore.QPointF(self.width()/2+p[0]*scale, self.height()/2+p[1]*scale)
        for a, b in EDGES:
            if estimate.valid[a] and estimate.valid[b]:
                color = "#70dfff" if a in (0, 2, 6) and b in (2, 4, 6) else "#ffb267"
                painter.setPen(QtGui.QPen(QtGui.QColor(color), 3))
                painter.drawLine(project(points[a]), project(points[b]))
        painter.setPen(QtGui.QColor("white"))
        for i in range(8):
            if estimate.valid[i]:
                p = project(points[i])
                painter.drawEllipse(p, 4, 4)
                painter.drawText(p+QtCore.QPointF(6, -4), LABELS[i])
        origin = QtCore.QPointF(55, self.height()-38)
        for axis, color, label in zip(np.eye(3), ("#ff5555", "#55dd55", "#5588ff"), ("X", "Y", "Z")):
            delta = rotation @ axis
            end = origin+QtCore.QPointF(30*delta[0], 30*delta[1])
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
            painter.drawLine(origin, end)
            painter.drawText(end+QtCore.QPointF(3, 0), label)


class RGBDPoseDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RGB-D 上半身三维观测 · 诊断窗口")
        self.resize(850, 680)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("相机光学系：X 向右、Y 向下、Z 向前，单位米。深度是皮肤/衣物表面，不是精确关节中心。绿色2D点有效，橙色点被深度检查拒绝。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QtWidgets.QLabel("等待观测")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.canvas = MetricSkeletonWidget()
        layout.addWidget(self.canvas, 1)
        self.table = QtWidgets.QTableWidget(8, 7)
        self.table.setHorizontalHeaderLabels(("部位", "二维分数", "X/Y/Z（米）", "深度支持", "跨度mm", "状态", "深度像素"))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(265)
        layout.addWidget(self.table)
        self.lengths = QtWidgets.QLabel("骨段长度：等待")
        self.lengths.setWordWrap(True)
        layout.addWidget(self.lengths)
        close = QtWidgets.QPushButton("关闭诊断窗口（不停止采集）")
        close.clicked.connect(self.hide)
        layout.addWidget(close)

    def update_estimate(self, estimate):
        self.canvas.estimate = estimate
        self.canvas.update()
        self.status.setText(f"有效 {int(estimate.valid.sum())}/8 点 · 推理 {estimate.inference_ms:.1f} ms · "
                            + estimate.diagnostics.get("reject_reason", "真实深度；不使用网络预测Z"))
        for i in range(8):
            p = estimate.points_camera_m[i]
            xyz = "/".join(f"{v:+.3f}" for v in p) if estimate.valid[i] else "—"
            pixel = estimate.depth_pixels[i]
            values = (LABELS[i], f"{estimate.confidence[INDICES[i]]:.2f}", xyz,
                      f"{estimate.support[i]:.0%}", f"{estimate.spread_mm[i]:.1f}" if np.isfinite(estimate.spread_mm[i]) else "—",
                      estimate.reasons[i], "/".join(f"{v:.1f}" for v in pixel) if np.isfinite(pixel).all() else "—")
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if j == 5:
                    item.setForeground(QtGui.QColor("#39b96b" if estimate.valid[i] else "#d58b26"))
                self.table.setItem(i, j, item)
        labels = {"left_upper_arm": "左上臂", "left_forearm": "左前臂", "right_upper_arm": "右上臂", "right_forearm": "右前臂", "shoulder_width": "肩宽"}
        self.lengths.setText(" · ".join(f"{labels[k]}：{v*100:.1f} cm" if v is not None else f"{labels[k]}：不可用"
                                         for k, v in estimate.bone_lengths_m.items()))
