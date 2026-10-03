"""Resizable camera display, detached without creating another capture pipeline."""
from PySide6 import QtCore, QtGui, QtWidgets


class CameraPreview(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__("等待相机画面", parent)
        self._original = QtGui.QPixmap()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #191b20; color: white")

    def display_image(self, image):
        self._original = QtGui.QPixmap.fromImage(image)
        self._rescale()

    def _rescale(self):
        if not self._original.isNull():
            self.setPixmap(self._original.scaled(
                self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class CameraPreviewWindow(QtWidgets.QDialog):
    returning = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("相机画面 · 可拖至另一显示器")
        self.resize(960, 600)
        self.setModal(False)
        self.preview_layout = QtWidgets.QVBoxLayout(self)
        button = QtWidgets.QPushButton("返回主界面（不停止相机或录制）")
        button.clicked.connect(self.close)
        self.preview_layout.addWidget(button)

    def closeEvent(self, event):
        self.returning.emit()
        super().closeEvent(event)
