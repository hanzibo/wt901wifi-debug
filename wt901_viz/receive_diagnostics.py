"""Qt controls for bounded, read-only UDP acquisition recording."""
from pathlib import Path
import time
from PySide6 import QtCore, QtWidgets
from wt901.receive_recording import ReceiveRecorder


class ReceiveDiagnosticsPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()

    def __init__(self, source, parent=None, root=None):
        super().__init__(parent)
        self.source = source
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "recordings/wrist_udp"
        self.recorder = None
        self._sink = None
        self._detached = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 2)
        row = QtWidgets.QHBoxLayout()
        self.button = QtWidgets.QPushButton("记录收包诊断")
        self.button.setEnabled(hasattr(source, "add_diagnostic_sink"))
        self.button.setToolTip("只观察现有 UDP 接收；不切换网络、不发送设备或机器人指令")
        self.button.clicked.connect(self.toggle)
        row.addWidget(self.button)
        self.label = QtWidgets.QLineEdit()
        self.label.setMaxLength(80)
        self.label.setPlaceholderText("测试标签：例如路由器Wi-Fi/省电关闭（不要填写密码）")
        row.addWidget(self.label, 1)
        self.duration = QtWidgets.QSpinBox()
        self.duration.setRange(10, 600)
        self.duration.setValue(120)
        self.duration.setSuffix(" 秒后自动停止")
        row.addWidget(self.duration)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel("收包诊断：未录制；默认记录 120 秒，也可手动停止。")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_recording)

    def toggle(self):
        if self.recorder is not None:
            self.stop()
            return
        recorder = None
        try:
            recorder = ReceiveRecorder(self.root, dict(test_label=self.label.text().strip(),
                listen_endpoint=list(self.source.sock.getsockname()),
                auto_stop_s=self.duration.value(),
                measurement="existing UDP socket; application reads and optional kernel timestamps"))
            self.recorder = recorder
            self._sink = recorder.record
            self._detached = False
            self.source.add_diagnostic_sink(self._sink)
        except Exception as exc:
            if self._sink is not None:
                self.source.remove_diagnostic_sink(self._sink)
            if recorder is not None:
                recorder.stop("start_error")
            if recorder is None:
                self.status.setText(f"不能开始收包诊断：{exc}")
                self.recorder = None
                return
            self._detached = True
            recorder.error = f"开始记录失败：{exc}"
        self.label.setEnabled(False)
        self.duration.setEnabled(False)
        self.button.setText("停止并保存收包诊断")
        self.timer.start()
        self.update_recording()

    def stop(self, reason="user"):
        if self.recorder is None:
            return
        if not self._detached:
            self.source.remove_diagnostic_sink(self._sink)
            self._detached = True
        self.recorder.stop(reason)
        self.button.setEnabled(False)
        self.status.setText(f"正在后台保存诊断：{self.recorder.path}")

    def update_recording(self):
        recorder = self.recorder
        if recorder is None:
            return
        elapsed = time.monotonic() - recorder.started
        if not self._detached and (elapsed >= self.duration.value() or recorder.error):
            self.stop("write_error" if recorder.error else "duration_complete")
        if recorder.finished:
            if not self._detached:
                self.stop("writer_finished")
            self.status.setText((f"诊断保存失败：{recorder.error}；部分文件位于 {recorder.path}"
                if recorder.error else
                f"诊断已保存：{recorder.path}（events.jsonl / summary.json / summary.md；"
                f"丢弃诊断事件 {recorder.dropped}）"))
            self.recorder = None
            self._sink = None
            self.timer.stop()
            self.label.setEnabled(True)
            self.duration.setEnabled(True)
            self.button.setEnabled(True)
            self.button.setText("记录收包诊断")
            self.finished.emit()
        elif not self._detached:
            self.status.setText(f"正在记录 {elapsed:.0f}/{self.duration.value()} 秒；"
                                f"丢弃诊断事件 {recorder.dropped}；目录：{recorder.path}")
