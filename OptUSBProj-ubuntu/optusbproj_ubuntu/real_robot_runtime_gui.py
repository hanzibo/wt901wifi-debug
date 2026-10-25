"""One-click read-only observer; never calls a robot motion/serial method."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import uuid

from PySide6 import QtCore, QtWidgets


class RuntimeRecordingSession:
    def __init__(self, owner, output_root=None):
        self.owner = owner
        root = Path(__file__).resolve().parents[2]
        self.script = root / 'scripts' / 'robot_runtime_probe.py'
        self.output_root = Path(output_root) if output_root else (
            root / 'OptUSBProj-ubuntu' / 'recordings' / 'runtime_probe')
        self.process = None
        self.output = None
        self.stopping = False
        self.close_pending = False
        self.last_context = None
        self.console = ''
        self.panel = QtWidgets.QGroupBox('运行性能记录（只读，不启动或停止机器人）')
        layout = QtWidgets.QVBoxLayout(self.panel)
        self.button = QtWidgets.QPushButton('开始性能记录（CPU / 内存 / 控制日志关联）')
        self.button.setToolTip('先开始记录，再正常启动单臂或双臂；结束后再次点击保存。最多记录 1 小时。')
        self.label = QtWidgets.QLabel('未记录；不会改变发送频率、运动限制或安全许可。')
        self.label.setWordWrap(True)
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.button); layout.addWidget(self.label)
        self.button.clicked.connect(self.toggle)
        self.timer = QtCore.QTimer(self.panel)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)

    @property
    def running(self):
        return self.process is not None

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.running: return
        self.output = self.output_root / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6])
        self.stopping = self.close_pending = False
        self.last_context = None
        self.console = ''
        process = QtCore.QProcess(self.panel)
        self.process = process
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_output)
        process.started.connect(self.started)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.error)
        self.button.setEnabled(False)
        self.label.setText('正在启动只读记录器；不操作串口或 IMU。')
        process.start(sys.executable, [str(self.script), '--pid', str(os.getpid()),
                      '--duration', '3600', '--interval', '1', '--output', str(self.output), '--gui-control'])

    def started(self):
        self.started_at = time.monotonic()
        self.button.setText('停止性能记录并保存（不停止机器人）')
        self.button.setEnabled(True)
        self.timer.start()
        self.tick()

    def read_output(self):
        if self.process is not None:
            self.console = (self.console + bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace'))[-16000:]

    def send(self, data):
        if self.process is not None:
            self.process.write((json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8'))

    def context(self):
        d = self.owner
        engine = getattr(d.viewer, 'engine', None)
        worker = d.worker
        path = getattr(getattr(worker, 'recorder', None), 'path', None)
        mode = getattr(d, 'mode_select', None)
        config = dict(mode=mode.currentText() if mode is not None else '单肘 L4',
                      frequency_hz=d.frequency.value(), manual_mapping=d.manual.isChecked(),
                      control_log=str(path) if path is not None else None,
                      assignments=dict(getattr(engine, 'assignments', {})),
                      algorithm_modes=dict(getattr(engine, 'algorithm_modes', {})))
        checkpoints = getattr(d, 'checkpoints', None)
        if checkpoints is not None and d._shared_mode():
            config['parameters'] = {key:value for key,value in checkpoints.snapshot(validated=False).items()
                                    if key not in ('id', 'created_at', 'updated_at')}
            config['checkpoint_label'] = checkpoints.label.text()
        for side, name in (('left', 'axis_rows'), ('right', 'right_axis_rows')):
            rows = getattr(d, name, None)
            if rows is not None:
                config[side] = [dict(enabled=r[0].isChecked(), lower=r[1].value(), upper=r[2].value(),
                                    speed=r[3].value(), acceleration=r[4].value(),
                                    direction=r[5].currentText()) for r in rows]
        config['mirror_enabled'] = bool(getattr(d, 'mirror_check', None) and d.mirror_check.isChecked())
        return config

    def tick(self):
        if not self.running or self.stopping: return
        try:
            now = time.monotonic()
            context = self.context()
            signature = json.dumps(context, sort_keys=True, ensure_ascii=False)
            stamp = dict(wall_time=datetime.now().astimezone().isoformat(), monotonic_s=now)
            if signature != self.last_context:
                self.send(dict(event='gui_context', **stamp, **context))
                self.last_context = signature
            engine = getattr(self.owner.viewer, 'engine', None)
            received = dict(getattr(engine, 'received_at', {}))
            self.send(dict(event='gui_state', **stamp, busy=bool(self.owner.busy),
                           connected=bool(self.owner.connected),
                           shared_phase=getattr(self.owner, 'shared_phase', None),
                           status=self.owner.status.text(),
                           device_ages_ms={key:max(0., now-at)*1000 for key,at in received.items()}))
            self.label.setText(f'记录中：{now-self.started_at:.0f} 秒（最多 1 小时）\n目录：{self.output}\n'
                               '请正常完成随动；先安全停止机器人，再点击停止记录。')
        except Exception as exc:
            # Observation failures are visible but must never stop robot motion.
            self.label.setText(f'资源仍在记录，但界面上下文采集失败：{exc}\n目录：{self.output}')

    def stop(self):
        if not self.running or self.stopping: return
        self.tick()
        self.stopping = True
        self.timer.stop()
        self.button.setEnabled(False)
        self.label.setText(f'正在保存记录；机器人状态不变。\n目录：{self.output}')
        self.process.write(b'stop\n')

    def prepare_close(self):
        if not self.running: return True
        self.close_pending = True
        self.stop()
        return False

    def error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.console += '\n记录器无法启动：'+self.process.errorString()
            self.finished(-1, QtCore.QProcess.CrashExit)

    def finished(self, code, _status):
        if self.process is None: return
        self.timer.stop()
        self.read_output()
        process, self.process = self.process, None
        process.deleteLater()
        self.stopping = False
        summary = self.output / 'summary.json'
        self.button.setText('开始性能记录（CPU / 内存 / 控制日志关联）')
        self.button.setEnabled(True)
        if code == 0 and summary.exists():
            overview = ''
            try:
                stats = json.loads(summary.read_text(encoding='utf-8'))['summary']
                cpu, hottest, memory = (stats.get(key) for key in
                    ('root_cpu_percent', 'hottest_thread_percent', 'tree_pss_mib'))
                if cpu:
                    overview += f'程序 CPU P95 {cpu["p95"]:.1f}%（100%=一个逻辑核）'
                if hottest:
                    overview += f'；最忙线程 P95 {hottest["p95"]:.1f}%'
                if memory:
                    overview += f'；进程树 PSS 峰值 {memory["max"]:.0f} MiB'
                overview = (overview or '记录不足一个采样周期；建议重新记录至少数分钟。')+'\n'
            except (OSError, ValueError, KeyError, TypeError):
                overview = '摘要无法显示，请检查 summary.json。\n'
            self.label.setText(f'性能记录已保存：{self.output}\n'+overview+
                               '含 samples.jsonl 和 summary.json；请把此目录发给我分析。机器人状态未改变。')
        else:
            self.label.setText(f'性能记录失败或不完整（不影响机器人控制）：\n{self.console[-2000:]}\n目录：{self.output}')
        if self.close_pending:
            QtCore.QTimer.singleShot(0, self.owner.close)
