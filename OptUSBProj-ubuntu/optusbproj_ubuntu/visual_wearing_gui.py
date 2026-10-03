"""Reuse guided capture, but keep wearing maps separate from heading feedback."""
import json
from pathlib import Path
import sys
import uuid

from PySide6 import QtCore, QtWidgets

from .camera_calibration_gui import CameraCalibrationDialog
from .pose import GYRO_TILT
from .visual_wearing import FORMAT


class VisualWearingDialog(CameraCalibrationDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.setWindowTitle('视觉辅助穿戴标定 · 右上臂 / 右前臂（实验）')
        self.resize(820, 790)
        self.report = None
        self.backup = None
        self.auto_analyze.setText('录制后计算视觉辅助穿戴映射（不自动应用）')
        self.replay.setText('双人体回放 · 现有映射 vs 视觉辅助')
        self.apply.setText('应用所选合格穿戴映射（不启用航向纠偏）')
        layout = self.layout()
        status_index = layout.indexOf(self.status)
        layout.removeWidget(self.status)
        status_scroll = QtWidgets.QScrollArea()
        status_scroll.setWidgetResizable(True)
        status_scroll.setMinimumHeight(130)
        status_scroll.setMaximumHeight(210)
        status_scroll.setWidget(self.status)
        layout.insertWidget(status_index, status_scroll)
        note = layout.itemAt(0).widget()
        note.setText('首版仅右臂：自然下垂无需强行贴紧身体。前举/侧举时肘伸直，站直且保持朝向。相机拟合真实骨轴，再归一到模型中立位；不代表解剖学扭转已标定。首次动作拟合，重复动作独立验收。固定相机和绑带；无需先做原IMU标定，不自动应用或启用机器人。')
        selection = QtWidgets.QHBoxLayout()
        self.part_checks = {}
        for part, label in (('right_upper_arm', '右上臂'), ('right_forearm', '右前臂')):
            check = QtWidgets.QCheckBox(label)
            check.setEnabled(False)
            self.part_checks[part] = check
            selection.addWidget(check)
        self.baseline_combo = QtWidgets.QComboBox()
        self.baseline_combo.addItem('对比：录制时现有/手动映射', 'existing')
        self.baseline_combo.addItem('对比：同数据原 IMU 平举标定', 'legacy')
        selection.addWidget(self.baseline_combo)
        layout.addLayout(selection)
        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel('双模型回放速度'))
        self.replay_speed = QtWidgets.QDoubleSpinBox()
        self.replay_speed.setRange(.1, 4.)
        self.replay_speed.setSingleStep(.25)
        self.replay_speed.setValue(1.)
        self.replay_speed.setSuffix(' 倍')
        speed_row.addWidget(self.replay_speed)
        layout.addLayout(speed_row)
        self.restore_button = QtWidgets.QPushButton('撤销本窗口应用 · 恢复应用前映射与中立位')
        self.restore_button.setEnabled(False)
        self.restore_button.clicked.connect(self.restore_previous)
        layout.addWidget(self.restore_button)
        # Keep the complete form usable on small/high-DPI screens. The inner
        # status scroll handles long diagnostics; the outer scroll prevents
        # Qt from compressing/overlapping the action buttons.
        content = QtWidgets.QWidget()
        content.setLayout(layout)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setWidget(content)
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(scroll)

    def readiness_error(self):
        error = super().readiness_error()
        if error:
            return error
        if getattr(self.owner, 'camera_worker', None) is None or not getattr(self.owner.camera_worker, 'capture_epoch', None):
            return '相机采集会话尚未就绪，请启动Gemini并等待预热。'
        if self.owner.x1_preview_dialog is not None and self.owner.x1_preview_dialog.sender is not None:
            return '请先停止 X1 随动，再做穿戴标定。'
        for part in self.part_checks:
            if self.owner.engine.algorithm_modes[part] != GYRO_TILT:
                return '两段设备均需选择“陀螺＋加计”，不使用在线视觉纠偏流拟合安装。'
        return None

    def monitor_error(self):
        error = super().monitor_error()
        if error:
            return error
        if getattr(self.owner.camera_worker, 'capture_epoch', None) != getattr(self, 'reference_camera_epoch', None):
            return '相机采集会话发生变化，请重新录制。'
        for part in self.part_checks:
            if self.owner.engine.algorithm_modes[part] != GYRO_TILT:
                return '录制期间姿态模式改变，请重录。'
        if self.owner.x1_preview_dialog is not None and self.owner.x1_preview_dialog.sender is not None:
            return 'X1 随动已开启，本轮停止。'
        return None

    def begin(self):
        self.reference_camera_epoch = getattr(self.owner.camera_worker, 'capture_epoch', None)
        super().begin()
        if self.active:
            self.report = None
            for check in self.part_checks.values():
                check.setEnabled(False); check.setChecked(False)

    def analyze_recording(self):
        if self.state != 'complete':
            return
        self.analysis_path = self.recording_path.parent / 'analysis_wearing' / (
            self.recording_path.name + '-visual-' + uuid.uuid4().hex[:6])
        if self.process is not None:
            self.process.deleteLater()
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.analysis_finished)
        self.process.errorOccurred.connect(self.analysis_error)
        self.process.setProgram(sys.executable)
        script = Path(__file__).resolve().parent.parent / 'visual_wearing_analyze.py'
        self.process.setArguments(['-u', str(script), str(self.recording_path), '--output', str(self.analysis_path)])
        self.state = 'analyzing'
        for button in (self.start, self.analyze, self.apply, self.replay):
            button.setEnabled(False)
        for check in self.part_checks.values():
            check.setEnabled(False)
        self.log = ''
        self.status.setText('正在拟合真实骨轴；不要求中立骨段竖直，不修改现有映射。')
        self.process.start()

    def analysis_finished(self, code, _status):
        self.read_log(); self.state = 'complete'
        self.start.setEnabled(True); self.analyze.setEnabled(True)
        try:
            if code != 0:
                raise ValueError(self.log[-1500:])
            report = json.loads((self.analysis_path / 'report.json').read_text())
            if report.get('format') != FORMAT:
                raise ValueError('分析输出格式不匹配')
        except (OSError, ValueError) as exc:
            self.status.setText('分析失败；原始录制保留。\n' + str(exc))
            return
        self.report = report
        lines = []
        for part, check in self.part_checks.items():
            result = report['parts'][part]
            good = result['status'] == '通过'
            check.setEnabled(good); check.setChecked(good)
            label = '右上臂' if part == 'right_upper_arm' else '右前臂'
            text = f"{label}：{result['status']}"
            if 'basis_matrix' in result:
                text += (f" · 中立偏离竖直 {result['neutral_tilt_deg']:.1f}°"
                         f" · 两套骨轴差 {result['bone_axis_difference_deg']:.1f}°"
                         f" · 映射变化 {result['mapping_offset_deg']:.1f}°\n"
                         f"首次拟合 中位/P90 {result['fit_median_deg']:.1f}°/{result['fit_p90_deg']:.1f}°")
                for name, title in (('forward_repeat', '再前举'), ('lateral_repeat', '再侧举'), ('validation', '独立动作')):
                    metrics = result['checks'].get(name)
                    if metrics:
                        text += f"；{title} {metrics['median_deg']:.1f}°/{metrics['p90_deg']:.1f}°"
            lines.append(text + '\n' + '；'.join(result['warnings'] + result.get('notes', [])))
        self.status.setText('\n'.join(lines) + f'\n结果：{self.analysis_path}\n误差是视觉一致性，不是GT；请先回放并检查轴向转动和肘部。')
        self.apply.setEnabled(any(c.isChecked() for c in self.part_checks.values()))
        self.replay.setEnabled(any('basis_matrix' in p for p in report['parts'].values()))

    def apply_live(self):
        if self.report is None:
            return
        parts = [p for p, c in self.part_checks.items() if c.isChecked()]
        if not parts:
            QtWidgets.QMessageBox.warning(self, '未应用', '请勾选至少一个合格部位。'); return
        answer = QtWidgets.QMessageBox.question(self, '确认应用穿戴映射',
            '将更新所选部位的轴映射和本次中立姿态。保留原相机航向纠偏参数，不自动启用融合或机器人。自然下垂偏角归一到模型标准中立位；轴向扭转仍须实测。是否应用？')
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            # The owner validates all parts atomically before changing anything.
            backup = self.owner.apply_visual_wearing_report(self.report, parts)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, '穿戴标定未应用', str(exc)); return
        self.backup = backup
        self.restore_button.setEnabled(True); self.apply.setEnabled(False)
        self.status.setText(self.status.text() + '\n已临时应用所选穿戴映射；请验证屈肘及长轴转动。')

    def restore_previous(self):
        if self.backup is None:
            return
        try:
            self.owner.restore_visual_wearing(self.backup)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, '无法撤销', str(exc)); return
        self.backup = None; self.restore_button.setEnabled(False)
        self.apply.setEnabled(self.report is not None and any(
            c.isChecked() for c in self.part_checks.values()))
        self.status.setText(self.status.text() + '\n已恢复应用前映射和中立位。')

    def open_replay(self):
        if self.report is None:
            return
        try:
            from .fusion_gui import FusionWindow
            from .fusion_replay import FusionSessionReplay
            manifest = json.loads((self.recording_path / 'manifest.json').read_text())
            candidates = [r for r in self.report['parts'].values() if 'basis_matrix' in r]
            baselines = []
            legacy = self.baseline_combo.currentData() == 'legacy'
            for result in candidates:
                baseline = dict(result)
                old = result['legacy_fit']
                if legacy and 'basis_matrix' not in old:
                    raise ValueError('原IMU标定无法求解，请选择录制时现有映射作为对比。')
                baseline['basis_matrix'] = old['basis_matrix'] if legacy else result['existing_basis_matrix']
                baselines.append(baseline)
            modes = dict(manifest.get('algorithm_modes_at_start', {}))
            for result in candidates:
                modes[result['part']] = GYRO_TILT
            source = FusionSessionReplay(self.recording_path, speed=self.replay_speed.value())
            try:
                window = FusionWindow(source, self.owner.settings_path,
                    manifest['assignments_at_start'], manifest.get('axis_mappings_at_start', {}),
                    algorithm_modes=modes, calibration_candidates=candidates,
                    comparison_shared_calibrations=baselines,
                    manual_mapping_source='同数据原IMU标定' if legacy else '录制时现有/手动映射')
            except Exception:
                source.close(); raise
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.owner._dialogs.append(window)
            window.destroyed.connect(lambda: self.owner._dialogs.remove(window)
                                     if window in self.owner._dialogs else None)
            window.show()
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, '回放失败', str(exc))
