"""Reuse guided capture, but keep wearing maps separate from heading feedback."""
import json
from pathlib import Path
import sys
import uuid

from PySide6 import QtCore, QtWidgets

from .camera_calibration_gui import CameraCalibrationDialog, calibration_steps
from .pose import GYRO_TILT
from .visual_wearing import FORMAT, PASS, REVIEW, needs_review, PART_LABELS

CAPTURE_SCOPES = {
    'right': ('right_upper_arm', 'right_forearm'),
    'left': ('left_upper_arm', 'left_forearm'),
    'bilateral': tuple(PART_LABELS),
}


def wearing_steps(preparation, hold, neutral, validation, scope):
    steps = calibration_steps(preparation, hold, neutral, validation)
    arm = {'right': '右臂', 'left': '左臂', 'bilateral': '双臂'}[scope]
    side = '同时向左右两侧平举（T字姿势）' if scope == 'bilateral' else (
        '向身体左侧平举' if scope == 'left' else '向身体右侧平举')
    instructions = {
        'neutral': f'站直、面向相机，{arm}自然下垂，肘腕伸直；两肩和髋完整入镜。',
        'forward': f'{arm}肘腕伸直，向正前方平举到水平；保持身体朝向。',
        'down': f'{arm}缓慢放下，回到自然下垂。',
        'lateral': f'{arm}肘腕伸直，{side}；不要耸肩或侧倾。',
        'forward_repeat': f'{arm}再次前平举，肘腕伸直（独立复测）。',
        'lateral_repeat': f'{arm}再次{side}，肘腕伸直（独立复测）。',
        'neutral_return': f'{arm}放下，恢复最初下垂和手背方向，站直静止。',
        'validation': f'{arm}分别或同时重复前举、侧举、屈肘、轻微转臂；每种姿态停留2–3秒，保持入镜。',
    }
    return [(key, title.replace('右臂', arm), instructions.get(key, instruction), prep, duration)
            for key, title, instruction, prep, duration in steps]


class VisualWearingDialog(CameraCalibrationDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.setWindowTitle('视觉辅助穿戴标定 · 双臂 / 单臂（实验）')
        self.resize(820, 790)
        self.report = None
        self.backup = None
        self._preview_opened = False
        self._application_done = False
        self.auto_analyze.setText('录制后计算视觉辅助穿戴映射（不自动应用）')
        from .calibration_options import add_prior_control
        add_prior_control(self, owner, visual=True)
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
        note.setText('支持左右上臂/前臂：自然下垂无需强行贴紧身体。前举/侧举时肘腕伸直，站直且保持朝向。双臂联合时同一轮动作采集四台设备，各部位独立拟合和验收。固定相机和绑带；无需先做原IMU标定，不自动应用或启用机器人，解剖学扭转仍需实测。')
        selection = QtWidgets.QGridLayout()
        self.part_checks = {}
        for index, (part, label) in enumerate(PART_LABELS.items()):
            check = QtWidgets.QCheckBox(label)
            check.setEnabled(False)
            self.part_checks[part] = check
            check.toggled.connect(self.update_apply_gate)
            selection.addWidget(check, index//2, index%2)
        self.baseline_combo = QtWidgets.QComboBox()
        self.baseline_combo.addItem('对比：录制时现有/手动映射', 'existing')
        self.baseline_combo.addItem('对比：同数据原 IMU 平举标定', 'legacy')
        selection.addWidget(self.baseline_combo, 2, 0, 1, 2)
        layout.addLayout(selection)
        self.review_confirmation = QtWidgets.QCheckBox('我已查看本次双模型回放，并确认需复核部位的动作方向和轴向运动可接受')
        self.review_confirmation.setEnabled(False)
        self.review_confirmation.toggled.connect(self.update_apply_gate)
        layout.addWidget(self.review_confirmation)
        self.load_recording_button = QtWidgets.QPushButton('加载已有引导录制 · 离线重新计算')
        self.load_recording_button.clicked.connect(self.load_recording)
        layout.addWidget(self.load_recording_button)
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
        self.update_plan()

    def capture_parts(self):
        scope = self.scope_combo.currentData() if hasattr(self, 'scope_combo') else 'right'
        return CAPTURE_SCOPES[scope]

    def capture_steps(self):
        scope = self.scope_combo.currentData() if hasattr(self, 'scope_combo') else 'right'
        return wearing_steps(self.preparation.value(), self.hold.value(),
                             self.neutral.value(), self.validation.value(), scope)

    def update_plan(self):
        super().update_plan()
        if hasattr(self, 'scope_combo'):
            names = '、'.join(PART_LABELS[p] for p in self.capture_parts())
            self.plan.setText(f'参与部位：{names}。' + self.plan.text())
            if self.state in ('idle', 'complete', 'cancelled'):
                self.instruction.setText(f'确认{names}设备在线且选择陀螺＋加计；开启三维观测。')

    def guide_extra(self):
        return {'workflow': 'visual_assisted_wearing', 'capture_scope': self.capture_scope,
                'participants': list(self.devices)}

    def readiness_error(self):
        error = super().readiness_error()
        if error:
            return error
        if getattr(self.owner, 'camera_worker', None) is None or not getattr(self.owner.camera_worker, 'capture_epoch', None):
            return '相机采集会话尚未就绪，请启动Gemini并等待预热。'
        if self.owner.x1_preview_dialog is not None and self.owner.x1_preview_dialog.sender is not None:
            return '请先停止 X1 随动，再做穿戴标定。'
        if len({self.owner.engine.assignments[p] for p in self.capture_parts()}) != len(self.capture_parts()):
            return '参与部位必须分配不同设备，不可重复使用同一ID。'
        for part in self.capture_parts():
            if self.owner.engine.algorithm_modes[part] != GYRO_TILT:
                return f'{PART_LABELS[part]}需选择“陀螺＋加计”，不使用在线视觉纠偏流拟合安装。'
        return None

    def set_controls(self, running):
        super().set_controls(running)
        if hasattr(self, 'load_recording_button'):
            self.load_recording_button.setEnabled(not running)
        if hasattr(self, 'scope_combo'):
            self.scope_combo.setEnabled(not running)

    def load_recording(self):
        if self.active or self.state == 'analyzing':
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, '选择完整引导录制目录',
            str(Path(__file__).resolve().parent.parent / 'recordings'))
        if not directory:
            return
        path = Path(directory)
        try:
            guide = json.loads((path / 'camera_guide.json').read_text())
            from .calibration_options import capture_prior_weight
            weight = capture_prior_weight(guide)
            if guide.get('status') != 'complete' or not all((path / name).is_file() for name in
                    ('manifest.json', 'runtime_pose.jsonl', 'vision.jsonl', 'imu.csv')):
                raise ValueError('请选择包含实际姿态和三维观测的完整引导录制目录')
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, '无法加载', str(exc)); return
        self.recording_path = path
        from .calibration_options import capture_prior_weight
        self.prior_weight_spin.blockSignals(True)
        self.prior_weight_spin.setValue(weight)
        self.prior_weight_spin.blockSignals(False)
        self.prior_weight_spin.update_prior_hint(self.prior_weight_spin.value())
        self.state = 'complete'
        self.analyze_recording()

    def monitor_error(self):
        error = super().monitor_error()
        if error:
            return error
        if getattr(self.owner.camera_worker, 'capture_epoch', None) != getattr(self, 'reference_camera_epoch', None):
            return '相机采集会话发生变化，请重新录制。'
        for part in self.devices:
            if self.owner.engine.algorithm_modes[part] != GYRO_TILT:
                return '录制期间姿态模式改变，请重录。'
        if self.owner.x1_preview_dialog is not None and self.owner.x1_preview_dialog.sender is not None:
            return 'X1 随动已开启，本轮停止。'
        return None

    def begin(self):
        self.capture_scope = self.scope_combo.currentData()
        self.reference_camera_epoch = getattr(self.owner.camera_worker, 'capture_epoch', None)
        super().begin()
        if self.active:
            self.report = None
            self._preview_opened = False
            self._application_done = False
            self.review_confirmation.setChecked(False)
            self.review_confirmation.setEnabled(False)
            for check in self.part_checks.values():
                check.setEnabled(False); check.setChecked(False)

    def analyze_recording(self):
        if self.state != 'complete':
            return
        self.prior_capture_label.setText(f'本次原IMU对照 λ={self.prior_weight_spin.value():.3f}（不影响视觉主拟合）')
        from .calibration_options import set_prior_enabled
        set_prior_enabled(self, False)
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
        self.process.setArguments(['-u', str(script), str(self.recording_path), '--output', str(self.analysis_path),
                                   '--prior-weight', str(self.prior_weight_spin.value())])
        self.state = 'analyzing'
        self.load_recording_button.setEnabled(False)
        self.scope_combo.setEnabled(False)
        self.report = None
        self._preview_opened = False
        self._application_done = False
        self.review_confirmation.setChecked(False)
        self.review_confirmation.setEnabled(False)
        for button in (self.start, self.analyze, self.apply, self.replay):
            button.setEnabled(False)
        for check in self.part_checks.values():
            check.setEnabled(False)
        self.log = ''
        self.status.setText('正在拟合真实骨轴；不要求中立骨段竖直，不修改现有映射。')
        self.process.start()

    def analysis_finished(self, code, _status):
        from .calibration_options import set_prior_enabled
        set_prior_enabled(self, True)
        self.read_log(); self.state = 'complete'
        self.start.setEnabled(True); self.analyze.setEnabled(True)
        self.load_recording_button.setEnabled(True)
        self.scope_combo.setEnabled(True)
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
        self.step_label.setText('分析完成')
        lines = []
        for part, check in self.part_checks.items():
            result = report['parts'].get(part)
            if result is None:
                check.setEnabled(False); check.setChecked(False)
                continue
            good = result['status'] in (PASS, REVIEW)
            check.setEnabled(good); check.setChecked(good)
            label = PART_LABELS[part]
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
            lines.append(text + '\n' + '；'.join(result['warnings'] + result.get('review_reasons', []) + result.get('notes', [])))
        self.status.setText('\n'.join(lines) + f'\n结果：{self.analysis_path}\n误差是视觉一致性，不是GT；请先回放并检查轴向转动和肘部。')
        assessment = report.get('posture_assessment', {})
        details = []
        for name, window in assessment.get('windows', {}).items():
            def value(key, statistic):
                metric = window[key]
                number = metric.get(statistic)
                return '不可靠' if not metric.get('reliable') or number is None else f'{number:.1f}°'
            details.append(f"{name}：倾斜P90 {value('tilt', 'p90_abs_deg')}；"
                           f"肩线中位 {value('shoulder_heading', 'median_deg')}；"
                           f"髋线中位 {value('hip_heading', 'median_deg')}；{window['heading_conclusion']}")
        if details:
            self.status.setText(self.status.text() + '\n分窗口动作诊断：\n' + '\n'.join(details))
        self.update_apply_gate()
        self.replay.setEnabled(any('basis_matrix' in p for p in report['parts'].values()))

    def analysis_error(self, error):
        from .calibration_options import set_prior_enabled
        set_prior_enabled(self, True)
        super().analysis_error(error)
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.load_recording_button.setEnabled(True)
            self.scope_combo.setEnabled(True)

    def update_apply_gate(self):
        if not hasattr(self, 'review_confirmation'):
            return
        selected = [p for p, c in self.part_checks.items() if c.isChecked() and self.report is not None
                    and self.report['parts'].get(p, {}).get('status') in (PASS, REVIEW)]
        review = self.report is not None and any(needs_review(self.report['parts'][p]) for p in selected)
        self.review_confirmation.setEnabled(bool(review and self._preview_opened))
        self.apply.setEnabled(bool(self.report is not None and selected and not self._application_done and
                             (not review or (self._preview_opened and self.review_confirmation.isChecked()))))
        self.review_confirmation.setToolTip('先点击双人体回放，检查完成后再确认。仅影响人工复核结果；失败结果仍禁止应用。')

    def apply_live(self):
        if self.report is None:
            return
        parts = [p for p, c in self.part_checks.items() if c.isChecked()]
        if not parts:
            QtWidgets.QMessageBox.warning(self, '未应用', '请勾选至少一个合格部位。'); return
        review = any(needs_review(self.report['parts'][p]) for p in parts)
        acknowledged = self._preview_opened and self.review_confirmation.isChecked()
        if review and not acknowledged:
            QtWidgets.QMessageBox.warning(self, '未应用', '请先打开本次双模型回放，检查后勾选人工复核确认。'); return
        answer = QtWidgets.QMessageBox.question(self, '确认应用穿戴映射',
            '将更新所选部位的轴映射和本次中立姿态。保留原相机航向纠偏参数，不自动启用融合或机器人。自然下垂偏角归一到模型标准中立位；轴向扭转仍须实测。是否应用？')
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            # The owner validates all parts atomically before changing anything.
            if review:
                backup = self.owner.apply_visual_wearing_report(self.report, parts, review_acknowledged=acknowledged)
            else:
                backup = self.owner.apply_visual_wearing_report(self.report, parts)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, '穿戴标定未应用', str(exc)); return
        self.backup = backup
        self._application_done = True
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
        self._application_done = False
        self.update_apply_gate()
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
            self._preview_opened = True
            self.update_apply_gate()
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, '回放失败', str(exc))
