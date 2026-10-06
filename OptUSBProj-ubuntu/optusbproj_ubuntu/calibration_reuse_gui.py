"""Small guided recovery capture: no RGB video/depth recording or robot startup."""
from datetime import datetime
from pathlib import Path
import time
import uuid

import numpy as np
from PySide6 import QtCore, QtWidgets

from .calibration_reuse import (analyze_recovery, apply_recovery, context_error,
    attempt_still_bias, import_camera_report, import_recording, load_profile, plan_bias_reuse, profile_from_state, write_json)
from .camera_calibration_gui import PART_LABELS
from .pose import GYRO_RGBD


class CalibrationReuseDialog(QtWidgets.QDialog):
    def __init__(self, owner):
        super().__init__(owner, QtCore.Qt.WindowType.Window)
        self.owner = owner;self.state = 'idle';self.profile = None;self.result = None
        self.samples = [];self.windows = [];self.checks = {};self.index = 0
        self.root = Path(__file__).resolve().parents[1]/'calibrations'/'reusable'
        self.setWindowTitle('历史标定 · 保存与快速恢复');self.resize(800, 720)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('设备未换位、绑带未滑动、相机未移动时使用。复用安装几何，尝试学习零偏、采集中立位并恢复相机对齐；零偏重估失败不阻断，不恢复旧积分器或累计修正。先停止 X1，参与臂段请选择“陀螺＋加计”。')
        note.setWordWrap(True);layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.profiles = QtWidgets.QComboBox();row.addWidget(self.profiles, 1)
        self.load = QtWidgets.QPushButton('加载所选');self.load.clicked.connect(self.load_selected);row.addWidget(self.load)
        self.browse = QtWidgets.QPushButton('配置 / 相机报告…');self.browse.clicked.connect(self.browse_profile);row.addWidget(self.browse)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.save = QtWidgets.QPushButton('保存当前已应用标定');self.save.clicked.connect(self.save_current);row.addWidget(self.save)
        self.import_button = QtWidgets.QPushButton('导入历史联合录制…');self.import_button.clicked.connect(self.import_history);row.addWidget(self.import_button)
        layout.addLayout(row)
        self.info = QtWidgets.QLabel('先保存一次有效标定；已有数据可选择历史相机 report.json（需保留其原录制中的穿戴快照），或导入完整已应用快照的联合录制。');self.info.setWordWrap(True);layout.addWidget(self.info)
        self.confirm = QtWidgets.QCheckBox('确认安装未移动、设备参数未改；中立位掌心/拇指方向与历史一致')
        layout.addWidget(self.confirm)
        form = QtWidgets.QFormLayout()
        self.delay = self.spin(form, '开始前倒计时', 0, 120, 10)
        self.prep = self.spin(form, '每步准备时间', 2, 30, 4)
        self.still = self.spin(form, '参考恢复准备时间（尽量安静，可支撑双臂）', 3, 30, 10)
        self.hold = self.spin(form, '每个动作静止采样', 3, 15, 3)
        layout.addLayout(form)
        self.skip_bias = QtWidgets.QCheckBox('跳过本次显式零偏重估（后台自动学习仍运行）')
        self.skip_bias.setToolTip('不沿用历史零偏，不重置当前积分器。尚未学习零偏的设备会明确警告；后续验收仍须通过。')
        layout.addWidget(self.skip_bias)
        self.force_bias = QtWidgets.QCheckBox('强制重新估计零偏（默认自动复用当前已学习结果）')
        self.force_bias.setToolTip('只有当前会话已学习的零偏可自动复用；积分器重建后重新检查。不会恢复历史文件中的零偏。')
        self.force_bias.toggled.connect(lambda checked: self.skip_bias.setChecked(False) if checked else None)
        self.skip_bias.toggled.connect(lambda checked: self.force_bias.setChecked(False) if checked else None)
        layout.addWidget(self.force_bias)
        self.label = QtWidgets.QLabel('等待开始');self.label.setWordWrap(True)
        self.label.setStyleSheet('font-size:26px;font-weight:bold;color:#329fcc');layout.addWidget(self.label)
        self.instruction = QtWidgets.QLabel('流程：参考准备（可选零偏重估）→ 自然下垂 → 前举 → 侧举 → 再前举验收 → 回到中立位。微动导致重估失败不会停止恢复。');self.instruction.setWordWrap(True);layout.addWidget(self.instruction)
        row = QtWidgets.QHBoxLayout()
        self.start = QtWidgets.QPushButton('开始快速恢复');self.start.clicked.connect(self.begin);row.addWidget(self.start)
        self.cancel = QtWidgets.QPushButton('取消本轮');self.cancel.clicked.connect(self.abort);self.cancel.setEnabled(False);row.addWidget(self.cancel)
        layout.addLayout(row)
        self.summary = QtWidgets.QPlainTextEdit();self.summary.setReadOnly(True);layout.addWidget(self.summary, 1)
        self.part_row = QtWidgets.QGridLayout();layout.addLayout(self.part_row)
        self.apply = QtWidgets.QPushButton('应用所选合格部位（不自动开启融合或机器人）')
        self.apply.clicked.connect(self.apply_selected);self.apply.setEnabled(False);layout.addWidget(self.apply)
        self.timer = QtCore.QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self.tick)
        self.refresh_profiles()

    @property
    def active(self):return self.state in ('delay', 'prepare', 'hold', 'analyzing')

    def spin(self, form, label, low, high, value):
        spin = QtWidgets.QSpinBox();spin.setRange(low, high);spin.setValue(value);spin.setSuffix(' 秒');form.addRow(label, spin);return spin

    def refresh_profiles(self):
        self.profiles.clear()
        for path in sorted(self.root.glob('*.json'), reverse=True):
            self.profiles.addItem(path.stem, str(path))

    def set_profile(self, profile, source):
        self.profile = profile;self.result = None;self.apply.setEnabled(False);self.confirm.setChecked(False)
        self.clear_checks()
        parts = '；'.join(f'{PART_LABELS[p]} {d["device_id"]}' for p, d in profile['parts'].items())
        date = profile.get('source_started_at', profile['created_at'])
        self.info.setText(f'标定/源录制时间：{date}\n{parts}\n来源：{source}')
        self.summary.setPlainText('历史安装关系已读取，尚未应用。请保持与历史中立位一致，先进行快速恢复。')

    def load_selected(self):
        if self.profiles.currentData():self.read_profile(Path(self.profiles.currentData()))

    def read_profile(self, path):
        try:self.set_profile(load_profile(path), path)
        except (OSError, ValueError, KeyError, TypeError) as exc:self.error(str(exc))

    def browse_profile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择历史标定配置', str(self.root), 'JSON (*.json)')
        if not path:return
        try:
            import json
            raw = json.loads(Path(path).read_text(encoding='utf-8'))
            if raw.get('format') == 'rgbd-heading-fusion-experiment-v1':
                profile = import_camera_report(Path(path));target = self.root/(self.filename('imported-report')+'.json')
                write_json(target, profile);self.refresh_profiles();self.set_profile(profile, target)
            else:self.read_profile(Path(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:self.error(f'导入相机报告失败：{exc}')

    def filename(self, prefix):return f'{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'

    def save_current(self):
        o = self.owner
        if o.replay_mode or o._pose_change_error():self.error('请在无录制、无机器人随动和无其他标定的实时界面保存');return
        try:
            profile = profile_from_state(o.engine, o.live_fusion, getattr(o.camera_worker, 'calibration', None), o.vision_metadata)
            path = self.root/(self.filename('profile')+'.json');write_json(path, profile)
            self.refresh_profiles();self.set_profile(profile, path);self.summary.setPlainText(f'已保存：{path}\n下次程序启动后加载此配置并做快速恢复。')
        except (OSError, ValueError, KeyError, TypeError) as exc:self.error(str(exc))

    def import_history(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择历史联合录制目录', str(self.root.parents[1]/'recordings'))
        if not path:return
        try:
            profile = import_recording(Path(path));target = self.root/(self.filename('imported')+'.json')
            write_json(target, profile);self.refresh_profiles();self.set_profile(profile, target)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error(f'导入失败：{exc}\n需包含已应用连续穿戴映射、相机标定及对应 runtime_pose；单独的标定动作录制不一定具备这些快照。可在当前标定完成后点“保存当前已应用标定”。')

    def error(self, text):self.summary.setPlainText(text)

    def clear_checks(self):
        while self.part_row.count():
            item = self.part_row.takeAt(0)
            if item.widget():item.widget().deleteLater()
        self.checks = {}

    def busy(self, enabled):
        for widget in (self.profiles, self.load, self.browse, self.save, self.import_button,
                       self.confirm, self.delay, self.prep, self.still, self.hold, self.skip_bias, self.force_bias, self.start):widget.setEnabled(not enabled)
        self.cancel.setEnabled(enabled)
        self.owner._sync_pose_controls()

    def begin(self):
        if self.profile is None:self.error('请先加载历史配置');return
        if not self.confirm.isChecked():self.error('请确认固定安装和历史中立姿态条件');return
        error = self.owner._pose_change_error()
        if error:self.error(error);return
        error = context_error(self.profile, self.owner)
        if error:self.error(error);return
        o = self.owner
        now = time.monotonic_ns()
        if o.last_pose_estimate is None or now-o.last_pose_estimate.captured_mono_ns > 500_000_000:self.error('请等待新的可信三维观测');return
        self.epochs = {d['device_id']: o.engine.epochs[d['device_id']] for d in self.profile['parts'].values()}
        self.camera_epoch = o.camera_worker.capture_epoch;self.vision_epoch = o.vision_epoch
        self.samples = [];self.windows = [];self.result = None;self.gravity = {};self.last_stamp = -1;self.index = 0
        self.clear_checks();self.apply.setEnabled(False)
        self.capture_skip_bias = self.skip_bias.isChecked();self.capture_force_bias = self.force_bias.isChecked()
        try:
            self.bias_pending, self.bias_outcomes = plan_bias_reuse(o.live_fusion, self.epochs, force=self.capture_force_bias, skip=self.capture_skip_bias)
        except ValueError as exc:self.error(str(exc));return
        self.bias = {d: r['bias_dps'] for d, r in self.bias_outcomes.items()}
        self.steps = [('still', '参考恢复准备', '尽量安静，可支撑双臂，不要故意缓慢旋转；微小动作不阻断流程。仅可信片段更新零偏，不清零设备。', self.still.value()),
                      ('neutral', '共同中立位', '站直，双臂自然下垂，掌心和拇指方向与历史标定一致；两肩、髋、肘腕完整入镜。', 5),
                      ('forward', '双臂前平举', '肘伸直，双臂向前举；保持躯干朝向，不耸肩。', self.hold.value()),
                      ('lateral', '双臂侧平举', '肘伸直，双臂向左右展开；保持躯干朝向，避免遮挡肘腕。', self.hold.value()),
                      ('forward_check', '再次前举 · 独立验收', '重新前举并静止；此步不参与 Q 拟合，仅验证。', self.hold.value()),
                      ('return', '回到中立位', '自然下垂，掌心和拇指恢复首次中立位方向。', 5)]
        if not self.bias_pending:self.steps = [s for s in self.steps if s[0] != 'still']
        self.state = 'delay';self.deadline = time.monotonic()+self.delay.value()
        if self.capture_skip_bias:plan = '用户选择跳过显式重估；未学习设备仍有漂移风险，直接准备共同中立位。'
        elif self.capture_force_bias:plan = '按选择强制重估参与设备；未找到可信片段时保留当前零偏。'
        elif not self.bias_pending:plan = '全部参与设备已学习，复用当前结果；倒计时后直接准备共同中立位。'
        else:plan = '仅未学习设备尝试重估；其他设备复用当前结果。'
        self.summary.setPlainText('\n'.join(f'{d}：{r["message"]}' for d,r in self.bias_outcomes.items())+
            '\n'+plan)
        self.busy(True);self.timer.start();self.tick()

    def tick(self):
        if not self.active:return
        error = context_error(self.profile, self.owner, epochs=self.epochs, camera_epoch=self.camera_epoch, vision_epoch=self.vision_epoch)
        if error:self.abort(error);return
        remaining = max(0, self.deadline-time.monotonic())
        if self.state == 'delay':self.label.setText(f'开始前 {remaining:.1f} 秒')
        else:
            phase, title, instruction, duration = self.steps[self.index]
            self.label.setText(f'{title} · {"准备" if self.state == "prepare" else "静止采样"} {remaining:.1f} 秒')
            self.instruction.setText(instruction)
        if remaining > 0:return
        if self.state in ('delay', 'hold'):
            if self.state == 'hold':
                self.windows[-1]['end_ns'] = time.monotonic_ns()
                phase = self.steps[self.index][0]
                try:
                    if phase == 'still':
                        end = self.windows[-1]['end_ns']
                        # Background learning may have completed since begin.
                        pending, ready = plan_bias_reuse(self.owner.live_fusion, self.bias_pending, force=self.capture_force_bias)
                        self.bias_outcomes.update(ready)
                        self.bias_outcomes.update(attempt_still_bias(self.owner.live_fusion, pending, self.windows[-1]['start_ns'], end))
                        self.bias = {d: r['bias_dps'] for d, r in self.bias_outcomes.items()}
                        self.summary.setPlainText('\n'.join(f'{d}：{r["message"]}' for d, r in self.bias_outcomes.items()))
                    if phase == 'neutral':
                        self.gravity = self.owner.live_fusion.lock_gravity(self.epochs, self.windows[-1]['start_ns'], self.windows[-1]['end_ns'])
                except (ValueError, KeyError, TypeError) as exc:self.abort(str(exc));return
                self.index += 1
                if self.index == len(self.steps):self.finish();return
            self.state = 'prepare';self.deadline = time.monotonic()+self.prep.value()
        else:
            self.state = 'hold';self.deadline = time.monotonic()+self.steps[self.index][3]
            self.windows.append({'phase': self.steps[self.index][0], 'start_ns': time.monotonic_ns(), 'end_ns': None})

    def on_estimate(self, estimate):
        if not self.active:return
        stamp = estimate.captured_mono_ns
        if stamp <= self.last_stamp or time.monotonic_ns()-stamp > 500_000_000:return
        self.last_stamp = stamp
        window = next((w for w in reversed(self.windows) if w['start_ns'] <= stamp and (w['end_ns'] is None or stamp <= w['end_ns'])), None)
        if window is None or window['phase'] == 'still':return
        imu = {};modes = {}
        for device in self.epochs:
            r = self.owner.live_fusion.at(device, stamp)
            if r is not None:
                imu[device] = r.tolist()
                modes[device] = {m: v.tolist() for m, v in self.owner.engine.mode_current.get(device, {}).items() if m != GYRO_RGBD}
        self.samples.append({'phase': window['phase'], 'stamp_ns': stamp, 'vision': estimate.to_record(), 'imu': imu, 'modes': modes})

    def finish(self):
        self.timer.stop();self.state = 'analyzing'
        run = self.root/'runs'/self.filename('recovery')
        try:
            write_json(run/'capture.json', {'windows': self.windows, 'samples': self.samples, 'gravity': self.gravity, 'bias_dps': self.bias,
                'bias_outcomes': self.bias_outcomes, 'skip_explicit_bias_learning': self.capture_skip_bias,
                'force_bias_learning': self.capture_force_bias,
                'epochs': self.epochs, 'camera_epoch': self.camera_epoch, 'vision_epoch': self.vision_epoch, 'profile': self.profile})
            self.result = analyze_recovery(self.profile, self.samples, self.owner.engine, self.gravity,
                camera_epoch=self.camera_epoch, vision_epoch=self.vision_epoch, vision_config=self.owner.vision_metadata)
            self.result['bias_outcomes'] = self.bias_outcomes
            write_json(run/'report.json', self.result);self.report_path = run/'report.json'
            self.show_result();self.state = 'complete';self.label.setText('恢复验证完成 · 尚未应用')
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.result = None;self.error(f'恢复失败：{exc}\n采样目录（若写入成功）：{run}');self.state = 'failed'
            if (run/'capture.json').exists() and not (run/'report.json').exists():
                try:write_json(run/'report.json', {'workflow': 'historical_calibration_recovery', 'accepted': False, 'reason': str(exc)})
                except OSError:pass
        self.busy(False)

    def show_result(self):
        lines = [f'{d}：{r["message"]}' for d, r in self.result.get('bias_outcomes', {}).items()]
        labels = {'neutral': '中立', 'forward': '前举', 'lateral': '侧举', 'forward_check': '再次前举', 'return': '回中立'}
        for index, (part, data) in enumerate(self.result['parts'].items()):
            check = QtWidgets.QCheckBox(f'{PART_LABELS[part]} · {"合格" if data["accepted"] else "需重新验证/标定"}')
            check.setEnabled(data['accepted']);check.setChecked(data['accepted']);self.part_row.addWidget(check, index//2, index%2);self.checks[part] = check
            errors = '，'.join(f'{labels.get(p,p)} P90 {e:.1f}°' for p, e in data.get('phase_p90_deg', {}).items())
            checks = f'中立系差 {data["neutral_alignment_deg"]:.1f}°，返回中立差 {data["return_error_deg"]:.1f}°' if 'neutral_alignment_deg' in data else ''
            lines.append(f'{PART_LABELS[part]}：{errors}\n{checks}\n'+('；'.join(data.get('warnings', [])) or '通过短动作一致性验收（非真值精度）'))
        self.summary.setPlainText('\n\n'.join(lines)+f'\n\n结果：{self.report_path}')
        self.apply.setEnabled(any(d['accepted'] for d in self.result['parts'].values()))

    def apply_selected(self):
        if self.result is None:return
        if not self.confirm.isChecked():self.error('请再次确认相机与绑带未移动、中立姿态一致');return
        error = context_error(self.profile, self.owner, epochs=self.epochs, camera_epoch=self.camera_epoch, vision_epoch=self.vision_epoch)
        if error:self.error(error);return
        if self.owner._pose_change_error():self.error('请先结束其他标定、录制或机器人随动');return
        try:
            parts = [p for p, check in self.checks.items() if check.isChecked()]
            apply_recovery(self.owner, self.result, parts)
            self.apply.setEnabled(False);self.label.setText('已应用 · 未自动开启融合或机器人')
            self.summary.appendPlainText('请在合格部位的算法下拉框自行选择“陀螺＋加计＋RGB-D”，先检查人体模型，再决定是否启动 X1。')
        except (ValueError, KeyError) as exc:self.error(str(exc))

    def abort(self, reason='用户取消；未应用穿戴/相机标定，已完成的静止零偏学习保留'):
        self.timer.stop();self.state = 'idle';self.result = None;self.apply.setEnabled(False)
        self.label.setText('恢复已停止');self.error(reason if isinstance(reason, str) else '用户取消；未应用标定');self.busy(False)

    def shutdown(self):
        if self.active:self.abort()
        self.timer.stop()

    def closeEvent(self, event):self.shutdown();event.accept()
