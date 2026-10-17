"""Preset integration: parameter-only transactions, never a hardware command."""
from copy import deepcopy
from PySide6 import QtCore, QtWidgets
from .real_robot_checkpoints import CheckpointStore, FIELDS, MODES, create, draft, validate
from .real_robot_start_settings import START_LABELS
from .real_robot_stop_gesture import GESTURE_KINDS


class CheckpointSession:
    def __init__(self, owner, store=None):
        self.owner = owner
        self.store = store or CheckpointStore()
        self.loaded = {}
        self.panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self.panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QtWidgets.QPushButton('参数 Checkpoint…')
        self.label = QtWidgets.QLabel('尚未加载参数预设')
        self.label.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.button); layout.addWidget(self.label, 1)
        self.button.clicked.connect(self.open)
        for rows in (owner.axis_rows, owner.right_axis_rows):
            for row in rows:
                for widget in row[1:5]: widget.valueChanged.connect(self.changed)
        for row in owner.end_session.rows:
            for widget in row: widget.valueChanged.connect(self.changed)
        end = owner.end_session
        end.enable.toggled.connect(self.changed)
        end.kind.currentIndexChanged.connect(self.changed)
        end.order.textChanged.connect(self.changed)
        for widget in (end.seconds, end.speed, end.acceleration, end.dwell):
            widget.valueChanged.connect(self.changed)
        owner.mode_select.currentIndexChanged.connect(self.changed)
        for widget in owner.auto_start.widgets.values(): widget.valueChanged.connect(self.changed)
        for widget in (owner.manual, owner.shared_offset, owner.right_options[1]):
            widget.toggled.connect(self.changed)

    def input_widgets(self):
        d = self.owner
        return {'left': d.shared_offset, **({'right': d.right_options[1]} if d._bilateral_mode() else {})}

    def input_values(self):
        return dict(manual_mapping=self.owner.manual.isChecked(),
                    palm_offset_enabled={s:w.isChecked() for s,w in self.input_widgets().items()})

    def mode(self):
        if not self.owner._shared_mode():
            raise ValueError('仅左臂 / 双臂仿真同源 IK 支持 Checkpoint')
        return 'bilateral_shared_ik' if self.owner._bilateral_mode() else 'left_shared_ik'

    def available(self):
        d = self.owner
        if (not d._shared_mode() or d.busy or d.ik_worker is not None
                or d.end_session.returning or d.end_session.await_home or d.auto_start.pending):
            return False
        if d.worker is not None:
            if d.worker.active or d.worker.pending_start is not None or d.worker.quit_requested:
                return False
            if d.worker.return_snapshot()[2]: return False
        return True

    def controls(self):
        self.panel.setVisible(self.owner._shared_mode())
        self.button.setEnabled(self.available())

    def parameters(self):
        mode = self.mode()
        joints = {}
        for side in MODES[mode]:
            rows = self.owner.axis_rows if side == 'left' else self.owner.right_axis_rows
            for i, row in enumerate(rows, 1):
                joints[f'{side[0].upper()}{i}'] = dict(zip(FIELDS, (w.value() for w in row[1:5])))
        return joints, self.owner.end_session.checkpoint_values()

    def snapshot(self, name='新预设', notes='', *, validated=True):
        joints, gesture = self.parameters()
        return (create if validated else draft)(self.mode(), joints, gesture, name, notes,
                                                start_settings=self.owner.auto_start.values(),
                                                input_settings=self.input_values(),
                                                **self.owner.end_session.checkpoint_options())

    def changed(self, *_):
        if not self.owner._shared_mode(): return
        saved = self.loaded.get(self.mode())
        if saved is None:
            self.label.setText('尚未加载参数预设')
        else:
            joints, gesture = self.parameters()
            dirty = joints != saved['joints'] or gesture != saved['gesture']
            dirty |= any(saved[k] != v for k, v in self.owner.end_session.checkpoint_options().items())
            dirty |= saved['start_settings'] != self.owner.auto_start.values()
            dirty |= saved['input_settings'] != self.input_values()
            self.label.setText(f'最近加载：{saved["name"]}'+('（已修改，未保存到预设）' if dirty else '（数值一致）'))

    def differences(self, data):
        data = validate(data, self.mode())
        joints, gesture = self.parameters()
        lines = []
        labels = dict(lower='最小角', upper='最大角', speed='速度', acceleration='加速度',
                      angle='停止姿态参考角', tolerance='容差')
        for previous, following in ((joints, data['joints']), (gesture, data['gesture'])):
            for axis, values in following.items():
                for field, value in values.items():
                    before = previous.get(axis, {}).get(field)
                    if before != value:
                        old = '不参与检测' if before is None else f'{before:.2f}'
                        lines.append(f'{axis} {labels[field]}：{old} → {value:.2f}')
            for axis in previous.keys()-following.keys():
                lines.append(f'{axis}：不再参与停止姿态检测')
        labels.update(enabled='启用停止信号', seconds='持续秒数', order='每侧回位顺序',
                      speed='回位速度', acceleration='回位加速度', dwell='步间停留秒', kind='停止姿态类型')
        for section, previous in self.owner.end_session.checkpoint_options().items():
            for key, value in data[section].items():
                if previous[key] != value:
                    old = GESTURE_KINDS[previous[key]] if key == 'kind' else previous[key]
                    new = GESTURE_KINDS[value] if key == 'kind' else value
                    lines.append(f'{labels[key]}：{old} → {new}')
        for key, previous in self.owner.auto_start.values().items():
            if previous != data['start_settings'][key]:
                lines.append(f'{START_LABELS[key]}：{previous} → {data["start_settings"][key]}')
        before = self.input_values(); after = data['input_settings']
        switches = [('使用手动映射', before['manual_mapping'], after['manual_mapping'])]
        switches += [(('左手' if s=='left' else '右手')+'手掌固定转向',
                      before['palm_offset_enabled'][s], value) for s,value in after['palm_offset_enabled'].items()]
        for label, old, new in switches:
            if old != new: lines.append(f'{label}：{"开启" if old else "关闭"} → {"开启" if new else "关闭"}')
        return '\n'.join(lines) or '数值与当前界面一致。仍会清除接管参考和安全确认。'

    def apply(self, data):
        if not self.available():
            raise ValueError('运行、减速、断流等待、IK 或回位确认期间不能加载')
        data = validate(data, self.mode())
        d = self.owner
        updates = []
        for side in MODES[self.mode()]:
            rows = d.axis_rows if side == 'left' else d.right_axis_rows
            for i, row in enumerate(rows, 1):
                values = data['joints'][f'{side[0].upper()}{i}']
                updates.extend((widget, values[field]) for widget, field in zip(row[1:5], FIELDS))
        updates.extend(d.end_session.checkpoint_updates(data['gesture'], data['stop_signal']['kind']))
        updates.extend((w, data['start_settings'][k]) for k, w in d.auto_start.widgets.items())
        end = d.end_session
        updates.append((end.seconds, data['stop_signal']['seconds']))
        updates.extend((getattr(end, key), data['return_settings'][key])
                       for key in ('speed', 'acceleration', 'dwell'))
        # Validate widget envelopes before touching ANY value: no Qt clamping.
        for widget, value in updates:
            if not widget.minimum() <= value <= widget.maximum():
                raise ValueError('预设数值超出当前界面范围；未加载')
        blockers = [QtCore.QSignalBlocker(w) for w, _ in updates]
        blockers += [QtCore.QSignalBlocker(d.safety), QtCore.QSignalBlocker(d.end_session.enable)]
        blockers.append(QtCore.QSignalBlocker(end.order))
        blockers.append(QtCore.QSignalBlocker(end.kind))
        blockers += [QtCore.QSignalBlocker(w) for w in (d.manual, *self.input_widgets().values())]
        try:
            for widget, value in updates: widget.setValue(value)
            d.safety.setChecked(False)
            end.enable.setChecked(data['stop_signal']['enabled'])
            end.kind.setCurrentIndex(end.kind.findData(data['stop_signal']['kind']))
            end.order.setText(','.join(map(str, data['return_settings']['order'])))
            d.manual.setChecked(data['input_settings']['manual_mapping'])
            for side, widget in self.input_widgets().items():
                widget.setChecked(data['input_settings']['palm_offset_enabled'][side])
        finally:
            del blockers
        d.end_session.clear_checkpoint_confirmation()
        d._invalidate_capture()
        d.arm_history.clear(); d.history.clear()
        d.arm_sample = d.sample = None
        self.loaded[self.mode()] = deepcopy(data)
        self.changed()
        d.start_notice.setText('参数预设已加载；请重新核对输出轴、安全许可及停止姿态参考，再采集目标。')
        d._arm_event('checkpoint_loaded', checkpoint_id=data['id'], name=data['name'],
                     checkpoint=data, parameter_only=True)
        d._controls()

    def open(self):
        if not self.available(): return
        from .real_robot_checkpoint_manager import CheckpointManager
        try:
            CheckpointManager(self).exec()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self.owner, 'Checkpoint 无法打开', str(exc))
