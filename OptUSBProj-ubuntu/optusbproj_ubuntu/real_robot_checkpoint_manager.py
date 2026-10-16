"""Modal preset editor. Saving a file never applies it or starts a robot."""
from copy import deepcopy
from datetime import datetime
from PySide6 import QtCore, QtWidgets
from .real_robot_checkpoints import FIELDS, MODES, HOLD_RANGE, gesture_axes, create, validate
from .real_robot_protocol import ARM_LIMITS
from .real_robot_start_settings import START_RANGES, START_LABELS
from .real_robot_stop_gesture import GESTURE_KINDS


class CheckpointManager(QtWidgets.QDialog):
    def __init__(self, session):
        super().__init__(session.owner)
        self.session, self.mode = session, session.mode()
        self.current = None
        self.dirty = False
        self.loading = False
        self.selected_id = None
        self.setWindowTitle('参数 Checkpoint · '+('双臂同源 IK' if self.mode == 'bilateral_shared_ik' else '左臂同源 IK'))
        self.resize(1000, 720)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel('保存关节、停止信号、回位参数及接管容差 / 倒计时 / 等待时间；不保存方向、频率、掌向、输出许可、自动启动授权或运行状态。\n'
                                '保存只写文件；加载只改界面，不发送指令。加载后安全和停止姿态参考必须重新确认。')
        info.setWordWrap(True); layout.addWidget(info)
        splitter = QtWidgets.QSplitter()
        self.items = QtWidgets.QListWidget()
        self.items.setMinimumWidth(220)
        splitter.addWidget(self.items)
        editor = QtWidgets.QWidget(); form = QtWidgets.QVBoxLayout(editor)
        self.name = QtWidgets.QLineEdit(); self.name.setMaxLength(80)
        self.notes = QtWidgets.QPlainTextEdit(); self.notes.setMaximumHeight(70)
        fields = QtWidgets.QFormLayout(); fields.addRow('名称', self.name); fields.addRow('备注', self.notes)
        form.addLayout(fields)
        self.tabs = QtWidgets.QTabWidget(); self.joint_widgets = {}
        for side in MODES[self.mode]:
            page = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(page)
            for col, title in enumerate(('关节', '最小 °', '最大 °', '速度 °/s', '加速度 °/s²')):
                grid.addWidget(QtWidgets.QLabel(title), 0, col)
            for i in range(1, 7):
                axis = f'{side[0].upper()}{i}'
                index = (6 if side == 'left' else 0)+i-1
                lo, hi = ARM_LIMITS[index]
                grid.addWidget(QtWidgets.QLabel(axis), i, 0)
                self.joint_widgets[axis] = {}
                for col, key in enumerate(FIELDS, 1):
                    widget = self.spin(lo if col < 3 else .1, hi if col < 3 else 250.)
                    self.joint_widgets[axis][key] = widget; grid.addWidget(widget, i, col)
            self.tabs.addTab(page, '左臂' if side == 'left' else '右臂')
        form.addWidget(self.tabs)
        self.stop_kind = QtWidgets.QComboBox()
        for key, label in GESTURE_KINDS.items(): self.stop_kind.addItem(label+'停止', key)
        kind_form = QtWidgets.QFormLayout(); kind_form.addRow('停止姿态（二选一）', self.stop_kind)
        form.addLayout(kind_form)
        self.stop_enabled = QtWidgets.QCheckBox('启用停止信号（加载后仍须重新确认停止姿态参考）')
        self.stop_seconds = self.spin(*HOLD_RANGE)
        form.addWidget(self.stop_enabled)
        duration = QtWidgets.QFormLayout(); duration.addRow('持续时间（秒）', self.stop_seconds)
        form.addLayout(duration)
        group = QtWidgets.QGroupBox('结束手势（协议角，不是场景角）')
        grid = QtWidgets.QGridLayout(group); self.gesture_widgets = {}; self.gesture_labels = {}
        for col, title in enumerate(('关节', '停止姿态参考角 °', '容差 ±°')):
            grid.addWidget(QtWidgets.QLabel(title), 0, col)
        for row, axis in enumerate(gesture_axes(self.mode, 'front_raise'), 1):
            index = (6 if axis[0] == 'L' else 0)+int(axis[1:])-1
            angle, tolerance = self.spin(*ARM_LIMITS[index]), self.spin(.1, 30.)
            angle.setValue(90.); tolerance.setValue(5.)
            self.gesture_widgets[axis] = dict(angle=angle, tolerance=tolerance)
            self.gesture_labels[axis] = QtWidgets.QLabel(axis)
            grid.addWidget(self.gesture_labels[axis], row, 0)
            grid.addWidget(angle, row, 1); grid.addWidget(tolerance, row, 2)
        form.addWidget(group)
        returning = QtWidgets.QGroupBox('顺序回位参数（不会自动执行回位）')
        return_form = QtWidgets.QFormLayout(returning)
        self.return_order = QtWidgets.QLineEdit()
        return_form.addRow('每侧顺序（1–6 各一次）', self.return_order)
        self.return_widgets = {}
        for key, label, lo, hi in (('speed', '回位速度 °/s', .1, 250.),
                                    ('acceleration', '回位加速度 °/s²', .1, 250.),
                                    ('dwell', '步间停留秒（不是到位反馈）', 0., 10.)):
            self.return_widgets[key] = self.spin(lo, hi)
            return_form.addRow(label, self.return_widgets[key])
        form.addWidget(returning)
        starting = QtWidgets.QGroupBox('接管参数（加载不会启动或授予自动运行许可）')
        start_form = QtWidgets.QFormLayout(starting)
        self.start_widgets = {}
        for key, bounds in START_RANGES.items():
            self.start_widgets[key] = self.spin(*bounds)
            start_form.addRow(START_LABELS[key], self.start_widgets[key])
        form.addWidget(starting); form.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(editor)
        splitter.addWidget(scroll); splitter.setStretchFactor(1, 3); layout.addWidget(splitter, 1)
        self.message = QtWidgets.QLabel(); self.message.setWordWrap(True)
        self.message.setTextFormat(QtCore.Qt.PlainText); layout.addWidget(self.message)
        self.file_errors = QtWidgets.QLabel(); self.file_errors.setWordWrap(True)
        self.file_errors.setTextFormat(QtCore.Qt.PlainText); layout.addWidget(self.file_errors)
        buttons = QtWidgets.QHBoxLayout()
        for title, method in (('从当前界面新建', self.new), ('保存修改', self.save),
                              ('另存为', self.save_as), ('加载到当前界面', self.load),
                              ('删除', self.delete), ('关闭', self.reject)):
            button = QtWidgets.QPushButton(title); button.clicked.connect(method); buttons.addWidget(button)
        layout.addLayout(buttons)
        self.items.currentItemChanged.connect(self.select)
        self.name.textChanged.connect(self.edited); self.notes.textChanged.connect(self.edited)
        self.stop_enabled.toggled.connect(self.edited)
        self.stop_kind.currentIndexChanged.connect(self.kind_changed)
        self.stop_seconds.valueChanged.connect(self.edited)
        self.return_order.textChanged.connect(self.edited)
        for widget in self.return_widgets.values(): widget.valueChanged.connect(self.edited)
        for widget in self.start_widgets.values(): widget.valueChanged.connect(self.edited)
        for table in (self.joint_widgets, self.gesture_widgets):
            for values in table.values():
                for widget in values.values(): widget.valueChanged.connect(self.edited)
        self.refresh()
        self.new()

    @staticmethod
    def spin(lo, hi):
        widget = QtWidgets.QDoubleSpinBox(); widget.setRange(lo, hi); widget.setDecimals(2)
        return widget

    def edited(self, *_):
        if not self.loading:
            self.dirty = True
            self.message.setText('编辑尚未保存；不会自动覆盖文件或主窗口。')

    def kind_changed(self):
        active = gesture_axes(self.mode, self.stop_kind.currentData())
        for axis, values in self.gesture_widgets.items():
            for widget in (self.gesture_labels[axis], *values.values()):
                widget.setVisible(axis in active); widget.setEnabled(axis in active)
        self.edited()

    def discard(self):
        return not self.dirty or QtWidgets.QMessageBox.question(
            self, '放弃未保存修改？', '继续将放弃编辑区尚未保存的修改。',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel) == QtWidgets.QMessageBox.Yes

    def refresh(self, ident=None):
        rows, errors = self.session.store.list(self.mode)
        blocker = QtCore.QSignalBlocker(self.items)
        self.items.clear()
        for data in rows:
            date = datetime.fromisoformat(data['updated_at']).astimezone().strftime('%m-%d %H:%M:%S')
            item = QtWidgets.QListWidgetItem(f'{data["name"]}\n{date} · {data["id"][:8]}')
            item.setSizeHint(QtCore.QSize(220, 48))
            item.setData(QtCore.Qt.UserRole, data['id'])
            item.setToolTip(f'{data["name"]}\n{date}\n{data["notes"]}')
            self.items.addItem(item)
            if data['id'] == ident: self.items.setCurrentItem(item)
        del blocker
        self.selected_id = ident
        self.file_errors.setText('以下损坏预设已跳过：\n'+'\n'.join(errors) if errors else '')
        self.file_errors.setVisible(bool(errors))

    def populate(self, data):
        self.loading = True
        try:
            self.current = deepcopy(data)
            self.name.setText(data['name']); self.notes.setPlainText(data['notes'])
            self.stop_enabled.setChecked(data['stop_signal']['enabled'])
            self.stop_kind.setCurrentIndex(self.stop_kind.findData(data['stop_signal']['kind']))
            self.kind_changed()
            self.stop_seconds.setValue(data['stop_signal']['seconds'])
            order = data['return_settings']['order']
            self.return_order.setText(','.join(map(str, order)) if isinstance(order, list) else order)
            for key, widget in self.return_widgets.items(): widget.setValue(data['return_settings'][key])
            for key, widget in self.start_widgets.items(): widget.setValue(data['start_settings'][key])
            for widgets, values in ((self.joint_widgets, data['joints']), (self.gesture_widgets, data['gesture'])):
                for axis, row in widgets.items():
                    if axis not in values: continue
                    for key, widget in row.items(): widget.setValue(values[axis][key])
        finally:
            self.loading = False
        self.dirty = False

    def read(self):
        if self.current is None: raise ValueError('请先选择预设，或修正主窗口数值后新建')
        data = deepcopy(self.current)
        data['name'], data['notes'] = self.name.text().strip(), self.notes.toPlainText()
        data['stop_signal'] = dict(enabled=self.stop_enabled.isChecked(), seconds=self.stop_seconds.value(),
                                   kind=self.stop_kind.currentData())
        data['start_settings'] = {k: w.value() for k, w in self.start_widgets.items()}
        try:
            order = [int(x.strip()) for x in self.return_order.text().split(',')]
        except ValueError:
            raise ValueError('每侧回位顺序必须包含 1–6 各一次') from None
        data['return_settings'] = dict(order=order, **{k: w.value() for k, w in self.return_widgets.items()})
        for field, widgets in (('joints', self.joint_widgets), ('gesture', self.gesture_widgets)):
            data[field] = {axis: {key: w.value() for key, w in row.items()} for axis, row in widgets.items()
                           if field != 'gesture' or axis in gesture_axes(self.mode, self.stop_kind.currentData())}
        return validate(data, self.mode)

    def select(self, item, previous):
        if item is None: return
        if not self.discard():
            blocker = QtCore.QSignalBlocker(self.items); self.items.setCurrentItem(previous); del blocker
            return
        try:
            ident = item.data(QtCore.Qt.UserRole)
            self.populate(self.session.store.load(self.mode, ident)); self.selected_id = ident
            self.message.setText('已选择预设；尚未加载到主窗口。')
        except (OSError, ValueError) as exc: self.error(exc)

    def new(self):
        if not self.discard(): return
        try:
            self.populate(self.session.snapshot(validated=False))
            self.refresh()
            self.message.setText('已复制当前界面数值；请命名并保存。')
        except (OSError, ValueError) as exc: self.error(exc)

    def save(self):
        try:
            data = self.session.store.save(self.read(), overwrite=self.selected_id is not None)
            self.populate(data); self.refresh(data['id'])
            self.message.setText('已保存文件，主窗口未改变。')
        except (OSError, ValueError) as exc: self.error(exc)

    def save_as(self):
        name, accepted = QtWidgets.QInputDialog.getText(self, '另存为新预设', '新名称：', text=self.name.text()+' 副本')
        if not accepted: return
        try:
            old = self.read()
            data = create(self.mode, old['joints'], old['gesture'], name.strip(), old['notes'],
                          stop_signal=old['stop_signal'], return_settings=old['return_settings'],
                          start_settings=old['start_settings'])
            data = self.session.store.save(data)
            self.populate(data); self.refresh(data['id'])
            self.message.setText('已另存新文件，原预设和主窗口未改变。')
        except (OSError, ValueError) as exc: self.error(exc)

    def load(self):
        try:
            if self.dirty: raise ValueError('请先保存编辑，再加载；也可重新选择已有预设')
            if self.selected_id is None: raise ValueError('请先选择或保存预设')
            data = self.session.store.load(self.mode, self.selected_id)
            differences = self.session.differences(data)
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, '确认覆盖参数',
                '加载将恢复停止姿态类型及开关，但清除接管参考、安全确认和停止姿态参考确认。\n不会自动发送或启动。\n展开详细信息核对差异。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, self)
            box.setDefaultButton(QtWidgets.QMessageBox.Cancel); box.setDetailedText(differences)
            if box.exec() != QtWidgets.QMessageBox.Yes: return
            self.session.apply(data)
            self.accept()
        except (OSError, ValueError) as exc: self.error(exc)

    def delete(self):
        if self.selected_id is None: return
        if QtWidgets.QMessageBox.question(self, '删除预设？',
                '将当前预设移入本地 trash 目录，可从文件恢复。编辑区修改也将丢弃。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Cancel) != QtWidgets.QMessageBox.Yes: return
        try:
            destination = self.session.store.delete(self.mode, self.selected_id)
            self.dirty = False; self.refresh(); self.new()
            self.message.setText(f'已移入回收目录：{destination}')
        except (OSError, ValueError) as exc: self.error(exc)

    def error(self, exc):
        self.message.setText(str(exc))

    def reject(self):
        if self.discard(): super().reject()
