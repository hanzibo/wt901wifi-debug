"""End-gesture and manual-return UI/session, isolated from the IK dialog."""
from dataclasses import replace
from PySide6 import QtWidgets
from .real_robot_protocol import decode, ARM_LIMITS
from .real_robot_bilateral import BilateralLimits
from .real_robot_stop_gesture import (GestureConfig, EndGesture, ALL_GESTURE_INDICES,
                                      GESTURE_KINDS, gesture_indices, axis_name)
from .real_robot_return import ReturnConfig, ORDER
from .real_robot_checkpoints import HOLD_RANGE


class EndSession:
    def __init__(self, owner):
        self.owner = owner
        self.returning = False
        self.await_home = False
        self.configured = False
        self.panel = QtWidgets.QGroupBox('结束手势与手动顺序回位（同源 IK；无实际角度反馈）')
        layout = QtWidgets.QVBoxLayout(self.panel)
        self.kind = QtWidgets.QComboBox()
        for key, label in GESTURE_KINDS.items(): self.kind.addItem(label+'停止', key)
        kind_form = QtWidgets.QFormLayout(); kind_form.addRow('停止姿态（二选一）', self.kind)
        layout.addLayout(kind_form)
        self.kind.currentIndexChanged.connect(self.kind_changed)
        self.enable = QtWidgets.QCheckBox('启用结束信号')
        self.enable.toggled.connect(self.changed)
        layout.addWidget(self.enable)
        self.seconds = self.spin(*HOLD_RANGE, 7.)
        self.seconds.valueChanged.connect(lambda *_: self.changed())
        duration_form = QtWidgets.QFormLayout()
        duration_form.addRow('结束信号持续时间（秒）', self.seconds)
        layout.addLayout(duration_form)
        grid = QtWidgets.QGridLayout()
        self.gesture_grid = grid
        self.rows = []
        self.gesture_rows = {}; self.row_labels = {}
        for column,text in enumerate(('协议轴','停止姿态参考角 °','容差 ±°')):
            grid.addWidget(QtWidgets.QLabel(text),0,column)
        for row, index in enumerate(ALL_GESTURE_INDICES, 1):
            name = axis_name(index)
            lo,hi = ARM_LIMITS[index]
            angle = self.spin(lo,hi,90.)
            tolerance = self.spin(.1,30.,5.)
            self.rows.append((angle,tolerance))
            self.gesture_rows[index] = (angle, tolerance)
            self.row_labels[index] = QtWidgets.QLabel(name)
            grid.addWidget(self.row_labels[index],row,0)
            grid.addWidget(angle,row,1);grid.addWidget(tolerance,row,2)
        layout.addLayout(grid)
        self.capture = QtWidgets.QPushButton('采集最后发送指令为停止姿态参考（须核对实际姿态）')
        self.validate = QtWidgets.QPushButton('确认手工填写的停止姿态参考角')
        self.capture.clicked.connect(lambda:self.configure(True))
        self.validate.clicked.connect(lambda:self.configure(False))
        layout.addWidget(self.capture);layout.addWidget(self.validate)
        self.gesture_label = QtWidgets.QLabel('停止姿态参考未确认；功能默认关闭。计时期间仍随动。')
        self.gesture_label.setWordWrap(True);layout.addWidget(self.gesture_label)
        self.side_order = QtWidgets.QComboBox()
        self.side_order.addItems(['先左臂，再右臂','先右臂，再左臂'])
        self.order = QtWidgets.QLineEdit(','.join(map(str,ORDER)))
        self.speed = self.spin(.1,250.,10.)
        self.acceleration = self.spin(.1,250.,20.)
        self.dwell = self.spin(0.,10.,.5)
        self.manual = QtWidgets.QCheckBox('逐步人工确认（默认开启；连续模式须先验收路径）')
        self.manual.setChecked(True)
        form = QtWidgets.QFormLayout()
        for label,widget in (('双臂先后',self.side_order),('每侧顺序（1–6 各一次）',self.order),
                             ('回位速度 °/s',self.speed),('回位加速度 °/s²',self.acceleration),
                             ('步间停留秒（不是到位反馈）',self.dwell)):
            form.addRow(label,widget)
        layout.addLayout(form);layout.addWidget(self.manual)
        self.return_button = QtWidgets.QPushButton('人工确认稳定与路径，启动顺序回位')
        self.next_button = QtWidgets.QPushButton('确认上一轴实际稳定，执行下一步')
        self.continuous_button = QtWidgets.QPushButton('一键连续顺序回位（一次确认，依次执行）')
        self.confirm_button = QtWidgets.QPushButton('确认真机实际已回上电姿态（不自动随动）')
        self.finger_return_button = QtWidgets.QPushButton('手指平滑回零（臂回位后，二次确认）')
        self.return_button.clicked.connect(lambda:self.start_return())
        self.continuous_button.clicked.connect(lambda:self.start_return(continuous=True))
        self.next_button.clicked.connect(self.next_step)
        self.confirm_button.clicked.connect(self.confirm_home)
        self.finger_return_button.clicked.connect(self.start_finger_return)
        for widget in (self.return_button,self.continuous_button,self.next_button,self.finger_return_button,self.confirm_button): layout.addWidget(widget)
        self.return_label = QtWidgets.QLabel('回位手动触发，不自动运行；顺序不保证无碰撞。')
        self.return_label.setWordWrap(True);layout.addWidget(self.return_label)
        self.configuration_widgets = [self.kind,self.enable,self.seconds,self.capture,self.validate,self.side_order,self.order,
                                      self.speed,self.acceleration,self.dwell,self.manual,
                                      *(w for row in self.rows for w in row)]

    def spin(self,lo,hi,value):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(lo,hi);widget.setValue(value)
        return widget

    def changed(self):
        self.owner._invalidate_capture()

    def kind_changed(self):
        self.configured = False
        self.reference = None
        self.gesture_label.setText('停止姿态已切换；请重新核对并确认参考角。')
        self.changed()
        self.owner._controls()

    def indices(self):
        return gesture_indices(self.kind.currentData(), bilateral=self.owner._bilateral_mode())

    def gesture_config(self):
        rows = [self.gesture_rows[i] for i in self.indices()]
        return GestureConfig(tuple(row[0].value() for row in rows),
                             tuple(row[1].value() for row in rows), seconds=self.seconds.value(),
                             indices=self.indices(), kind=self.kind.currentData())

    def checkpoint_options(self):
        try:
            order = [int(x.strip()) for x in self.order.text().split(',')]
        except ValueError:
            order = self.order.text()  # Keep invalid editor drafts correctable.
        return dict(stop_signal=dict(enabled=self.enable.isChecked(), seconds=self.seconds.value(),
                                     kind=self.kind.currentData()),
                    return_settings=dict(order=order,
                                         speed=self.speed.value(), acceleration=self.acceleration.value(),
                                         dwell=self.dwell.value()))

    def checkpoint_values(self):
        return {axis_name(i): dict(angle=self.gesture_rows[i][0].value(), tolerance=self.gesture_rows[i][1].value())
                for i in self.indices()}

    def checkpoint_updates(self, values, kind=None):
        updates = []
        for index in gesture_indices(self.kind.currentData() if kind is None else kind,
                                     bilateral=self.owner._bilateral_mode()):
            row = self.gesture_rows[index]; name = axis_name(index)
            updates.extend(((row[0], values[name]['angle']), (row[1], values[name]['tolerance'])))
        return updates

    def clear_checkpoint_confirmation(self):
        self.configured = False
        self.reference = None
        self.gesture_label.setText('已加载预设；停止开关已恢复，但须重新确认停止姿态参考才可启动检测。')

    def reset(self):
        self.returning = self.await_home = False
        self.return_label.setText('回位资格已清除；连接异常或关闭后须重新上电确认。')

    def configure(self, use_frame):
        d = self.owner
        if d.busy: return
        try:
            if use_frame:
                frame,_,_ = d.worker.return_snapshot() if d.worker is not None else (None,None,False)
                if frame is None: raise ValueError('本次连接尚无成功写出的指令；可手工填写已验证角度')
                q,_ = decode(frame)
                for i in self.indices(): self.gesture_rows[i][0].setValue(q[i])
            config = self.gesture_config()
            if QtWidgets.QMessageBox.warning(d,'停止姿态参考仅是指令角',
                    '确认填写的检测轴参考是已核对的实际结束姿态对应协议角？\n'
                    '不是实际角度反馈，也不是防碰撞或硬件急停。',
                    QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel,QtWidgets.QMessageBox.Cancel) != QtWidgets.QMessageBox.Yes:
                return
            self.configured = True
            self.reference = config
            self.gesture_label.setText(f'{GESTURE_KINDS[config.kind]}参考已确认；{config.seconds:g} 秒连续停留。计时期间仍随动。')
            self.changed()
        except Exception as exc: self.gesture_label.setText(str(exc))

    def gesture(self, limits):
        if not self.enable.isChecked(): return None
        if not self.configured:
            raise ValueError('须先确认所选停止姿态的参考角')
        config = self.gesture_config()
        if config != self.reference:
            raise ValueError('停止姿态参数已变化；须重新确认参考')
        local = [limits.indices.index(i) for i in config.indices]
        if not all(limits.enabled[i] for i in local):
            raise ValueError('结束检测轴必须同时获得输出许可')
        if any(not limits.lower[i] <= a <= limits.upper[i] for i,a in zip(local,config.angles)):
            raise ValueError('停止姿态参考角不在本次软件范围内')
        return EndGesture(config)

    def tick(self):
        if self.returning and self.owner.worker is not None:
            self.owner.worker.heartbeat()
            self.owner._controls()
            return True
        return False

    def controls(self):
        d = self.owner
        self.panel.setVisible(d._shared_mode())
        bilateral = d._bilateral_mode()
        selected = self.indices()
        self.enable.setText('启用'+GESTURE_KINDS[self.kind.currentData()]+'结束信号：'+
                            '/'.join(map(axis_name, selected))+f' 同时在容差内保持 {self.seconds.value():g} 秒')
        for i, row in self.gesture_rows.items():
            for widget in (self.row_labels[i], *row): widget.setVisible(i in selected)
        idle = not d.busy and d.ik_worker is None
        for widget in self.configuration_widgets: widget.setEnabled(idle and not self.await_home)
        for i, row in self.gesture_rows.items():
            for widget in row: widget.setEnabled(idle and not self.await_home and i in selected)
        self.side_order.setEnabled(bilateral and idle and not self.await_home)
        frame,state,pending = d.worker.return_snapshot() if d.worker is not None and hasattr(d.worker,'return_snapshot') else (None,None,False)
        ready = d.connected and d.worker is not None and d._shared_mode()
        self.return_button.setEnabled(ready and idle and state is not None and state.get('mode')=='paused' and not pending)
        self.continuous_button.setEnabled(self.return_button.isEnabled())
        self.next_button.setEnabled(ready and self.returning and state is not None and state.get('phase')=='await_step')
        self.confirm_button.setEnabled(ready and idle and self.await_home and pending)
        nonzero_fingers = frame is not None and any(decode(frame)[1])
        self.finger_return_button.setEnabled(ready and idle and self.await_home and pending and bool(nonzero_fingers))
        self.confirm_button.setText('臂已回位但手指非零：请先执行手指平滑回零' if self.await_home and nonzero_fingers
                                    else '确认真机实际已回上电姿态（不自动随动）')
        if nonzero_fingers: self.confirm_button.setEnabled(False)
        if self.await_home:
            d.capture_button.setEnabled(False);d.start_button.setEnabled(False)

    def start_return(self, *, continuous=False):
        d = self.owner
        if d.busy or d.ik_worker is not None or d.worker is None or not d._shared_mode(): return
        try:
            sides = (('left','right') if self.side_order.currentIndex()==0 else ('right','left')) if d._bilateral_mode() else ('left',)
            config = ReturnConfig(sides=sides,
                                  order=tuple(int(x.strip()) for x in self.order.text().split(',')),
                                  dwell=self.dwell.value(),manual=False if continuous else self.manual.isChecked())
            source = d._limits()
            def returning_limits(arm):
                return replace(arm,speed=(self.speed.value(),)*6,acceleration=(self.acceleration.value(),)*6)
            limits = (BilateralLimits(*(returning_limits(getattr(source,s)) for s in ('right','left')))
                      if d._bilateral_mode() else returning_limits(source))
            if QtWidgets.QMessageBox.warning(d,'手动顺序回位，不是防碰撞',
                    '确认真实机器人已停止且稳定？确认逐轴路径无障碍并有独立停止手段？\n'
                    f'每侧顺序 {config.order}；仅原有许可轴，回到协议 90°。\n'
                    + ('本次为连续执行：确认后按顺序自动执行所有轴，不再逐步询问。\n' if not config.manual else '本次为逐步人工确认。\n')+
                    '连续执行模式没有真实到位反馈。取消不会发送。',
                    QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel,QtWidgets.QMessageBox.Cancel) != QtWidgets.QMessageBox.Yes:
                return
            if d.busy or d.ik_worker is not None or d.worker is None:
                raise ValueError('确认期间控制状态已变化')
            d.arm_capture = None
            d.worker.begin_return(limits,config)
            d.arm_configuration = limits
            self.returning = True
            self.await_home = False
            d.busy = True
            self.return_label.setText('回位已提交；首帧保持起点。人体与 IMU 不控制回位。')
        except Exception as exc: self.return_label.setText('回位未启动：'+str(exc))
        d._controls()

    def start_finger_return(self):
        d=self.owner
        if d.busy or d.ik_worker is not None or d.worker is None or not self.await_home:return
        try:
            frame,state,pending=d.worker.return_snapshot()
            if not pending or frame is None or state is None or state.get('phase')!='complete':
                raise ValueError('须先完成手臂顺序回位')
            if QtWidgets.QMessageBox.warning(d,'二次确认：手指回协议 0，不是位置反馈',
                    '确认双臂已实际稳定回位，手中无物体，手指张开路径无障碍且有独立断电手段？\n'
                    '仅将本次曾明确授权且非零的手指平滑回到协议 0；双臂、头腰保持不动。\n'
                    '使用各指最后授权的速度/加速度及范围，不依赖 IMU。完成后仍须人工确认真实到位。',
                    QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel,QtWidgets.QMessageBox.Cancel)!=QtWidgets.QMessageBox.Yes:
                return
            if d.busy or d.ik_worker is not None or d.worker is None:raise ValueError('确认期间状态已改变')
            d.worker.begin_finger_return()
            self.returning=True;self.await_home=False;d.busy=True
            self.return_label.setText('手指平滑回零已提交；首帧保持起点，不自动恢复随动。')
        except Exception as exc:self.return_label.setText('手指回零未启动：'+str(exc))
        d._controls()

    def next_step(self):
        d = self.owner
        if not self.returning or d.worker is None: return
        if QtWidgets.QMessageBox.warning(d,'人工确认下一步',
                '确认上一轴实际已稳定，并确认下一轴运动路径安全？',
                QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel,QtWidgets.QMessageBox.Cancel) != QtWidgets.QMessageBox.Yes:
            return
        try: d.worker.advance_return()
        except Exception as exc: self.return_label.setText(str(exc))

    def confirm_home(self):
        d = self.owner
        if not self.await_home or d.worker is None: return
        if QtWidgets.QMessageBox.warning(d,'人工确认，不是反馈',
                '确认真机实际已稳定在全部 90° / 手指 0 的上电姿态？\n'
                '确认后仍须重新采集目标与明确启动，不自动随动。',
                QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel,QtWidgets.QMessageBox.Cancel) != QtWidgets.QMessageBox.Yes:
            return
        try:
            d.origin = d.last_frame = d.worker.confirm_home()
            d.arm_capture = None;d.arm_history.clear();d.arm_sample = None
            self.await_home = False
            self.return_label.setText('已人工确认回位；重新勾选安全许可、锁定目标、明确启动。')
        except Exception as exc: self.return_label.setText(str(exc))
        d._controls()

    def progress(self, state):
        if 'gesture_elapsed' in state:
            label = GESTURE_KINDS[state.get('gesture_kind', self.kind.currentData())]
            self.gesture_label.setText(f'{label}保持 {state["gesture_elapsed"]:.1f} / {state.get("gesture_seconds", self.seconds.value()):g} 秒；'+
                                      ('已触发暂停，等待制动结束' if state.get('gesture_triggered') else '计时期间仍随动'))
        if state.get('task') != 'return': return
        axis = state.get('return_axis') or '无剩余轴'
        self.return_label.setText(f'顺序回位：{state.get("phase")}；下一/当前轴 {axis}；'
                                 f'步骤 {min(state["return_step"],state["return_count"])} / {state["return_count"]}\n'+state.get('reason',''))
        if state.get('return_kind')=='fingers':
            self.return_label.setText('手指平滑回零：'+state['phase']+'；'+state.get('reason','')+'\n'+
                '\n'.join(f'{s}: {m["command"]} → {m["target"]}' for s,m in state['finger_return'].items()))
        if state['mode'] == 'paused':
            self.returning = False
            self.await_home = state.get('phase') == 'complete'
