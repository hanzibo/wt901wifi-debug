"""Explicit hand permissions and parameter-only presets; never opens a UART."""
import time
import math
from PySide6 import QtCore, QtWidgets
from .real_robot_hand import HandConfig, observation_valid
from .real_robot_hand_profiles import ROOT, save_profile, read_profile


class HandSession:
    def __init__(self, owner):
        self.owner=owner; self.observations={}; self.rows={}; self.labels={}; self.starts={}; self.stops={}
        self.residual_checks={}; self.errors={}
        self.panel=QtWidgets.QGroupBox('真机五指 · 双 IMU 离散开合（单独授权，默认关闭）')
        outer=QtWidgets.QVBoxLayout(self.panel)
        self.open_button=QtWidgets.QPushButton('五指端点 / Checkpoint / 输出许可…')
        self.open_button.clicked.connect(self.show); outer.addWidget(self.open_button)
        self.summary=QtWidgets.QLabel('手指未授权；先进入同源跟随，再在配置窗口逐指启用。')
        self.summary.setWordWrap(True); outer.addWidget(self.summary)
        self.dialog=QtWidgets.QDialog(owner); self.dialog.setWindowTitle('真机五指端点与离散开合'); self.dialog.resize(1050,650)
        layout=QtWidgets.QVBoxLayout(self.dialog)
        note=QtWidgets.QLabel('0–4095 是协议量，不是角度或已验证机械范围。首帧保持最后写出量。\n'
            '同一只手五指共用开合状态，但每指端点和许可独立。停止不是物理急停；不要夹持物体或靠近人员。\n'
            '先完成臂回位，再明确执行手指平滑回零，最后人工确认全机上电构型。无实际位置反馈。')
        note.setWordWrap(True); layout.addWidget(note)
        inputs=QtWidgets.QPushButton('打开双 IMU 开手 / 闭合参考与阈值配置…')
        inputs.clicked.connect(lambda:owner.viewer._show_gripper_gesture()); layout.addWidget(inputs)
        tabs=QtWidgets.QTabWidget(); layout.addWidget(tabs)
        for side,title in (('left','左手'),('right','右手')):
            page=QtWidgets.QWidget(); box=QtWidgets.QVBoxLayout(page); tabs.addTab(page,title)
            table=QtWidgets.QGridLayout(); box.addLayout(table)
            for col,text in enumerate(('许可','张开量','闭合量','最小量','最大量','速度 量/s','加速度 量/s²')):
                table.addWidget(QtWidgets.QLabel(text),0,col)
            self.rows[side]=[]
            for i in range(5):
                allowed=QtWidgets.QCheckBox(f'协议指{i+1}')
                row=[allowed]
                for value,cap in ((0,4095),(200,4095),(0,4095),(200,4095),(200,4095),(400,16380)):
                    spin=QtWidgets.QSpinBox(); spin.setRange(0 if len(row)<5 else 1,cap); spin.setValue(value); row.append(spin)
                self.rows[side].append(row)
                for col,w in enumerate(row): table.addWidget(w,i+1,col)
            label=QtWidgets.QLabel('尚无有效手势'); label.setWordWrap(True); self.labels[side]=label; box.addWidget(label)
            bypass=QtWidgets.QCheckBox('真机：允许离轴残差超过 20°仍应用开合（实验，默认关闭）')
            bypass.setToolTip('仅放宽本手离轴残差门限，可能误判开合；不绕过断流、数据新鲜度或设备参考检查。切换后须重新启用。')
            self.residual_checks[side]=bypass
            bypass.toggled.connect(lambda _,s=side:self.policy_changed(s));box.addWidget(bypass)
            start=QtWidgets.QPushButton('明确启用'+title+'勾选手指（当前开合预设）…')
            stop=QtWidgets.QPushButton('减速停用'+title+'手指（保持末次量）')
            start.clicked.connect(lambda _,s=side:self.enable(s)); stop.clicked.connect(lambda _,s=side:self.stop(s))
            box.addWidget(start); box.addWidget(stop); self.starts[side]=start; self.stops[side]=stop
        buttons=QtWidgets.QHBoxLayout(); layout.addLayout(buttons)
        for text,fn in (('保存新手型 Checkpoint',self.save),('加载手型 Checkpoint（清除许可）',self.load)):
            b=QtWidgets.QPushButton(text); b.clicked.connect(fn); buttons.addWidget(b)
        self.timer=QtCore.QTimer(self.dialog); self.timer.timeout.connect(self.controls); self.timer.start(100)
        self.controls()

    def show(self):
        self.controls(); self.dialog.show(); self.dialog.raise_()

    def config(self,side):
        rows=self.rows[side]
        return HandConfig(**{k:tuple(row[col].value() for row in rows) for col,k in enumerate(
            ('opened','closed','lower','upper','speed','acceleration'),1)},enabled=tuple(r[0].isChecked() for r in rows))

    def states(self):
        worker=self.owner.worker
        state=worker.hand_snapshot() if worker is not None and hasattr(worker,'hand_snapshot') else {}
        return state if isinstance(state,dict) else {}

    def receive(self,side,data):
        if bool(data.get('residual_bypass',False))!=self.residual_checks[side].isChecked():
            data={**data,'valid':False}
        self.observations[side]=dict(data)
        if self.owner.worker is not None: self.owner.worker.observe_hand(side,data)

    def sync_policy(self,side):
        gesture=getattr(self.owner.viewer,'gripper_gesture_dialog',None)
        if gesture is not None:
            gesture.panels[side].set_real_residual_policy(self.residual_checks[side].isChecked())

    def policy_changed(self,side):
        if self.owner.worker is not None:self.owner.worker.stop_hand(side)
        self.observations.pop(side,None);self.errors.pop(side,None)
        self.sync_policy(side);self.controls()

    def blockers(self,side):
        d=self.owner;data=self.observations.get(side,{});reasons=[]
        if not d._shared_mode():reasons.append('请选择仿真同源 IK 模式')
        if side=='right' and not d._bilateral_mode():reasons.append('左臂模式不许可右手')
        if d.worker is None:reasons.append('串口未连接')
        if not d.safety.isChecked():reasons.append('尚未确认安全许可')
        if not d.busy or d.shared_phase!='following':reasons.append('手臂尚未进入同源跟随（接管到位后仍须确认开始跟随）')
        elif d.shared_control_mode!='active':reasons.append('手臂已暂停或正在减速/等待输入')
        if not observation_valid(data,time.monotonic(),allow_residual=self.residual_checks[side].isChecked()):
            residual=data.get('residual_deg');stamp=data.get('stamp')
            if not data:reasons.append('尚未收到双 IMU 手势')
            elif isinstance(residual,(int,float)) and residual>20. and not self.residual_checks[side].isChecked():
                reasons.append('离轴残差超过 20°；严格模式拒绝')
            elif type(data.get('state')) is not bool:reasons.append('开合尚未判定；请采集参考并越过阈值保持')
            elif type(stamp) not in (int,float) or not -.1<=time.monotonic()-stamp<=.25:reasons.append('手势数据不新鲜（超过 250 ms 或时间异常）')
            else:reasons.append('手势无效：检查采样、设备分配、参考及残差策略是否同步')
        return reasons

    def ready(self,side):
        return not self.blockers(side)

    def enable(self,side):
        try:
            if not self.ready(side): raise ValueError('须先进入健康的同源跟随，并采集新鲜双 IMU 手势')
            config=self.config(side)
            if not any(config.enabled): raise ValueError('请先明确勾选待输出手指')
            signature=self.observations[side]['signature']
            bypass=self.residual_checks[side].isChecked()
            lines='\n'.join(f'协议指{i+1}: {config.opened[i]} ↔ {config.closed[i]}，'
                            f'范围 {config.lower[i]}–{config.upper[i]}，速度 {config.speed[i]}，加速度 {config.acceleration[i]}'
                            for i,e in enumerate(config.enabled) if e)
            if QtWidgets.QMessageBox.warning(self.dialog,'二次确认：允许本次手指运动',
                    ('左手' if side=='left' else '右手')+'：确认指序、端点、空间与独立断电手段？\n'+lines+
                    ('\n注意：本手已允许残差超过 20°，可能误判开合。' if bypass else '\n本手使用 20° 严格门限。')+
                    '\n从最后已写出量平滑接管；不测量位置或接触力。',
                    QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.Cancel,QtWidgets.QMessageBox.Cancel)!=QtWidgets.QMessageBox.Yes:
                return
            if (not self.ready(side) or self.observations[side]['signature']!=signature or self.config(side)!=config
                    or self.residual_checks[side].isChecked()!=bypass):
                raise ValueError('确认期间输入或配置改变，未启用')
            self.owner.worker.enable_hand(side,config,allow_residual=bypass)
            self.errors.pop(side,None)
        except (ValueError,RuntimeError) as exc:self.errors[side]='未启用：'+str(exc)
        self.controls()

    def stop(self,side):
        if self.owner.worker is not None: self.owner.worker.stop_hand(side)
        self.controls()

    def controls(self):
        if self.owner.closing:
            self.dialog.hide()
        states=self.states(); summary=[]
        for side,rows in self.rows.items():
            state=states.get(side,{})
            running=state.get('mode') in ('active','gap','braking')
            for row in rows:
                for w in row: w.setEnabled(not running)
            reasons=self.blockers(side)
            if not any(row[0].isChecked() for row in rows):reasons.append('尚未勾选待输出手指')
            self.starts[side].setEnabled(not running and not reasons)
            self.starts[side].setToolTip('；'.join(reasons) if reasons else '等待明确启用与二次确认')
            self.stops[side].setEnabled(running)
            data=self.observations.get(side,{})
            valid=observation_valid(data,time.monotonic(),allow_residual=self.residual_checks[side].isChecked())
            residual=data.get('residual_deg');stamp=data.get('stamp')
            residual_text=f'{residual:.1f}°' if type(residual) in (int,float) and math.isfinite(residual) else '未知'
            age_text=f'{(time.monotonic()-stamp)*1000:.0f} ms' if type(stamp) in (int,float) and stamp>0 else '未知'
            pose='闭合' if data.get('state') is True else '张开' if data.get('state') is False else '待判定'
            text=f'{"左手" if side=="left" else "右手"}手势：{pose}（{"有效" if valid else "无效/未就绪"}）；离轴残差 {residual_text}；数据年龄 {age_text}'
            text+='\n残差策略：'+('实验放宽 >20°（仍检查断流与参考）' if self.residual_checks[side].isChecked() else '严格 ≤20°')
            if state:
                text+=f'\n输出：{state["mode"]}；指令 {state["command"]} → {state["target"]}；'+state['reason']
            else:text+='\n输出：未授权'
            if not running:text+='\n'+('暂不可启用：'+'；'.join(reasons) if reasons else '输入及启动条件已满足，等待明确启用')
            if side in self.errors:text+='\n'+self.errors[side]
            self.labels[side].setText(text);summary.append(text)
        if summary: self.summary.setText('\n'.join(summary))

    def save(self):
        try:
            configs={s:self.config(s) for s in self.rows}
            name,ok=QtWidgets.QInputDialog.getText(self.dialog,'保存新手型 Checkpoint','名称')
            if ok:
                path=save_profile(name,configs); self.summary.setText('已保存参数，不含授权：'+str(path))
        except (ValueError,OSError) as exc: QtWidgets.QMessageBox.warning(self.dialog,'保存失败',str(exc))

    def apply(self,path):
        if any(s.get('mode')!='paused' for s in self.states().values()):
            raise ValueError('请先停用手指并等待指令侧减速结束')
        name,configs=read_profile(path)  # Validate all sides before changing any widget.
        for side,c in configs.items():
            for i,row in enumerate(self.rows[side]):
                row[0].setChecked(False)
                for k,w in zip(('opened','closed','lower','upper','speed','acceleration'),row[1:]):
                    w.setValue(getattr(c,k)[i])
        self.observations.clear()
        gesture=getattr(self.owner.viewer,'gripper_gesture_dialog',None)
        if gesture is not None:
            for panel in gesture.panels.values(): panel.reset_discrete()
        self.summary.setText('已加载 '+name+'；许可关闭。请重新确认端点与手势，不自动输出。')
        self.controls()

    def load(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self.dialog,'加载真机五指 Checkpoint',str(ROOT),'JSON (*.json)')
        if not path:return
        try:self.apply(path)
        except (ValueError,OSError,TypeError) as exc:QtWidgets.QMessageBox.warning(self.dialog,'加载失败',str(exc))
