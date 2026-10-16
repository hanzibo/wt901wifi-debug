"""CoppeliaSim preview UI and thin Qt worker adapter."""
import time
import json
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets
from .x1_live import MAX_JOINT_SPEED_DEG_S
from .scene_environment import DEFAULT_INSTALL,DEFAULT_SCENE,EnvironmentAction,launch_environment,port_in_use
from .scene_hand import HandMailbox
from .scene_robot_runtime import SceneRuntime
# Compatibility exports for existing callers and tests.
from .scene_robot_state import (
    Mailbox, dispatch_reason, joint_step, joint_steps, cycle_wait_ms,
    StaleImuInput, temporary_imu_gap, pose_sample,
)


class SceneWorker(QtCore.QThread):
    report=QtCore.Signal(object)
    failed=QtCore.Signal(str)
    def __init__(self,port,mailbox,speed,angle,logging=False,parent=None,*,wrist_speed=None,hand_mailbox=None):
        super().__init__(parent)
        self.port,self.mailbox,self.speed,self.angle=port,mailbox,speed,angle
        self.logging=logging
        self.wrist_speed=speed if wrist_speed is None else wrist_speed
        self.hand_mailbox=hand_mailbox or HandMailbox()
    def run(self):
        runtime = SceneRuntime(
            self.port, self.mailbox, self.speed, self.angle, self.logging,
            self.wrist_speed, self.hand_mailbox, self.report.emit, self.failed.emit,
            self.isInterruptionRequested, self.msleep)
        runtime.run()


class SceneRobotDialog(QtWidgets.QDialog):
    def __init__(self,viewer):
        super().__init__(viewer)
        self.viewer=viewer;self.worker=None;self.mailbox=Mailbox();self.signature=None
        self.connected=False
        self.config=None
        self.hand_mailbox=HandMailbox();self.hand_dialog=None;self.hand_angles={};self.hand_readback_at=0.
        self.environment_action=None;self.environment_process=None
        self.setWindowTitle('新机器人 · CoppeliaSim 双臂 / 手指实时预览')
        self.resize(750,600)
        layout=QtWidgets.QVBoxLayout(self)
        info=QtWidgets.QLabel('复用当前人体姿态及穿戴标定。仅停止状态运动学预览，不控制真机。\n'
            '请先打开已确认中立位的新场景并停止仿真；勿同时使用关节 demo。\n'
            '零角为屈肘姿态；参考中立位两侧关节4为 −90°。手指可独立配置并启用离散开合。')
        info.setWordWrap(True);layout.addWidget(info)
        self.install_path=QtWidgets.QLineEdit(str(DEFAULT_INSTALL))
        self.scene_path=QtWidgets.QLineEdit(str(DEFAULT_SCENE))
        for label,field,is_dir in (('安装目录',self.install_path,True),('启动场景',self.scene_path,False)):
            row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel(label));row.addWidget(field)
            choose=QtWidgets.QPushButton('选择…')
            choose.clicked.connect(lambda checked=False,f=field,d=is_dir:self.choose_environment_path(f,d))
            row.addWidget(choose);layout.addLayout(row)
        row=QtWidgets.QHBoxLayout()
        self.open_environment_button=QtWidgets.QPushButton('打开 CoppeliaSim 环境')
        self.close_environment_button=QtWidgets.QPushButton('关闭 CoppeliaSim 环境')
        row.addWidget(self.open_environment_button);row.addWidget(self.close_environment_button);layout.addLayout(row)
        self.environment_status=QtWidgets.QLabel('环境：可复用手动打开的实例；关闭前请保存场景。')
        self.environment_status.setWordWrap(True);layout.addWidget(self.environment_status)
        self.environment_gpu=QtWidgets.QCheckBox('新启动环境使用 NVIDIA GPU 渲染（默认开启；已有实例需重启才生效）')
        self.environment_gpu.setChecked(True);layout.addWidget(self.environment_gpu)
        self.open_environment_button.clicked.connect(self.open_environment)
        self.close_environment_button.clicked.connect(self.close_environment)
        form=QtWidgets.QFormLayout();layout.addLayout(form)
        self.port=QtWidgets.QSpinBox();self.port.setRange(1024,65535);self.port.setValue(23000)
        self.speed=QtWidgets.QSpinBox();self.speed.setRange(5,120);self.speed.setValue(int(MAX_JOINT_SPEED_DEG_S));self.speed.setSuffix(' °/s')
        self.wrist_speed=QtWidgets.QSpinBox();self.wrist_speed.setRange(5,720);self.wrist_speed.setValue(240);self.wrist_speed.setSuffix(' °/s')
        self.wrist_speed.setToolTip('左右手关节 5、6 共用；连接前设置。肩肘 1–4 限速不变。仅仿真预览，不是真机安全参数。')
        self.angle=QtWidgets.QSpinBox();self.angle.setRange(100,170);self.angle.setValue(150);self.angle.setSuffix(' °')
        form.addRow('本机 ZeroMQ 端口',self.port)
        form.addRow('肩肘关节 1–4 限速（左右共用）',self.speed)
        form.addRow('腕部关节 5–6 限速（左右共用）',self.wrist_speed)
        form.addRow('预览区间 ±（非机械安全限位）',self.angle)
        self.extra_limits=QtWidgets.QCheckBox('额外角度区间限制（默认关闭；始终遵守场景关节区间）')
        form.addRow(self.extra_limits)
        self.angle.setEnabled(False);self.extra_limits.toggled.connect(self.angle.setEnabled)
        self.task=QtWidgets.QComboBox()
        self.task.addItem('肘腕位置＋可选手掌朝向（X1 默认方案）','positions')
        self.task.addItem('骨段向量＋可选手掌朝向（对照方案）','vectors')
        self.scale=QtWidgets.QComboBox()
        self.scale.addItem('统一比例（X1 默认）','uniform')
        self.scale.addItem('上臂／前臂分别缩放','segments')
        form.addRow('任务目标',self.task);form.addRow('位置映射长度比例',self.scale)
        self.solver_mode=QtWidgets.QComboBox()
        self.solver_mode.addItem('分层：臂段 1–4 → 手腕 5–6（默认）','staged')
        self.solver_mode.addItem('六关节联合优化（原方案，对照）','joint')
        form.addRow('关节任务分配',self.solver_mode)
        self.segment_orientation=QtWidgets.QCheckBox('臂段朝向软约束（上臂＋前臂，无需手背设备）')
        self.segment_orientation.setChecked(True)
        self.segment_orientation.setToolTip('使用穿戴标定后的臂段旋转，减少伸直位轴向歧义；不是碰撞保护，轴向漂移仍可能影响结果。')
        self.segment_weight=QtWidgets.QDoubleSpinBox()
        self.segment_weight.setRange(0.,1.);self.segment_weight.setDecimals(3)
        self.segment_weight.setSingleStep(.025);self.segment_weight.setValue(.15)
        form.addRow(self.segment_orientation);form.addRow('臂段朝向残差权重（0 等于关闭）',self.segment_weight)
        self.left=QtWidgets.QCheckBox('左臂');self.right=QtWidgets.QCheckBox('右臂')
        self.left.setChecked(True);self.right.setChecked(True)
        self.lhand=QtWidgets.QCheckBox('左手掌朝向');self.rhand=QtWidgets.QCheckBox('右手掌朝向')
        self.manual=QtWidgets.QCheckBox('允许手动轴映射（实验；不要求穿戴标定）')
        row=QtWidgets.QHBoxLayout()
        for w in (self.left,self.right,self.lhand,self.rhand): row.addWidget(w)
        layout.addLayout(row);layout.addWidget(self.manual)
        self.palm_offset_controls={}
        for side,label in (('left','左'),('right','右')):
            enabled=QtWidgets.QCheckBox(label+'手掌目标固定转向（实验）')
            axis=QtWidgets.QComboBox();axis.addItems(['x','y','z'])
            angle=QtWidgets.QDoubleSpinBox();angle.setRange(-180,180);angle.setValue(90);angle.setSuffix(' °')
            plus=QtWidgets.QPushButton('+90°');minus=QtWidgets.QPushButton('−90°')
            plus.clicked.connect(lambda checked=False,w=angle:w.setValue(90))
            minus.clicked.connect(lambda checked=False,w=angle:w.setValue(-90))
            row=QtWidgets.QHBoxLayout()
            for w in (enabled,axis,angle,plus,minus):row.addWidget(w)
            layout.addLayout(row)
            self.palm_offset_controls[side]=(enabled,axis,angle,plus,minus)
        warning=QtWidgets.QLabel('固定转向仅在该侧“手掌朝向”开启时生效；轴为机器人中立手掌局部轴，不是人体轴。会改变中立掌面，无碰撞保护。停止跟随后调整。')
        warning.setWordWrap(True);layout.addWidget(warning)
        self.logging=QtWidgets.QCheckBox('记录角度、腕部目标速度 / 5、6 角速度 / 朝向误差及时序（连接前勾选）')
        layout.addWidget(self.logging)
        self.connect_button=QtWidgets.QPushButton('连接 / 重新读取场景')
        self.hand_config_button=QtWidgets.QPushButton('左右手指姿态配置…（可独立于双臂运行）')
        self.hand_config_button.clicked.connect(self.show_hand_config);layout.addWidget(self.hand_config_button)
        self.hand_status=QtWidgets.QLabel('手指：尚未连接；输出默认关闭');layout.addWidget(self.hand_status)
        hand_stop=QtWidgets.QPushButton('停止左右手指输出（保持当前位置）')
        hand_stop.clicked.connect(self.stop_hands);layout.addWidget(hand_stop)
        self.start=QtWidgets.QPushButton('启动所选手臂跟随');self.start.setEnabled(False)
        self.stop=QtWidgets.QPushButton('停止双臂发送 · 保持角度（手指独立）')
        self.export=QtWidgets.QPushButton('导出本次运动学配置与参考（诊断 JSON）')
        self.export.setEnabled(False)
        for w in (self.connect_button,self.start,self.stop,self.export):layout.addWidget(w)
        self.status=QtWidgets.QLabel('未连接。启动后可能逐步移向人体姿态；本版无碰撞保护。')
        self.status.setWordWrap(True);layout.addWidget(self.status)
        self.diagnostic=QtWidgets.QLabel('时序诊断：尚未启动');self.diagnostic.setWordWrap(True);layout.addWidget(self.diagnostic)
        self.values=QtWidgets.QPlainTextEdit();self.values.setReadOnly(True);layout.addWidget(self.values)
        self.connect_button.clicked.connect(self.connect_scene)
        self.start.clicked.connect(self.begin)
        self.stop.clicked.connect(self.halt)
        self.export.clicked.connect(self.export_config)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(50);self.timer.timeout.connect(self.sample)
        self.ingress_file=None
        self.ingress_timer=QtCore.QTimer(self);self.ingress_timer.setInterval(200)
        self.ingress_timer.timeout.connect(self.flush_ingress)
        self.input_gap=False

    def choose_environment_path(self,field,is_dir):
        if is_dir:path=QtWidgets.QFileDialog.getExistingDirectory(self,'选择 CoppeliaSim 安装目录',field.text())
        else:path,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择启动场景',field.text(),'CoppeliaSim (*.ttt)')
        if path:field.setText(path)

    def environment_busy(self,busy):
        for w in (self.open_environment_button,self.close_environment_button,self.install_path,self.scene_path,self.environment_gpu):
            w.setEnabled(not busy)
        if busy:
            self._environment_port_enabled=self.port.isEnabled();self.port.setEnabled(False)
        else:self.port.setEnabled(self._environment_port_enabled)

    def open_environment(self):
        if self.environment_action is not None:return
        if self.environment_process is not None and self.environment_process.poll() is None:
            self.environment_status.setText('本窗口启动的环境仍在运行／加载中；不会重复打开。');return
        self.environment_busy(True)
        self.environment_status.setText('检查所选端口是否已有环境…')
        action=EnvironmentAction(self.port.value(),parent=self);self.environment_action=action
        action.result.connect(self.environment_open_result)
        action.finished.connect(self.environment_action_finished);action.start()

    def environment_open_result(self,available,message):
        if available:self.environment_status.setText(message);return
        if port_in_use(self.port.value()):
            self.environment_status.setText('所选端口已占用但 API 未响应；请处理环境弹窗或等待，未重复启动。');return
        try:
            self.environment_process=launch_environment(self.install_path.text(),self.scene_path.text(),self.port.value(),
                                                        gpu=self.environment_gpu.isChecked())
            self.environment_status.setText('已启动进程'+('（已设置 NVIDIA GPU 渲染）' if self.environment_gpu.isChecked() else '')+
                '，正在加载场景；等待窗口加载完成后点击“连接 / 重新读取场景”。')
        except Exception as exc:self.environment_status.setText('启动失败：'+str(exc))

    def environment_action_finished(self):
        self.environment_action.deleteLater();self.environment_action=None;self.environment_busy(False)

    def close_environment(self):
        if self.environment_action is not None:return
        answer=QtWidgets.QMessageBox.question(self,'关闭 CoppeliaSim',
            '将停止跟随，并请求所选本机端口 %d 的 CoppeliaSim 正常退出。\n'
            '请先保存需要保留的场景；本程序不会自动保存，也不会强制结束进程。\n是否继续？'%self.port.value())
        if answer!=QtWidgets.QMessageBox.Yes:return
        self.halt(reason='用户关闭 CoppeliaSim 环境')
        if not self.shutdown():return
        self.connected=False;self.start.setEnabled(False);self.connect_button.setEnabled(True)
        for w in (self.port,self.speed,self.wrist_speed,self.logging,self.extra_limits):w.setEnabled(True)
        self.angle.setEnabled(self.extra_limits.isChecked())
        self.environment_busy(True)
        action=EnvironmentAction(self.port.value(),closing=True,parent=self);self.environment_action=action
        action.result.connect(lambda ok,message:self.environment_status.setText(message))
        action.finished.connect(self.environment_action_finished);action.start()

    def controls(self,running):
        for widgets in self.palm_offset_controls.values():
            for w in widgets:w.setEnabled(not running)
        for w in (self.left,self.right,self.lhand,self.rhand,self.manual,self.task,self.scale,self.solver_mode,
                  self.segment_orientation,self.segment_weight):w.setEnabled(not running)
        self.start.setEnabled(self.connected and not running)

    def show_hand_config(self):
        if self.hand_dialog is None:
            from .scene_hand_gui import HandPoseDialog
            self.hand_dialog=HandPoseDialog(self)
            if self.connected:self.hand_dialog.bind_scene()
        self.hand_dialog.show();self.hand_dialog.raise_()

    def revoke_hand_input(self,side):
        gesture=getattr(self.viewer,'gripper_gesture_dialog',None)
        if gesture is not None:gesture.panels[side].send_scene.setChecked(False)

    def stop_hands(self):
        self.hand_mailbox.stop()
        if self.hand_dialog is not None:self.hand_dialog.stop_all()
        for side in ('left','right'):self.revoke_hand_input(side)
        self.hand_status.setText('手指输出已停止；保持当前位置')

    def receive_hand(self,side,data):
        gesture=getattr(self.viewer,'gripper_gesture_dialog',None)
        panel=None if gesture is None else gesture.panels[side]
        if not data.get('enabled'):
            requests,_=self.hand_mailbox.snapshot()
            if requests.get(side,{}).get('source')=='gesture':self.hand_mailbox.stop(side)
            return
        try:
            if not self.connected:raise ValueError('请先连接新机器人场景，再重新启用手指输出')
            if self.hand_dialog is None:raise ValueError('请先打开手指配置并确认两组端点')
            self.hand_dialog.endpoint(side,'open')  # Validate both endpoints even before a state is available.
            if not data.get('valid') or data.get('state') not in (False,True):
                self.hand_mailbox.stop(side)
                if panel:panel.scene_status.setText('暂停手指：手势数据无效或状态待判定，保持当前位置')
                return
            key='closed' if data['state'] else 'open'
            q=self.hand_dialog.endpoint(side,key)
            self.hand_dialog.preview.pop(side,None)
            self.hand_mailbox.set(side,q,data['stamp'],self.hand_dialog.widgets[side][1].value(),'gesture')
            if panel:panel.scene_status.setText('手势目标：'+('闭合' if data['state'] else '张开')+'；等待场景回读')
        except (ValueError,KeyError) as exc:
            self.hand_mailbox.stop(side)
            if panel:
                panel.send_scene.setChecked(False);panel.scene_status.setText(str(exc))

    def connect_scene(self):
        physical = getattr(self.viewer, 'real_robot_dialog', None)
        if physical is not None and (getattr(physical, 'mirror_worker', None) is not None
                or (getattr(physical, 'busy', False) and physical._shared_mode())):
            self.status.setText('真机同源控制占用场景；请先停止同源控制并关闭镜像。')
            return
        self.stop_hands()
        self.halt()
        if not self.shutdown(): return
        self.connected=False;self.start.setEnabled(False);self.connect_button.setEnabled(False)
        for w in (self.port,self.speed,self.wrist_speed,self.angle,self.logging,self.extra_limits):w.setEnabled(False)
        self.status.setText('正在读取场景关节链；不会改变关节角…')
        self.worker=SceneWorker(self.port.value(),self.mailbox,self.speed.value(),
            self.angle.value() if self.extra_limits.isChecked() else None,self.logging.isChecked(),self,
            wrist_speed=self.wrist_speed.value(),hand_mailbox=self.hand_mailbox)
        worker=self.worker
        self.worker.report.connect(lambda data: self.report(data) if self.worker is worker else None)
        self.worker.failed.connect(lambda text: self.failure(text) if self.worker is worker else None)
        self.worker.start()

    def flush_ingress(self):
        if self.ingress_file is None:return
        try:
            for row in self.viewer.source.take_diagnostics():
                self.ingress_file.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.ingress_file.flush()
        except OSError as exc:
            self.status.setText('接收诊断日志写入失败：'+str(exc))
            self.close_ingress()

    def close_ingress(self):
        self.ingress_timer.stop()
        if self.ingress_file is not None:
            self.ingress_file.close();self.ingress_file=None
        source=getattr(self.viewer,'source',None)
        if hasattr(source,'enable_diagnostics'):source.enable_diagnostics(False)

    def selected(self):
        sides=[s for s,w in (('left',self.left),('right',self.right)) if w.isChecked()]
        hands=[s for s,w in (('left',self.lhand),('right',self.rhand)) if w.isChecked() and s in sides]
        if not sides:raise ValueError('至少选择一侧手臂')
        return sides,hands

    def palm_offsets(self):
        return {s:{'enabled':widgets[0].isChecked(),'axis':widgets[1].currentText(),'degrees':widgets[2].value()}
                for s,widgets in getattr(self,'palm_offset_controls',{}).items()}

    def begin(self):
        try:
            physical = getattr(self.viewer, 'real_robot_dialog', None)
            if physical is not None and (getattr(physical, 'mirror_worker', None) is not None
                    or (getattr(physical, 'busy', False) and physical._shared_mode())):
                raise ValueError('真机同源控制占用仿真；请先停止同源控制并关闭镜像')
            sides,hands=self.selected()
            data,self.signature=pose_sample(self.viewer.engine,sides,hands,self.manual.isChecked())
            data.update(task=self.task.currentData(),scale=self.scale.currentData(),
                        solver_mode=self.solver_mode.currentData(),
                        palm_offsets=self.palm_offsets(),
                        segment_orientation=self.segment_orientation.isChecked() and self.segment_weight.value()>0,
                        segment_weight=self.segment_weight.value())
            self.mailbox.resume(data);self.controls(True);self.timer.start()
            self.last_sample_at=data['at']
            self.input_gap=False
            self.status.setText('跟随中；IMU 超过 250 ms 保持，1 s 内恢复可自动继续；持续断流或标定变化需重新启动')
        except Exception as exc:self.status.setText(str(exc))

    def sample(self):
        try:
            sides,hands=self.selected()
            data,signature=pose_sample(self.viewer.engine,sides,hands,self.manual.isChecked())
            if signature!=self.signature:raise ValueError('设备分配、姿态模式或标定已改变，请重新启动')
            data['gui_interval_ms']=(data['at']-self.last_sample_at)*1000
            self.last_sample_at=data['at']
            data.update(task=self.task.currentData(),scale=self.scale.currentData(),
                        solver_mode=self.solver_mode.currentData() if hasattr(self,'solver_mode') else 'staged',
                        palm_offsets=SceneRobotDialog.palm_offsets(self),
                        segment_orientation=self.segment_orientation.isChecked() and self.segment_weight.value()>0,
                        segment_weight=self.segment_weight.value())
            self.mailbox.set(True,data)
            if self.input_gap:
                self.input_gap=False
                self.status.setText('IMU 数据恢复：已核对原标定/模式，按原限速继续跟随，不回放积压目标')
                source=getattr(self.viewer,'source',None)
                if hasattr(source,'diagnostic_event'):source.diagnostic_event('follow_input_resumed')
        except Exception as exc:
            if temporary_imu_gap(exc):
                self.mailbox.set(True,None)
                self.status.setText('短暂 IMU 断流：保持当前角度，等待自动恢复；'+str(exc))
                if not self.input_gap:
                    source=getattr(self.viewer,'source',None)
                    if hasattr(source,'diagnostic_event'):source.diagnostic_event('follow_input_waiting',reason=str(exc))
                self.input_gap=True
                return
            source=getattr(self.viewer,'source',None)
            if hasattr(source,'diagnostic_event'):
                source.diagnostic_event('follow_input_rejected',reason=str(exc),
                    device_ages_ms={d:(time.monotonic()-at)*1000 for d,at in self.viewer.engine.received_at.items()})
            self.halt(reason='输入检查失败：'+str(exc));self.status.setText('已停止：'+str(exc))

    def halt(self,checked=False,reason='用户/窗口主动停止'):
        self.input_gap=False
        self.timer.stop();self.mailbox.set(False,reason=reason);self.controls(False)
        self.status.setText('已停止发送；保持当前角度。已在途的单次通信不能撤回。')

    @QtCore.Slot(object)
    def report(self,data):
        if 'hand_angles' in data:
            self.hand_angles={s:np.array(q) for s,q in data['hand_angles'].items()}
            self.hand_readback_at=data['hand_readback_mono_s']
            active=data.get('hand_requests',{})
            self.hand_status.setText('手指场景回读正常；当前输出：'+('、'.join(('左' if s=='left' else '右')+'手 '+r['source'] for s,r in active.items()) or '未启用'))
            gesture=getattr(self.viewer,'gripper_gesture_dialog',None)
            if gesture is not None:
                for s,r in active.items():
                    if r['source']=='gesture':gesture.panels[s].scene_status.setText('CoppeliaSim 已回读该侧十五关节；按配置端点过渡中')
        if 'diagnostic' in data:self.diagnostic.setText('时序诊断：'+data['diagnostic'])
        if data.get('recovering'):
            self.stop_hands()
            self.connected=False;self.halt(reason='通信超时，正在恢复连接')
            self.connect_button.setEnabled(False)
            self.status.setText('通信超时：暂停发送，正在重建连接并核对场景；不会盲目重发角度。')
        if data.get('recovered'):
            self.connected=True;self.controls(False);self.connect_button.setEnabled(True)
            self.status.setText('通信已恢复并读回实际角度；请点击启动继续跟随。原标定与运动学参考保留。')
        if data.get('connected'):
            self.connected=True;self.connect_button.setEnabled(True);self.controls(False)
            self.config=data['config'];self.export.setEnabled(True)
            if self.hand_dialog is not None:self.hand_dialog.bind_scene()
            self.status.setText('已连接；轴变换与机器人参考已读取，尚未发送角度。')
            if data.get('log'):
                self.status.setText(self.status.text()+'\n日志：'+data['log'])
                source=getattr(self.viewer,'source',None)
                if hasattr(source,'enable_diagnostics'):
                    path=Path(data['log']).with_suffix('.ingress.jsonl')
                    try:
                        self.ingress_file=path.open('w',encoding='utf-8')
                        source.enable_diagnostics(True)
                        source.diagnostic_event('ingress_begin',note='GUI recvfrom 时间，不是内核到达时间；未记录原始数据负载')
                        self.ingress_timer.start()
                        self.status.setText(self.status.text()+'\n收包诊断：'+str(path))
                    except OSError as exc:self.status.setText('接收诊断日志创建失败：'+str(exc))
        if 'angles' in data:
            lines=[]
            for side,q in data['angles'].items():
                lines.append(('左' if side=='left' else '右')+'臂实际角度 °：'+', '.join(f'{v:.1f}' for v in np.rad2deg(q)))
                wrist=data.get('wrist_diagnostics',{}).get(side)
                if wrist is not None:
                    fmt=lambda value:'—' if value is None else f'{value:.1f}'
                    lines.append('腕部诊断：目标速度 '+fmt(wrist['target_speed_deg_s'])+' °/s；J5 '+
                        fmt(wrist['joint5_velocity_deg_s'])+' °/s；J6 '+fmt(wrist['joint6_velocity_deg_s'])+' °/s')
                    if wrist['target_relative_forearm_speed_deg_s'] is not None:
                        lines.append('相对目标前臂的掌面速度：'+fmt(wrist['target_relative_forearm_speed_deg_s'])+' °/s')
                    lines.append('腕部单步限速触边 J5/J6：'+str(wrist['wrist_step_limit_active'])+
                                 '；场景角度限位触边：'+str(wrist['wrist_joint_limit_active']))
                if side in data.get('errors',{}):
                    err,orient=data['errors'][side]
                    label='肘腕位置' if data.get('task')=='positions' else '骨段端点'
                    lines.append(f'{label}误差：{err*1000:.1f} mm；掌面误差：'+('未启用' if orient is None else f'{orient:.1f}°'))
                    if data.get('solver_mode')=='staged':
                        lines.append('分层求解：手掌任务不驱动关节 1–4'+
                            ('；腕部目标未完全达到（不借肩部代偿）' if orient is not None and orient>5 else ''))
                    segment_error=data.get('segment_errors',{}).get(side,{})
                    if segment_error:
                        lines.append('臂段朝向误差：上臂 %.1f°／前臂 %.1f°'%(
                            segment_error['upper_arm'],segment_error['forearm']))
                    else:lines.append('臂段朝向约束：未启用')
            self.values.setPlainText('\n'.join(lines))
        if 'warning' in data and not self.mailbox.blocked:self.status.setText(data['warning'])

    def failure(self,text):
        self.stop_hands()
        self.connected=False;self.halt(reason='连接/求解失败：'+text);self.status.setText('连接/求解停止：'+text)
        self.connect_button.setEnabled(True)
        for w in (self.port,self.speed,self.wrist_speed,self.logging,self.extra_limits):w.setEnabled(True)
        self.angle.setEnabled(self.extra_limits.isChecked())

    def export_config(self):
        if self.config is None:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'保存运动学诊断配置','scene_robot_config.json','JSON (*.json)')
        if path:
            try:
                Path(path).write_text(json.dumps(self.config,ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError as exc:self.status.setText('保存失败：'+str(exc))

    def shutdown(self):
        self.stop_hands()
        if self.hand_dialog is not None:self.hand_dialog.hide()
        if self.worker is not None:
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self.status.setText('通信仍在退出，请稍后再试');return False
            self.worker=None
        self.flush_ingress();self.close_ingress()
        return True

    def closeEvent(self,event):
        if self.environment_action is not None:
            self.environment_status.setText('环境操作正在进行，请稍后关闭配置窗口。');event.ignore();return
        self.halt()
        if self.shutdown():
            self.connected=False;self.start.setEnabled(False);self.connect_button.setEnabled(True)
            for w in (self.port,self.speed,self.wrist_speed,self.logging,self.extra_limits):w.setEnabled(True)
            self.angle.setEnabled(self.extra_limits.isChecked())
            event.accept()
        else:event.ignore()
