"""Guided RGB-D/IMU prefix calibration and validation capture; no device writes."""
import json
from pathlib import Path
import sys
import time
import uuid

from PySide6 import QtCore, QtWidgets


def calibration_steps(preparation, hold, neutral, validation):
    """Label actual phases; individual analyzers choose their training windows."""
    return [
        ('neutral','中立位','站直、面向相机，右臂自然下垂；肩髋完整入镜。',preparation,neutral),
        ('forward','右臂前平举','肘伸直，右臂向正前方抬到水平；躯干不动。',preparation,hold),
        ('down','放下右臂','右臂缓慢放下，回到自然下垂。',preparation,hold),
        ('lateral','右臂侧平举','肘伸直，右臂向右侧抬到水平；不要耸肩或侧倾。',preparation,hold),
        ('forward_repeat','再次前平举','肘伸直，再次前平举到水平。',preparation,hold),
        ('lateral_repeat','再次侧平举','肘伸直，再次侧平举到水平。',preparation,hold),
        ('neutral_return','回到中立位','右臂放下，站直静止。',preparation,neutral),
        ('validation','独立验证','重复前举、侧举、屈肘、轻微转身；水平/斜向姿态停留2–3秒。',0,validation),
        ('tail','结束保护','动作已完成，请保持静止，等待自动结束。末尾3秒会剔除。',0,4),
    ]


class CameraCalibrationDialog(QtWidgets.QDialog):
    def __init__(self, owner):
        super().__init__(owner,QtCore.Qt.WindowType.Window)
        self.owner=owner
        self.setWindowTitle('相机标定 · 右臂联合录制引导')
        self.resize(760,650)
        self.state='idle';self.recording_path=None;self.analysis_path=None
        self.steps=[];self.events=[];self.index=0;self.deadline=0.;self.started_ns=None
        self.devices={};self.process=None;self.log=''
        self.gravity_reference={};self.reference_epochs={}
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('相机移动或设备重新穿戴后重新执行。当前仅标定右上臂＋右前臂到相机的骨段方向；无需先标定STL轴映射。固定相机和绑带，不发送设备清零指令，不覆盖人体/机器人配置。')
        note.setWordWrap(True);layout.addWidget(note)
        form=QtWidgets.QFormLayout();self.spins=[]
        self.delay=self.spin(form,'开始前倒计时',0,120,10)
        self.preparation=self.spin(form,'每个动作准备时间',2,30,4)
        self.hold=self.spin(form,'每个动作静止保持',3,20,4)
        self.neutral=self.spin(form,'中立位静止保持',5,30,8)
        self.validation=self.spin(form,'独立验证录制时长',10,600,90)
        layout.addLayout(form)
        self.auto_analyze=QtWidgets.QCheckBox('完成后自动计算离线标定与融合对比（不应用到实时模型）')
        self.auto_analyze.setChecked(True);layout.addWidget(self.auto_analyze)
        self.plan=QtWidgets.QLabel();self.plan.setWordWrap(True);layout.addWidget(self.plan)
        for spin in self.spins:spin.valueChanged.connect(self.update_plan)
        self.step_label=QtWidgets.QLabel('等待开始');self.step_label.setWordWrap(True)
        self.step_label.setStyleSheet('font-size:28px; font-weight:bold; color:#329fcc;')
        layout.addWidget(self.step_label)
        self.instruction=QtWidgets.QLabel('先启动 Gemini 相机和三维观测，确认右上臂、右前臂设备在线。')
        self.instruction.setWordWrap(True);self.instruction.setStyleSheet('font-size:20px;');layout.addWidget(self.instruction)
        self.progress=QtWidgets.QProgressBar();layout.addWidget(self.progress)
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        controls=QtWidgets.QHBoxLayout()
        self.start=QtWidgets.QPushButton('开始引导标定');self.start.clicked.connect(self.begin);controls.addWidget(self.start)
        self.cancel=QtWidgets.QPushButton('取消本轮');self.cancel.setEnabled(False);self.cancel.clicked.connect(self.abort);controls.addWidget(self.cancel)
        layout.addLayout(controls)
        self.analyze=QtWidgets.QPushButton('计算本次标定');self.analyze.setEnabled(False);self.analyze.clicked.connect(self.analyze_recording);layout.addWidget(self.analyze)
        self.replay=QtWidgets.QPushButton('打开三路离线对比');self.replay.setEnabled(False);self.replay.clicked.connect(self.open_replay);layout.addWidget(self.replay)
        self.apply=QtWidgets.QPushButton('应用相机标定到本次实时会话');self.apply.setEnabled(False);self.apply.clicked.connect(self.apply_live);layout.addWidget(self.apply)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self.tick)
        self.update_plan()

    @property
    def active(self):
        return self.state in ('delay','prepare','hold')

    def spin(self,form,label,minimum,maximum,value):
        spin=QtWidgets.QSpinBox();spin.setRange(minimum,maximum);spin.setValue(value);spin.setSuffix(' 秒')
        form.addRow(label,spin);self.spins.append(spin);return spin

    def update_plan(self):
        steps=calibration_steps(self.preparation.value(),self.hold.value(),self.neutral.value(),self.validation.value())
        total=sum(s[3]+s[4] for s in steps)
        self.plan.setText(f'流程：中立 → 前举 → 放下 → 侧举 → 再前举 → 再侧举 → 中立 → 独立验证 → 结束保护。录制约 {total} 秒，另有开始前 {self.delay.value()} 秒倒计时。')

    def readiness_error(self):
        o=self.owner
        other=getattr(o,'camera_calibration_dialog',None)
        if other is not None and other is not self and (other.active or other.state=='analyzing'):
            return '另一个标定流程正在执行，请先完成或取消。'
        if o.replay_mode or o.camera_source!='orbbec':return '仅支持实时 Gemini RGB-D 模式。'
        if o.session is not None or o._record_start_deadline is not None:return '已有录制/预约，请先结束或取消。'
        if o.last_rgbd_observation is None or time.monotonic_ns()-o.last_rgbd_observation.captured_mono_ns>1e9:return '相机尚未就绪或帧已过期。'
        if not o.rgbd_pose_enabled.isChecked() or o.last_pose_estimate is None or time.monotonic_ns()-o.last_pose_estimate.captured_mono_ns>1e9:return '请开启三维观测并等待模型就绪。'
        for part in ('right_upper_arm','right_forearm'):
            device=o.engine.assignments.get(part)
            if not device or not o.engine.is_live(device,max_age=1):return f'{part} 未分配在线设备；请检查电源、热点和部位分配。'
        return None

    def set_controls(self,running):
        for spin in self.spins:spin.setEnabled(not running)
        self.auto_analyze.setEnabled(not running)
        self.start.setEnabled(not running);self.cancel.setEnabled(running)
        for button in (self.owner.session_button,self.owner.timed_session_button,self.owner.cancel_session_button):
            button.setEnabled(not running and not self.owner.replay_mode)
        if not running:self.owner.cancel_session_button.setEnabled(False)

    def begin(self):
        error=self.readiness_error()
        if error:self.status.setText(error);return
        self.devices={k:self.owner.engine.assignments[k] for k in ('right_upper_arm','right_forearm')}
        self.reference_epochs={d:getattr(self.owner.engine,'epochs',{}).get(d) for d in self.devices.values()}
        self.gravity_reference={}
        self.steps=calibration_steps(self.preparation.value(),self.hold.value(),self.neutral.value(),self.validation.value())
        self.events=[];self.recording_path=None;self.analysis_path=None;self.started_ns=None;self.index=0
        self.progress.setRange(0,len(self.steps));self.progress.setValue(0)
        self.analyze.setEnabled(False);self.replay.setEnabled(False)
        self.apply.setEnabled(False)
        self.state='delay';self.deadline=time.monotonic()+self.delay.value()
        self.set_controls(True);self.timer.start();self.tick()

    def monitor_error(self):
        for part,device in self.devices.items():
            if self.owner.engine.assignments.get(part)!=device:return '设备分配发生变化，本轮无效。'
            if getattr(self.owner.engine,'epochs',{}).get(device)!=self.reference_epochs.get(device):return f'{device} 积分参考系发生变化，请重录。'
            if not self.owner.engine.is_live(device,max_age=2):return f'{device} 超过2秒没有数据，本轮已停止。'
        if self.owner.last_rgbd_observation is None or time.monotonic_ns()-self.owner.last_rgbd_observation.captured_mono_ns>2e9:return 'RGB-D 数据中断，本轮已停止。'
        if not self.owner.rgbd_pose_enabled.isChecked():return '三维观测已关闭，本轮已停止。'
        if self.owner.last_pose_estimate is None or time.monotonic_ns()-self.owner.last_pose_estimate.captured_mono_ns>3e9:return '三维推理结果超过3秒未更新，本轮已停止。'
        return None

    def tick(self):
        if not self.active:return
        error=self.monitor_error()
        if error:self.abort(error);return
        now=time.monotonic()
        if self.state=='delay':
            self.step_label.setText(f'开始倒计时：{max(0,self.deadline-now):.1f} 秒')
            self.instruction.setText('离开电脑，进入画面，站直；准备中立姿态。此阶段还未录制。')
            if now<self.deadline:return
            if not self.owner._start_session():self.abort('无法创建联合录制');return
            self.started_ns=self.owner.session.started_mono_ns;self.recording_path=self.owner.session.path
            self.enter_phase('prepare' if self.steps[0][3] else 'hold');return
        if self.owner.session is None or self.owner.session.path!=self.recording_path:
            self.abort('录制已被外部结束；本轮不执行标定。');return
        key,title,instruction,prep,hold=self.steps[self.index]
        remaining=max(0,self.deadline-now)
        phase_text='准备' if self.state=='prepare' else ('执行' if key in ('validation','tail') else '静止采样')
        self.step_label.setText(f'{self.index+1}/{len(self.steps)} · {title} · '+phase_text+f' {remaining:.1f} 秒')
        self.instruction.setText(instruction+(' 请调整姿态，尚未进入静止采样。' if self.state=='prepare' else (' 请保持静止。' if key not in ('validation','tail') else '')))
        elapsed=(time.monotonic_ns()-self.started_ns)/1e9
        self.status.setText(f'已录制 {elapsed:.1f} 秒；目录 {self.recording_path}')
        self.progress.setRange(0,len(self.steps));self.progress.setValue(self.index)
        if now<self.deadline:return
        self.events[-1]['end_s']=elapsed
        if not self.save_guide('running'):return
        if self.state=='prepare':self.enter_phase('hold');return
        if key=='neutral':
            neutral=self.events[-1]
            start_ns=self.started_ns+round((neutral['start_s']+.5)*1e9)
            end_ns=self.started_ns+round((neutral['end_s']-.2)*1e9)
            try:self.gravity_reference=self.owner.prepare_camera_gravity(list(self.devices.values()),start_ns,end_ns)
            except ValueError as exc:self.abort('重力初始化未通过：'+str(exc));return
            if not self.save_guide('running'):return
        self.index+=1
        if self.index==len(self.steps):self.finish();return
        self.enter_phase('prepare' if self.steps[self.index][3] else 'hold')

    def enter_phase(self,phase):
        self.state=phase
        key,title,instruction,prep,hold=self.steps[self.index]
        stamp=time.monotonic_ns()
        self.deadline=stamp/1e9+(prep if phase=='prepare' else hold)
        self.events.append({'step':key,'phase':phase,'start_s':(stamp-self.started_ns)/1e9,'end_s':None})
        if not self.save_guide('running'):return
        phase_text='准备' if phase=='prepare' else ('执行' if key in ('validation','tail') else '静止采样')
        self.step_label.setText(f'{self.index+1}/{len(self.steps)} · {title} · '+phase_text)
        self.instruction.setText(instruction)
        QtWidgets.QApplication.beep()

    def save_guide(self,status,reason=None):
        if self.recording_path is None:return True
        payload={'format':'rgbd-guided-camera-calibration-v1','status':status,'reason':reason,
                 'devices':self.devices,'events':self.events,'recording_started_mono_ns':self.started_ns,
                 'gravity_reference':self.gravity_reference,
                 'settings':{'delay_s':self.delay.value(),'preparation_s':self.preparation.value(),'hold_s':self.hold.value(),'neutral_s':self.neutral.value(),'validation_s':self.validation.value()},
                 'note':'actual host monotonic phase times; prepare phases not static samples; no live mapping writes'}
        try:
            (self.recording_path/'camera_guide.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        except OSError as exc:
            if self.active:self.abort('无法保存动作时间窗：'+str(exc))
            else:self.status.setText('无法保存动作时间窗：'+str(exc))
            return False
        return True

    def abort(self,reason='用户取消本轮'):
        if isinstance(reason,bool):reason='用户取消本轮'
        if not self.active:return
        self.timer.stop();self.state='cancelled'
        if self.started_ns is not None and self.events and self.events[-1]['end_s'] is None:self.events[-1]['end_s']=(time.monotonic_ns()-self.started_ns)/1e9
        if self.owner.session is not None and self.recording_path==self.owner.session.path:self.owner._stop_session()
        self.save_guide('cancelled',reason);self.set_controls(False)
        self.step_label.setText('本轮已停止');self.status.setText(str(reason)+('；已有原始数据保留。' if self.recording_path else '；未创建录制。'))

    def finish(self):
        self.timer.stop();self.state='complete';self.owner._stop_session(auto=True)
        self.set_controls(False);self.progress.setValue(len(self.steps))
        if not self.save_guide('complete'):return
        self.step_label.setText('录制完成');self.status.setText(f'已保存 {self.recording_path}；末尾3秒已截断。')
        try:manifest=json.loads((self.recording_path/'manifest.json').read_text())
        except (OSError,ValueError) as exc:
            self.state='failed';self.status.setText('无法检查录制完整性：'+str(exc));return
        if manifest.get('error'):
            self.state='failed';self.status.setText('录制不完整：'+str(manifest['error']));return
        self.analyze.setEnabled(True)
        if self.auto_analyze.isChecked():self.analyze_recording()

    def analysis_windows(self):
        holds={e['step']:e for e in self.events if e['phase']=='hold' and e['end_s'] is not None}
        neutral=holds['neutral'];end=holds['neutral_return']
        # Exclude the first 0.5s settling and last 0.2s of the static neutral.
        return (neutral['start_s']+.5,neutral['end_s']-.2),(neutral['start_s'],end['end_s'])

    def analyze_recording(self):
        if self.state!='complete':return
        neutral,train=self.analysis_windows()
        self.analysis_path=self.recording_path.parent/'analysis_rgbd'/(self.recording_path.name+'-guided-'+uuid.uuid4().hex[:6])
        script=Path(__file__).resolve().parent.parent/'rgbd_fuse_offline.py'
        if self.process is not None:self.process.deleteLater()
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.analysis_finished)
        self.process.errorOccurred.connect(self.analysis_error)
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-u',str(script),str(self.recording_path),'--output',str(self.analysis_path),'--neutral',f'{neutral[0]:.6f}:{neutral[1]:.6f}','--train',f'{train[0]:.6f}:{train[1]:.6f}'])
        if (self.recording_path/'runtime_pose.jsonl').is_file() and (self.recording_path/'runtime_pose.jsonl').stat().st_size:
            self.process.setArguments(self.process.arguments()+['--live-state'])
        self.state='analyzing';self.start.setEnabled(False);self.analyze.setEnabled(False);self.log=''
        self.status.setText('正在按实际时间窗计算离线标定；不会改动实时映射。');self.process.start()

    def read_log(self):
        self.log=(self.log+bytes(self.process.readAllStandardOutput()).decode('utf-8','replace'))[-12000:]

    def analysis_error(self,error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:
            self.state='complete';self.start.setEnabled(True);self.analyze.setEnabled(True);self.status.setText('分析进程启动失败：'+self.process.errorString())

    def analysis_finished(self,code,_status):
        self.read_log();self.state='complete';self.start.setEnabled(True);self.analyze.setEnabled(True)
        report_path=self.analysis_path/'report.json'
        if code!=0 or not report_path.is_file():self.status.setText('离线分析失败，原始录制保留。\n'+self.log[-1500:]);return
        report=json.loads(report_path.read_text());parts=report['parts']
        self.apply.setEnabled(report.get('orientation_source')=='live_runtime_pose' and any(p.get('accepted') for p in parts.values()))
        messages=[f'{part}：'+('标定通过' if p.get('accepted') else '需重录 / '+p.get('reason','方向拟合未通过')) for part,p in parts.items()]
        self.status.setText('；'.join(messages)+f'\n分析目录：{self.analysis_path}\n通过仅代表方向标定可用，并不保证融合优于纯IMU。')
        self.replay.setEnabled(True)

    def apply_live(self):
        try:report=json.loads((self.analysis_path/'report.json').read_text())
        except (OSError,ValueError) as exc:
            QtWidgets.QMessageBox.warning(self,'标定未应用','无法读取标定结果：'+str(exc));return
        answer=QtWidgets.QMessageBox.question(self,'确认应用','请先停止机器人随动并保持安全姿态。应用只绑定本次参考系，不修改轴映射，也不自动切换算法。之后可先在人体模型中选择右臂“陀螺＋加计＋RGB-D”验证。是否应用？')
        if answer!=QtWidgets.QMessageBox.StandardButton.Yes:return
        try:self.owner.apply_camera_report(report)
        except ValueError as exc:
            self.status.setText('未应用：'+str(exc))
            QtWidgets.QMessageBox.warning(self,'标定未应用',str(exc)+'\n没有启用融合模式。')
            return
        self.status.setText('已应用到本次会话。请先在人体现有算法下确认中立位，再选择右上臂/前臂RGB-D融合模式；应用不会自动启用机器人。')
        self.apply.setEnabled(False)
        QtWidgets.QMessageBox.information(self,'标定已应用','已应用到本次实时参考系。现在可以选择已通过部位的RGB-D融合模式；未自动启动机器人。')

    def open_replay(self):
        script=Path(__file__).resolve().parent.parent/'rgbd_fusion_viewer.py'
        ok,_pid=QtCore.QProcess.startDetached(sys.executable,[str(script),str(self.analysis_path)])
        if not ok:self.status.setText('无法启动对比窗口')

    def shutdown(self):
        self.abort('主窗口关闭，本轮停止')
        if self.process is not None and self.process.state()!=QtCore.QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1000):self.process.kill();self.process.waitForFinished(1000)
        self.close()

    def closeEvent(self,event):
        if self.active:
            answer=QtWidgets.QMessageBox.question(self,'标定正在录制','关闭将取消本轮并保留已录制数据，是否关闭？',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No)
            if answer!=QtWidgets.QMessageBox.StandardButton.Yes:event.ignore();return
            self.abort()
        super().closeEvent(event)
