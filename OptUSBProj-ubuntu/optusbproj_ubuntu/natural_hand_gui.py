"""Left-palm manual neutral offset prototype, display only."""
from copy import deepcopy
import time
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtCore,QtWidgets
from .axis_mapping import mapping_matrix


class NaturalHandDialog(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host,QtCore.Qt.WindowType.Window)
        self.host=host;self.key='left_hand';self.candidate=None;self.stage=None;self.samples=[]
        self.previous=deepcopy(host.engine.natural_hand_references.get(self.key))
        self.setWindowTitle('左手自然掌面参考 · 手动原型（仅人体显示）');self.resize(700,640)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('穿戴标定应用后会独立保存标准朝向的B、N；仅手动映射需先采集标准基准。普通归零不覆盖它们。然后自然下垂（掌心向内偏后）、手腕伸直，采集自然参考并调节偏角。角度是人工估计，不是真值或视觉纠偏；X1不变。')
        note.setWordWrap(True);layout.addWidget(note)
        self.confirm=QtWidgets.QCheckBox('确认左手自然下垂、手腕伸直；设备和绑带未移动')
        layout.addWidget(self.confirm)
        self.standard_confirm=QtWidgets.QCheckBox('仅手动映射：采集标准基准时，确认手背向前、手指向下')
        layout.addWidget(self.standard_confirm)
        form=QtWidgets.QFormLayout();self.delay=QtWidgets.QSpinBox();self.delay.setRange(0,120);self.delay.setValue(5);self.delay.setSuffix(' 秒')
        self.hold=QtWidgets.QSpinBox();self.hold.setRange(2,10);self.hold.setValue(2);self.hold.setSuffix(' 秒')
        self.angle=QtWidgets.QDoubleSpinBox();self.angle.setRange(-180,180);self.angle.setSuffix(' °');self.angle.setSingleStep(1)
        if self.previous is not None:self.angle.setValue(self.previous.get('manual_offset_deg',0))
        form.addRow('开始前倒计时',self.delay);form.addRow('参考静止采样',self.hold)
        form.addRow('绕模型 +Z 长轴偏角（右手定则）',self.angle);layout.addLayout(form)
        self.capture=QtWidgets.QPushButton('倒计时采集自然参考');self.capture.clicked.connect(self.begin);layout.addWidget(self.capture)
        self.standard_capture=QtWidgets.QPushButton('采集标准朝向基准（仅手动映射；穿戴标定会自动保存）')
        self.standard_capture.clicked.connect(lambda:self.begin(standard=True));layout.addWidget(self.standard_capture)
        self.preview=QtWidgets.QPushButton('实时预览偏角（未应用）');self.preview.clicked.connect(self.show_preview);layout.addWidget(self.preview)
        self.angle.valueChanged.connect(self.update_angle)
        self.apply=QtWidgets.QPushButton('应用自然掌面参考（只影响人体显示）');self.apply.clicked.connect(self.apply_reference);layout.addWidget(self.apply)
        self.resume=QtWidgets.QPushButton('恢复已保存自然显示');self.resume.clicked.connect(self.resume_reference);layout.addWidget(self.resume)
        self.standard=QtWidgets.QPushButton('切回标准化显示 / 取消预览');self.standard.clicked.connect(self.standard_display);layout.addWidget(self.standard)
        self.undo=QtWidgets.QPushButton('撤销本窗口应用');self.undo.clicked.connect(self.undo_reference);layout.addWidget(self.undo)
        self.status=QtWidgets.QLabel('尚未采样。先点击确认，再开始倒计时。');self.status.setWordWrap(True);layout.addWidget(self.status)
        layout.addStretch()
        self.timer=QtCore.QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self.tick)
        self.preview.setEnabled(False);self.apply.setEnabled(False);self.undo.setEnabled(False)

    @property
    def active(self):return self.stage is not None

    def error(self):
        h=self.host;e=h.engine
        if h.replay_mode:return '回放中不可重新建立参考'
        if getattr(h,'session',None) is not None or getattr(h,'_record_start_deadline',None) is not None or getattr(h,'recorder',None) is not None:return '请先结束录制或预约'
        if getattr(h,'_timed_zero_deadline',None) is not None:return '请先取消定时归零'
        for name in ('calibration_dialog','camera_calibration_dialog','_combined_calibration_dialog','_reuse_calibration_dialog','_visual_wearing_dialog'):
            d=getattr(h,name,None)
            if d is not None and (getattr(d,'active',False) or getattr(d,'state',None)=='analyzing' or (name=='calibration_dialog' and d.isVisible())):return '请先完成或关闭其他标定窗口'
        device=e.assignments[self.key]
        if not device or not e.is_live(device,max_age=2):return '左手设备未分配或数据超过2秒未更新'
        if e.algorithm_modes[self.key] not in ('vendor_rpy','gyro_tilt'):return '左手请选择厂商RPY或陀螺＋加计'
        if e.current_for(device,e.algorithm_modes[self.key]) is None:return '左手当前算法尚无姿态数据'
        return None

    def dirty(self):self.host._render_dirty=True

    def begin(self,_checked=False,*,standard=False):
        error=self.error()
        if error:self.status.setText(error);return
        if standard and not self.standard_confirm.isChecked():self.status.setText('请确认标准朝向：手背向前、手指向下');return
        if not standard and not self.confirm.isChecked():self.status.setText('请先确认自然下垂和绑带状态');return
        e=self.host.engine;device=e.assignments[self.key];mode=e.algorithm_modes[self.key]
        self.capture_kind='standard' if standard else 'natural'
        try:
            if standard:
                if self.key in e.continuous_basis:raise ValueError('已有穿戴标定，请重新应用原标定结果；不能单独替换与B配对的N')
                b=mapping_matrix(e.axis_mappings[self.key]);r=e.current_for(device,mode)
                self.seed=dict(part=self.key,device_id=device,mode=mode,epoch=e.epochs[device],mapping_basis=b.tolist(),basis=b.tolist(),anchor=r.tolist())
            else:self.seed=e.make_natural_hand_reference(self.key,e.current_for(device,mode),np.eye(3))
        except ValueError as exc:self.status.setText(str(exc));return
        e.natural_hand_previews.pop(self.key,None);self.dirty()
        self.candidate=None;self.samples=[];self.last_stamp=None
        self.stage='delay';self.deadline=time.monotonic()+self.delay.value()
        for control in (self.capture,self.standard_capture,self.preview,self.apply,self.angle,self.resume,self.standard,self.undo):control.setEnabled(False)
        self.timer.start();self.tick()

    def tick(self):
        if not self.active:return
        error=self.error() or self.host.engine.natural_hand_error(self.seed)
        if error:self.finish_capture(error);return
        now=time.monotonic()
        if self.stage=='delay':
            pose='标准朝向：手背向前' if self.capture_kind=='standard' else '自然朝向：掌心向内偏后'
            self.status.setText(f'准备{pose}，剩余 {max(0,self.deadline-now):.1f} 秒；尚未采样。')
            if now>=self.deadline:self.stage='hold';self.sample_start=now;self.deadline=now+self.hold.value()
            return
        e=self.host.engine;device=self.seed['device_id'];stamp=e.received_at.get(device)
        if stamp is not None and stamp>=self.sample_start and 0<=now-stamp<=.5 and stamp!=self.last_stamp:
            self.samples.append(e.current_for(device,self.seed['mode']).copy());self.last_stamp=stamp
        self.status.setText(f'静止采样：剩余 {max(0,self.deadline-now):.1f} 秒，已抽样 {len(self.samples)} 帧。')
        if now<self.deadline:return
        if len(self.samples)<12:self.finish_capture('同期新姿态不足12帧，请检查数据后重采');return
        rotations=Rotation.from_matrix(self.samples);mean=rotations.mean()
        spread=float(np.percentile((mean.inv()*rotations).magnitude()*180/np.pi,90))
        if spread>3:self.finish_capture(f'参考姿态波动P90 {spread:.1f}°，超过3°；请支撑并保持自然下垂后重采');return
        if self.capture_kind=='standard':
            try:e.capture_manual_standard_hand(self.key,mean.as_matrix())
            except ValueError as exc:self.finish_capture(str(exc));return
            self.finish_capture('手动映射的标准朝向基准已独立保存，普通归零不会覆盖。下一步摆自然朝向，采集自然参考。');return
        self.candidate=deepcopy(self.seed)
        # Keep the original mapping frame used at countdown start.
        self.candidate['basis']=(np.asarray(self.seed['basis']) @ np.asarray(self.seed['anchor']).T @ mean.as_matrix()).tolist()
        self.candidate['anchor']=mean.as_matrix().tolist()
        self.candidate['spread_p90_deg']=spread;self.candidate['sample_count']=len(self.samples)
        self.update_angle();self.finish_capture(f'参考已采集，波动P90 {spread:.1f}°。调节偏角并点击预览，再决定是否应用。')

    def finish_capture(self,message):
        self.stage=None;self.timer.stop();self.status.setText(message)
        for control in (self.capture,self.standard_capture,self.angle,self.resume,self.standard):control.setEnabled(True)
        self.preview.setEnabled(self.candidate is not None);self.apply.setEnabled(self.candidate is not None)

    def update_angle(self):
        if self.candidate is None:return
        self.candidate['neutral_pose']=Rotation.from_euler('z',self.angle.value(),degrees=True).as_matrix().tolist()
        self.candidate['manual_offset_deg']=self.angle.value()
        if self.key in self.host.engine.natural_hand_previews:self.show_preview()

    def show_preview(self):
        if self.candidate is None:return
        error=self.error()
        if error:self.status.setText(error);return
        try:self.host.engine.set_natural_hand_reference(self.candidate,preview=True)
        except ValueError as exc:self.status.setText(str(exc));return
        self.dirty();self.status.setText('正在临时预览，改变角度可实时观察；未保存，关闭窗口会取消预览。请勿录制。')

    def apply_reference(self):
        if self.candidate is None:return
        error=self.error()
        if error:self.status.setText(error);return
        try:self.host.engine.set_natural_hand_reference(self.candidate)
        except ValueError as exc:self.status.setText(str(exc));return
        self.host.engine.natural_hand_previews.pop(self.key,None);self.undo.setEnabled(True);self.dirty()
        self.status.setText('已应用自然显示。普通归零会切回标准化显示，但保留此参考；可按恢复按钮再次显示。X1不变。')

    def resume_reference(self):
        error=self.error()
        if error:self.status.setText(error);return
        data=deepcopy(self.host.engine.natural_hand_references.get(self.key))
        if data is None:self.status.setText('没有已保存的自然参考，请重新采样');return
        data['enabled']=True
        try:self.host.engine.set_natural_hand_reference(data)
        except ValueError as exc:self.status.setText(str(exc));return
        self.host.engine.natural_hand_previews.pop(self.key,None);self.dirty();self.status.setText('已恢复原采样基准的自然显示，不使用最后一次普通归零基准。')

    def standard_display(self):
        self.host.engine.natural_hand_previews.pop(self.key,None)
        data=self.host.engine.natural_hand_references.get(self.key)
        if data:data['enabled']=False
        self.dirty();self.status.setText('已切回标准化显示；已保存自然参考未删除。')

    def undo_reference(self):
        error=self.error()
        if error:self.status.setText(error);return
        if self.previous is not None:
            try:self.host.engine.set_natural_hand_reference(self.previous)
            except ValueError as exc:self.status.setText('旧参考已失效，未恢复：'+str(exc));return
        else:self.host.engine.natural_hand_references.pop(self.key,None)
        self.host.engine.natural_hand_previews.pop(self.key,None);self.undo.setEnabled(False);self.dirty();self.status.setText('已恢复打开窗口前的显示参考。')

    def closeEvent(self,event):
        self.stage=None;self.timer.stop();self.host.engine.natural_hand_previews.pop(self.key,None);self.dirty();event.accept()
