"""Guided left-palm C0 observation/comparison. Explicit apply, never correction."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6 import QtCore,QtGui,QtWidgets
from .palm_worker import PalmWorker,DEFAULT_HAND_MODEL
from .palm_geometry import estimate_c0,distance,angle
from .visual_wearing import body_frame
from .natural_hand import proper

REPORT_ROOT=Path(__file__).resolve().parents[1]/'calibrations'/'palm_visual'


class PalmVisualDialog(QtWidgets.QDialog):
    def __init__(self,host):
        super().__init__(host,QtCore.Qt.WindowType.Window)
        self.host=host;self.worker=None;self.phase='idle';self.report=None;self.seed=None;self.K=None
        self.body_rows=[];self.imu_rows=[];self.palm_rows=[];self.fit_rows=[];self.check_rows=[];self.windows=[];self.seen=set();self.latest=None
        self.diagnostic_images=[];self.diagnostic_counts={}
        self.setWindowTitle('左手视觉自然掌面 C0 · 手动对比（实验）');self.resize(760,820)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('先应用左手标准穿戴标定（手动映射先采标准基准），左手选择陀螺＋加计。相机和绑带固定；站直、朝向与标准标定一致。首版仅观测表面掌面，不是真值，不做长期纠偏，不改变X1；视觉结果不自动应用。')
        note.setWordWrap(True);layout.addWidget(note)
        self.confirm=QtWidgets.QCheckBox('确认标准基准有效，身体朝向一致；采样时手掌展开、手腕伸直')
        layout.addWidget(self.confirm)
        form=QtWidgets.QFormLayout()
        self.delay=QtWidgets.QSpinBox();self.delay.setRange(2,60);self.delay.setValue(5);self.delay.setSuffix(' 秒')
        self.prepare=QtWidgets.QSpinBox();self.prepare.setRange(3,30);self.prepare.setValue(5);self.prepare.setSuffix(' 秒')
        self.hold=QtWidgets.QSpinBox();self.hold.setRange(4,20);self.hold.setValue(5);self.hold.setSuffix(' 秒')
        self.manual=QtWidgets.QDoubleSpinBox();self.manual.setRange(-180,180);self.manual.setSuffix(' °')
        saved=host.engine.natural_hand_references.get('left_hand',{})
        self.manual.setValue(saved.get('manual_offset_deg',saved.get('comparison_manual_offset_deg',0)))
        self.manual.valueChanged.connect(self.update_manual_comparison)
        for label,widget in (('开始倒计时',self.delay),('动作准备',self.prepare),('静止保持',self.hold),('手动估计长轴偏角（对照）',self.manual)):form.addRow(label,widget)
        layout.addLayout(form)
        self.instruction=QtWidgets.QLabel('中立站直 / 自然朝向 → 斜前举掌心向相机 → 换一个可见掌面姿态独立复测 → 回到自然朝向。');self.instruction.setWordWrap(True);layout.addWidget(self.instruction)
        self.image=QtWidgets.QLabel('掌部裁剪画面及 RGB 三轴：X手背、Y右手系横向、Z指向手腕。');self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter);self.image.setMinimumHeight(160);layout.addWidget(self.image)
        self.detection_status=QtWidgets.QLabel('黄色：手部21点；青色圈：人体左腕；品红圈：手部腕点。二维检出不代表三维合格。')
        self.detection_status.setWordWrap(True);layout.addWidget(self.detection_status)
        self.status=QtWidgets.QLabel('等待开始；主界面先启动Gemini和人体三维观测。');self.status.setWordWrap(True);layout.addWidget(self.status)
        row=QtWidgets.QHBoxLayout();self.start=QtWidgets.QPushButton('开始引导观察与估计');self.cancel=QtWidgets.QPushButton('取消本轮')
        self.start.clicked.connect(self.begin);self.cancel.clicked.connect(lambda:self.abort('已取消，原标定未改变'))
        row.addWidget(self.start);row.addWidget(self.cancel);layout.addLayout(row)
        row=QtWidgets.QHBoxLayout();self.manual_view=QtWidgets.QPushButton('预览手动 C0');self.visual_view=QtWidgets.QPushButton('预览视觉 C0')
        self.manual_view.clicked.connect(lambda:self.preview(False));self.visual_view.clicked.connect(lambda:self.preview(True))
        row.addWidget(self.manual_view);row.addWidget(self.visual_view);layout.addLayout(row)
        self.apply=QtWidgets.QPushButton('明确应用视觉自然参考（只影响人体显示）');self.apply.clicked.connect(self.apply_visual);layout.addWidget(self.apply)
        self.undo=QtWidgets.QPushButton('撤销视觉应用 / 恢复本窗口前的自然参考');self.undo.setEnabled(False);self.undo.clicked.connect(self.undo_visual);layout.addWidget(self.undo)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self.tick)
        self.set_controls(False)

    @property
    def active(self):return self.phase in ('delay','neutral','prepare_train','train','prepare_check','check','prepare_return','return')

    def set_controls(self,running):
        self.start.setEnabled(not running);self.cancel.setEnabled(running)
        for widget in (self.delay,self.prepare,self.hold,self.manual,self.confirm):widget.setEnabled(not running)
        ready=self.report is not None and not running
        self.manual_view.setEnabled(ready);self.visual_view.setEnabled(ready)
        self.apply.setEnabled(ready and self.report.get('accepted',False))

    def binding_error(self):
        h=self.host
        if h.replay_mode:return '回放中不可重做视觉参考'
        if h.session is not None or h._record_start_deadline is not None:return '请先结束联合录制/预约'
        if h.camera_source!='orbbec' or h.camera_worker is None or not h.rgbd_pose_enabled.isChecked():return '请启动Gemini和人体三维观测'
        if h.engine.algorithm_modes['left_hand']!='gyro_tilt':return '左手请选择陀螺＋加计，本版同步使用纯IMU轨迹'
        device=h.engine.assignments['left_hand']
        if not device or not h.engine.is_live(device,max_age=1):return '左手设备未分配或数据过期'
        if self.seed is not None:
            error=h.engine.natural_hand_error(self.seed)
            if error:return error
            if h.camera_worker.capture_epoch!=self.camera_epoch or h.vision_epoch!=self.vision_epoch:return '相机/人体视觉会话已变化，请重录'
        return None

    def begin(self):
        if self.worker is not None:self.status.setText('旧手部推理仍在退出，请稍后重试');return
        self.seed=None
        error=self.host._pose_change_error() or self.binding_error()
        if error:self.status.setText(error);return
        if not self.confirm.isChecked():self.status.setText('请先确认标准基准和身体朝向');return
        if not DEFAULT_HAND_MODEL.is_file():self.status.setText('缺少本地 hand_landmarker.task，请先下载');return
        e=self.host.engine;device=e.assignments['left_hand']
        try:self.seed=e.make_natural_hand_reference('left_hand',e.current_for(device,'gyro_tilt'),np.eye(3))
        except ValueError as exc:self.status.setText(str(exc));return
        self.camera_epoch=self.host.camera_worker.capture_epoch;self.vision_epoch=self.host.vision_epoch
        self.manual_C0=Rotation.from_euler('z',self.manual.value(),degrees=True).as_matrix()
        self.manual_angle=self.manual.value();self.before=deepcopy(e.natural_hand_references.get('left_hand'))
        self.body_rows=[];self.imu_rows=[];self.palm_rows=[];self.fit_rows=[];self.check_rows=[];self.windows=[];self.seen=set();self.report=None;self.K=None;self.latest=None
        self.path=None;self.undo.setEnabled(False)
        self.diagnostic_images=[];self.diagnostic_counts={};self.image.clear()
        self.detection_status.setText('等待本轮掌部观测；黄色21点 / 青色人体腕点 / 品红手部腕点。')
        self.worker=PalmWorker(self);self.worker.error.connect(self.abort);self.worker.finished.connect(self.worker_finished);self.worker.start()
        self.set_controls(True);self.enter('delay',self.delay.value());self.timer.start()

    def worker_finished(self):
        worker=self.worker;self.worker=None
        if worker is not None:worker.deleteLater()

    def enter(self,phase,duration):
        self.phase=phase;self.phase_start=time.monotonic_ns();self.deadline=self.phase_start+int(duration*1e9)
        self.windows.append({'phase':phase,'start_ns':self.phase_start,'end_ns':self.deadline})
        instructions={
            'delay':'离开鼠标，站直，保持与标准穿戴标定相同朝向；左手自然下垂、掌心向内偏后。',
            'neutral':'自然下垂静止；肩髋完整入镜。此时只采设备参考和身体坐标，不要求看清手掌。',
            'prepare_train':'左臂斜前举，展开手掌，掌心朝向相机，手腕伸直；肩髋保持可见。',
            'train':'保持可见掌面静止，正在拟合；画面应看到红绿蓝三轴。',
            'prepare_check':'换一个略有不同的斜举高度/掌面朝向，掌心仍向相机，手腕伸直；不要转身。',
            'check':'独立复测，保持可见掌面静止；这些帧不参与拟合。',
            'prepare_return':'左臂放下，回到最初自然朝向。',
            'return':'自然下垂静止；检查返回参考的IMU一致性。'}
        self.instruction.setText(instructions[phase]);QtWidgets.QApplication.beep()

    def submit(self,observation):
        if self.worker is None or self.phase not in ('prepare_train','train','prepare_check','check'):return
        body=self.host.last_pose_estimate
        if body is not None:self.worker.submit(observation,body.to_record())

    def on_body(self,estimate):
        if self.phase=='neutral' and self.phase_start+500_000_000<=estimate.captured_mono_ns<=self.deadline-200_000_000:
            row=estimate.to_record()
            if not self.body_rows or row['captured_mono_ns']!=self.body_rows[-1]['captured_mono_ns']:self.body_rows.append(row)

    def consume(self,row):
        stamp=row['captured_mono_ns']
        image=row.pop('frame_rgb',None)
        if image is not None:
            h,w=image.shape[:2];q=QtGui.QImage(image.data,w,h,image.strides[0],QtGui.QImage.Format.Format_RGB888).copy()
            self.image.setPixmap(QtGui.QPixmap.fromImage(q).scaled(480,240,QtCore.Qt.AspectRatioMode.KeepAspectRatio))
            stage=row.get('stage','unknown')
            if not row['accepted'] and len(self.diagnostic_images)<12 and self.diagnostic_counts.get(stage,0)<3:
                detail={key:row[key] for key in ('reason','detected_hand_count','wrist_distance_px','wrist_gate_px','roi_xyxy') if key in row}
                detail['guide_phase']=self.phase
                self.diagnostic_images.append((stamp,stage,q.scaled(640,480,QtCore.Qt.AspectRatioMode.KeepAspectRatio),detail))
                self.diagnostic_counts[stage]=self.diagnostic_counts.get(stage,0)+1
        self.latest=row
        if image is None:
            self.image.clear();self.image.setText('当前掌面无效：'+row.get('reason','未知'))
        distance_text=(f"；腕点距离 {row['wrist_distance_px']:.1f} / 门限 {row['wrist_gate_px']:.1f} px" if 'wrist_distance_px' in row else '')
        self.detection_status.setText(f"二维检出 {row.get('detected_hand_count','未知')} 个；关联 {'通过' if row.get('association_valid') else '未通过'}；三维 {'有效' if row['accepted'] else '未通过'}{distance_text}\n{row.get('reason','')}（采样帧，非当前实时姿态）")
        phase=next((w['phase'] for w in self.windows if w['phase'] in ('train','check') and w['start_ns']+500_000_000<=stamp<=w['end_ns']-200_000_000),None)
        if phase is None or stamp in self.seen:return
        self.seen.add(stamp);self.palm_rows.append(row)
        if not row['accepted'] or self.K is None:return
        r=self.host.live_fusion.at(self.seed['device_id'],stamp)
        if r is None:row.update(accepted=False,reason='无同期纯IMU/间隔超过100ms');return
        b=np.asarray(self.seed['basis']);anchor=np.asarray(self.seed['anchor'])
        d=b@(anchor.T@r)@b.T;c=self.K@proper(row['R_camera_hand'])
        sample={'stamp_ns':stamp,'D':d.tolist(),'C':c.tolist()}
        (self.fit_rows if phase=='train' else self.check_rows).append(sample)

    def tick(self):
        if not self.active:return
        error=self.binding_error()
        if error:self.abort(error);return
        if self.worker is not None:
            row=self.worker.take_latest()
            if row is not None:self.consume(row)
        now=time.monotonic_ns();e=self.host.engine;device=self.seed['device_id']
        if self.phase in ('neutral','return'):
            stamp=e.received_at.get(device)
            if stamp is not None and self.phase_start+500_000_000<=int(stamp*1e9)<=self.deadline-200_000_000 and (not self.imu_rows or self.imu_rows[-1][0]!=stamp):
                self.imu_rows.append((stamp,e.current_for(device,'gyro_tilt').copy()))
        reason=(self.latest or {}).get('reason','等待掌部观测')
        self.status.setText(f'{self.phase} 剩余 {max(0,(self.deadline-now)/1e9):.1f}秒 · 拟合 {len(self.fit_rows)} 帧 · 独立复测 {len(self.check_rows)} 帧\n{reason}')
        if now<self.deadline:return
        try:
            if self.phase=='delay':self.enter('neutral',self.hold.value())
            elif self.phase=='neutral':
                if len(self.imu_rows)<12:raise ValueError('中立纯IMU有效抽样不足12帧')
                self.K,self.body_quality=body_frame(self.body_rows)
                rotations=Rotation.from_matrix([r for _,r in self.imu_rows]);mean=rotations.mean()
                spread=float(np.percentile((mean.inv()*rotations).magnitude()*180/np.pi,90))
                if spread>3:raise ValueError('自然中立IMU波动超过3°')
                self.seed=e.make_natural_hand_reference('left_hand',mean.as_matrix(),np.eye(3));self.imu_rows=[]
                self.enter('prepare_train',self.prepare.value())
            elif self.phase=='prepare_train':self.enter('train',self.hold.value())
            elif self.phase=='train':self.enter('prepare_check',self.prepare.value())
            elif self.phase=='prepare_check':self.enter('check',self.hold.value())
            elif self.phase=='check':self.enter('prepare_return',self.prepare.value())
            elif self.phase=='prepare_return':self.enter('return',self.hold.value())
            else:self.finish()
        except (ValueError,KeyError,OSError) as exc:self.abort(str(exc))

    def finish(self):
        report=estimate_c0(self.fit_rows,self.check_rows,self.manual_C0)
        returned=[distance(r,self.seed['anchor']) for _,r in self.imu_rows]
        report['return_median_deg']=float(np.median(returned)) if returned else None
        if len(returned)<12 or np.median(returned)>12:report.update(accepted=False,reason=report.get('reason','')+'；返回自然参考不一致/帧不足：可能是未复位或IMU漂移')
        report.update(format='visual-palm-c0-v1',reference=self.seed,K_model_from_camera=self.K.tolist(),
            body_quality=self.body_quality,camera_epoch=self.camera_epoch,vision_epoch=self.vision_epoch,
            windows=self.windows,training=self.fit_rows,validation=self.check_rows,palm_observations=self.palm_rows)
        report['clock_note']='用主机单调时钟插值匹配；不是硬件同步。纯IMU积分仍使用设备时钟。'
        report['manual_estimate_offset_deg']=self.manual_angle
        path=REPORT_ROOT/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        path.mkdir(parents=True,exist_ok=False)
        report['diagnostic_screenshots']=self.save_diagnostics(path)
        (path/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
        self.report=report;self.path=path;self.phase='complete';self.timer.stop();self.stop_worker();self.set_controls(False)
        axes='/'.join(f'{v:.1f}°' for v in report['axis_difference_deg'])
        z_label=f"{report['closest_z_offset_deg']:.1f}°" if report['closest_z_offset_deg'] is not None else '无法唯一确定'
        self.status.setText(f'视觉结果：'+('独立验收通过' if report['accepted'] else '需重录')+
            f"\n手动/视觉 SO(3)差 {report['manual_visual_so3_deg']:.1f}°；X/Y/Z轴差 {axes}"
            f"\n视觉最近长轴偏角 {z_label}；仅长轴拟合剩余差 {report['z_only_residual_deg']:.1f}°"
            f"\n独立复测 median/P90 {report['validation_median_deg']:.1f}/{report['validation_p90_deg']:.1f}°；自然长轴倾斜 {report['neutral_long_axis_tilt_deg']:.1f}°"
            f"\n返回自然参考 {report['return_median_deg']}°；{report.get('reason','')}\n{path}")

    def selected_reference(self,visual):
        if self.report is None:raise ValueError('请先完成估计')
        data=deepcopy(self.seed);data['neutral_pose']=self.report['C0_visual' if visual else 'C0_manual']
        data['source']='rgbd-palm-surface' if visual else 'manual-long-axis-offset'
        if not visual:data['manual_offset_deg']=self.manual_angle
        data['comparison_manual_offset_deg']=self.manual_angle
        data['report_path']=str(self.path/'report.json');return data

    def update_manual_comparison(self,value):
        if self.report is None:return
        self.manual_angle=float(value);self.manual_C0=Rotation.from_euler('z',value,degrees=True).as_matrix()
        visual=np.asarray(self.report['C0_visual'])
        self.report.update(C0_manual=self.manual_C0.tolist(),manual_estimate_offset_deg=float(value),
            manual_visual_so3_deg=distance(self.manual_C0,visual),
            axis_difference_deg=[angle(self.manual_C0[:,i],visual[:,i]) for i in range(3)])
        try:(self.path/'report.json').write_text(json.dumps(self.report,ensure_ascii=False,allow_nan=False,indent=2))
        except (OSError,ValueError) as exc:self.status.setText('手动对照已更新，但报告写入失败：'+str(exc));return
        self.status.setText(f"手动对照改为 {value:+.1f}°；视觉拟合不变。SO(3)差 {self.report['manual_visual_so3_deg']:.1f}°"
            f"\n独立复测 median/P90 {self.report['validation_median_deg']:.1f}/{self.report['validation_p90_deg']:.1f}°；{self.path}")
        preview=self.host.engine.natural_hand_previews.get('left_hand')
        if preview and preview.get('source')=='manual-long-axis-offset':self.preview(False)

    def preview(self,visual):
        error=self.binding_error()
        if error:self.status.setText(error);return
        try:self.host.engine.set_natural_hand_reference(self.selected_reference(visual),preview=True)
        except ValueError as exc:self.status.setText(str(exc));return
        self.host._render_dirty=True
        self.instruction.setText('正在预览：'+('视觉完整 C0' if visual else '手动长轴 C0')+'；未应用，关闭窗口会取消临时显示。不是真值对照。')

    def apply_visual(self):
        if self.report is None or not self.report['accepted']:return
        error=self.binding_error()
        if error:self.status.setText(error);return
        answer=QtWidgets.QMessageBox.question(self,'应用视觉参考','以视觉完整C0替换左手自然参考？只影响人体显示，不启用纠偏或机器人。',QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No,QtWidgets.QMessageBox.StandardButton.No)
        if answer!=QtWidgets.QMessageBox.StandardButton.Yes:return
        error=self.binding_error()
        if error:self.status.setText(error);return
        try:self.host.engine.set_natural_hand_reference(self.selected_reference(True))
        except ValueError as exc:self.status.setText(str(exc));return
        self.host.engine.natural_hand_previews.pop('left_hand',None);self.host._render_dirty=True
        self.undo.setEnabled(True)
        self.instruction.setText('已明确应用视觉自然参考；没有启用长期纠偏，X1仍使用标准化姿态。')

    def undo_visual(self):
        if self.active or self.host.session is not None:self.status.setText('请先结束采集/录制');return
        if self.before is None:self.host.engine.natural_hand_references.pop('left_hand',None)
        else:
            try:self.host.engine.set_natural_hand_reference(self.before)
            except ValueError as exc:self.status.setText('原参考已失效，未恢复：'+str(exc));return
        self.host.engine.natural_hand_previews.pop('left_hand',None);self.host._render_dirty=True;self.undo.setEnabled(False)
        self.instruction.setText('已恢复本轮开始前的自然参考。')

    def stop_worker(self):
        if self.worker is not None:self.worker.stop()

    def save_diagnostics(self,path):
        saved=[]
        for stamp,stage,image,detail in self.diagnostic_images:
            name=f'diagnostic-{stage}-{stamp}.jpg'
            if image.save(str(path/name),'JPG',85):saved.append(dict(detail,file=name,captured_mono_ns=stamp,stage=stage))
        return saved

    def abort(self,message):
        if self.active and self.seed is not None:
            path=REPORT_ROOT/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            try:
                path.mkdir(parents=True,exist_ok=False)
                data=dict(format='visual-palm-c0-failed-v1',accepted=False,reason=str(message),reference=self.seed,
                    K_model_from_camera=None if self.K is None else self.K.tolist(),windows=self.windows,
                    training=self.fit_rows,validation=self.check_rows,palm_observations=self.palm_rows)
                data['diagnostic_screenshots']=self.save_diagnostics(path)
                (path/'report.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2));message=str(message)+f'；诊断：{path}'
            except (OSError,ValueError) as exc:message=str(message)+'；诊断写入失败：'+str(exc)
        self.phase='failed';self.timer.stop();self.stop_worker();self.set_controls(False);self.status.setText(str(message)+'；未自动改变标定')

    def shutdown(self):
        self.timer.stop();self.stop_worker()
        if self.worker is not None and not self.worker.wait(3000):return False
        self.host.engine.natural_hand_previews.pop('left_hand',None);self.host._render_dirty=True;self.phase='idle';return True

    def closeEvent(self,event):
        if self.shutdown():event.accept()
        else:self.status.setText('手部推理正在退出，请稍后关闭');event.ignore()
