"""Common guided recording with independent wearing and camera result selection."""
import json
from pathlib import Path
import sys
import uuid
from copy import deepcopy

from PySide6 import QtCore,QtWidgets

from .camera_calibration_gui import CameraCalibrationDialog,PART_LABELS
from .combined_calibration import apply_joint
from .camera_protocol import OBLIQUE_PROTOCOL, add_oblique_steps
from .recording_cleanup import validate_capture
from .visual_wearing import needs_review
from .visual_wearing_gui import wearing_steps
from .guided_calibration import ARM_PARTS
from .pose import GYRO_RGBD, GYRO_TILT
from .settings import save_config


class CombinedCalibrationDialog(CameraCalibrationDialog):
    def __init__(self,owner):
        super().__init__(owner)
        self.setWindowTitle('双臂一体标定 · 穿戴映射＋相机纠偏');self.resize(920,850)
        self.report=None;self._preview_opened=False;self.applied=False
        self._applied_camera_parts=()
        self._cleanup_process=None;self._cleanup_capture=None;self._deleted_capture=None
        self._camera_preview_processes=[]
        self.delay.setValue(5)
        self.preparation.setValue(4)
        self.hold.setValue(4)
        self.neutral.setValue(5)
        self.validation.setValue(10)
        from .calibration_options import add_prior_control
        add_prior_control(self, owner)
        layout=self.layout()
        layout.itemAt(0).widget().setText('一套动作、一次录制，两路分别拟合和验收。原IMU穿戴保留正前举/侧举；相机改用斜前举/侧举，避开正面遮挡。可选旧视觉穿戴算法仍要求正前举可见。重复动作只验收，不参与拟合；不自动应用或启动机器人。')
        row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel('穿戴算法'))
        self.method=QtWidgets.QComboBox()
        self.method.addItem('原 IMU 平举标定（默认）','imu');self.method.addItem('RGB-D 视觉辅助穿戴标定','visual')
        row.addWidget(self.method);layout.insertLayout(2,row)
        self.auto_analyze.setText('录制后计算两类标定（不自动应用）')
        status_index=layout.indexOf(self.status);layout.removeWidget(self.status)
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignTop)
        self.status.setMinimumHeight(140)
        status_scroll=QtWidgets.QScrollArea();status_scroll.setWidgetResizable(True)
        status_scroll.setMinimumHeight(90);status_scroll.setMaximumHeight(130);status_scroll.setWidget(self.status)
        layout.insertWidget(status_index,status_scroll)
        self.analyze.setText('计算穿戴＋相机两类结果')
        self.replay.setText('穿戴双模型回放 · 现有 vs 本轮候选')
        self.replay.clicked.disconnect();self.replay.clicked.connect(self.open_wearing_preview)
        self.apply.setText('应用勾选合格结果（不自动启用融合或机器人）')
        self.enable_rgbd=QtWidgets.QPushButton('一键切换已应用合格部位为陀螺＋加计＋RGB-D')
        self.enable_rgbd.setEnabled(False)
        self.enable_rgbd.setToolTip('仅切换本轮已应用的合格相机部位；未通过、未应用或参考失效的部位跳过，保留原模式。不启动X1。')
        self.enable_rgbd.clicked.connect(self.enable_bilateral_rgbd)
        layout.insertWidget(layout.indexOf(self.apply)+1,self.enable_rgbd)
        self.enable_gyro=QtWidgets.QPushButton('一键切回双臂陀螺＋加计（重力纠倾斜）')
        self.enable_gyro.setToolTip('切换左右上臂、前臂的显示/控制模式，不归零、不重置积分器，不删除穿戴或相机标定。需先停止录制和X1随动。')
        self.enable_gyro.clicked.connect(self.enable_bilateral_gyro)
        layout.insertWidget(layout.indexOf(self.enable_rgbd)+1,self.enable_gyro)
        self.table=QtWidgets.QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(['部位','穿戴结果','应用穿戴','相机结果','应用相机'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(190);layout.addWidget(self.table)
        self.review=QtWidgets.QCheckBox('已查看本轮回放，确认需人工复核的视觉穿戴结果可接受')
        layout.addWidget(self.review)
        self.camera_replay=QtWidgets.QPushButton('相机纠偏三路离线对比')
        self.camera_replay.setEnabled(False);self.camera_replay.clicked.connect(self.open_camera_preview)
        self.delete_recording=QtWidgets.QPushButton('删除本次原始录制（释放磁盘空间）')
        self.delete_recording.setEnabled(False)
        self.delete_recording.setToolTip('本轮计算完成并生成报告后才能操作，包括全部未通过的结果。永久删除原始数据，保留报告和内存结果；不能再回放或重新分析。')
        self.delete_recording.clicked.connect(self.delete_current_capture)
        cleanup_row=QtWidgets.QHBoxLayout();cleanup_row.addWidget(self.camera_replay);cleanup_row.addWidget(self.delete_recording)
        layout.addLayout(cleanup_row)
        self.wear_checks={};self.camera_checks={}

    def capture_steps(self):
        scope=self.scope_combo.currentData()
        steps=wearing_steps(self.preparation.value(),self.hold.value(),self.neutral.value(),self.validation.value(),scope)
        return add_oblique_steps(steps,self.preparation.value(),self.hold.value(),scope)

    def guide_extra(self):
        return {'workflow':'combined_wearing_camera_calibration','capture_scope':self.capture_scope,
                'participants':list(self.devices),'wearing_method':self.capture_method,
                'camera_pose_protocol':OBLIQUE_PROTOCOL}

    def begin(self):
        error=self.owner._pose_change_error()
        if self.active or self.state=='analyzing':error='当前采集/分析尚未结束'
        if self._cleanup_running():error='正在删除原始录制，请等待完成'
        if not error:error=self.readiness_error()
        if error:
            self._show_start_error(error);return
        self.capture_method=self.method.currentData()
        # Parent must accept startup before retiring previous results. A second
        # readiness check can still fail at an age boundary; preserve those too.
        if not super().begin():
            self._show_start_error(self.status.text());return
        self.report=None;self.applied=False;self._preview_opened=False
        self._applied_camera_parts=()
        self.enable_rgbd.setEnabled(False)
        self._cleanup_capture=None;self._deleted_capture=None;self.delete_recording.setEnabled(False)
        self.table.setRowCount(0);self.review.setChecked(False);self.camera_replay.setEnabled(False)

    def _show_start_error(self,error):
        if '不要用视觉纠偏流拟合' in error:
            error+='\n请点击“一键切回双臂陀螺＋加计”，再开始录制。'
        message='未开始录制：'+error+'\n上次标定结果保留。'
        self.status.setText(message)
        QtWidgets.QMessageBox.warning(self,'无法开始标定录制',message)

    def set_controls(self,running):
        super().set_controls(running)
        if hasattr(self,'method'):self.method.setEnabled(not running)
        if hasattr(self,'enable_rgbd'):self.enable_rgbd.setEnabled(self.applied and bool(self._applied_camera_parts) and not running)
        if hasattr(self,'enable_gyro'):self.enable_gyro.setEnabled(not running and not self.owner.replay_mode)

    def analyze_recording(self):
        if self._cleanup_running() or self._deleted_capture is not None:return
        if self.state!='complete':return
        self.delete_recording.setEnabled(False);self._cleanup_capture=None
        self.analysis_method=self.method.currentData()
        self.prior_capture_label.setText(f'本次计算 λ={self.prior_weight_spin.value():.3f}（结果报告记录实际使用值）')
        self._preview_opened=False;self.review.setChecked(False);self.applied=False
        self._applied_camera_parts=()
        self.enable_rgbd.setEnabled(False)
        self.analysis_path=self.recording_path.parent/'analysis_combined'/(self.recording_path.name+'-'+uuid.uuid4().hex[:6])
        if self.process is not None:self.process.deleteLater()
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log);self.process.finished.connect(self.analysis_finished)
        self.process.errorOccurred.connect(self.analysis_error)
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-u',str(Path(__file__).resolve().parents[1]/'combined_calibration_analyze.py'),
            str(self.recording_path),'--output',str(self.analysis_path),'--wearing-method',self.analysis_method,
            '--prior-weight', str(self.prior_weight_spin.value())])
        self.state='analyzing';self.report=None;self.log='';self.table.setRowCount(0)
        self.set_controls(True);self.cancel.setEnabled(False)
        for button in (self.analyze,self.replay,self.apply,self.camera_replay):button.setEnabled(False)
        self.status.setText('同一录制正在分别计算穿戴和相机结果；不修改实时模型。')
        self.process.start()

    def analysis_error(self,error):
        super().analysis_error(error)
        if error==QtCore.QProcess.ProcessError.FailedToStart:self.set_controls(False)

    def analysis_finished(self,code,_status):
        self.read_log();self.state='complete';self.set_controls(False);self.analyze.setEnabled(True)
        try:
            if code!=0:raise ValueError(self.log[-2500:])
            self.report=json.loads((self.analysis_path/'report.json').read_text());self.show_results()
            self._authorize_cleanup()
        except (OSError,ValueError,KeyError) as exc:self.status.setText('联合分析失败；原始数据保留：'+str(exc))

    def show_results(self):
        wearing=self.report['wearing']['parts'];camera=self.report['camera']['parts']
        self.table.setRowCount(len(self.devices));self.wear_checks={};self.camera_checks={};messages=[]
        for row,part in enumerate(self.devices):
            w=wearing.get(part,{});c=camera.get(part,{})
            self.table.setItem(row,0,QtWidgets.QTableWidgetItem(PART_LABELS[part]))
            self.table.setItem(row,1,QtWidgets.QTableWidgetItem(w.get('status','需重录')))
            self.table.setItem(row,3,QtWidgets.QTableWidgetItem('通过' if c.get('accepted') else '需重录'))
            for column,values,checks in ((2,w,self.wear_checks),(4,c,self.camera_checks)):
                check=QtWidgets.QCheckBox();check.setEnabled(bool(values.get('accepted')))
                check.setChecked(bool(values.get('accepted')));self.table.setCellWidget(row,column,check);checks[part]=check
            reason=c.get('reason') or ('独立复测通过' if c.get('accepted') else '拟合未通过')
            for phase,label in (('forward_repeat','再次前举'),('oblique_repeat','再次斜前举'),('lateral_repeat','再次侧举'),('neutral_return','回中立')):
                reason=reason.replace(phase,label)
            message=f'{PART_LABELS[part]} · 穿戴：'+('；'.join(w.get('warnings',[])) or w.get('status','失败'))+' · 相机：'+reason
            checks=c.get('independent_validation',{})
            for phase,label in (('forward_repeat','再次前举'),('oblique_repeat','再次斜前举'),('lateral_repeat','再次侧举'),('neutral_return','回中立')):
                metric=checks.get(phase)
                if metric:
                    detail=f"{label} {metric['samples']}帧"
                    if 'median_deg' in metric:detail+=f" · 中位/P90 {metric['median_deg']:.2f}°/{metric['p90_deg']:.2f}°"
                    message+='\n  '+detail
            diagnostic=c.get('camera_diagnostics',{})
            neutral=diagnostic.get('windows',{}).get('neutral',{})
            if 'median_deg' in neutral:message+=f"\n  首次中立预测/视觉残差 {neutral['median_deg']:.2f}°（不是回位角）"
            returned=diagnostic.get('return_vs_initial',{})
            if returned:
                message+=(f"\n  首尾变化：IMU姿态 {returned['imu_orientation_change_deg']:.2f}°"
                          f" · 预测骨向 {returned['predicted_direction_change_deg']:.2f}°"
                          f" · 视觉骨向 {returned['visual_direction_change_deg']:.2f}°"
                          f" · 肘腕/端点深度落差变化 {returned['endpoint_depth_difference_change_m']*100:+.2f} cm")
                message+='\n  上述是两路一致性诊断，不是GT；相机验收仍要求中位<12°、P90<25°且≥8帧。'
            messages.append(message)
            self.table.item(row,1).setToolTip(message);self.table.item(row,3).setToolTip(message)
        self.status.setText('\n'.join(messages)+f'\n分析目录：{self.analysis_path}')
        self.apply.setEnabled(any(d.get('accepted') for d in list(wearing.values())+list(camera.values())))
        self.replay.setEnabled(any(d.get('basis_matrix') for d in wearing.values()))
        self.camera_replay.setEnabled((self.analysis_path/'camera'/'comparison.jsonl').is_file())

    def apply_live(self):
        if self.report is None or self.applied:return
        if self.owner._pose_change_error():self.status.setText('请先停止其他录制/标定及X1随动');return
        wear=[p for p,c in self.wear_checks.items() if c.isChecked()]
        camera=[p for p,c in self.camera_checks.items() if c.isChecked()]
        review_needed=self.report['wearing_method']=='visual' and any(needs_review(self.report['wearing']['parts'][p]) for p in wear)
        if review_needed and not (self._preview_opened and self.review.isChecked()):
            self.status.setText('视觉穿戴存在需复核结果，请先看双模型回放并明确勾选确认。');return
        answer=QtWidgets.QMessageBox.question(self,'确认应用','只应用勾选合格结果；穿戴先应用，相机随后应用。未选择相机的穿戴部位会停用旧相机结果。不自动切换算法或启动机器人。是否继续？')
        if answer!=QtWidgets.QMessageBox.StandardButton.Yes:return
        try:apply_joint(self.owner,self.report,wear,camera,review_acknowledged=self.review.isChecked())
        except (ValueError,KeyError) as exc:self.status.setText('未应用：'+str(exc));return
        self.applied=True;self.apply.setEnabled(False)
        # Snapshot actual successful application, not later checkbox edits.
        self._applied_camera_parts=tuple(camera)
        self.enable_rgbd.setEnabled(bool(camera))
        self.status.setText('已应用所选合格结果。'+
            (f'可一键启用 {len(camera)} 个已应用相机部位的RGB-D融合。' if camera else '本轮未应用相机结果，不能启用融合。')+
            '请先观察人体模型，没有自动启动机器人。')

    def enable_bilateral_rgbd(self):
        if not self.applied:return
        owner=self.owner
        error=owner._pose_change_error()
        if self.active or self.state=='analyzing' or self._cleanup_running():error='请先结束采集、分析或删除操作'
        if owner.recorder is not None:error='请先结束录制'
        if error:
            self.status.setText('未切换：'+error);return
        try:
            # Skip ineligible parts, but commit all eligible changes atomically.
            trial=deepcopy(owner.engine)
            selected=[];skipped=[]
            for part in ARM_PARTS:
                data=self.report['camera']['parts'].get(part,{})
                if part not in self._applied_camera_parts:
                    reason='相机未通过' if data and not data.get('accepted') else '本轮未应用相机结果'
                    skipped.append(f'{PART_LABELS[part]}：{reason}');continue
                try:
                    if not data.get('accepted'):raise ValueError('相机结果未通过')
                    device=data['device_id']
                    if trial.assignments.get(part)!=device or trial.epochs.get(device)!=data.get('reference_epoch'):
                        raise ValueError('设备绑定/参考系已变化，需重新标定')
                    if owner.algorithm_combos[part].findData(GYRO_RGBD)<0:raise ValueError('界面缺少RGB-D模式')
                    trial.set_algorithm_mode(part,GYRO_RGBD)
                except (ValueError,KeyError) as exc:
                    skipped.append(f'{PART_LABELS[part]}：{exc}');continue
                selected.append(part)
            if not selected:
                self.status.setText('未切换：没有本轮已应用且仍有效的相机部位。\n跳过：'+'；'.join(skipped));return
            save_config(owner.settings_path,trial.assignments,trial.axis_mappings,trial.algorithm_modes)
        except (ValueError,OSError,KeyError) as exc:
            self.status.setText('双臂未切换，原模式保持不变：'+str(exc));return
        for field in ('algorithm_modes','current','neutral','natural_hand_previews','natural_hand_references'):
            setattr(owner.engine,field,getattr(trial,field))
        for part in selected:
            combo=owner.algorithm_combos[part];combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(GYRO_RGBD));combo.blockSignals(False)
        owner._render_dirty=True
        self.status.setText(f'已切换 {len(selected)}/4：'+ '、'.join(PART_LABELS[p] for p in selected)+
            ' → 陀螺＋加计＋RGB-D。'+ ('\n跳过（保留原模式）：'+'；'.join(skipped) if skipped else '')+
            '\n请先观察人体随动；未启动X1。')

    def enable_bilateral_gyro(self):
        error=self.owner._pose_change_error()
        if self.active or self.state=='analyzing' or self._cleanup_running():error='请先结束采集、分析或删除操作'
        if self.owner.recorder is not None:error='请先结束录制'
        if error:
            self.status.setText('未切换：'+error)
            QtWidgets.QMessageBox.warning(self,'无法切回纯IMU模式',self.status.text());return
        try:
            trial=deepcopy(self.owner.engine)
            for part in ARM_PARTS:
                if self.owner.algorithm_combos[part].findData(GYRO_TILT)<0:raise ValueError(f'{PART_LABELS[part]}：界面缺少陀螺＋加计模式')
                try:trial.set_algorithm_mode(part,GYRO_TILT)
                except ValueError as exc:raise ValueError(f'{PART_LABELS[part]}：{exc}') from exc
            save_config(self.owner.settings_path,trial.assignments,trial.axis_mappings,trial.algorithm_modes)
        except (ValueError,OSError,KeyError) as exc:
            self.status.setText('未切换，原模式保持不变：'+str(exc))
            QtWidgets.QMessageBox.warning(self,'无法切回纯IMU模式',self.status.text());return
        for field in ('algorithm_modes','current','neutral','natural_hand_previews','natural_hand_references'):
            setattr(self.owner.engine,field,getattr(trial,field))
        for part in ARM_PARTS:
            combo=self.owner.algorithm_combos[part];combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(GYRO_TILT));combo.blockSignals(False)
        self.owner._render_dirty=True
        self.status.setText('左右上臂、前臂已切回陀螺＋加计（重力纠倾斜）。标定和积分器保留，没有归零。请确认相机/设备就绪，再点击开始录制。')

    def _authorize_cleanup(self):
        # Freeze the exact computed session identity, not a later mutable path.
        try:
            target=Path(self.recording_path).resolve()
            if target != Path(self.report['session']).resolve():raise ValueError('报告与录制路径不一致')
            manifest=json.loads((target/'manifest.json').read_text())
            validate_capture(target,manifest['started_mono_ns'])
            self._cleanup_capture=(target,manifest['started_mono_ns']);self.delete_recording.setEnabled(True)
        except (OSError,ValueError,KeyError,TypeError):
            self._cleanup_capture=None;self.delete_recording.setEnabled(False)

    def open_wearing_preview(self):
        if self._cleanup_running() or self._deleted_capture is not None:return
        if self.report is None:return
        from .fusion_gui import FusionWindow
        from .fusion_replay import FusionSessionReplay
        candidates=[]
        for part,data in self.report['wearing']['parts'].items():
            if data.get('basis_matrix'):
                item=dict(data,part=part)
                if self.report['wearing_method']=='visual':item['method']='visual_bone_v1'
                candidates.append(item)
        source=None
        try:
            source=FusionSessionReplay(self.recording_path)
            window=FusionWindow(source,self.owner.settings_path,source.manifest.get('assignments_at_start',{}),
                source.manifest.get('axis_mappings_at_start',{}),algorithm_modes=source.manifest.get('algorithm_modes_at_start',{}),
                calibration_candidates=candidates,manual_mapping_source='同一录制：原映射 vs 本轮穿戴候选')
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose);self.owner._dialogs.append(window)
            window.destroyed.connect(lambda:self.owner._dialogs.remove(window) if window in self.owner._dialogs else None)
            window.show();self._preview_opened=True
        except (ValueError,KeyError,OSError) as exc:
            if source is not None:source.close()
            self.status.setText('回放失败：'+str(exc))

    def open_camera_preview(self):
        if self._cleanup_running() or self._deleted_capture is not None:return
        process=QtCore.QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([str(Path(__file__).resolve().parents[1]/'rgbd_fusion_viewer.py'),str(self.analysis_path/'camera')])
        self._camera_preview_processes.append(process)
        process.finished.connect(lambda *_:self._release_preview_process(process))
        process.errorOccurred.connect(lambda error:self._release_preview_process(process) if error==QtCore.QProcess.ProcessError.FailedToStart else None)
        process.start()

    def _release_preview_process(self,process):
        if process in self._camera_preview_processes:self._camera_preview_processes.remove(process)
        process.deleteLater()

    def _cleanup_running(self):
        return self._cleanup_process is not None and self._cleanup_process.state()!=QtCore.QProcess.ProcessState.NotRunning

    def cleanup_error(self):
        if self.report is None or self._cleanup_capture is None:return '请先完成本轮计算并生成结果'
        if self.active or self.state=='analyzing' or self._cleanup_running():return '采集、分析或删除仍在进行'
        if self.owner.session is not None or self.owner._record_start_deadline is not None:return '请先结束录制/预约'
        target,token=self._cleanup_capture
        if self.recording_path is None or Path(self.recording_path).resolve()!=target:return '当前录制已改变'
        if self.analysis_path is None or Path(self.analysis_path).resolve().is_relative_to(target):return '分析目录必须位于原录制之外'
        if self._camera_preview_processes:return '请先关闭本轮相机三路回放窗口'
        for window in self.owner._dialogs:
            source=getattr(window,'source',None)
            if getattr(source,'is_replay',False) and Path(source.path).resolve()==target:
                if not getattr(getattr(source,'imu_file',None),'closed',False):return '请先关闭正在使用本轮录制的人体回放窗口'
        try:validate_capture(target,token)
        except (OSError,ValueError,KeyError,TypeError) as exc:return str(exc)
        return None

    def delete_current_capture(self):
        error=self.cleanup_error()
        if error:self.status.setText('未删除：'+error);return
        target,token=self._cleanup_capture
        answer=QtWidgets.QMessageBox.question(self,'永久删除本次原始录制',
            f'仅永久删除以下原始录制目录：\n{target}\n\n分析报告保留在：\n{self.analysis_path}\n\n内存结果和已应用标定不受影响；尚未应用的合格结果仍可在本窗口应用。需人工复核的视觉结果请先完成回放确认。请确认其他独立程序也未使用该目录。删除后不能回放或重新分析，无法从回收站恢复。是否删除？',
            QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No,QtWidgets.QMessageBox.StandardButton.No)
        if answer!=QtWidgets.QMessageBox.StandardButton.Yes:return
        error=self.cleanup_error()  # Re-check after the confirmation's nested event loop.
        if error:self.status.setText('未删除：'+error);return
        process=QtCore.QProcess(self);self._cleanup_process=process
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.setProgram(sys.executable)
        process.setArguments([str(Path(__file__).resolve().parents[1]/'cleanup_calibration_recording.py'),str(target),'--started-mono-ns',str(token)])
        process.finished.connect(self._cleanup_finished)
        process.errorOccurred.connect(self._cleanup_error)
        for button in (self.delete_recording,self.start,self.analyze,self.replay,self.camera_replay):button.setEnabled(False)
        self.status.setText('正在后台删除本次原始录制；实时标定保持不变。')
        process.start()

    def _cleanup_error(self,error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:self._cleanup_finished(1,None)

    def _cleanup_finished(self,code,_status):
        process=self._cleanup_process
        if process is None:return
        log=bytes(process.readAllStandardOutput()).decode('utf-8','replace')[-2000:]
        target=self._cleanup_capture[0]
        self._deleted_capture=target;self.start.setEnabled(True)
        if code==0:
            self.delete_recording.setEnabled(False);self._cleanup_capture=None
            self.status.setText(f'已永久删除原始录制：{target}\n分析报告仍保留：{self.analysis_path}\n内存结果和实时标定未改变，合格结果仍可应用；原始回放与重新分析已禁用。')
        else:
            self.delete_recording.setEnabled(target.exists())
            self.status.setText('删除失败，可能已有部分文件移除；实时标定未改变。\n'+log)
        self._cleanup_process=None;process.deleteLater()

    def closeEvent(self,event):
        if self._cleanup_running():self.status.setText('删除正在进行，请等待完成后关闭。');event.ignore();return
        super().closeEvent(event)
