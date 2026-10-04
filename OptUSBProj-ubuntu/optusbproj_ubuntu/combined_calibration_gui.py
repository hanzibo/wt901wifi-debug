"""Common guided recording with independent wearing and camera result selection."""
import json
from pathlib import Path
import sys
import uuid

from PySide6 import QtCore,QtWidgets

from .camera_calibration_gui import CameraCalibrationDialog,PART_LABELS
from .combined_calibration import apply_joint
from .visual_wearing import needs_review
from .visual_wearing_gui import wearing_steps


class CombinedCalibrationDialog(CameraCalibrationDialog):
    def __init__(self,owner):
        super().__init__(owner)
        self.setWindowTitle('双臂一体标定 · 穿戴映射＋相机纠偏');self.resize(920,850)
        self.report=None;self._preview_opened=False;self.applied=False
        self.validation.setValue(20)
        layout=self.layout()
        layout.itemAt(0).widget().setText('一套动作、一次录制，两路分别拟合和验收。默认沿用原IMU双臂穿戴算法，可选视觉辅助；重复动作只验收，不参与拟合。保持当前积分器运行，不强制重新学习零偏；不自动应用或启动机器人。')
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
        self.table=QtWidgets.QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(['部位','穿戴结果','应用穿戴','相机结果','应用相机'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(190);layout.addWidget(self.table)
        self.review=QtWidgets.QCheckBox('已查看本轮回放，确认需人工复核的视觉穿戴结果可接受')
        layout.addWidget(self.review)
        self.camera_replay=QtWidgets.QPushButton('相机纠偏三路离线对比')
        self.camera_replay.setEnabled(False);self.camera_replay.clicked.connect(self.open_camera_preview);layout.addWidget(self.camera_replay)
        self.wear_checks={};self.camera_checks={}

    def capture_steps(self):
        return wearing_steps(self.preparation.value(),self.hold.value(),self.neutral.value(),self.validation.value(),self.scope_combo.currentData())

    def guide_extra(self):
        return {'workflow':'combined_wearing_camera_calibration','capture_scope':self.capture_scope,
                'participants':list(self.devices),'wearing_method':self.capture_method}

    def begin(self):
        error=self.owner._pose_change_error()
        if error:self.status.setText(error);return
        self.capture_method=self.method.currentData();self.report=None;self.applied=False;self._preview_opened=False
        self.table.setRowCount(0);self.review.setChecked(False);self.camera_replay.setEnabled(False)
        super().begin()

    def set_controls(self,running):
        super().set_controls(running)
        if hasattr(self,'method'):self.method.setEnabled(not running)

    def analyze_recording(self):
        if self.state!='complete':return
        self.analysis_method=self.method.currentData()
        self._preview_opened=False;self.review.setChecked(False);self.applied=False
        self.analysis_path=self.recording_path.parent/'analysis_combined'/(self.recording_path.name+'-'+uuid.uuid4().hex[:6])
        if self.process is not None:self.process.deleteLater()
        self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log);self.process.finished.connect(self.analysis_finished)
        self.process.errorOccurred.connect(self.analysis_error)
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-u',str(Path(__file__).resolve().parents[1]/'combined_calibration_analyze.py'),
            str(self.recording_path),'--output',str(self.analysis_path),'--wearing-method',self.analysis_method])
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
            for phase,label in (('forward_repeat','再次前举'),('lateral_repeat','再次侧举'),('neutral_return','回中立')):
                reason=reason.replace(phase,label)
            message=f'{PART_LABELS[part]} · 穿戴：'+('；'.join(w.get('warnings',[])) or w.get('status','失败'))+' · 相机：'+reason
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
        self.status.setText('已应用所选合格结果。请先观察人体模型，再手动选择已应用相机部位的RGB-D融合算法；没有自动启动机器人。')

    def open_wearing_preview(self):
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
        ok,_=QtCore.QProcess.startDetached(sys.executable,[str(Path(__file__).resolve().parents[1]/'rgbd_fusion_viewer.py'),str(self.analysis_path/'camera')])
        if not ok:self.status.setText('无法启动相机纠偏对比窗口')
