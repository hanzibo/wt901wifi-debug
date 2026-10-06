"""Device-discovering gesture observer and opt-in discrete RViz preview."""
import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from itertools import permutations, product
import json
import math
from pathlib import Path
import sys
import socket
import time
from types import SimpleNamespace
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from wt901.gui_udp import GuiUdpSource
from wt901_viz.axis_mapping import mapping_matrix
from .gripper_gesture import PairBuffer, GestureCalibration, static_mean, DiscreteClosure
from .pose import PoseEngine, GYRO_TILT, VENDOR_RPY


class TracePlot(QtWidgets.QWidget):
    """Small native Qt plot; no additional graphing dependency."""
    def __init__(self):
        super().__init__();self.rows=[];self.setMinimumHeight(190)

    def set_rows(self, rows):
        self.rows=list(rows);self.update()

    def paintEvent(self,event):
        painter=QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),QtGui.QColor('#151c28'))
        area=QtCore.QRectF(55,35,max(1,self.width()-75),max(1,self.height()-70))
        painter.setPen(QtGui.QColor('#8192aa'));painter.drawRect(area)
        painter.drawText(12,20,'deg · 蓝：手势角 / 橙：离轴残差')
        painter.drawText(55,self.height()-10,'最近24秒 · 横轴：运行时间(s)')
        if not self.rows:return
        end=self.rows[-1][0];start=max(0.,end-24.);span=max(1.,end-start)
        rows=[row for row in self.rows if row[0]>=start]
        low=min(-10.,min(row[1] for row in rows));high=max(100.,max(max(row[1:]) for row in rows))
        for value in np.linspace(low,high,5):
            y=area.bottom()-(value-low)/(high-low)*area.height()
            painter.drawText(2,int(y+4),f'{value:.0f}')
        painter.drawText(55,self.height()-28,f'{start:.1f}')
        painter.drawText(self.width()-65,self.height()-28,f'{end:.1f}')
        for column,color in ((1,'#4caaff'),(2,'#ffac55')):
            points=QtGui.QPolygonF([QtCore.QPointF(area.left()+(t-start)/span*area.width(),
                area.bottom()-(row[column]-low)/(high-low)*area.height()) for row in rows for t in (row[0],)])
            painter.setPen(QtGui.QPen(QtGui.QColor(color),2));painter.drawPolyline(points)


class DemoSource:
    """Analytic common-hand + finger motion, no socket or hardware."""
    def __init__(self):
        self.start = time.monotonic()

    def poll_timed(self):
        now = time.monotonic(); t = now-self.start
        # First 3 seconds allow filter bias acquisition.
        theta = 0. if t < 3 else .65*(1-math.cos((t-3)*.6))
        speed = 0. if t < 3 else .39*math.sin((t-3)*.6)
        yaw = 0. if t < 3 else .4*math.sin((t-3)*.3)
        ys = 0. if t < 3 else .12*math.cos((t-3)*.3)
        hand = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
        finger = hand @ Rotation.from_rotvec([0, theta, 0]).as_matrix()
        rows = []
        for device, r, gyro in (('DEMO-手背', hand, [0,0,ys]),
                               ('DEMO-指段', finger, [-math.sin(theta)*ys,speed,math.cos(theta)*ys])):
            roll,pitch,yaw_deg = Rotation.from_matrix(r).as_euler('xyz',degrees=True)
            a = r.T @ [0.,0.,1.]; g = np.rad2deg(gyro)
            tick = int(t*1000)
            rows.append((int(now*1e9), SimpleNamespace(device_id=device,roll=roll,pitch=pitch,yaw=yaw_deg,
                hour=0,minute=(tick//60000)%60,second=(tick//1000)%60,millisecond=tick%1000,
                gyro_x=g[0],gyro_y=g[1],gyro_z=g[2],acc_x=a[0],acc_y=a[1],acc_z=a[2],battery_percent=100)))
        return rows

    def close(self):
        pass


class SharedGestureSource:
    """Bounded copy of the host's packets, never opens or closes its UDP socket."""
    def __init__(self):
        self.rows = deque(maxlen=2000)

    def feed(self, ns, packet, observation=None):
        self.rows.append((ns, packet, observation))

    def poll_timed(self):
        return [(stamp,packet) for stamp,packet,_ in self.poll_observed()]

    def poll_observed(self):
        rows = list(self.rows)
        self.rows.clear()
        return rows

    def close(self):
        self.rows.clear()


class GestureWindow(QtWidgets.QWidget):
    def __init__(self, source, output_root, parent=None, panel_joint=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.embedded = parent is not None
        self.preview_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.preview_socket.setblocking(False)
        self.preview_seq = 0
        self.last_send = 0.
        self.last_ack = None
        self.discrete = DiscreteClosure()
        self.role_validator = None
        self.host_bias_ready = {}
        self.source=source;self.output_root=Path(output_root);self.engine=PoseEngine()
        self.buffer=PairBuffer();self.calibration=None;self.capture=None
        self.record_file=None;self.record_path=None;self.last_log_stamp=None
        self.plot_rows=deque(maxlen=1200);self.last_plot_stamp=None;self.filtered=None;self.last_filter_stamp=None
        self.setWindowTitle('双 IMU 夹爪 · 离散开合（仅 RViz）');self.resize(1040,920)
        outer=QtWidgets.QVBoxLayout(self)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);outer.addWidget(scroll)
        content=QtWidgets.QWidget();scroll.setWidget(content)
        layout=QtWidgets.QVBoxLayout(content)
        note=QtWidgets.QLabel(('复用主窗口设备数据，不另占UDP1399。' if self.embedded else '独占UDP1399，请关闭人体／手腕GUI。')+
            '支撑手等待静止学习；先采开手，再采舒适闭合。\n'
            '默认开手时两设备外壳XYZ同向；否则调整安装映射。双阈值不消除漂移；不控制真实机器人。')
        note.setWordWrap(True);layout.addWidget(note)
        if panel_joint is not None:note.hide()
        form=QtWidgets.QFormLayout();layout.addLayout(form)
        self.hand=QtWidgets.QComboBox();self.finger=QtWidgets.QComboBox()
        for combo in (self.hand,self.finger):combo.addItem('请选择在线设备','')
        form.addRow('手背设备',self.hand);form.addRow('拇指／四指设备',self.finger)
        self.mode=QtWidgets.QComboBox();self.mode.addItem('陀螺＋加计（推荐）',GYRO_TILT);self.mode.addItem('厂商RPY（对照）',VENDOR_RPY)
        form.addRow('两设备共同姿态算法',self.mode)
        self.mount=QtWidgets.QComboBox()
        self.mount.addItem('同向：指段 +X/+Y/+Z → 手背 +X/+Y/+Z', ('+X','+Y','+Z'))
        for axes in permutations('XYZ'):
            for signs in product('+-',repeat=3):
                mapping=tuple(s+a for s,a in zip(signs,axes))
                try:mapping_matrix(mapping)
                except ValueError:continue
                if mapping!=('+X','+Y','+Z'):self.mount.addItem('指段 +X/+Y/+Z → 手背 '+ '/'.join(mapping),mapping)
        form.addRow('开手时安装轴映射（近似）',self.mount)
        self.label=QtWidgets.QComboBox();self.label.addItems(['拇指绑法','四指绑法'])
        form.addRow('记录实验标签',self.label)
        self.delay=QtWidgets.QDoubleSpinBox();self.delay.setRange(0,30);self.delay.setValue(3);self.delay.setSuffix(' 秒')
        self.hold=QtWidgets.QDoubleSpinBox();self.hold.setRange(.5,10);self.hold.setValue(1.5);self.hold.setSuffix(' 秒')
        form.addRow('采样准备倒计时',self.delay);form.addRow('端点静止采样',self.hold)
        buttons=QtWidgets.QGridLayout();layout.addLayout(buttons)
        self.open_button=QtWidgets.QPushButton('倒计时采集开手参考');self.closed_button=QtWidgets.QPushButton('倒计时采集闭合参考')
        self.cancel=QtWidgets.QPushButton('取消采样');self.reset=QtWidgets.QPushButton('清除手势参考（不重置积分器）')
        for index,b in enumerate((self.open_button,self.closed_button,self.cancel,self.reset)):
            buttons.addWidget(b,index//2,index%2)
        self.open_button.clicked.connect(lambda:self.start_capture('open'));self.closed_button.clicked.connect(lambda:self.start_capture('closed'))
        self.cancel.clicked.connect(self.cancel_capture);self.reset.clicked.connect(self.invalidate)
        self.status=QtWidgets.QLabel('等待设备数据；发现后自动列出，不自动分配。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.devices=QtWidgets.QLabel();self.devices.setWordWrap(True);layout.addWidget(self.devices)
        self.metric_label=QtWidgets.QLabel('尚未采集开合参考');self.metric_label.setWordWrap(True);self.metric_label.setStyleSheet('font-size:18px;');layout.addWidget(self.metric_label)
        self.bar=QtWidgets.QProgressBar();self.bar.setRange(0,1000);self.bar.setFormat('平滑闭合量 %p%（非夹紧力）');layout.addWidget(self.bar)
        self.warn_only=QtWidgets.QCheckBox('实验：离轴残差仅提示，不阻止更新')
        self.warn_only.setToolTip('勾选后，残差超过20°仍更新闭合量；不绕过断流、同期数据或旋转分支检查。')
        self.warn_only.toggled.connect(self.residual_policy_changed);layout.addWidget(self.warn_only)
        thresholds=QtWidgets.QFormLayout();layout.addLayout(thresholds)
        self.open_threshold=QtWidgets.QDoubleSpinBox();self.closed_threshold=QtWidgets.QDoubleSpinBox()
        for spin,value in ((self.open_threshold,30),(self.closed_threshold,70)):
            spin.setRange(0,100);spin.setValue(value);spin.setSuffix(' %')
        self.dwell=QtWidgets.QDoubleSpinBox();self.dwell.setRange(0,2);self.dwell.setSingleStep(.05);self.dwell.setValue(.15);self.dwell.setSuffix(' 秒')
        thresholds.addRow('打开阈值（≤，默认30%）',self.open_threshold)
        thresholds.addRow('闭合阈值（≥，默认70%）',self.closed_threshold)
        thresholds.addRow('越过阈值持续确认时间',self.dwell)
        self.state_label=QtWidgets.QLabel('离散状态：待判定；两阈值之间保持状态');layout.addWidget(self.state_label)
        output=QtWidgets.QHBoxLayout();layout.addLayout(output)
        self.send_preview=QtWidgets.QCheckBox('发送离散开合至 RViz（不控制真机）');output.addWidget(self.send_preview)
        self.target_joint=QtWidgets.QComboBox();self.target_joint.addItem('右夹爪 · 57','57');self.target_joint.addItem('左夹爪 · 7','7');output.addWidget(self.target_joint)
        if panel_joint is not None:
            self.target_joint.setCurrentIndex(self.target_joint.findData(panel_joint))
            self.target_joint.setEnabled(False)
        self.reverse=QtWidgets.QCheckBox('反向映射');output.addWidget(self.reverse)
        self.preview_status=QtWidgets.QLabel('RViz输出关闭；请勿同时启用手机网页夹爪输出。');layout.addWidget(self.preview_status)
        if self.embedded and panel_joint is None:
            bridge=QtWidgets.QPushButton('打开 X1 预览配置…（启动 ROS 接收 / RViz 模型）')
            bridge.clicked.connect(parent._show_x1_preview);layout.addWidget(bridge)
        for spin in (self.open_threshold,self.closed_threshold,self.dwell):spin.valueChanged.connect(self.reset_discrete)
        self.target_joint.currentIndexChanged.connect(lambda *_:self.send_preview.setChecked(False))
        self.reverse.toggled.connect(lambda *_:self.send_preview.setChecked(False))
        self.send_preview.toggled.connect(self.preview_toggled)
        self.plot=TracePlot()
        layout.addWidget(self.plot,1)
        self.record=QtWidgets.QPushButton('开始轻量诊断记录（无图像）');self.record.clicked.connect(self.toggle_record);layout.addWidget(self.record)
        self.config_widgets=(self.hand,self.finger,self.mode,self.mount,self.label,self.delay,self.hold,
                             self.open_threshold,self.closed_threshold,self.dwell)
        for combo in (self.hand,self.finger,self.mode,self.mount):combo.currentIndexChanged.connect(self.invalidate)
        self.start=time.monotonic();self.last_flush=self.start
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(20)

    def invalidate(self,*_):
        self.reset_discrete()
        self.calibration=None;self.capture=None;self.filtered=None;self.last_filter_stamp=None
        self.buffer=PairBuffer()
        self.status.setText('手势参考已清除；请重新采集开手与闭合。');self.metric_label.setText('未标定');self.bar.setValue(0)
        self.plot_rows.clear();self.plot.set_rows([])
        self.log_event('reference_cleared');self.controls()

    def reset_discrete(self,*_):
        self.send_preview.setChecked(False)
        try:
            self.discrete=DiscreteClosure(self.open_threshold.value()/100,self.closed_threshold.value()/100,self.dwell.value())
            self.state_label.setText('离散状态：待判定；两阈值之间保持，必须重新越过阈值确认')
        except ValueError as exc:
            self.discrete=None;self.state_label.setText(str(exc))

    def preview_toggled(self, enabled):
        self.last_ack=None
        self.preview_status.setText('等待有效离散状态 / ROS 接收反馈；请停用手机输出。' if enabled else 'RViz输出关闭；机器人保持末次目标。')

    def update_preview(self, now, allowed):
        while True:
            try:
                raw,peer=self.preview_socket.recvfrom(65536)
            except BlockingIOError:break
            try:
                ack=json.loads(raw)
                stamp=ack.get('sent_mono_ns')
                if peer==('127.0.0.1',13991) and ack.get('type')=='gripper_status' and type(stamp) is int and 0 <= time.monotonic_ns()-stamp <= 500_000_000:
                    joint=ack['joints'][self.target_joint.currentData()]
                    self.last_ack=now
                    self.preview_status.setText(f"ROS已收到 · 关节 {self.target_joint.currentData()} · 目标 {joint['target']:.2f} rad · 当前 {joint['position']:.2f} rad（仅预览）")
            except (ValueError,KeyError,TypeError):pass
        if not self.send_preview.isChecked():return
        if not allowed or self.discrete is None or self.discrete.state is None:
            self.preview_status.setText('停止发送：数据无效／采样中／离散状态待判定；保持末次位置')
            return
        if now-self.last_send >= .05:
            self.preview_seq+=1;self.last_send=now
            # Preserve the phone preview convention: open=1 rad, closed=0 rad.
            position=float(self.discrete.state) if self.reverse.isChecked() else float(not self.discrete.state)
            frame=dict(version=1,type='gripper',seq=self.preview_seq,sent_mono_ns=time.monotonic_ns(),joint=self.target_joint.currentData(),position=position)
            self.preview_socket.sendto(json.dumps(frame).encode(),('127.0.0.1',13991))
        if self.last_ack is None or now-self.last_ack>1:
            self.preview_status.setText('已发送，但尚无 ROS 接收确认：请启动/更新 ROS 桥，并停用其他夹爪输入')

    def controls(self):
        for widget in self.config_widgets:widget.setEnabled(self.capture is None and self.record_file is None)
        self.open_button.setEnabled(self.capture is None)
        self.closed_button.setEnabled(self.capture is None and self.calibration is not None)
        self.cancel.setEnabled(self.capture is not None);self.reset.setEnabled(self.capture is None)

    def log_event(self, kind, **fields):
        if self.record_file:
            self.record_file.write(json.dumps(dict(kind=kind,host_mono_ns=time.monotonic_ns(),**fields),ensure_ascii=False,allow_nan=False)+'\n')

    def residual_policy_changed(self, enabled):
        try:self.log_event('residual_policy',warn_only=enabled,threshold_deg=20.)
        except OSError as exc:
            self.stop_record();self.status.setText('记录错误：'+str(exc))

    def selected_pair(self, now):
        if self.role_validator is not None:self.role_validator()
        hand,finger=self.hand.currentData(),self.finger.currentData()
        if not hand or not finger:raise ValueError('请先选择手背设备和拇指／四指设备')
        if hand==finger:raise ValueError('手背和指段必须选择两台不同设备')
        if self.mode.currentData()==GYRO_TILT:
            for device in (hand,finger):
                if not self.bias_ready(device):raise ValueError('等待两设备静止零偏学习：支撑手保持不动，不能缓慢转动')
        return self.buffer.pair(hand,finger,now)

    def bias_ready(self, device):
        if device in self.host_bias_ready:return self.host_bias_ready[device]
        estimator=self.engine._tilt_filters.get(device)
        return estimator is not None and estimator.bias_ready

    def start_capture(self, kind):
        self.send_preview.setChecked(False)
        try:
            self.selected_pair(time.monotonic())
            if kind=='closed' and self.calibration is None:raise ValueError('先采集开手参考')
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self,'不能开始采样',str(exc));return
        self.capture=dict(kind=kind,state='prepare',deadline=time.monotonic()+self.delay.value(),rows=[],stamp=None)
        self.controls()

    def cancel_capture(self):
        self.capture=None;self.status.setText('已取消采样，旧参考保持。');self.controls()

    def update_capture(self, pair, now):
        c=self.capture
        if c is None:return
        if now>=c['deadline']:
            if c['state']=='prepare':c['state']='hold';c['deadline']=now+self.hold.value()
            else:
                try:
                    h=static_mean([r[0] for r in c['rows']]);f=static_mean([r[1] for r in c['rows']])
                    if c['kind']=='open':self.calibration=GestureCalibration(h,f,mapping_matrix(self.mount.currentData()))
                    else:self.calibration.set_closed(h,f)
                    self.filtered=None;self.last_filter_stamp=None
                    self.reset_discrete()
                    self.log_event('calibration',calibration=self.calibration.snapshot(),endpoint=c['kind'])
                    self.status.setText('参考采集成功；请测试保持指型、整体转腕时是否出现串扰。')
                except ValueError as exc:self.status.setText('采集未通过，旧参考保留：'+str(exc))
                self.capture=None;self.controls();return
        if c['state']=='hold' and pair[0]!=c['stamp']:
            c['rows'].append(pair[1:]);c['stamp']=pair[0]
        name='开手／展开' if c['kind']=='open' else '舒适闭合'
        self.status.setText(f"{name}：{'准备' if c['state']=='prepare' else '保持静止采样'} {max(0,c['deadline']-now):.1f}秒 · {len(c['rows'])}组同期样本")

    def tick(self):
        now=time.monotonic()
        output_allowed=False
        try:
            rows=(self.source.poll_observed() if hasattr(self.source,'poll_observed') else
                  [(ns,packet,None) for ns,packet in self.source.poll_timed()])
            for ns,packet,observation in rows:
                device=packet.device_id;stamp=ns/1e9
                last=self.engine.received_at.get(device)
                previous=self.engine.packets.get(device)
                packet_time=lambda p:p.hour*3600+p.minute*60+p.second+p.millisecond/1000.
                clock_jump=False
                if previous is not None:
                    delta=packet_time(packet)-packet_time(previous)
                    if delta < -43200:delta+=86400
                    clock_jump=delta < -.1 or delta > 1.
                epoch_changed=(observation is not None and device in self.engine.epochs and
                               self.engine.epochs[device]!=observation['epoch'])
                if last is not None and (stamp-last>.5 or clock_jump or epoch_changed):
                    self.engine.invalidate_neutral(device);self.buffer.rows.pop(device,None)
                    if device in (self.hand.currentData(),self.finger.currentData()):self.invalidate()
                if observation is None:
                    self.host_bias_ready.pop(device,None)
                    self.engine.feed(packet,received_at=stamp)
                else:
                    # Read-only per-packet snapshot. Never integrate a second time,
                    # nor use the host's *latest* pose for older queued timestamps.
                    self.engine.packets[device]=packet;self.engine.received_at[device]=stamp
                    self.engine.epochs[device]=observation['epoch']
                    self.engine.mode_current[device]={mode:r.copy() for mode,r in observation['rotations'].items()}
                    self.host_bias_ready[device]=observation['bias_ready']
                rotation=self.engine.current_for(device,self.mode.currentData())
                if rotation is None:
                    self.buffer.rows.pop(device,None)
                    continue
                self.buffer.append(device,stamp,rotation)
                self.log_event('imu',device_id=device,received_mono_ns=ns,packet=asdict(packet) if is_dataclass(packet) else vars(packet),
                    epoch=self.engine.epochs[device],rotation=rotation.tolist(),
                    pose_source='host' if observation is not None else 'independent')
                for combo in (self.hand,self.finger):
                    if combo.findData(device)<0:combo.addItem(device,device)
            summaries=[]
            for device,stamp in self.engine.received_at.items():
                p=self.engine.packets[device]
                ready=self.bias_ready(device)
                learning=('复用主窗口：零偏就绪' if ready else '主窗口静止学习中') if device in self.host_bias_ready else ('零偏就绪' if ready else '静止学习中')
                summaries.append(f"{device} · {max(0,now-stamp)*1000:.0f}ms · 电量 {getattr(p,'battery_percent','?')}% · "+learning)
            self.devices.setText('；'.join(summaries))
            pair=self.selected_pair(now)
            self.update_capture(pair,now)
            if self.calibration:
                result=self.calibration.evaluate(*pair[1:]);angle=result.angle_deg;residual=result.residual_deg
                self.metric_label.setText(f'手势角 {angle:+.1f}° · 离轴残差 {residual:.1f}°'+
                    (' · 尚未采闭合端点' if result.closure is None else f' · 原始闭合量 {result.closure*100:.1f}%'))
                valid=residual<=20.
                update_allowed=valid or self.warn_only.isChecked()
                if result.closure is not None and update_allowed and pair[0]!=self.last_filter_stamp:
                    dt=0. if self.last_filter_stamp is None else pair[0]-self.last_filter_stamp
                    alpha=1. if self.filtered is None else 1-math.exp(-dt/.12)
                    self.filtered=result.closure if self.filtered is None else self.filtered+alpha*(result.closure-self.filtered)
                    self.last_filter_stamp=pair[0];self.bar.setValue(round(self.filtered*1000))
                    if self.discrete is not None and self.capture is None:
                        state=self.discrete.update(self.filtered,pair[0])
                        self.state_label.setText('离散状态：'+('待判定' if state is None else '闭合（100%）' if state else '打开（0%）')+' · 阈值之间保持状态')
                output_allowed=result.closure is not None and update_allowed and self.capture is None and self.discrete is not None
                if not output_allowed and self.discrete is not None:self.discrete.pause()
                if self.capture is None:
                    warning=('实验模式，仅提示，闭合量继续更新。' if self.warn_only.isChecked() else '平滑闭合量保持。')
                    self.status.setText('离轴残差>20°：'+warning+'请检查非标定动作／绑带／漂移。' if not valid else '观测中；双阈值离散判断，不控制真实机器人。')
                if pair[0]!=self.last_plot_stamp:
                    self.plot_rows.append((pair[0]-self.start,angle,residual));self.last_plot_stamp=pair[0]
                    self.plot.set_rows(self.plot_rows)
                if pair[0]!=self.last_log_stamp:
                    self.log_event('gesture',pair_mono_ns=round(pair[0]*1e9),angle_deg=angle,residual_deg=residual,
                        closure=result.closure,filtered_closure=self.filtered,valid=valid,
                        residual_warn_only=self.warn_only.isChecked(),update_allowed=update_allowed,
                        discrete_state=None if self.discrete is None else self.discrete.state)
                    self.last_log_stamp=pair[0]
            if self.record_file and now-self.last_flush>1:self.record_file.flush();self.last_flush=now
        except ValueError as exc:
            if self.discrete is not None:self.discrete.pause()
            self.status.setText(str(exc))
            if self.capture:self.cancel_capture();self.status.setText('采样取消：'+str(exc))
        except OSError as exc:
            output_allowed=False
            if self.discrete is not None:self.discrete.pause()
            self.stop_record();self.status.setText('接收／记录错误：'+str(exc))
        try:self.update_preview(now,output_allowed)
        except OSError as exc:self.send_preview.setChecked(False);self.preview_status.setText('预览发送错误：'+str(exc))
        self.controls()

    def toggle_record(self):
        if self.record_file:self.stop_record();return
        try:
            self.selected_pair(time.monotonic())
            path=self.output_root/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            path.mkdir(parents=True);self.record_file=(path/'diagnostics.jsonl').open('x',encoding='utf-8');self.record_path=path
            self.log_event('session',format='imu-gripper-observer-v1',hand=self.hand.currentData(),finger=self.finger.currentData(),
                mode=self.mode.currentData(),label=self.label.currentText(),installation=list(self.mount.currentData()),joint=self.target_joint.currentData(),
                calibration=None if self.calibration is None else self.calibration.snapshot(),timebase='host_monotonic; IMU integration uses device packet time',
                simulation=isinstance(self.source,DemoSource) or getattr(self.source,'simulation',False))
            self.log_event('residual_policy',warn_only=self.warn_only.isChecked(),threshold_deg=20.)
            self.log_event('discrete_policy',open_threshold=self.open_threshold.value()/100,closed_threshold=self.closed_threshold.value()/100,dwell_s=self.dwell.value())
            self.record.setText('停止记录：'+str(path));self.controls()
        except (ValueError,OSError) as exc:
            self.stop_record();QtWidgets.QMessageBox.warning(self,'不能记录',str(exc))

    def stop_record(self):
        if self.record_file:
            try:self.record_file.close()
            except OSError:pass
        self.record_file=None;self.record.setText('开始轻量诊断记录（无图像）')
        if self.record_path:self.status.setText('记录已保存：'+str(self.record_path))
        self.controls()

    def closeEvent(self,event):
        self.send_preview.setChecked(False);self.stop_record()
        if self.embedded:
            self.hide();event.ignore()
        else:
            self.shutdown();event.accept()

    def shutdown(self):
        self.send_preview.setChecked(False);self.timer.stop();self.stop_record();self.source.close();self.preview_socket.close()


class BilateralDemoSource(DemoSource):
    """Four synthetic device IDs for exercising both panels, not real validation."""
    def poll_timed(self):
        rows=super().poll_timed()
        return [(stamp,SimpleNamespace(**{**vars(packet),'device_id':side+'-'+packet.device_id}))
                for side in ('左','右') for stamp,packet in rows]


class BilateralGestureWindow(QtWidgets.QWidget):
    """One upstream receiver, two independent gesture/state/output channels."""
    def __init__(self, source, output_root, parent=None):
        super().__init__(parent,QtCore.Qt.WindowType.Window)
        self.source=source;self.embedded=parent is not None;self._closed=False
        self.setWindowTitle('左右手四 IMU 夹爪 · 独立离散开合（仅 RViz）')
        self.resize(1450,940)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('左、右各选手背＋指段，共四台不同设备；每侧独立采集开手/闭合、设置阈值及启用输出。\n'
            +('只读复用主窗口姿态及静止学习，不重复积分。' if self.embedded else '独占 UDP1399，请关闭其他人体/手腕 GUI。')+
            '左侧固定 RViz 关节7，右侧固定57；只做预览，不控制真机，仍不消除长期漂移。')
        note.setWordWrap(True);layout.addWidget(note)
        self.assignment_status=QtWidgets.QLabel('请选择设备；可先测试单侧。');layout.addWidget(self.assignment_status)
        top=QtWidgets.QHBoxLayout();layout.addLayout(top)
        if self.embedded:
            bridge=QtWidgets.QPushButton('打开 X1 预览配置…（启动 ROS 接收 / RViz）')
            bridge.clicked.connect(parent._show_x1_preview);top.addWidget(bridge)
        stop=QtWidgets.QPushButton('停用左右夹爪输出（保持末次位置）');stop.clicked.connect(self.stop_outputs);top.addWidget(stop)
        split=QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal);layout.addWidget(split,1)
        self.panels={}
        for side,title,joint in (('left','左手 → RViz 左夹爪 · 7','7'),('right','右手 → RViz 右夹爪 · 57','57')):
            group=QtWidgets.QGroupBox(title);column=QtWidgets.QVBoxLayout(group)
            queue=SharedGestureSource();queue.simulation=isinstance(source,DemoSource)
            panel=GestureWindow(queue,Path(output_root)/side,self,panel_joint=joint)
            panel.setWindowFlags(QtCore.Qt.WindowType.Widget);panel.timer.stop()
            panel.role_validator=lambda panel=panel:self.validate_roles(panel)
            panel.plot.setMinimumHeight(150)
            column.addWidget(panel);split.addWidget(group);self.panels[side]=panel
        split.setSizes([700,700])
        for panel in self.panels.values():
            for combo in (panel.hand,panel.finger):combo.currentIndexChanged.connect(self.assignments_changed)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(20)

    def validate_roles(self, active_panel=None):
        ids=[combo.currentData() for panel in self.panels.values() for combo in (panel.hand,panel.finger) if combo.currentData()]
        if len(ids)!=len(set(ids)):
            raise ValueError('设备分配冲突：四个角色不能共用同一台 IMU，请重新选择')
        if self.embedded and active_panel is not None:
            host=self.parentWidget().engine
            for device in (active_panel.hand.currentData(),active_panel.finger.currentData()):
                if device in active_panel.host_bias_ready and active_panel.engine.epochs.get(device)!=host.epochs.get(device):
                    raise ValueError('主窗口积分参考系已改变，等待新数据并重新采集开合参考')

    def feed_host_packet(self, stamp, packet, engine):
        device=packet.device_id;estimator=engine._tilt_filters.get(device)
        observation=dict(epoch=engine.epochs.get(device),
            bias_ready=bool(estimator is not None and estimator.bias_ready),
            rotations={mode:r.copy() for mode in (VENDOR_RPY,GYRO_TILT)
                       if (r:=engine.current_for(device,mode)) is not None})
        self.source.feed(stamp,packet,observation)

    def assignments_changed(self,*_):
        try:
            self.validate_roles()
            self.assignment_status.setText('设备分配无冲突；每侧选择两台不同设备后采集开合参考。')
        except ValueError as exc:
            self.assignment_status.setText(str(exc));self.stop_outputs()

    def stop_outputs(self):
        for panel in self.panels.values():panel.send_preview.setChecked(False)

    def tick(self):
        if self._closed:return
        try:rows=(self.source.poll_observed() if hasattr(self.source,'poll_observed') else
                  [(stamp,packet,None) for stamp,packet in self.source.poll_timed()])
        except (OSError,ValueError) as exc:
            self.stop_outputs();self.assignment_status.setText('接收失败：'+str(exc));return
        for panel in self.panels.values():
            for stamp,packet,observation in rows:panel.source.feed(stamp,packet,observation)
            panel.tick()

    def closeEvent(self,event):
        self.stop_outputs()
        for panel in self.panels.values():panel.stop_record()
        if self.embedded:self.hide();event.ignore()
        else:self.shutdown();event.accept()

    def shutdown(self):
        if self._closed:return
        self._closed=True;self.timer.stop();self.stop_outputs()
        for panel in self.panels.values():panel.shutdown()
        self.source.close()


def main():
    parser=argparse.ArgumentParser(description='双IMU夹爪离散开合，可选RViz预览，不控制真实机器人')
    parser.add_argument('--listen-ip',default='0.0.0.0');parser.add_argument('--listen-port',type=int,default=1399)
    parser.add_argument('--demo',action='store_true',help='合成数据，无UDP／硬件')
    parser.add_argument('--single',action='store_true',help='旧版单侧配置窗口')
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'recordings/imu_gripper')
    args=parser.parse_args();app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    try:source=(DemoSource() if args.single else BilateralDemoSource()) if args.demo else GuiUdpSource(port=args.listen_port,host=args.listen_ip)
    except OSError as exc:
        QtWidgets.QMessageBox.critical(None,'无法监听UDP',f'{exc}\n请关闭占用UDP1399的人体／手腕GUI。');return 1
    window=(GestureWindow if args.single else BilateralGestureWindow)(source,args.output_root);window.show();return app.exec()
