"""Device-discovering 2D observer for thumb/finger gesture experiments."""
import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from itertools import permutations, product
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from wt901.gui_udp import GuiUdpSource
from wt901_viz.axis_mapping import mapping_matrix
from .gripper_gesture import PairBuffer, GestureCalibration, static_mean
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


class GestureWindow(QtWidgets.QWidget):
    def __init__(self, source, output_root):
        super().__init__()
        self.source=source;self.output_root=Path(output_root);self.engine=PoseEngine()
        self.buffer=PairBuffer();self.calibration=None;self.capture=None
        self.record_file=None;self.record_path=None;self.last_log_stamp=None
        self.plot_rows=deque(maxlen=1200);self.last_plot_stamp=None;self.filtered=None;self.last_filter_stamp=None
        self.setWindowTitle('双 IMU 夹爪手势观察 · 拇指／四指（不控制机器人）');self.resize(1040,740)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('独占UDP1399，请关闭人体／手腕GUI。支撑手等待静止学习；先采开手，再采舒适闭合。\n'
            '默认开手时两设备外壳XYZ同向；否则调整下方安装映射。相对漂移不会自动抵消；不写设备、不连接X1。')
        note.setWordWrap(True);layout.addWidget(note)
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
        buttons=QtWidgets.QHBoxLayout();layout.addLayout(buttons)
        self.open_button=QtWidgets.QPushButton('倒计时采集开手参考');self.closed_button=QtWidgets.QPushButton('倒计时采集闭合参考')
        self.cancel=QtWidgets.QPushButton('取消采样');self.reset=QtWidgets.QPushButton('清除手势参考（不重置积分器）')
        for b in (self.open_button,self.closed_button,self.cancel,self.reset):buttons.addWidget(b)
        self.open_button.clicked.connect(lambda:self.start_capture('open'));self.closed_button.clicked.connect(lambda:self.start_capture('closed'))
        self.cancel.clicked.connect(self.cancel_capture);self.reset.clicked.connect(self.invalidate)
        self.status=QtWidgets.QLabel('等待设备数据；发现后自动列出，不自动分配。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.devices=QtWidgets.QLabel();self.devices.setWordWrap(True);layout.addWidget(self.devices)
        self.metric_label=QtWidgets.QLabel('尚未采集开合参考');self.metric_label.setStyleSheet('font-size:20px;');layout.addWidget(self.metric_label)
        self.bar=QtWidgets.QProgressBar();self.bar.setRange(0,1000);self.bar.setFormat('平滑闭合量 %p%（非夹紧力）');layout.addWidget(self.bar)
        self.warn_only=QtWidgets.QCheckBox('实验：离轴残差仅提示，不阻止更新')
        self.warn_only.setToolTip('勾选后，残差超过20°仍更新闭合量；不绕过断流、同期数据或旋转分支检查。')
        self.warn_only.toggled.connect(self.residual_policy_changed);layout.addWidget(self.warn_only)
        self.plot=TracePlot()
        layout.addWidget(self.plot,1)
        self.record=QtWidgets.QPushButton('开始轻量诊断记录（无图像）');self.record.clicked.connect(self.toggle_record);layout.addWidget(self.record)
        self.config_widgets=(self.hand,self.finger,self.mode,self.mount,self.label,self.delay,self.hold)
        for combo in (self.hand,self.finger,self.mode,self.mount):combo.currentIndexChanged.connect(self.invalidate)
        self.start=time.monotonic();self.last_flush=self.start
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(20)

    def invalidate(self,*_):
        self.calibration=None;self.capture=None;self.filtered=None;self.last_filter_stamp=None
        self.buffer=PairBuffer()
        self.status.setText('手势参考已清除；请重新采集开手与闭合。');self.metric_label.setText('未标定');self.bar.setValue(0)
        self.plot_rows.clear();self.plot.set_rows([])
        self.log_event('reference_cleared');self.controls()

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
        hand,finger=self.hand.currentData(),self.finger.currentData()
        if self.mode.currentData()==GYRO_TILT:
            for device in (hand,finger):
                f=self.engine._tilt_filters.get(device)
                if f is None or not f.bias_ready:raise ValueError('等待两设备静止零偏学习：支撑手保持不动，不能缓慢转动')
        return self.buffer.pair(hand,finger,now)

    def start_capture(self, kind):
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
        try:
            for ns,packet in self.source.poll_timed():
                device=packet.device_id;stamp=ns/1e9
                last=self.engine.received_at.get(device)
                previous=self.engine.packets.get(device)
                packet_time=lambda p:p.hour*3600+p.minute*60+p.second+p.millisecond/1000.
                clock_jump=False
                if previous is not None:
                    delta=packet_time(packet)-packet_time(previous)
                    if delta < -43200:delta+=86400
                    clock_jump=delta < -.1 or delta > 1.
                if last is not None and (stamp-last>.5 or clock_jump):
                    self.engine.invalidate_neutral(device);self.buffer.rows.pop(device,None)
                    if device in (self.hand.currentData(),self.finger.currentData()):self.invalidate()
                self.engine.feed(packet,received_at=stamp)
                self.buffer.append(device,stamp,self.engine.current_for(device,self.mode.currentData()))
                self.log_event('imu',device_id=device,received_mono_ns=ns,packet=asdict(packet) if is_dataclass(packet) else vars(packet),
                    epoch=self.engine.epochs[device],rotation=self.engine.current_for(device,self.mode.currentData()).tolist())
                for combo in (self.hand,self.finger):
                    if combo.findData(device)<0:combo.addItem(device,device)
            summaries=[]
            for device,stamp in self.engine.received_at.items():
                p=self.engine.packets[device];f=self.engine._tilt_filters.get(device)
                summaries.append(f"{device} · {max(0,now-stamp)*1000:.0f}ms · 电量 {getattr(p,'battery_percent','?')}% · "+('零偏就绪' if f and f.bias_ready else '静止学习中'))
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
                if self.capture is None:
                    warning=('实验模式，仅提示，闭合量继续更新。' if self.warn_only.isChecked() else '平滑闭合量保持。')
                    self.status.setText('离轴残差>20°：'+warning+'请检查非标定动作／绑带／漂移。' if not valid else '观测中；只显示手势量，不控制机器人。')
                if pair[0]!=self.last_plot_stamp:
                    self.plot_rows.append((pair[0]-self.start,angle,residual));self.last_plot_stamp=pair[0]
                    self.plot.set_rows(self.plot_rows)
                if pair[0]!=self.last_log_stamp:
                    self.log_event('gesture',pair_mono_ns=round(pair[0]*1e9),angle_deg=angle,residual_deg=residual,
                        closure=result.closure,filtered_closure=self.filtered,valid=valid,
                        residual_warn_only=self.warn_only.isChecked(),update_allowed=update_allowed)
                    self.last_log_stamp=pair[0]
            if self.record_file and now-self.last_flush>1:self.record_file.flush();self.last_flush=now
        except ValueError as exc:
            self.status.setText(str(exc))
            if self.capture:self.cancel_capture();self.status.setText('采样取消：'+str(exc))
        except OSError as exc:
            self.stop_record();self.status.setText('接收／记录错误：'+str(exc))
        self.controls()

    def toggle_record(self):
        if self.record_file:self.stop_record();return
        try:
            self.selected_pair(time.monotonic())
            path=self.output_root/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            path.mkdir(parents=True);self.record_file=(path/'diagnostics.jsonl').open('x',encoding='utf-8');self.record_path=path
            self.log_event('session',format='imu-gripper-observer-v1',hand=self.hand.currentData(),finger=self.finger.currentData(),
                mode=self.mode.currentData(),label=self.label.currentText(),installation=list(self.mount.currentData()),
                calibration=None if self.calibration is None else self.calibration.snapshot(),timebase='host_monotonic; IMU integration uses device packet time',
                simulation=isinstance(self.source,DemoSource))
            self.log_event('residual_policy',warn_only=self.warn_only.isChecked(),threshold_deg=20.)
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
        self.timer.stop();self.stop_record();self.source.close();event.accept()


def main():
    parser=argparse.ArgumentParser(description='双IMU夹爪手势观察，不控制机器人')
    parser.add_argument('--listen-ip',default='0.0.0.0');parser.add_argument('--listen-port',type=int,default=1399)
    parser.add_argument('--demo',action='store_true',help='合成数据，无UDP／硬件')
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'recordings/imu_gripper')
    args=parser.parse_args();app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    try:source=DemoSource() if args.demo else GuiUdpSource(port=args.listen_port,host=args.listen_ip)
    except OSError as exc:
        QtWidgets.QMessageBox.critical(None,'无法监听UDP',f'{exc}\n请关闭占用UDP1399的人体／手腕GUI。');return 1
    window=GestureWindow(source,args.output_root);window.show();return app.exec()
