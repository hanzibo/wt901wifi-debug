"""Editable hand endpoints. No direct simulator API calls on the GUI thread."""
import json
import time
from pathlib import Path
import numpy as np
from PySide6 import QtCore,QtWidgets
from .scene_hand import hand_pose


class HandPoseDialog(QtWidgets.QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.profiles={};self.widgets={};self.preview={}
        self.setWindowTitle('新机器人 · 左右手指张开 / 闭合配置（仅仿真）');self.resize(1000,700)
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('每侧五条手指链 × 三个关节；按场景 jointLH/RH 编号显示，不猜测手指名称。\n'
            '先连接场景，再配置并确认两组端点。全部 90° 不代表合理握拳；无碰撞或抓取力保护。\n'
            '编辑或采集会停用该侧输出。预览与 IMU 控制互斥；停止保持当前位置。')
        note.setWordWrap(True);layout.addWidget(note)
        tabs=QtWidgets.QTabWidget();layout.addWidget(tabs)
        for side,label in (('left','左手'),('right','右手')):
            page=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(page);tabs.addTab(page,label)
            table=QtWidgets.QTableWidget(15,4);table.setHorizontalHeaderLabels(['场景关节','张开 °','闭合 °','场景回读 °'])
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            spins=[]
            for i in range(15):
                name='joint'+('LH' if side=='left' else 'RH')+str(i//3+1)+str(i%3+1)
                item=QtWidgets.QTableWidgetItem(name);item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable);table.setItem(i,0,item)
                pair=[]
                for col in (1,2):
                    spin=QtWidgets.QDoubleSpinBox();spin.setRange(0,90);spin.setDecimals(2)
                    spin.valueChanged.connect(lambda _,s=side:self.edited(s))
                    table.setCellWidget(i,col,spin);pair.append(spin)
                spins.append(pair)
                actual=QtWidgets.QTableWidgetItem('—');actual.setFlags(actual.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable);table.setItem(i,3,actual)
            box.addWidget(table)
            speed=QtWidgets.QSpinBox();speed.setRange(5,720);speed.setValue(120);speed.setSuffix(' °/s')
            speed.valueChanged.connect(lambda _,s=side:self.stop(s))
            box.addWidget(QtWidgets.QLabel('手指过渡速度（该侧十五关节共用；独立于肩肘和腕部）'));box.addWidget(speed)
            for key,title in (('open','张开'),('closed','闭合')):
                row=QtWidgets.QHBoxLayout();box.addLayout(row)
                for text,fn in ((f'读取当前场景为{title}',lambda _,s=side,k=key:self.capture(s,k)),
                                (f'确认表格{title}端点',lambda _,s=side,k=key:self.confirm(s,k)),
                                (f'预览{title}',lambda _,s=side,k=key:self.begin_preview(s,k))):
                    button=QtWidgets.QPushButton(text);button.clicked.connect(fn);row.addWidget(button)
            status=QtWidgets.QLabel('尚未连接 / 未确认端点');status.setWordWrap(True);box.addWidget(status)
            self.widgets[side]=(spins,speed,status)
            setattr(self,side+'_table',table)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        for text,fn in (('停止所有手指输出',self.stop_all),('保存手型 JSON',self.save),('加载手型 JSON',self.load)):
            b=QtWidgets.QPushButton(text);b.clicked.connect(fn);row.addWidget(b)
        for source,target,text in (('left','right','复制左手配置到右手'),('right','left','复制右手配置到左手')):
            b=QtWidgets.QPushButton(text);b.clicked.connect(lambda _,s=source,t=target:self.copy_side(s,t));row.addWidget(b)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)

    def stop(self,side):
        self.preview.pop(side,None);self.owner.hand_mailbox.stop(side)
        self.owner.revoke_hand_input(side)

    def stop_all(self):
        for side in self.widgets:self.stop(side)

    def edited(self,side):
        self.stop(side)
        if side in self.profiles:self.profiles[side]['confirmed']={'open':False,'closed':False}
        self.widgets[side][2].setText('已编辑；请重新确认张开和闭合端点')

    def limits(self,side):
        if not self.owner.connected or self.owner.config is None:raise ValueError('请先连接 CoppeliaSim 场景')
        return self.owner.config['scene']['hands'][side]['limits']

    def values(self,side,key):
        col=0 if key=='open' else 1
        return hand_pose(np.deg2rad([p[col].value() for p in self.widgets[side][0]]),self.limits(side))

    def confirm(self,side,key):
        try:
            self.stop(side);q=self.values(side,key)
            p=self.profiles.setdefault(side,{'confirmed':{'open':False,'closed':False}})
            p[key]=q.tolist();p['confirmed'][key]=True
            self.widgets[side][2].setText('端点确认：'+str(p['confirmed']))
        except (ValueError,KeyError) as exc:self.widgets[side][2].setText(str(exc))

    def capture(self,side,key):
        try:
            self.limits(side)
            if time.monotonic()-self.owner.hand_readback_at>.5:raise ValueError('场景回读不新鲜，请等待连接恢复')
            q=hand_pose(self.owner.hand_angles[side],self.limits(side))
            self.stop(side)
            col=0 if key=='open' else 1
            for pair,v in zip(self.widgets[side][0],np.rad2deg(q)):
                blocker=QtCore.QSignalBlocker(pair[col]);pair[col].setValue(v);del blocker
            self.confirm(side,key)
        except (ValueError,KeyError) as exc:self.widgets[side][2].setText(str(exc))

    def endpoint(self,side,key,require_both=True):
        p=self.profiles.get(side,{})
        keys=('open','closed') if require_both else (key,)
        if not all(p.get('confirmed',{}).get(k,False) for k in keys):raise ValueError('请先配置并确认所需手型端点')
        for k in keys:
            if not np.allclose(self.values(side,k),p[k],atol=1e-8):raise ValueError('端点已改变，请重新确认')
        q=self.values(side,key)
        return q

    def begin_preview(self,side,key):
        try:
            q=self.endpoint(side,key,False);self.stop(side)
            self.preview[side]=q;self.tick()
            self.widgets[side][2].setText('手型预览中；可停止保持，不保存场景')
        except ValueError as exc:self.widgets[side][2].setText(str(exc))

    def tick(self):
        fresh=time.monotonic()-self.owner.hand_readback_at<=.5
        for side,q in self.owner.hand_angles.items():
            table=getattr(self,side+'_table')
            for i,v in enumerate(np.rad2deg(q)):table.item(i,3).setText(f'{v:.2f}' if fresh else '过期')
        if not self.owner.connected:
            if self.preview:self.stop_all()
            return
        for side,q in self.preview.items():
            self.owner.hand_mailbox.set(side,q,time.monotonic(),self.widgets[side][1].value(),'preview')

    def bind_scene(self):
        self.stop_all()
        for side,(spins,_,status) in self.widgets.items():
            for pair,limits in zip(spins,self.limits(side)):
                for spin in pair:
                    blocker=QtCore.QSignalBlocker(spin);spin.setRange(*np.rad2deg(limits));del blocker
            # A new scene connection requires explicit endpoint confirmation.
            if side in self.profiles:self.profiles[side]['confirmed']={'open':False,'closed':False}
            status.setText('已连接；请采集或确认张开 / 闭合端点')

    def save(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'保存手型','hand_poses.json','JSON (*.json)')
        if not path:return
        try:
            data={'version':1,'sides':{}}
            for side in self.profiles:
                if not all(self.profiles[side]['confirmed'].values()):continue
                data['sides'][side]={'open_deg':np.rad2deg(self.endpoint(side,'open')).tolist(),
                    'closed_deg':np.rad2deg(self.endpoint(side,'closed')).tolist(),
                    'paths':[o['path'] for o in self.owner.config['scene']['hands'][side]['objects']],
                    'speed_deg_s':self.widgets[side][1].value()}
            if not data['sides']:raise ValueError('至少确认一侧的两组端点后才能保存')
            Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        except (OSError,ValueError) as exc:QtWidgets.QMessageBox.warning(self,'保存失败',str(exc))

    def load(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'加载手型','','JSON (*.json)')
        if not path:return
        self.stop_all()
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if data['version']!=1:raise ValueError('不支持的手型格式')
            checked={}
            if not data['sides'] or not set(data['sides'])<=set(self.widgets):raise ValueError('无效手侧')
            for side in data['sides']:
                row=data['sides'][side];limits=self.limits(side)
                if row['paths']!=[o['path'] for o in self.owner.config['scene']['hands'][side]['objects']]:raise ValueError('手指对象路径不匹配')
                if not 5<=row['speed_deg_s']<=720:raise ValueError('无效手指速度')
                checked[side]=(hand_pose(np.deg2rad(row['open_deg']),limits),hand_pose(np.deg2rad(row['closed_deg']),limits),row['speed_deg_s'])
            for side,(opened,closed,speed) in checked.items():
                for pair,a,b in zip(self.widgets[side][0],np.rad2deg(opened),np.rad2deg(closed)):
                    pair[0].setValue(a);pair[1].setValue(b)
                self.widgets[side][1].setValue(speed)
                self.profiles[side]={'open':self.values(side,'open').tolist(),'closed':self.values(side,'closed').tolist(),
                                     'confirmed':{'open':False,'closed':False}}
                self.widgets[side][2].setText('已加载；请检查并确认两组端点，不自动输出')
        except (OSError,ValueError,KeyError,TypeError) as exc:QtWidgets.QMessageBox.warning(self,'加载失败',str(exc))

    def copy_side(self,source,target):
        try:
            opened=self.endpoint(source,'open');closed=self.endpoint(source,'closed')
            for q in (opened,closed):hand_pose(q,self.limits(target))
            self.stop(target)
            for pair,a,b in zip(self.widgets[target][0],np.rad2deg(opened),np.rad2deg(closed)):
                pair[0].setValue(a);pair[1].setValue(b)
            self.widgets[target][1].setValue(self.widgets[source][1].value())
            self.profiles[target]={'open':self.values(target,'open').tolist(),'closed':self.values(target,'closed').tolist(),
                                   'confirmed':{'open':False,'closed':False}}
            self.widgets[target][2].setText('已复制数值；左右安装轴可能不同，请检查并确认端点')
        except ValueError as exc:self.widgets[target][2].setText(str(exc))

    def closeEvent(self,event):
        self.stop_all();event.accept()
