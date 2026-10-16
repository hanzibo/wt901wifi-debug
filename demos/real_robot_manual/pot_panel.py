"""Read-only Nano input and explicitly armed finger output through the sole writer."""
import json
import os
import time
from dataclasses import asdict
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QGroupBox, QGridLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QCheckBox, QPushButton, QMessageBox, QFileDialog)
from .potentiometers import FingerMap, parse_a4
from .finger_live import FingerTrajectory
from .protocol import decode


def same_port(a, b):
    return bool(a and b and os.path.realpath(a) == os.path.realpath(b))


class PotPanel(QGroupBox):
    def __init__(self, owner):
        super().__init__('A4 单电位器 → 指定一根手指（默认仅预览）')
        self.owner=owner;self.connection=None;self.buffer=b'';self.latest=None
        self.received_at=0.;self.previous=None;self.count=0;self.gaps=0
        self.clock_ms=0;self.clock_offset=None;self.stop_requested=False
        layout=QGridLayout(self)
        self.port=QComboBox();self.port.setEditable(True)
        self.port.setEditText('/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0')
        self.refresh=QPushButton('刷新 Nano 串口');self.connect=QPushButton('连接 Nano（只读）')
        self.disconnect=QPushButton('断开 Nano');self.side=QComboBox();self.side.addItems(['左手','右手'])
        self.finger=QComboBox();self.finger.addItems([f'协议指{i+1}' for i in range(5)])
        bar=QHBoxLayout()
        for w in (self.port,self.refresh,self.connect,self.disconnect,self.side,self.finger):bar.addWidget(w)
        layout.addLayout(bar,0,0,1,8)
        self.info=QLabel('仅读取 A4；兼容已烧录 POT5 固件，不使用 A0–A3。不能同时打开串口监视器。')
        layout.addWidget(self.info,1,0,1,8)
        for col,name in enumerate(('许可','输入','ADC 开','ADC 闭','输出开','输出闭','当前 ADC','目标量')):
            layout.addWidget(QLabel(name),2,col)
        self.rows=[];self.controls=[self.side,self.finger]
        for i in range(1):
            enabled=QCheckBox('许可所选指');channel=QLabel('A4（固定）')
            spins=[]
            for maximum,value in ((1023,0),(1023,0),(4095,0),(4095,200)):
                spin=QSpinBox();spin.setRange(0,maximum);spin.setValue(value);spins.append(spin)
            raw=QLabel('—');target=QLabel('未标定')
            row=[enabled,channel,*spins,raw,target];self.rows.append(row)
            for col,w in enumerate(row):layout.addWidget(w,3+i,col)
            for w in [enabled,*spins]:
                self.controls.append(w)
                signal=w.toggled if isinstance(w,QCheckBox) else w.currentIndexChanged if isinstance(w,QComboBox) else w.valueChanged
                signal.connect(self.changed)
        buttons=QHBoxLayout();layout.addLayout(buttons,8,0,1,8)
        for text,fn in (('采集当前为张开',lambda:self.capture(2)),('采集当前为闭合',lambda:self.capture(3)),
                        ('保存标定',self.save),('加载标定',self.load)):
            b=QPushButton(text);b.clicked.connect(fn);buttons.addWidget(b);self.controls.append(b)
        self.start=QPushButton('明确启动 A4 → 所选指');self.stop=QPushButton('减速停止（不自动恢复）')
        layout.addWidget(self.start,9,0,1,4);layout.addWidget(self.stop,9,4,1,4)
        self.output=QLabel('其余通道冻结为最后确认／写出指令；RH/LH 指序需逐指验证。输出量不是角度。')
        self.output.setWordWrap(True);layout.addWidget(self.output,10,0,1,8)
        self.refresh.clicked.connect(self.enumerate_ports);self.connect.clicked.connect(self.open_input)
        self.disconnect.clicked.connect(self.close_input);self.start.clicked.connect(self.start_output)
        self.stop.clicked.connect(lambda:self.stop_output());self.side.currentIndexChanged.connect(self.changed)
        self.finger.currentIndexChanged.connect(self.changed)
        self.timer=QTimer(self);self.timer.setInterval(20);self.timer.timeout.connect(self.poll);self.timer.start()
        self.update_controls()

    def maps(self):
        r=self.rows[0]
        return [FingerMap(4,*(s.value() for s in r[2:6]),r[0].isChecked() and i==self.finger.currentIndex()) for i in range(5)]

    def target_values(self, maps):
        return [m.target(self.latest) for m in maps]

    def changed(self):
        if not self.owner.pot_active:self.owner.edited()

    def fresh(self):
        return self.latest is not None and time.monotonic()-self.received_at <= .5

    def capture(self,column):
        if not self.fresh():self.info.setText('采集失败：需要新鲜 Nano 数据');return
        self.rows[0][column].setValue(self.latest[4])

    def enumerate_ports(self):
        from serial.tools import list_ports
        current=self.port.currentText();self.port.clear()
        for p in list_ports.comports():self.port.addItem(p.device)
        self.port.setEditText(current)

    def open_input(self):
        if self.connection:return
        try:
            port=self.port.currentText().strip()
            if not port.startswith('/dev/') or '..' in port:raise ValueError('请选择 /dev/ 下的 Nano 端口')
            if self.owner.worker and same_port(port,self.owner.worker.port):raise ValueError('Nano 与机器人不能使用同一串口')
            from serial.tools import list_ports
            for p in list_ports.comports():
                if same_port(port,p.device) and p.vid==0x10c4:raise ValueError('这是 CP2102 机器人串口，不是 Nano')
            import serial
            self.connection=serial.Serial(port,115200,timeout=0,exclusive=True)
            self.buffer=b'';self.latest=None;self.previous=None;self.count=self.gaps=0
            self.clock_ms=0;self.clock_offset=None
            self.info.setText('已打开 Nano，等待启动及数据；未写入任何字节')
        except Exception as exc:self.info.setText('Nano 连接失败：'+str(exc))
        self.update_controls()

    def close_input(self):
        self.stop_output('Nano 已断开')
        if self.connection:self.connection.close()
        self.connection=None;self.latest=None;self.update_controls()

    def poll(self):
        newest=None
        try:
            if self.connection:
                n=self.connection.in_waiting
                if n>8192:raise RuntimeError('Nano 数据积压，停止并重新连接')
                self.buffer+=self.connection.read(n)
                if len(self.buffer)>16384:raise RuntimeError('Nano 输入行过长')
                newest=None
                while b'\n' in self.buffer:
                    line,self.buffer=self.buffer.split(b'\n',1);parsed=parse_a4(line)
                    if parsed is None:
                        if line.startswith(b'BOOT'):self.stop_output('Nano 重启，须人工重新启动')
                        continue
                    seq,uptime,values=parsed
                    if self.previous:
                        ds=(seq-self.previous[0]) & 0xffffffff;dt=(uptime-self.previous[1]) & 0xffffffff
                        if ds==0:continue
                        if ds>0x7fffffff or dt>0x7fffffff:
                            self.stop_output('Nano 序号／时钟回退，须重新启动')
                            self.clock_ms=0;self.clock_offset=None
                        else:
                            self.gaps+=max(0,ds-1);self.clock_ms+=dt
                    self.previous=(seq,uptime);self.count+=1;newest=parsed
                if newest:
                    self.latest=newest[2]
                    now=time.monotonic()
                    # Preserve board elapsed time across GUI stalls: buffered old packets
                    # must not refresh the output watchdog merely because read() ran now.
                    offset=now-self.clock_ms/1000
                    self.clock_offset=offset if self.clock_offset is None else min(offset,self.clock_offset)
                    self.received_at=min(now,self.clock_offset+self.clock_ms/1000)
                    if self.owner.worker:self.owner.worker.record('pot_input',sequence=newest[0],uptime_ms=newest[1],input_channel='A4',adc_a4=self.latest[4],received_monotonic=self.received_at,gaps=self.gaps)
            if self.latest:
                age=time.monotonic()-self.received_at
                self.info.setText(f'A4 ADC {self.latest[4]} · {self.count} 帧 · 丢序 {self.gaps} · 年龄 {age*1000:.0f} ms'+(' · 输入过期' if age>.5 else ''))
                self.rows[0][6].setText(str(self.latest[4]))
                try:
                    values=self.target_values(self.maps())
                    self.rows[0][7].setText(str(values[self.finger.currentIndex()]))
                    if (newest and self.owner.pot_active and not self.stop_requested and self.fresh()
                            and not self.owner.motion.stopping):
                        self.owner.worker.retarget_live({'values':values,'deadline':self.received_at+.5})
                except ValueError as exc:
                    if self.owner.pot_active:self.stop_output(str(exc))
                    else:
                        for r in self.rows:r[7].setText('未标定')
        except Exception as exc:
            self.close_input();self.info.setText('Nano 读取失败：'+str(exc))
        self.update_controls()

    def update_controls(self):
        active=self.owner.pot_active
        for w in self.controls:w.setEnabled(not active)
        self.port.setEnabled(self.connection is None);self.refresh.setEnabled(self.connection is None)
        self.connect.setEnabled(self.connection is None);self.disconnect.setEnabled(self.connection is not None)
        self.start.setEnabled(self.owner.connected and self.owner.origin is not None and
            self.owner.arm.isChecked() and self.owner.confirm.isChecked() and
            not self.owner.executing and not self.owner.live_mode.isChecked() and self.fresh())
        self.stop.setEnabled(active)

    def start_output(self):
        if not self.start.isEnabled():return
        try:
            owner=self.owner;maps=self.maps();values=self.target_values(maps)
            if same_port(self.port.currentText(),owner.worker.port):raise ValueError('两个串口相同')
            motion=FingerTrajectory(owner.origin,'left' if self.side.currentIndex()==0 else 'right',maps,
                owner.finger_speed.value(),owner.finger_acceleration.value(),owner.frequency.value())
            if QMessageBox.warning(self,'五指真机发送确认',
                f'仅控制{self.side.currentText()}{self.finger.currentText()}，输入固定 A4；其他通道保持最后指令。\n'
                '输出量不是角度，0…200 也不是已验证安全范围。\n'
                '确认指序、运动空间和断电手段后，是否开始？',
                QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
            if not self.fresh():raise ValueError('确认期间数据过期，请重试')
            values=self.target_values(maps)
            motion.set_target({'values':values,'deadline':self.received_at+.5})
            self.stop_requested=False
            owner.motion=motion;owner.sent_steps=0;owner.pot_active=owner.executing=True
            owner.permissions();owner.worker.record('pot_mapping',side=motion.side,maps=[asdict(m) for m in maps])
            owner.worker.submit_motion(motion)
        except Exception as exc:
            if self.owner.pot_active:self.owner.on_failed(str(exc))
            QMessageBox.warning(self,'无法启动',str(exc))

    def stop_output(self,reason='用户停止电位器随动'):
        if self.owner.pot_active:
            self.stop_requested=True
            if self.owner.motion.stopping:return
            try:self.owner.worker.retarget_live({'stop':True,'reason':reason})
            except Exception as exc:self.owner.on_failed(str(exc))

    def save(self):
        try:
            maps=self.maps()
            path,_=QFileDialog.getSaveFileName(self,'保存标定','','JSON (*.json)')
            if path:
                with open(path,'w',encoding='utf-8') as f:json.dump({'version':2,'side':self.side.currentIndex(),'finger':self.finger.currentIndex(),'mapping':asdict(maps[self.finger.currentIndex()])},f,ensure_ascii=False,indent=2)
        except Exception as exc:QMessageBox.warning(self,'保存失败',str(exc))

    def load(self):
        try:
            path,_=QFileDialog.getOpenFileName(self,'加载标定','','JSON (*.json)')
            if not path:return
            with open(path,encoding='utf-8') as f:data=json.load(f)
            if data['version']!=2 or type(data['side']) is not int or data['side'] not in (0,1):raise ValueError('需要 A4 单通道标定（版本2），旧五路配置请重新标定')
            if type(data['finger']) is not int or not 0<=data['finger']<5:raise ValueError('目标手指无效')
            m=FingerMap(**data['mapping'])
            if m.channel!=4:raise ValueError('单通道输入必须为 A4')
            self.side.setCurrentIndex(data['side'])
            self.finger.setCurrentIndex(data['finger'])
            for r in self.rows:
                r[0].setChecked(False)
                for w,v in zip(r[2:6],(m.opened,m.closed,m.output_open,m.output_closed)):w.setValue(v)
            self.changed() # Never restore output permission from disk.
        except Exception as exc:QMessageBox.warning(self,'加载失败',str(exc))
