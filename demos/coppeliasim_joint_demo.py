"""Independent stopped-scene joint inspector. No IMU, ROS or hardware commands."""
import argparse
import json
import math
import sys
import threading
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import zmq

WRITE_HZ = 30
READ_HZ = 5


class LatestTargets:
    """Bounded latest-value mailbox, never a backlog of obsolete drag events."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}
        self.enabled = False

    def enable(self, enabled):
        with self.lock:
            self.enabled = enabled
            self.pending.clear()

    def put(self, path, value):
        with self.lock:
            if self.enabled:self.pending[path] = value

    def take(self):
        with self.lock:
            result, self.pending = self.pending, {}
            return result if self.enabled else {}


class Backend:
    def __init__(self, host='127.0.0.1', port=23000):
        self.client = RemoteAPIClient(host, port)
        self.client.socket.setsockopt(zmq.RCVTIMEO, 2500)
        self.client.socket.setsockopt(zmq.SNDTIMEO, 2500)
        self.client.socket.setsockopt(zmq.LINGER, 0)
        self.sim = self.client.require('sim')
        self.rows = []
        self.initial = {}
        self.inspect()

    def guard(self):
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            raise ValueError('仿真正在运行：本版只允许停止状态下的运动学预览，请先停止仿真')

    def inspect(self):
        self.guard()
        s = self.sim
        for h in s.getObjectsInTree(s.handle_scene, s.object_joint_type, 0):
            if s.getJointType(h) != s.joint_revolute_subtype:
                continue
            cyclic, interval = s.getJointInterval(h)
            dependent, offset, multiplier = s.getJointDependency(h)
            position = s.getJointPosition(h)
            mode = s.getJointMode(h)
            if isinstance(mode, (tuple,list)):mode = mode[0]
            mode_name = '动态' if mode == s.jointmode_dynamic else '运动学' if mode == s.jointmode_kinematic else str(mode)
            row = dict(handle=h, path=s.getObjectAlias(h, 1), mode=mode_name,
                       lower=-math.pi if cyclic else interval[0],
                       upper=math.pi if cyclic else interval[0]+interval[1],
                       cyclic=cyclic, dependent=dependent, position=position)
            self.rows.append(row)
            self.initial[row['path']] = position
        self.rows.sort(key=lambda r:r['path'])

    def read(self):
        return self.exchange({}, read=True)

    def exchange(self, changes, *, read=False):
        """One RPC; validate the full batch BEFORE applying any joint angle.

        Execute ephemeral Lua in CoppeliaSim, no persistent scene script added.
        Simulation state validation and writes share the same server invocation.
        """
        rows = {r['path']:r for r in self.rows}
        items = []
        for path,value in changes.items():
            r = rows[path]
            if r['dependent'] >= 0:
                raise ValueError('从属关节不能独立控制')
            if not math.isfinite(value) or not r['lower'] <= value <= r['upper']:
                raise ValueError('角度超出场景配置区间')
            items.append('{path='+json.dumps(path,ensure_ascii=False)+',h='+str(r['handle'])+',v='+repr(float(value))+'}')
        objects = ','.join('{path='+json.dumps(r['path'],ensure_ascii=False)+',h='+str(r['handle'])+'}' for r in self.rows)
        code = '''local sim=require('sim')
if sim.getSimulationState()~=sim.simulation_stopped then
    return {ok=false,error='仿真正在运行，请停止仿真后重新连接'} end
local objects={%s}
local changes={%s}
for _,r in ipairs(objects) do
    if sim.getObject(r.path)~=r.h then return {ok=false,error='场景对象已改变，请重新连接'} end
end
for _,r in ipairs(changes) do
    local master=sim.getJointDependency(r.h)
    if master>=0 then return {ok=false,error='关节已变为从属关节'} end
    local cyclic,iv=sim.getJointInterval(r.h)
    if not cyclic and (r.v<iv[1] or r.v>iv[1]+iv[2]) then
        return {ok=false,error='场景关节区间已改变'} end
end
for _,r in ipairs(changes) do sim.setJointPosition(r.h,r.v) end
local values={}
if %s then for _,r in ipairs(objects) do values[r.path]=sim.getJointPosition(r.h) end end
return {ok=true,values=values}
''' % (objects,','.join(items),'true' if read else 'false')
        success, result = self.sim.executeLuaCode(code)
        if not success or not isinstance(result,dict):
            raise RuntimeError('批量通信失败：'+str(result))
        if not result.get('ok'):
            raise ValueError(result.get('error','场景检查失败'))
        return result['values'] if read else None

    def set_position(self, path, value):
        return self.exchange({path:value},read=True)[path]

    def close(self):
        self.client.socket.close(linger=0)
        self.client.context.term()


def smoke(host, port):
    backend = Backend(host, port)
    try:
        original = backend.read()
        path = '/BaseLinkHull/jointL4'
        target = original[path]+math.radians(5)
        try:
            actual = backend.set_position(path, target)
            assert abs(actual-target) < 1e-6, (target, actual)
            assert len(backend.rows) == 42, len(backend.rows)
            print(f'PASS: {len(backend.rows)} joints; L4 +5deg set/readback verified', flush=True)
        finally:
            backend.set_position(path, original[path])
        assert abs(backend.read()[path]-original[path]) < 1e-6
        print('PASS: original L4 angle restored; scene not saved', flush=True)
    finally:
        backend.close()


def benchmark(host, port):
    backend=Backend(host,port)
    try:
        values=backend.read()
        values={r['path']:values[r['path']] for r in backend.rows if r['dependent'] < 0}
        elapsed=[]
        start=time.monotonic()
        for i in range(60):
            tick=time.monotonic()
            result=backend.exchange(values,read=i%6==0)
            elapsed.append((time.monotonic()-tick)*1000)
            if result is not None:
                assert all(abs(result[k]-v)<1e-6 for k,v in values.items())
            time.sleep(max(0,1/WRITE_HZ-(time.monotonic()-tick)))
        duration=time.monotonic()-start
        ordered=sorted(elapsed)
        print(f'PASS: 60 full-joint batches, unchanged angles; {60/duration:.1f} Hz; '
              f'RPC mean {sum(elapsed)/len(elapsed):.1f} ms, P95 {ordered[56]:.1f} ms, max {max(elapsed):.1f} ms')
    finally:backend.close()


def gui(host, port, smoke_test=False):
    from PySide6 import QtCore, QtWidgets

    class Worker(QtCore.QThread):
        inventory = QtCore.Signal(object)
        values = QtCore.Signal(object)
        status = QtCore.Signal(str)

        def __init__(self):
            super().__init__()
            self.commands = LatestTargets()
            self.end = threading.Event()

        def run(self):
            backend = None
            try:
                self.status.emit('正在连接并读取场景…')
                backend = Backend(host, port)
                self.inventory.emit(backend.rows)
                next_read = 0.
                stats_at = time.monotonic()
                writes = 0
                rpc_ms = 0.
                while not self.end.is_set():
                    started = time.monotonic()
                    pending = self.commands.take()
                    read = started >= next_read
                    if pending or read:
                        values = backend.exchange(pending,read=read)
                        rpc_ms = (time.monotonic()-started)*1000
                        writes += bool(pending)
                        if read:
                            self.values.emit(values)
                            next_read = time.monotonic()+1/READ_HZ
                    now = time.monotonic()
                    if now-stats_at >= 1:
                        self.status.emit(f'已连接 · 批量写入上限 {WRITE_HZ} Hz / 读回 {READ_HZ} Hz · '
                            f'近期写入 {writes/(now-stats_at):.1f} Hz · RPC {rpc_ms:.1f} ms · 不保存场景')
                        stats_at,writes = now,0
                    self.end.wait(max(0,1/WRITE_HZ-(time.monotonic()-started)))
            except Exception as exc:
                self.status.emit(f'连接/读取停止：{exc}；请停止仿真或检查场景后重新连接')
            finally:
                if backend:
                    backend.close()

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('新机器人 · CoppeliaSim 关节验证（独立 Demo）')
            self.resize(1100, 820)
            self.worker = None
            self.controls = {}
            self.initial = {}
            self.last = {}
            layout = QtWidgets.QVBoxLayout(self)
            note = QtWidgets.QLabel('先保持 CoppeliaSim 仿真停止。读取场景角度作为初始目标，不自动归零。\n'
                '勾选后才允许滑杆写入；区间来自场景，不代表实机安全限位。动态模式下的直接摆姿不验证物理可执行性。')
            note.setWordWrap(True)
            layout.addWidget(note)
            self.status = QtWidgets.QLabel('未连接')
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            buttons = QtWidgets.QHBoxLayout()
            connect = QtWidgets.QPushButton('连接 / 重新读取关节')
            connect.clicked.connect(self.connect)
            buttons.addWidget(connect)
            self.enable = QtWidgets.QCheckBox('允许写入仿真关节角（实验）')
            self.enable.toggled.connect(self.toggle)
            buttons.addWidget(self.enable)
            restore = QtWidgets.QPushButton('恢复连接时角度')
            restore.clicked.connect(self.restore)
            buttons.addWidget(restore)
            layout.addLayout(buttons)
            self.tabs = QtWidgets.QTabWidget()
            layout.addWidget(self.tabs)
            self.connect()

        def connect(self):
            self.enable.setChecked(False)
            if self.worker and self.worker.isRunning():
                self.worker.end.set()
                if not self.worker.wait(5000):
                    self.status.setText('旧连接仍在结束，请稍后重试')
                    return
            self.worker = Worker()
            self.worker.inventory.connect(self.build)
            self.worker.values.connect(self.update_values)
            self.worker.status.connect(self.status.setText)
            self.worker.finished.connect(lambda:self.enable.setChecked(False))
            self.worker.start()

        def build(self, rows):
            self.tabs.clear()
            self.controls.clear()
            self.initial = {r['path']:r['position'] for r in rows}
            for label, match in [('双臂',lambda n:'jointLH' not in n and 'jointRH' not in n),
                                 ('左手',lambda n:'jointLH' in n),('右手',lambda n:'jointRH' in n)]:
                table = QtWidgets.QTableWidget(0, 5)
                table.setHorizontalHeaderLabels(['关节 / 场景模式', '角度滑杆', '目标 °', '实际 °', '区间 °'])
                table.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
                table.horizontalHeader().setSectionResizeMode(1,QtWidgets.QHeaderView.ResizeMode.Stretch)
                for r in filter(lambda r:match(r['path']),rows):
                    i = table.rowCount(); table.insertRow(i)
                    name = r['path']; lower,upper = map(math.degrees,(r['lower'],r['upper']))
                    table.setItem(i,0,QtWidgets.QTableWidgetItem(name.rsplit('/',1)[-1]+f" · mode={r['mode']}"))
                    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                    slider.setRange(math.ceil(lower*10),math.floor(upper*10))
                    spin = QtWidgets.QDoubleSpinBox();spin.setDecimals(2);spin.setRange(lower,upper)
                    spin.setSuffix(' °');spin.setValue(math.degrees(r['position']))
                    slider.setValue(round(spin.value()*10))
                    actual = QtWidgets.QLabel(f"{spin.value():.2f}")
                    table.setCellWidget(i,1,slider);table.setCellWidget(i,2,spin);table.setCellWidget(i,3,actual)
                    table.setItem(i,4,QtWidgets.QTableWidgetItem(f'{lower:.1f} … {upper:.1f}'))
                    slider.valueChanged.connect(lambda v,s=spin:s.setValue(v/10))
                    spin.valueChanged.connect(lambda v,n=name,s=slider:self.changed(n,v,s))
                    self.controls[name]=(slider,spin,actual,r['dependent'] < 0)
                self.tabs.addTab(table,f'{label} · {table.rowCount()}')
            self.toggle(False)

        def changed(self, name, degrees, slider):
            blocker = QtCore.QSignalBlocker(slider)
            slider.setValue(round(degrees*10))
            if self.enable.isChecked() and self.worker and self.worker.isRunning():
                self.worker.commands.put(name,math.radians(degrees))

        def toggle(self, enabled):
            if enabled and (not self.worker or not self.worker.isRunning() or not self.controls):
                self.enable.setChecked(False)
                self.status.setText('连接尚未就绪，请先连接并读取关节')
                return
            for name,(slider,spin,actual,independent) in self.controls.items():
                slider.setEnabled(enabled and independent);spin.setEnabled(enabled and independent)
                if not enabled and name in self.last:
                    blocker = QtCore.QSignalBlocker(spin)
                    spin.setValue(math.degrees(self.last[name]))
                    blocker2 = QtCore.QSignalBlocker(slider)
                    slider.setValue(round(spin.value()*10))
            if not enabled and self.worker:
                self.worker.commands.enable(False)
            elif self.worker:
                self.worker.commands.enable(True)

        def update_values(self, values):
            self.last = values
            for name,value in values.items():
                if name in self.controls:self.controls[name][2].setText(f'{math.degrees(value):.2f}')
            if not self.enable.isChecked():self.toggle(False)

        def restore(self):
            if not self.enable.isChecked():
                self.status.setText('先勾选允许写入；恢复操作会改变场景中的关节姿态')
                return
            for name,value in self.initial.items():
                if self.controls[name][3]:self.controls[name][1].setValue(math.degrees(value))

        def closeEvent(self,event):
            self.enable.setChecked(False)
            if self.worker:
                self.worker.end.set()
                if not self.worker.wait(5000):
                    event.ignore();self.status.setText('正在结束网络连接，请稍后关闭');return
            event.accept()

    app = QtWidgets.QApplication(sys.argv)
    window = Window();window.show()
    if smoke_test:
        def check():
            assert len(window.controls) == 42, len(window.controls)
            assert window.tabs.count() == 3
            assert not window.enable.isChecked()
            assert all(not c[0].isEnabled() for c in window.controls.values())
            window.grab().save('/tmp/coppeliasim-joint-demo.png')
            print('PASS: GUI inventory 42, three tabs, writes disabled',flush=True)
            window.close();app.quit()
        QtCore.QTimer.singleShot(3000,check)
    return app.exec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=23000)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--gui-smoke-test',action='store_true')
    parser.add_argument('--benchmark-test',action='store_true')
    args = parser.parse_args()
    if args.smoke_test:smoke(args.host,args.port)
    elif args.benchmark_test:benchmark(args.host,args.port)
    else:sys.exit(gui(args.host,args.port,args.gui_smoke_test))
