"""Separate management window sharing the controller's ONE UDP source."""
import json
import time

from PySide6 import QtCore, QtWidgets


class ImuManager(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.source = viewer.source
        self.pending = None
        self.setWindowTitle('WT901 · Jetson IMU 接入与管理（独立窗口）')
        self.resize(1050, 620)
        layout = QtWidgets.QVBoxLayout(self)
        self.network = QtWidgets.QLabel('正在读取板子网卡 IPv4…')
        self.network.setTextFormat(QtCore.Qt.PlainText)
        self.network.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.network.setWordWrap(True)
        layout.addWidget(self.network)
        note = QtWidgets.QLabel(
            'PC 辅助配置：先把 IMU 切回 AP → PC 连接设备热点 → 写入路由器 2.4 GHz SSID / 密码，'
            'UDP 目标填写下方板子有线 IPv4（不是路由器 .1、不是 PC .7），端口通常 1399。\n'
            '板子不创建热点。列表仅由实际收到的 WT 数据生成；与机器人窗口共享单个 UDP 收包源，可同时打开。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['设备 ID', '分配部位', '设备 IP:源端口', '收包 Hz',
                                               '距末包 ms', '电量估计', 'RSSI dBm', '状态'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.controls)
        layout.addWidget(self.table, 1)
        self.status = QtWidgets.QLabel('等待 IMU。平均 Hz 不代表低延迟；请同时观察距末包时间。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.status)
        self.ap_button = QtWidgets.QPushButton('将选中的在线 IMU 切回 AP / UDP…（需要确认）')
        self.ap_button.clicked.connect(self.switch_ap)
        layout.addWidget(self.ap_button)
        self.migrate_button = QtWidgets.QPushButton('批量迁移在线 IMU 到电脑热点 / 其他接收端…')
        self.migrate_button.clicked.connect(self.migrate_network)
        layout.addWidget(self.migrate_button)
        self.refresh_button = QtWidgets.QPushButton('刷新板子网卡 IPv4')
        self.refresh_button.clicked.connect(self.refresh_network)
        layout.addWidget(self.refresh_button)
        self.process = QtCore.QProcess(self)
        self.process.finished.connect(self.network_ready)
        self.process.errorOccurred.connect(self.network_error)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.render)
        self.timer.start()
        self.network_timer = QtCore.QTimer(self)
        self.network_timer.setInterval(5000)
        self.network_timer.timeout.connect(self.refresh_network)
        self.network_timer.start()
        self.refresh_network()
        self.render()

    def refresh_network(self):
        if self.process.state() == QtCore.QProcess.NotRunning:
            self.process.start('ip', ['-j', '-4', 'addr', 'show', 'scope', 'global'])

    def network_error(self, error):
        self.network.setText('无法启动 ip 命令，请在系统网络设置中查询板子有线 IPv4。')

    def network_ready(self, code, *_):
        try:
            if code:
                raise ValueError('ip 命令失败')
            rows = json.loads(bytes(self.process.readAllStandardOutput()))
            ips = [f"{r['ifname']}: {a['local']}/{a['prefixlen']}"
                   for r in rows for a in r.get('addr_info', []) if a.get('family') == 'inet']
            port = self.source.sock.getsockname()[1]
            self.network.setText('板子当前 IPv4：\n'+'\n'.join(ips)+
                                 f'\n实际监听：0.0.0.0:{port}（所有网卡）；填写有线网卡对应的单播 IPv4。')
        except (ValueError, KeyError, TypeError) as exc:
            self.network.setText(f'读取 IPv4 失败：{exc}')

    def selected(self):
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item else None

    def allowed(self):
        d = self.viewer.control
        return (d.worker is None and not d.busy and d.ik_worker is None
                and not d.auto_start.pending and not d.end_session.returning)

    def controls(self):
        key = self.selected()
        self.ap_button.setEnabled(self.allowed() and key in self.source.seen_endpoints())
        self.ap_button.setToolTip('须先在真机控制窗口制动并断开串口；不允许控制期间切断 IMU 输入。')
        self.migrate_button.setEnabled(self.allowed() and bool(self.source.seen_endpoints()))

    def migrate_network(self):
        from wt901_viz.network_migration import NetworkMigrationDialog
        if not self.allowed():
            self.status.setText('请先制动并断开真机串口。')
            return
        dialog = NetworkMigrationDialog(self.source, self, can_switch=lambda key: self.allowed(),
                                        default_target=1)
        dialog.exec()
        dialog.deleteLater()

    def render(self):
        selected = self.selected()
        live = self.source.seen_endpoints()
        known = self.source.seen_endpoints(max_age=float('inf'))
        stats = self.source.receive_stats()
        labels = {'left_upper_arm':'左上臂', 'left_forearm':'左前臂', 'left_hand':'左手背',
                  'right_upper_arm':'右上臂', 'right_forearm':'右前臂', 'right_hand':'右手背'}
        blockers = QtCore.QSignalBlocker(self.table)
        self.table.setRowCount(len(known))
        for row, key in enumerate(sorted(known)):
            ip, port, stamp = known[key]
            packet = self.source.latest_packet(key)
            stat = stats.get(key)
            rate = '观察中' if stat is None or stat.rate_hz is None else f'{stat.rate_hz:.1f}'
            assigned = [labels.get(k,k) for k,v in self.viewer.config['assignments'].items() if v == key]
            battery = packet.battery_percent if packet else None
            values = [key, '/'.join(assigned) or '未分配', f'{ip}:{port}', rate,
                      f'{(time.monotonic()-stamp)*1000:.0f}',
                      '—' if battery is None else f'{battery}%',
                      str(packet.rssi) if packet else '—', '在线' if key in live else '离线 / 已超时']
            for column, value in enumerate(values):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            if key == selected:
                self.table.selectRow(row)
        del blockers
        self.controls()
        if self.pending and self.pending[0] not in live and time.monotonic()-self.pending[1] > 3:
            self.status.setText(f'{self.pending[0]} 已超过 3 秒没有数据；这不等于 AP 已验证。'
                                '请在 PC Wi-Fi 列表查找该 ID 命名的设备热点。')

    def switch_ap(self):
        key = self.selected()
        if not self.allowed() or key not in self.source.seen_endpoints():
            self.status.setText('未发送：设备离线，或真机串口 / 启动 / IK 尚未退出。请先制动并断开串口。')
            return
        try:
            ip = self.source.device_ip(key)
        except RuntimeError as exc:
            self.status.setText(f'未发送：{exc}')
            return
        answer = QtWidgets.QMessageBox.question(self, '确认将 IMU 还原为设备 AP',
            f'仅操作 {key}（{ip}）。它将离开路由器 Wi-Fi，停止向板子发送数据，'
            '并尝试创建以设备 ID 命名的热点。\n之后需 PC 连接设备热点重新配置。'
            '\nUDP 没有确认回执，发送成功不等于切换成功。确认继续？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        # Recheck freshness and safety after the modal confirmation.
        if not self.allowed():
            self.status.setText('未发送：确认期间真机状态改变。')
            return
        try:
            self.source.switch_to_ap_udp(key)
            self.pending = (key, time.monotonic())
            self.status.setText(f'已向 {key} / {ip}:9250 发送 AP/UDP 指令；没有设备确认回执，'
                                '请在 PC Wi-Fi 列表验证设备热点。本操作不控制机器人。')
        except (OSError, RuntimeError) as exc:
            self.status.setText(f'未能发送 AP/UDP 指令：{exc}')
