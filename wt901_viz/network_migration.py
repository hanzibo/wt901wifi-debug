"""Shared PC/Jetson batch STA migration. No network switching or extra socket."""
from PySide6 import QtCore, QtWidgets
from wt901.network_config import build_sta_command


class NetworkMigrationDialog(QtWidgets.QDialog):
    def __init__(self, source, parent=None, can_switch=None, default_target=0):
        super().__init__(parent)
        self.source, self.can_switch = source, can_switch
        self.pending = []
        self.target = None
        self.sent = self.failed = 0
        self.setWindowTitle('批量迁移在线 IMU · Wi-Fi / UDP')
        self.resize(900, 650)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            '直接向当前在线 IMU 写入新的 Wi-Fi 和 UDP 地址，无需先恢复 AP。\n'
            '目标 Wi-Fi 必须为 2.4 GHz，目标接收程序须已打开。迁移到电脑前先开启电脑热点；'
            '迁出电脑热点时，保持热点开启直到整批发送完毕。本窗口不切换主机网络。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.profile = QtWidgets.QComboBox()
        self.profile.addItems(['开发板 · HUAWEI-509', '电脑热点 · WT901-LAN', '自定义'])
        self.ssid = QtWidgets.QLineEdit()
        self.ip = QtWidgets.QLineEdit()
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password.setPlaceholderText('本次使用，不保存到文件或日志')
        self.password_confirm = QtWidgets.QLineEdit()
        self.password_confirm.setEchoMode(QtWidgets.QLineEdit.Password)
        form = QtWidgets.QFormLayout()
        for label, widget in [('目标预设', self.profile), ('Wi-Fi SSID', self.ssid),
                              ('UDP 目标 IPv4（接收设备，不是路由器）', self.ip),
                              ('UDP 目标端口', self.port), ('目标 Wi-Fi 密码', self.password),
                              ('确认密码', self.password_confirm)]:
            form.addRow(label, widget)
        layout.addLayout(form)
        self.profile.currentIndexChanged.connect(self.apply_profile)
        self.profile.setCurrentIndex(default_target)
        self.apply_profile(default_target)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['迁移', '设备 ID', '当前设备 IP', '发送结果'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        self.refresh_button = QtWidgets.QPushButton('刷新在线设备（默认全选）')
        self.refresh_button.clicked.connect(self.refresh)
        layout.addWidget(self.refresh_button)
        self.ready = QtWidgets.QCheckBox('我已断开真机控制，目标 Wi-Fi 已开启，并已核对目标接收地址')
        layout.addWidget(self.ready)
        self.status = QtWidgets.QLabel('UDP 无确认回执；请在目标接收端核对每个 ID 的实时数据。')
        self.status.setTextFormat(QtCore.Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QtWidgets.QHBoxLayout()
        self.send_button = QtWidgets.QPushButton('确认并批量写入勾选设备…')
        self.send_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton('停止后续发送')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_remaining)
        close = QtWidgets.QPushButton('关闭')
        close.clicked.connect(self.reject)
        for widget in (self.send_button, self.stop_button, close):
            actions.addWidget(widget)
        layout.addLayout(actions)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.send_next)
        self.refresh()

    def apply_profile(self, index):
        if index in (0, 1):
            ssid, ip = [('HUAWEI-509', '192.168.3.8'), ('WT901-LAN', '10.42.0.1')][index]
            self.ssid.setText(ssid)
            self.ip.setText(ip)
            self.port.setValue(1399)

    def refresh(self):
        if self.target is not None:
            return
        live = self.source.seen_endpoints()
        self.table.setRowCount(len(live))
        for row, (key, endpoint) in enumerate(sorted(live.items())):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(row, 0, item)
            for col, text in enumerate((key, endpoint[0], '尚未发送'), 1):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

    def permitted(self, keys):
        return self.can_switch is None or all(self.can_switch(key) for key in keys)

    def start(self):
        if self.target is not None:
            return
        selected = [(self.table.item(row, 1).text(), self.table.item(row, 2).text(), row)
                    for row in range(self.table.rowCount())
                    if self.table.item(row, 0).checkState() == QtCore.Qt.Checked]
        if not selected or not self.ready.isChecked():
            self.status.setText('请勾选在线设备，并确认真机已断开及目标网络已就绪。')
            return
        ssid, ip, port = self.ssid.text(), self.ip.text().strip(), self.port.value()
        password = self.password.text()
        try:
            if password != self.password_confirm.text():
                raise ValueError('两次输入的密码不一致')
            build_sta_command(ssid, password, ip, port)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if not self.permitted([key for key, _, _ in selected]):
            self.status.setText('未发送：请先退出控制或设备参数操作。')
            return
        answer = QtWidgets.QMessageBox.question(
            self, '确认批量迁移 IMU',
            f'向以下 {len(selected)} 台设备写入 Wi-Fi：{ssid}\nUDP 目标：{ip}:{port}\n'
            + '\n'.join(key for key, _, _ in selected)
            + '\n设备将停止向原接收端发送数据。密码或地址错误可能需要恢复 AP。'
              '\n本窗口仅确认发送，不确认目标端接入成功。继续？',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        if not self.ready.isChecked() or not self.permitted([key for key, _, _ in selected]):
            self.status.setText('未发送：确认期间控制状态已改变。')
            return
        self.pending = selected
        self.target = (ssid, password, ip, port)
        self.sent = self.failed = 0
        self.password.clear()
        self.password_confirm.clear()
        self.set_busy(True)
        for _, _, row in selected:
            self.table.item(row, 3).setText('等待发送')
        self.timer.start()

    def set_busy(self, busy):
        for widget in (self.profile, self.ssid, self.ip, self.port, self.password,
                       self.password_confirm, self.table, self.refresh_button,
                       self.ready, self.send_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def send_next(self):
        if not self.pending or self.target is None:
            self.finish()
            return
        key, expected_ip, row = self.pending[0]
        if not self.permitted([key]):
            self.cancel_remaining('控制状态改变，已停止后续发送')
            return
        self.pending.pop(0)
        try:
            self.source.configure_network(key, expected_ip, *self.target)
            self.table.item(row, 3).setText('已发送；目标端待验证')
            self.sent += 1
        except (OSError, RuntimeError, ValueError):
            # Exception text from a transport must never expose the credential payload.
            self.table.item(row, 3).setText('未发送：离线、IP 变化或发送异常；刷新后重试')
            self.failed += 1
        self.status.setText(f'已发送 {self.sent}，失败 {self.failed}，剩余 {len(self.pending)}。'
                            '请在目标端核对实时数据；不会自动重试。')
        if not self.pending:
            self.finish()

    def finish(self):
        self.timer.stop()
        self.target = None
        self.set_busy(False)
        self.ready.setChecked(False)
        self.status.setText(f'发送结束：已发送 {self.sent}，失败 {self.failed}。'
                            '发送不等于接入成功，请在目标端核对设备 ID 和实时数据。')

    def cancel_remaining(self, reason='已停止后续发送；已发送的配置不会撤回'):
        for _, _, row in self.pending:
            self.table.item(row, 3).setText('已取消，未发送')
        self.pending = []
        self.finish()
        self.status.setText(reason + f'。已发送 {self.sent}，失败 {self.failed}。')

    def done(self, result):
        self.cancel_remaining()
        self.password.clear()
        self.password_confirm.clear()
        super().done(result)
