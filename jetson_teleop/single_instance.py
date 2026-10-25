"""Two desktop entries, one process/socket; IPC only opens a window, never motion."""
import os
from pathlib import Path

from PySide6 import QtCore, QtNetwork


def endpoint(port):
    root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.RuntimeLocation)
    if not root:
        root = f'/tmp/wt901-jetson-{os.getuid()}'
        Path(root).mkdir(mode=0o700, exist_ok=True)
    return str(Path(root)/f'wt901-jetson-{port}.sock')


def forward(name, action):
    if action not in ('imu', 'control'):
        raise ValueError('未知窗口')
    client = QtNetwork.QLocalSocket()
    client.connectToServer(name)
    if not client.waitForConnected(300):
        return False
    client.write((action+'\n').encode('ascii'))
    client.flush()
    client.waitForBytesWritten(300)
    client.disconnectFromServer()
    return True


class InstanceServer(QtCore.QObject):
    requested = QtCore.Signal(str)

    def __init__(self, name):
        super().__init__()
        self.server = QtNetwork.QLocalServer(self)
        self.server.setSocketOptions(QtNetwork.QLocalServer.UserAccessOption)
        # Only remove stale same-user endpoints; live instances are handled by forward().
        QtNetwork.QLocalServer.removeServer(name)
        if not self.server.listen(name):
            raise RuntimeError('无法创建单实例本地接口：'+self.server.errorString())
        self.server.newConnection.connect(self.accept)

    def accept(self):
        while self.server.hasPendingConnections():
            client = self.server.nextPendingConnection()
            buffer = bytearray()

            def read(client=client, buffer=buffer):
                buffer.extend(bytes(client.readAll()))
                if b'\n' in buffer or len(buffer) > 32:
                    if bytes(buffer) in (b'imu\n', b'control\n'):
                        self.requested.emit(bytes(buffer).strip().decode('ascii'))
                    client.disconnectFromServer()

            client.readyRead.connect(read)
            client.disconnected.connect(client.deleteLater)
            read()

    def close(self):
        self.server.close()
