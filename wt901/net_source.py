"""通过 WiFi (TCP/UDP) 接收 WT901WIFI 姿态数据。

设备作为客户端主动连到本机端口 (默认 1399), 推的仍是同样的 54 字节数据包,
所以复用 protocol.extract_packets / AttitudePacket, 只换 IO 层。

串口被固件限制在约 2Hz, 而 WiFi 通道支持 RATE 寄存器配置到最高 200Hz。
"""

from __future__ import annotations

import queue
import socket
import threading
from typing import Iterator

from .protocol import AttitudePacket, extract_packets

DEFAULT_PORT = 1399


class _BaseNetSource:
    def __init__(self) -> None:
        self._queue: queue.Queue[AttitudePacket] = queue.Queue()
        self._stop = threading.Event()

    # -- 供 GUI 定时器调用, 非阻塞 ----------------------------------------
    def poll(self) -> list[AttitudePacket]:
        out: list[AttitudePacket] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def __iter__(self) -> Iterator[AttitudePacket]:
        import time
        while not self._stop.is_set():
            pkts = self.poll()
            if pkts:
                yield from pkts
            else:
                time.sleep(0.005)

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _feed(self, data: bytes, buffer: bytearray) -> None:
        buffer.extend(data)
        for raw in extract_packets(buffer):
            self._queue.put(AttitudePacket.parse(raw))


class TcpAttitudeSource(_BaseNetSource):
    """TCP 服务器, 监听 port, 接受设备连接并解析数据流。

    支持同时接入多台设备 (每连接一个线程)。
    """

    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
        super().__init__()
        self.port = port
        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self._sock.bind((host, port))
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(client, addr),
                             daemon=True).start()

    def _handle(self, client: socket.socket, addr) -> None:
        client.settimeout(0.5)
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                try:
                    data = client.recv(2048)
                except socket.timeout:
                    continue
                if not data:
                    break
                self._feed(data, buffer)
        except OSError:
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass


class UdpAttitudeSource(_BaseNetSource):
    """UDP 服务器, 监听 port。UDP 无需建立连接, 设备可直接发过来。"""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
        super().__init__()
        self.port = port
        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # 加大接收缓冲, 避免渲染繁忙时内核丢包
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._buffers: dict[tuple, bytearray] = {}
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self._feed(data, self._buffers.setdefault(addr, bytearray()))
