"""把寄存器协议映射到 UDP, 复用串口版函数。

设备在 AP 或 STA 模式下监听 9250 端口 (LOCALPORT), 可向指定设备 IP
发送寄存器指令。只接受该 IP 的回包，避免多设备数据混入校准流程。

本类刻意模仿 pyserial 的接口 (write/flush/read/reset_input_buffer),
这样 registers.py 里的协议函数无需任何改动即可复用。
"""

from __future__ import annotations

import socket
import time

DEFAULT_DEVICE = ("192.168.4.1", 9250)
DEFAULT_LOCAL_PORT = 1399


class UdpTransport:
    def __init__(self, device: tuple[str, int] = DEFAULT_DEVICE,
                 local_port: int = DEFAULT_LOCAL_PORT,
                 timeout: float = 0.5):
        self.device = device
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", local_port))
        self.sock.settimeout(timeout)
        self._buf = bytearray()

    # -- pyserial 兼容接口 -------------------------------------------------
    def write(self, data) -> int:
        return self.sock.sendto(bytes(data), self.device)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._buf.clear()
        old_timeout = self.sock.gettimeout()
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(4096)
        except (socket.timeout, BlockingIOError):
            pass
        finally:
            self.sock.settimeout(old_timeout)

    def drain_available(self) -> bytes:
        """取走当前排队的、来自目标设备的 UDP 数据。"""
        old_timeout = self.sock.gettimeout()
        self.sock.setblocking(False)
        chunks = []
        try:
            while True:
                try:
                    data, addr = self.sock.recvfrom(65535)
                except (socket.timeout, BlockingIOError):
                    break
                if addr[0] == self.device[0]:
                    chunks.append(data)
        finally:
            self.sock.settimeout(old_timeout)
        return b"".join(chunks)

    def read(self, n: int = 1) -> bytes:
        old_timeout = self.sock.gettimeout()
        deadline = time.monotonic() + (old_timeout if old_timeout is not None else 0.5)
        try:
            while len(self._buf) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.sock.settimeout(remaining)
                try:
                    data, addr = self.sock.recvfrom(4096)
                except socket.timeout:
                    break
                if addr[0] == self.device[0]:
                    self._buf.extend(data)
        finally:
            self.sock.settimeout(old_timeout)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
