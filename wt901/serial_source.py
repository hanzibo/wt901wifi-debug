"""从串口读取 WT901WIFI 姿态数据。"""

from __future__ import annotations

import time
from typing import Iterator

import serial

from .protocol import AttitudePacket, extract_packets

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 9600


class SerialAttitudeSource:
    """连接 WT901WIFI 的 Type-C 串口, 逐包产出 AttitudePacket。

    用法::

        with SerialAttitudeSource() as src:
            for pkt in src:
                print(pkt.roll, pkt.pitch, pkt.yaw)
    """

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD,
                 timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self._serial = serial.Serial(port, baud, timeout=timeout)
        # 有些 CH340 设备需要 DTR/RTS 有效才开始发送
        self._serial.dtr = True
        self._serial.rts = True
        time.sleep(0.2)
        self._serial.reset_input_buffer()
        self._buffer = bytearray()

    def __iter__(self) -> Iterator[AttitudePacket]:
        while True:
            pkts = self.poll()
            if pkts:
                yield from pkts
            else:
                time.sleep(0.005)  # 无数据时让出 CPU, 避免忙等

    def poll(self) -> list[AttitudePacket]:
        """非阻塞地读取当前已到达的数据, 返回本次解出的所有数据包。

        适合放进 GUI 定时器里调用, 不会阻塞事件循环。
        """
        n = self._serial.in_waiting
        if n:
            self._buffer.extend(self._serial.read(n))
        return [AttitudePacket.parse(raw) for raw in extract_packets(self._buffer)]

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass

    def __enter__(self) -> "SerialAttitudeSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
