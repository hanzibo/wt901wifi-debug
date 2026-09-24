"""WT901WIFI 数据包解析。

串口(9600 8N1) 与 WiFi(TCP/UDP 1399) 输出的都是同一种 54 字节数据包：
    偏移  字段
    0-11  设备ID (12 字节 ASCII, 形如 WT5500011408)
    12-19 年 月 日 时 分 秒 毫秒L 毫秒H
    20-25 AX AY AZ            加速度 (int16 LE)
    26-31 GX GY GZ            角速度 (int16 LE)
    32-37 HX HY HZ            磁场   (int16 LE)
    38-43 Roll Pitch Yaw      姿态角 (int16 LE)
    44-45 温度
    46-47 电量
    48-49 RSSI
    50-51 版本
    52-53 0x0D 0x0A

数据流本身没有校验和, 只靠包头 'WT' 与结尾 CRLF 定界。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields

PACKET_LEN = 54
HEADER = b"WT"
TAIL = b"\r\n"

# WT901WIFI protocol, BAT register table. The packet carries centivolts,
# not a measured state-of-charge percentage. Keep its discrete bins intact.
BATTERY_PERCENT_STEPS = (
    (3.93, 90),
    (3.87, 75),
    (3.82, 60),
    (3.79, 50),
    (3.77, 40),
    (3.73, 30),
    (3.70, 20),
    (3.68, 15),
    (3.50, 10),
    (3.40, 5),
)


def battery_percent_from_voltage(voltage: float) -> int | None:
    """Estimate battery percentage from the vendor's discrete voltage table.

    None means no valid voltage was reported. This is not a fuel-gauge reading.
    The manual uses >3.96 V for 100% and 3.93-3.96 V for 90%.
    """
    if not math.isfinite(voltage) or voltage <= 0:
        return None
    if voltage > 3.96:
        return 100
    for minimum, percent in BATTERY_PERCENT_STEPS:
        if voltage >= minimum:
            return percent
    return 0


def _s16(low: int, high: int) -> int:
    """小端有符号 16 位。"""
    return struct.unpack("<h", bytes((low, high)))[0]


def _u16(low: int, high: int) -> int:
    return low | (high << 8)


@dataclass
class AttitudePacket:
    device_id: str

    # 片内时间
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    millisecond: int

    # 加速度 (g)
    acc_x: float
    acc_y: float
    acc_z: float

    # 角速度 (°/s)
    gyro_x: float
    gyro_y: float
    gyro_z: float

    # 磁场 (uT)
    mag_x: float
    mag_y: float
    mag_z: float

    # 姿态角 (°)
    roll: float
    pitch: float
    yaw: float

    temperature: float  # ℃
    battery_v: float  # V
    rssi: int  # dBm
    version: int

    @property
    def battery_percent(self) -> int | None:
        return battery_percent_from_voltage(self.battery_v)

    @classmethod
    def parse(cls, buf: bytes) -> "AttitudePacket":
        if len(buf) != PACKET_LEN:
            raise ValueError(f"数据包长度应为 {PACKET_LEN}, 实际 {len(buf)}")
        if not is_valid(buf):
            raise ValueError("不是合法的 WT 数据包")
        b = buf
        return cls(
            device_id=b[0:12].decode("ascii", "replace"),
            year=2000 + b[12],
            month=b[13],
            day=b[14],
            hour=b[15],
            minute=b[16],
            second=b[17],
            millisecond=_u16(b[18], b[19]),
            acc_x=_s16(b[20], b[21]) / 32768 * 16,
            acc_y=_s16(b[22], b[23]) / 32768 * 16,
            acc_z=_s16(b[24], b[25]) / 32768 * 16,
            gyro_x=_s16(b[26], b[27]) / 32768 * 2000,
            gyro_y=_s16(b[28], b[29]) / 32768 * 2000,
            gyro_z=_s16(b[30], b[31]) / 32768 * 2000,
            mag_x=_s16(b[32], b[33]) * 100 / 1024,
            mag_y=_s16(b[34], b[35]) * 100 / 1024,
            mag_z=_s16(b[36], b[37]) * 100 / 1024,
            roll=_s16(b[38], b[39]) / 32768 * 180,
            pitch=_s16(b[40], b[41]) / 32768 * 180,
            yaw=_s16(b[42], b[43]) / 32768 * 180,
            temperature=_s16(b[44], b[45]) / 100,
            battery_v=_u16(b[46], b[47]) / 100,
            rssi=_s16(b[48], b[49]),
            version=_u16(b[50], b[51]),
        )

    @property
    def time_str(self) -> str:
        return (
            f"{self.year:04d}-{self.month:02d}-{self.day:02d} "
            f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}."
            f"{self.millisecond:03d}"
        )

    def as_row(self) -> dict:
        """用于 CSV/表格输出。"""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["time"] = self.time_str
        return d

    def __str__(self) -> str:
        return (
            f"[{self.device_id}] "
            f"Roll={self.roll:8.2f} Pitch={self.pitch:8.2f} Yaw={self.yaw:8.2f} | "
            f"Acc=({self.acc_x:6.3f},{self.acc_y:6.3f},{self.acc_z:6.3f})g "
            f"Gyro=({self.gyro_x:7.2f},{self.gyro_y:7.2f},{self.gyro_z:7.2f})°/s "
            f"Mag=({self.mag_x:7.1f},{self.mag_y:7.1f},{self.mag_z:7.1f})uT | "
            f"T={self.temperature:5.2f}℃ Bat={self.battery_v:.2f}V "
            f"RSSI={self.rssi}dBm Ver={self.version}"
        )


def is_valid(buf: bytes) -> bool:
    """检查一个 54 字节缓冲是否是合法的 WT 数据包。"""
    return (
        len(buf) >= PACKET_LEN
        and buf[0:2] == HEADER
        and buf[52:54] == TAIL
    )


def extract_packets(buffer: bytearray) -> list[bytes]:
    """从字节缓冲中切出所有完整的 54 字节数据包, 就地删除已消费的字节。

    容错做法: 找不到包头就丢弃 1 字节继续找; 长度够但结尾不是 CRLF 时
    也丢弃 1 字节重新同步, 避免粘包/丢字节后永久错位。
    """
    packets: list[bytes] = []
    while True:
        # 对齐到包头
        if len(buffer) >= 2 and buffer[0:2] != HEADER:
            del buffer[0]
            continue
        if len(buffer) < PACKET_LEN:
            break
        if buffer[52:54] == TAIL:
            packets.append(bytes(buffer[:PACKET_LEN]))
            del buffer[:PACKET_LEN]
        else:
            # 长度够但尾部不对, 说明错位, 丢弃一个字节重来
            del buffer[0]
    return packets
