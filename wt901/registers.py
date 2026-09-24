"""WT901WIFI 寄存器读写协议。

写寄存器 (HEX):
    FF AA ADDR DATAL DATAH

读寄存器:
    FF AA 27 ADDR 00   ->   应答 55 5F + 4 个寄存器(L,H) + SUM   (共 11 字节)

注意: 所有设置类寄存器必须先解锁 (写 0xB588 到 0x69), 否则写入无效。
串口数据流是持续的 54 字节数据包, 寄存器应答混在其中, 因此接收时
必须靠包头 0x55 0x5F + 校验和来甄别, 不能直接按长度切。
"""

from __future__ import annotations

import time

UNLOCK_KEY = 0xB588
UNLOCK_REG = 0x69
CMD_READ = 0x27

# 常用寄存器
REG_SAVE = 0x00
REG_CALSW = 0x01
REG_KEY = 0x69
REG_BANDWIDTH = 0x1F
REG_GYRORANGE = 0x20
REG_ACCRANGE = 0x21
REG_ORIENT = 0x23
REG_AXIS6 = 0x24
REG_FILTK = 0x25
REG_ACCFILT = 0x2A
REG_WIFIMODE = 0x2D
REG_VERSION = 0x2E
REG_RATE = 0x68

SAVE = 0x0000
RESTART = 0x00FF
FACTORY_RESET = 0x0001

RESPONSE_HEADER = b"\x55\x5F"
RESPONSE_LEN = 11


def _checksum_ok(frame: bytes) -> bool:
    return (sum(frame[:10]) & 0xFF) == frame[10]


def write_register(ser, addr: int, value: int) -> None:
    """写一个寄存器 (不含解锁)。value 为 16 位。"""
    ser.write(bytes([0xFF, 0xAA, addr & 0xFF,
                     value & 0xFF, (value >> 8) & 0xFF]))
    ser.flush()


def unlock(ser) -> None:
    """解锁寄存器写入权限 (约 10 秒内有效)。"""
    write_register(ser, UNLOCK_REG, UNLOCK_KEY)


def save(ser) -> None:
    """保存当前配置到设备。"""
    write_register(ser, REG_SAVE, SAVE)


def read_registers(ser, addr: int, timeout: float = 1.5) -> list[int]:
    """读 addr 起的 4 个寄存器, 返回 4 个 16 位值。

    返回: [REG1, REG2, REG3, REG4], 即 addr, addr+1, addr+2, addr+3。
    """
    ser.reset_input_buffer()
    ser.write(bytes([0xFF, 0xAA, CMD_READ, addr & 0xFF, 0x00]))
    ser.flush()

    buf = bytearray()
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = ser.read(64)
        if chunk:
            buf.extend(chunk)
        # 在流里找带正确校验和的 55 5F 应答
        i = 0
        while True:
            i = buf.find(RESPONSE_HEADER, i)
            if i < 0:
                # 保留尾部 1 字节, 避免跨块漏掉包头
                if len(buf) > 1:
                    del buf[:-1]
                break
            if len(buf) - i >= RESPONSE_LEN:
                frame = bytes(buf[i:i + RESPONSE_LEN])
                if _checksum_ok(frame):
                    return [frame[2] | (frame[3] << 8),
                            frame[4] | (frame[5] << 8),
                            frame[6] | (frame[7] << 8),
                            frame[8] | (frame[9] << 8)]
                i += 1
            else:
                del buf[:i]
                break
    raise TimeoutError(f"读寄存器 0x{addr:02X} 超时, 未收到有效应答")


UNLOCK_SETTLE = 0.6   # 解锁后需等待约 0.6s, 否则紧接着的写入不生效


def set_rate(ser, hz: float) -> int:
    """设置回传频率 (Hz), 返回写入的寄存器值 (周期 ms)。

    寄存器值 = 1000/Hz, 单位 ms。例如 10Hz -> 100, 200Hz -> 5。
    """
    if not (0.2 <= hz <= 200):
        raise ValueError("频率需在 0.2 ~ 200 Hz 之间")
    period_ms = int(round(1000.0 / hz))
    period_ms = max(5, min(period_ms, 5000))
    unlock(ser)
    time.sleep(UNLOCK_SETTLE)
    write_register(ser, REG_RATE, period_ms)
    time.sleep(0.3)
    return period_ms


def rate_hz_from_register(value: int) -> float:
    """寄存器值 (ms) -> Hz。"""
    return 1000.0 / value if value else 0.0
