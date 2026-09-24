#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WT901WIFI 磁场诊断 / 校准 / 还原。

背景
----
yaw 由 9 轴算法用磁力计锚定; 硬磁偏置会淹没地磁 (B_earth ~46 uT),
表现为「转动后 yaw 被慢慢拉回」。

校准原理: 设备把样本拟合到球面 m = R^T·B + c, 球心即偏置 c。
若转动覆盖不足(样本球太小), 拟合退化 -> 会把整个磁场当偏置减掉,
结果 |m| 趋近 0, 比不校准更糟。所以本脚本先做覆盖度预检, 达标才校准。

用法:
    python magcal.py --check              # 只诊断, 不写任何寄存器 (默认 20s)
    python magcal.py --check --seconds 30
    python magcal.py                      # 先预检再校准 (预检不达标就不校准)
    python magcal.py --reset              # 还原: 磁偏移归零, MAGRANG 回 500
    python magcal.py --device-ip 10.42.0.174 --device-id WT5500011381 --probe
                                      # STA 热点模式下只读测试指定设备
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

from wt901.registers import (
    REG_CALSW,
    REG_VERSION,
    UNLOCK_SETTLE,
    read_registers,
    save,
    unlock,
    write_register,
)
from wt901.protocol import AttitudePacket, extract_packets

CAL_NORMAL = 0x00
CAL_ACC = 0x01
CAL_HEADING_ZERO = 0x04
CAL_MAG = 0x07

MAGOFF_ADDR = 0x0B        # HXOFF, HYOFF, HZOFF
MAGRANG_ADDR = 0x1C       # MAGRANGX/Y/Z, 出厂默认 500
MAGRANG_DEFAULT = 500
GOOD_SPAN_UT = 70.0       # 完整球面覆盖时每轴跨度应接近 2*|B| ~ 90 uT


def open_link(args):
    port = args.port
    if port is None and not args.net and not args.device_ip and os.path.exists("/dev/ttyUSB0"):
        port = "/dev/ttyUSB0"
    if port:
        import serial
        s = serial.Serial(port, 9600, timeout=0.5)
        s.dtr = s.rts = True
        time.sleep(0.2)
        s.reset_input_buffer()
        return s, f"串口 {port} @9600"
    from wt901.transport import UdpTransport
    if args.device_ip:
        return UdpTransport(device=(args.device_ip, 9250)), \
            f"WiFi STA (UDP -> {args.device_ip}:9250, 目标 {args.device_id})"
    return UdpTransport(), "WiFi (UDP -> 192.168.4.1:9250)"


def drain(link) -> bytes:
    """取走链路上『此刻已排队』的字节, 绝不停下来等下一个包。

    注意: 不能用『读到超时为止』的写法 —— 设备 100 Hz 持续输出,
    那样永远等不到空闲, 会死循环卡住。
    """
    if hasattr(link, "drain_available"):
        return link.drain_available()
    n = getattr(link, "in_waiting", 0)
    return link.read(n) if n else b""


def verify_device_id(link, expected_id: str, timeout: float = 4.0) -> None:
    """写入前用目标 IP 的数据包核对设备 ID。"""
    buf = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buf.extend(drain(link))
        for raw in extract_packets(buf):
            actual_id = AttitudePacket.parse(raw).device_id
            if actual_id != expected_id:
                raise RuntimeError(f"目标 IP 对应 {actual_id}，不是 {expected_id}；已停止")
            print(f"设备 ID 核对成功: {actual_id}")
            return
        time.sleep(0.02)
    raise TimeoutError(f"未从目标 IP 收到 {expected_id} 的数据；已停止，未写入")


def collect(link, seconds: float, show: bool = True):
    """在 seconds 秒内持续收包, 统计磁场/陀螺的跨度。返回统计字典。"""
    buf = bytearray()
    mlo, mhi = [1e9] * 3, [-1e9] * 3
    glo, ghi = [1e9] * 3, [-1e9] * 3
    mag_lo, mag_hi, gmax, n = 1e9, -1e9, 0.0, 0
    t_end = time.time() + seconds
    while True:
        left = t_end - time.time()
        if left <= 0:
            break
        buf.extend(drain(link))
        for raw in extract_packets(buf):
            p = AttitudePacket.parse(raw)
            mv = (p.mag_x, p.mag_y, p.mag_z)
            gv = (p.gyro_x, p.gyro_y, p.gyro_z)
            for i in range(3):
                mlo[i], mhi[i] = min(mlo[i], mv[i]), max(mhi[i], mv[i])
                glo[i], ghi[i] = min(glo[i], gv[i]), max(ghi[i], gv[i])
            m = math.sqrt(mv[0] ** 2 + mv[1] ** 2 + mv[2] ** 2)
            mag_lo, mag_hi = min(mag_lo, m), max(mag_hi, m)
            gmax = max(gmax, abs(gv[0]), abs(gv[1]), abs(gv[2]))
            n += 1
        if show:
            span = [(mhi[i] - mlo[i]) if n else 0.0 for i in range(3)]
            print(f"\r剩余 {left:4.0f}s | 陀螺峰值 {gmax:6.1f}°/s | 磁场跨度 "
                  f"X={span[0]:5.1f} Y={span[1]:5.1f} Z={span[2]:5.1f} uT | 包 {n:5d}",
                  end="", flush=True)
        time.sleep(0.005)
    if show:
        print()
    return {
        "n": n, "gmax": gmax, "mag_lo": mag_lo, "mag_hi": mag_hi,
        "mspan": [mhi[i] - mlo[i] for i in range(3)],
        "gspan": [ghi[i] - glo[i] for i in range(3)],
    }


def verdict(st: dict) -> str:
    """根据统计给出判读, 返回 'ok' / 'magnet' / 'noturned' / 'noda'. """
    if st["n"] < 20:
        return "noda"
    rotated = st["gmax"] > 100.0
    mag_ok = max(st["mspan"]) > 40.0
    if rotated and not mag_ok:
        return "magnet"
    if not rotated:
        return "noturned"
    return "ok"


def print_verdict(st: dict) -> None:
    tag = verdict(st)
    print(f"样本 {st['n']}  磁场 |m| {st['mag_lo']:.1f}~{st['mag_hi']:.1f} uT, 跨度 "
          f"X={st['mspan'][0]:.1f} Y={st['mspan'][1]:.1f} Z={st['mspan'][2]:.1f} uT;  "
          f"陀螺峰值 {st['gmax']:.0f}°/s")
    print("判读:")
    if tag == "noda":
        print("  * 几乎没收到数据 —— 检查 WiFi 连接 (nmcli dev wifi) / 是否已被别的程序占用端口。")
    elif tag == "magnet":
        print("  * 陀螺显示确实在转, 但磁场几乎不变 -> 磁源固定在设备本体上(或磁力计饱和)。")
        print("    这不是校准能修的, 先排除:")
        print("      - 设备/线缆是否紧贴磁铁、磁吸支架、音箱?")
        print("      - 握持时手腕上是否有磁吸物(表带/手机支架)?")
        print("      - 拿到另一个房间或室外再测, 看 |m| 是否降到 ~46 uT。")
    elif tag == "noturned":
        print("  * 陀螺显示没怎么转动 -> 这次没转起来, 请重测并转得更充分。")
    else:
        print("  * 磁场随转动变化正常 -> 可以校准 (硬磁偏置可被扣除)。")


def read_mag_range(link):
    try:
        return read_registers(link, MAGRANG_ADDR, timeout=1.5)[:3]
    except Exception as e:
        print(f"  (读 MAGRANG 失败: {e})", file=sys.stderr)
        return None


def do_reset(link) -> int:
    print("还原磁校准: 磁偏移归零, MAGRANG 回默认 500 ...")
    unlock(link)
    time.sleep(UNLOCK_SETTLE)
    for a in (MAGOFF_ADDR, MAGOFF_ADDR + 1, MAGOFF_ADDR + 2):
        write_register(link, a, 0)
        time.sleep(0.15)
    for i in range(3):
        write_register(link, MAGRANG_ADDR + i, MAGRANG_DEFAULT)
        time.sleep(0.15)
    save(link)
    time.sleep(0.3)
    print("完成。")
    return 0


def do_check(link, seconds: float) -> int:
    print("非侵入检测 (不写任何寄存器)。节奏:")
    print("  前 1/3 静止不动;  后 2/3 举离桌面/笔记本, 绕 X/Y/Z 各慢转 2~3 圈。\n")
    st = collect(link, seconds)
    print_verdict(st)
    return 0 if verdict(st) == "ok" else 1


def do_calibrate(link, seconds: float) -> int:
    # 阶段 1: 预检 (校准模式下设备可能停发数据, 所以覆盖度必须在正常模式下测)
    print("== 阶段 1/2: 覆盖度预检 (正常模式) ==")
    print("请举离桌面, 绕 X/Y/Z 各慢转 2~3 圈 ...\n")
    st = collect(link, 15.0)
    print_verdict(st)
    if verdict(st) != "ok":
        print("\n预检未通过, 不进入校准 (避免把磁场减没)。")
        return 1

    # 阶段 2: 正式校准
    print(f"\n== 阶段 2/2: 正式校准 (CALSW=0x07, {seconds:.0f} 秒) ==")
    before = read_mag_range(link)
    print(f"校准前 MAGRANG = {before}")
    entered = False
    try:
        unlock(link)
        time.sleep(UNLOCK_SETTLE)
        entered = True
        write_register(link, REG_CALSW, CAL_MAG)
        time.sleep(0.3)
        print("请继续绕 X/Y/Z 各慢转 2~3 圈 ...\n")
        collect(link, seconds)                  # 校准期间监控磁场覆盖
    finally:
        if entered:
            print("退出磁场校准并保存 ...")
            write_register(link, REG_CALSW, CAL_NORMAL)
            time.sleep(0.3)
            save(link)
            time.sleep(0.3)
    print(f"校准后 MAGRANG = {read_mag_range(link)}")
    print("完成。若 |m| 仍异常, 用 --reset 还原。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WT901WIFI 磁场诊断 / 校准 / 还原")
    ap.add_argument("--net", action="store_true", help="强制走设备 AP 模式 WiFi UDP")
    ap.add_argument("--port", help="强制走指定串口")
    ap.add_argument("--device-ip", help="STA 模式目标设备 IP；发送到该 IP 的 UDP 9250")
    ap.add_argument("--device-id", help="STA 模式必须指定设备 ID，写入前核对")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--probe", action="store_true", help="只读测试目标设备 ID 和寄存器应答")
    modes.add_argument("--check", action="store_true", help="只诊断, 不写寄存器")
    modes.add_argument("--reset", action="store_true", help="还原磁校准")
    ap.add_argument("--seconds", type=float, default=None, help="check 默认 20s, 校准默认 45s")
    args = ap.parse_args()
    if sum(bool(x) for x in (args.port, args.net, args.device_ip)) > 1:
        ap.error("--port、--net、--device-ip 只能选择一种链路")
    if args.device_ip and not args.device_id:
        ap.error("STA 无线模式必须同时指定 --device-id，以防误校准其他设备")
    if args.device_id and not args.device_ip:
        ap.error("--device-id 需与 --device-ip 一起使用")

    link, how = open_link(args)
    print(f"链路: {how}")
    try:
        if args.device_ip:
            verify_device_id(link, args.device_id)
        if args.probe:
            print(f"版本寄存器读取成功: {read_registers(link, REG_VERSION)}")
            print("只读测试成功，未启动校准、未写入寄存器。")
            return 0
        if args.check:
            return do_check(link, args.seconds or 20.0)
        if args.reset:
            return do_reset(link)
        return do_calibrate(link, args.seconds or 45.0)
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
