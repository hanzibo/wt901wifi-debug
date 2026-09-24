#!/usr/bin/env python3
"""WT901WIFI 配置工具: 查看/修改寄存器, 设置 WiFi 与服务器地址。

示例:
    python configure.py show                      # 读取当前配置
    python configure.py rate 50                   # 回传频率设为 50Hz
    python configure.py wifi --ssid MyWiFi --password 12345678 \
        --server-ip 192.168.1.100 --port 1399 --proto tcp
    python configure.py raw 0x25 50               # 直接写寄存器
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

from wt901.registers import (
    read_registers, unlock, write_register, save, set_rate,
    rate_hz_from_register, REG_RATE,
)
from wt901.serial_source import DEFAULT_BAUD, DEFAULT_PORT

BANDWIDTH = {0: "256Hz", 1: "188Hz", 2: "98Hz", 3: "42Hz",
             4: "20Hz", 5: "10Hz", 6: "5Hz"}
GYRORANGE = {0: "250°/s", 1: "500°/s", 2: "1000°/s", 3: "2000°/s"}
ACCRANGE = {0: "±2g", 1: "±4g", 2: "±8g", 3: "±16g"}


def open_link(args):
    """打开配置通道: 默认串口, --net 时走 UDP (设备 AP 模式)。"""
    if args.net:
        from wt901.transport import UdpTransport
        t = UdpTransport(local_port=args.listen_port)
        print(f"配置通道: UDP -> 192.168.4.1:9250 (本机 :{args.listen_port})")
        return t
    import serial
    s = serial.Serial(args.port, args.baud, timeout=0.3)
    s.dtr = True
    s.rts = True
    time.sleep(0.3)
    return s


def open_serial(args):
    return open_link(args)


def local_ips() -> list[tuple[str, str]]:
    """返回 [(网卡名, IPv4)], 排除回环/隧道/虚拟网卡。"""
    import fcntl
    import struct as _struct

    skip = ("lo", "tun", "tap", "wg", "xray", "ppp", "docker",
            "br-", "virbr", "veth", "zt")
    out: list[tuple[str, str]] = []
    try:
        for _idx, name in socket.if_nameindex():
            if name.startswith(skip):
                continue
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                packed = fcntl.ioctl(
                    sk.fileno(), 0x8915,  # SIOCGIFADDR
                    _struct.pack("256s", name[:15].encode()))
                out.append((name, socket.inet_ntoa(packed[20:24])))
            except OSError:
                pass
            finally:
                sk.close()
    except Exception:
        pass
    return out


def cmd_show(args) -> int:
    s = open_serial(args)
    try:
        def rd(addr):
            try:
                return read_registers(s, addr)
            except Exception:
                return None

        print(f"设备: {args.port} @ {args.baud}" if not args.net
              else f"设备: WiFi (UDP 192.168.4.1:9250)")
        print()
        r = rd(REG_RATE)
        if r:
            print(f"  RATE      回传频率   {rate_hz_from_register(r[0]):8.2f} Hz  "
                  f"(寄存器={r[0]}, 周期 {r[0]}ms)")
        r = rd(0x1F)
        if r:
            print(f"  BANDWIDTH 带宽       {BANDWIDTH.get(r[0], r[0])}")
        r = rd(0x20)
        if r:
            print(f"  GYRORANGE 陀螺量程   {GYRORANGE.get(r[0], r[0])}")
        r = rd(0x21)
        if r:
            print(f"  ACCRANGE  加计量程   {ACCRANGE.get(r[0], r[0])}")
        r = rd(0x24)
        if r:
            print(f"  AXIS6     算法       {'6轴(相对航向)' if r[0] else '9轴(绝对航向)'}")
        r = rd(0x2D)
        if r:
            print(f"  WIFIMODE  WiFi模式   {'AP(自建热点)' if r[0] == 2 else 'STA(连路由器)'}")
        r = rd(0x2E)
        if r:
            print(f"  VERSION   版本号     {r[0]}")
    finally:
        s.close()
    return 0


def cmd_rate(args) -> int:
    s = open_serial(args)
    try:
        print(f"设置回传频率为 {args.hz} Hz ...")
        period = set_rate(s, args.hz)
        time.sleep(0.5)
        save(s)
        time.sleep(0.5)
        back = read_registers(s, REG_RATE)[0]
        print(f"  写入周期 {period}ms, 回读 {back} -> "
              f"{rate_hz_from_register(back):.2f} Hz")
        print("  已保存。")
    finally:
        s.close()
    return 0


def cmd_raw(args) -> int:
    s = open_serial(args)
    try:
        addr = int(args.addr, 0)
        value = int(args.value, 0)
        unlock(s)
        time.sleep(0.6)
        write_register(s, addr, value)
        time.sleep(0.4)
        save(s)
        time.sleep(0.4)
        back = read_registers(s, addr)[0]
        print(f"写 0x{addr:02X} = {value}, 回读 {back}")
    finally:
        s.close()
    return 0


def cmd_wifi(args) -> int:
    if args.server_ip == "auto":
        ips = local_ips()
        if not ips:
            print("无法自动获取本机 IP, 请用 --server-ip 指定", file=sys.stderr)
            return 1
        print("检测到本机地址:")
        for name, ip in ips:
            print(f"  {name:12} {ip}")
        args.server_ip = ips[0][1]
        print(f"自动选用: {args.server_ip}\n")

    proto = args.proto.upper()
    cmd = f'IPWIFI:"{args.ssid}","{args.password}";{proto}{args.server_ip},{args.port}\r\n'
    print(f"将发送指令: {cmd.strip()}")

    s = open_serial(args)
    try:
        s.reset_input_buffer()
        s.write(cmd.encode("ascii"))
        s.flush()
        time.sleep(1.0)
        print("\n指令已发送。设备会切到 STA 模式并尝试连接路由器。")
        print(f"确认电脑与设备在同一 2.4GHz 局域网, 且本机 {args.server_ip} 的 "
              f"{args.port} 端口可被访问 (注意防火墙)。")
        print(f"然后用: python wt901_viz/viewer.py --net {args.proto} "
              f"--listen-port {args.port}")
    finally:
        s.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WT901WIFI 配置工具")
    ap.add_argument("--port", default=DEFAULT_PORT, help="串口设备")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--net", action="store_true",
                    help="改用 WiFi(UDP) 配置 (设备拔掉 USB / AP 模式时使用)")
    ap.add_argument("--listen-port", type=int, default=1399,
                    help="--net 时本机监听端口 (默认 1399)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="读取当前配置").set_defaults(func=cmd_show)

    p = sub.add_parser("rate", help="设置回传频率")
    p.add_argument("hz", type=float)
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("raw", help="直接写寄存器")
    p.add_argument("addr")
    p.add_argument("value")
    p.set_defaults(func=cmd_raw)

    p = sub.add_parser("wifi", help="设置 WiFi 与服务器地址 (STA 模式)")
    p.add_argument("--ssid", required=True, help="2.4GHz WiFi 名称")
    p.add_argument("--password", required=True)
    p.add_argument("--server-ip", default="auto",
                   help="接收数据的电脑 IP, auto = 自动检测")
    p.add_argument("--port", type=int, default=1399)
    p.add_argument("--proto", choices=("tcp", "udp"), default="tcp")
    p.set_defaults(func=cmd_wifi)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
