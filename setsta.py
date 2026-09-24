#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 WT901WIFI 切换到 STA 模式 (连路由器), 并指定把数据推给哪台电脑。

指令语法来自官方《WT901WIFI protocol.pdf》2.2.1 ASCII setting format:
    WITWIFI:"<SSID>","<password>"\\r\\n     设置 WiFi, 发完自动切 STA 模式 (2.4GHz)
    UDPIP:<电脑IP>,<端口>\\r\\n             指定 UDP 服务器 (推荐, 多设备用这个)
    TCPIP:<电脑IP>,<端口>\\r\\n             指定 TCP 服务器
    WIT<电脑IP>\\r\\n                       让设备知道服务器 IP (可配广播发现)

用法:
    # 先看会发什么 (不真正下发)
    python setsta.py --ssid MyWiFi --password 12345678 --server-ip 192.168.1.100 --dry-run

    # 通过 USB 串口配置 (推荐, 每台设备插一次最稳)
    python setsta.py --port /dev/ttyUSB0 --ssid MyWiFi --password 12345678

    # 通过设备自带 AP 热点配置 (电脑先连上名为 WTxxxx 的热点)
    python setsta.py --net --ssid MyWiFi --password 12345678

    # 已知设备已在同一局域网, 只广播本机 IP 让所有设备自动找到服务器
    python setsta.py --discover --server-ip 192.168.1.100
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

from configure import local_ips


def build_commands(args) -> list[str]:
    cmds = [f'WITWIFI:"{args.ssid}","{args.password}"\r\n']
    dest = f'{args.proto.upper()}IP:{args.server_ip},{args.server_port}\r\n'
    cmds.append(dest)
    if args.discover:
        cmds.append(f'WIT{args.server_ip}\r\n')
    return cmds


def send_serial(args, cmds):
    import serial
    s = serial.Serial(args.port, args.baud, timeout=0.5)
    s.dtr = s.rts = True
    time.sleep(0.2)
    try:
        for c in cmds:
            print(f"  -> 串口写入: {c.strip()}")
            s.write(c.encode("ascii"))
            s.flush()
            time.sleep(0.8)
    finally:
        s.close()


def send_net(args, cmds):
    from wt901.transport import UdpTransport
    t = UdpTransport()
    try:
        for c in cmds:
            print(f"  -> UDP 发往 {t.device}: {c.strip()}")
            t.write(c.encode("ascii"))
            time.sleep(0.8)
    finally:
        t.close()


def discover_broadcast(ip: str) -> None:
    """手册: 向 255.255.255.255:9250 广播本机 IP, 设备收到后会切到该 IP。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    msg = f"WIT{ip}\r\n".encode("ascii")
    s.sendto(msg, ("255.255.255.255", 9250))
    s.close()
    print(f"  -> 已广播 {msg!r} 到 255.255.255.255:9250")


def main() -> int:
    ap = argparse.ArgumentParser(description="WT901WIFI 切 STA 模式")
    ap.add_argument("--ssid", help="2.4GHz WiFi 名称")
    ap.add_argument("--password", help="WiFi 密码")
    ap.add_argument("--server-ip", default="auto", help="接收数据的电脑 IP, auto=自动检测")
    ap.add_argument("--server-port", type=int, default=1399, help="服务器端口, 默认 1399")
    ap.add_argument("--proto", choices=("udp", "tcp"), default="udp",
                    help="STA 下先用 UDP (手册强烈建议, 切 TCP 容易失联)")
    ap.add_argument("--port", help="USB 串口, 如 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--net", action="store_true", help="通过设备 AP 热点配置 (UDP -> 192.168.4.1:9250)")
    ap.add_argument("--discover", action="store_true", help="附带发送 WIT<ip> 广播发现指令")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要发送的指令")
    args = ap.parse_args()

    if args.server_ip == "auto":
        ips = local_ips()
        if not ips:
            print("无法自动检测本机 IP, 请显式指定 --server-ip", file=sys.stderr)
            return 1
        print("检测到本机地址:")
        for name, ip in ips:
            print(f"  {name:12} {ip}")
        args.server_ip = ips[0][1]
        print(f"自动选用: {args.server_ip}\n")

    if not args.ssid:
        # 只做发现广播
        if not args.discover:
            print("需要 --ssid/--password (或单独用 --discover)", file=sys.stderr)
            return 1
        cmds = [f'WIT{args.server_ip}\r\n']
    else:
        cmds = build_commands(args)

    print("将要发送的指令:")
    for c in cmds:
        print(f"  {c.strip()}")

    if args.dry_run:
        print("\n(--dry-run, 未实际下发)")
        return 0

    if args.port:
        print(f"\n通过串口 {args.port} 下发 ...")
        send_serial(args, cmds)
        print("\n完成: 设备已写入 WiFi 并切到 STA 模式。")
    elif args.net:
        print("\n通过设备 AP (192.168.4.1:9250) 下发 ...")
        send_net(args, cmds)
    else:
        print("\n(未指定 --port/--net, 走广播发现)")
        discover_broadcast(args.server_ip)
        return 0

    if args.discover:
        discover_broadcast(args.server_ip)

    print("\n注意:")
    print("  * 设备切到 STA 后, 192.168.4.1 就不存在了, 只能通过路由器局域网访问。")
    print(f"  * 电脑需与设备在同一 2.4GHz 局域网, 且放行本机 UDP {args.server_port}。")
    print(f"  * 本机接收: python multi9.py --listen-port {args.server_port}")
    print("  * 配置错了: 按住设备按键约 2 秒恢复 AP 模式, 或重新插 USB 串口配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
