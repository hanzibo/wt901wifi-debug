#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一台电脑同时接收多个 WT901WIFI。

前提: 各设备已切到 STA 模式并连上同一个 2.4GHz 局域网, 且都往本机 <ip>:1399 推送。
每个数据包都带 12 字节设备 ID, 因此**同一个端口即可区分多台设备**, 无需分配不同端口。

用法:
    python multi9.py                        # UDP 1399, 表格里每台设备一行
    python multi9.py --full                 # 每行显示全部 9 个量
    python multi9.py --net tcp              # 用 TCP 收 (设备需配 TCPIP:)
    python multi9.py --lines                # 滚动输出, 不原地刷新
"""
from __future__ import annotations

import argparse
import sys
import time


def make_source(args):
    if args.net == "tcp":
        from wt901.net_source import TcpAttitudeSource
        return TcpAttitudeSource(args.listen_port), f"TCP :{args.listen_port}"
    from wt901.net_source import UdpAttitudeSource
    return UdpAttitudeSource(args.listen_port), f"UDP :{args.listen_port}"


def main() -> int:
    ap = argparse.ArgumentParser(description="多台 WT901WIFI 终端查看")
    ap.add_argument("--net", choices=("udp", "tcp"), default="udp")
    ap.add_argument("--listen-port", type=int, default=1399)
    ap.add_argument("--hz", type=float, default=5.0, help="刷新率上限, 默认 5 Hz")
    ap.add_argument("--full", action="store_true", help="每行显示全部 9 个量")
    ap.add_argument("--lines", action="store_true", help="滚动输出")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="设备超过 N 秒没数据就不再显示 (0=一直显示)")
    args = ap.parse_args()

    src, how = make_source(args)
    inplace = (not args.lines) and sys.stdout.isatty()
    print(f"# 监听 {how}, Ctrl+C 退出", file=sys.stderr)

    devices: dict[str, tuple] = {}      # id -> (packet, last_seen, count)
    next_out = 0.0
    block = 0
    try:
        while True:
            for p in src.poll():
                if p.device_id in devices:
                    _, _, c = devices[p.device_id]
                else:
                    c = 0
                devices[p.device_id] = (p, time.time(), c + 1)

            now = time.time()
            if now < next_out:
                time.sleep(0.005)
                continue
            next_out = now + (1.0 / args.hz if args.hz > 0 else 0)

            if args.timeout > 0:
                for k in [k for k, v in devices.items() if now - v[1] > args.timeout]:
                    del devices[k]

            rows = []
            for did in sorted(devices):
                p, seen, cnt = devices[did]
                age = now - seen
                if args.full:
                    rows.append(
                        f"{did}  R={p.roll:8.2f} P={p.pitch:7.2f} Y={p.yaw:8.2f} | "
                        f"G=({p.gyro_x:7.2f},{p.gyro_y:7.2f},{p.gyro_z:7.2f}) | "
                        f"A=({p.acc_x:+6.3f},{p.acc_y:+6.3f},{p.acc_z:+6.3f}) | {cnt}")
                else:
                    rows.append(f"{did}  R={p.roll:8.2f} P={p.pitch:7.2f} Y={p.yaw:8.2f}"
                                f"   A=({p.acc_x:+6.3f},{p.acc_y:+6.3f},{p.acc_z:+6.3f})"
                                f"  {'在线' if age < 1.0 else f'陈旧 {age:4.1f}s'}  {cnt}")

            head = f"# 已发现 {len(devices)} 台设备   ({how})   {time.strftime('%H:%M:%S')}"
            lines = [head] + rows if rows else [head, "(等待数据 ...)"]
            if inplace:
                if block:
                    sys.stdout.write(f"\033[{block}A")
                sys.stdout.write("\n".join(f"{l:<100}" for l in lines) + "\n")
                block = len(lines)
            else:
                print("\n".join(lines))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
