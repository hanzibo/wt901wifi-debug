#!/usr/bin/env python3
"""WT901WIFI 实时姿态读取。

示例:
    python read_attitude.py                     # 默认 /dev/ttyUSB0 @9600
    python read_attitude.py --port /dev/ttyUSB1
    python read_attitude.py --csv attitude.csv  # 同时记录到 CSV
    python read_attitude.py --angles-only       # 只打印姿态角
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

from wt901 import AttitudePacket, SerialAttitudeSource
from wt901.serial_source import DEFAULT_BAUD, DEFAULT_PORT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="读取 WT901WIFI 姿态信息")
    p.add_argument("--port", default=DEFAULT_PORT, help=f"串口 (默认 {DEFAULT_PORT})")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"波特率 (默认 {DEFAULT_BAUD})")
    p.add_argument("--csv", metavar="FILE", help="把数据同时写入 CSV 文件")
    p.add_argument("--angles-only", action="store_true", help="只打印 Roll/Pitch/Yaw")
    p.add_argument("--hz", type=float, default=0.0,
                   help="屏幕刷新频率上限, 0 表示每包都打印")
    return p


def main() -> int:
    args = build_parser().parse_args()

    fh = None
    if args.csv:
        fh = open(args.csv, "w", newline="", encoding="utf-8")

    print(f"打开 {args.port} @ {args.baud} ... (Ctrl+C 退出)", file=sys.stderr)
    try:
        src = SerialAttitudeSource(args.port, args.baud)
    except Exception as e:
        print(f"打开串口失败: {e}", file=sys.stderr)
        print("提示: 确认设备已开机, 且端口正确 (ls /dev/ttyUSB*)", file=sys.stderr)
        return 1

    csv_writer = None
    interval = 1.0 / args.hz if args.hz > 0 else 0.0
    last = 0.0
    count = 0
    t0 = time.time()

    try:
        with src:
            for pkt in src:
                count += 1
                if csv_writer is None and args.csv:
                    csv_writer = csv.DictWriter(fh, fieldnames=list(pkt.as_row().keys()))
                    csv_writer.writeheader()

                if csv_writer is not None:
                    csv_writer.writerow(pkt.as_row())
                    fh.flush()

                now = time.time()
                if interval and now - last < interval:
                    continue
                last = now

                if args.angles_only:
                    print(f"Roll={pkt.roll:8.2f}  Pitch={pkt.pitch:8.2f}  Yaw={pkt.yaw:8.2f}")
                else:
                    print(pkt)
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()
        dt = time.time() - t0
        if dt > 0:
            print(f"\n共 {count} 包, {dt:.1f}s, 平均 {count/dt:.1f} Hz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
