#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WT901WIFI 终端数据打印 (含地磁量) —— 带详细注释版。

打印 12 个量:
    3 个姿态角   : Roll / Pitch / Yaw          (°)
    3 个角速度   : Gyro X / Y / Z              (°/s)   <- 设备原始输出
    3 个线加速度 : Acc X / Y / Z               (g)
    3 个磁场     : Mag X / Y / Z               (µT)    <- 地磁量

注意: 设备输出的是「角速度」而非「角加速度」。若确实要角加速度,
加 --ang-accel, 脚本用陀螺差分算出 (°/s²), 会明显更抖。

用法:
    python print9.py                     # 自动: 有 /dev/ttyUSB0 就用串口, 没有或打不开则走 WiFi UDP 1399
    python print9.py --net               # 强制 WiFi (设备推送到本机 1399)
    python print9.py --port /dev/ttyUSB0
    python print9.py --hz 10             # 刷新率上限, 默认 10 Hz (设备本身 100 Hz)
    python print9.py --lines             # 每包一行滚动输出 (默认原地刷新)
    python print9.py --ang-accel         # 中间那组改成角加速度
    python print9.py --count 200         # 打印 200 条后退出
    python print9.py --csv out.csv       # 同时写 CSV

===============================================================================
数据是怎么从 IMU 跑到屏幕上的?
===============================================================================

关键点先说三条, 后面代码里都会对应到:

  1) **IMU 是发送方, 电脑是接收方**。设备上电后作为客户端, 把数据「主动推」
     到电脑的 UDP 1399 端口 (地址由设备的 UDPIP/UDPPORT 寄存器决定)。
     所以本程序做的事是「开一个 socket 等它发过来」, 而不是「去问设备要」。
     这也是为什么设备必须先配好、且和电脑在同一网络里。

  2) **真正的阻塞式收数据发生在后台线程里**, 主线程(打印循环)永远不碰 socket。
     线程收到字节流后解析成对象, 塞进一个 queue.Queue。
     主线程只是从队列里「取走已经解析好的对象」。

  3) **本文件其实只负责「显示」**。把字节变成数字的活儿在 wt901/ 包里:
       wt901/protocol.py     —— 54 字节数据包 → AttitudePacket 对象
       wt901/net_source.py   —— UDP/TCP 接收 + 后台线程
       wt901/serial_source.py—— 串口接收

完整链路:

    ┌─────────────┐   每 10ms 一个 54 字节包
    │  WT901WIFI  │ ─────────────────────────────┐
    │  (IMU 设备) │                              │  UDP 到 电脑IP:1399
    └─────────────┘                              ▼
                         ┌──────────────────────────────────────────┐
                         │ 【后台线程】UdpAttitudeSource._serve()    │  ← net_source.py
                         │   socket.recvfrom(2048)  阻塞等字节       │
                         │        │                                 │
                         │        ▼                                 │
                         │   _feed(data, buffer)                    │
                         │     buffer.extend(data)   ← 滚动字节缓冲  │
                         │        │                                 │
                         │        ▼                                 │
                         │   extract_packets(buffer)  ← protocol.py │
                         │     按 'WT' 开头 + CRLF 结尾切出 54 字节 │
                         │        │                                 │
                         │        ▼                                 │
                         │   AttitudePacket.parse(raw)              │
                         │     int16 小端 → 物理量 (g / °/s / µT / °)│
                         │        │                                 │
                         │        ▼                                 │
                         │   queue.put(pkt)          ← 线程安全的队列│
                         └────────────────┬─────────────────────────┘
                                          │
                         ┌────────────────▼─────────────────────────┐
                         │ 【主线程】src.poll()                      │  ← 本文件
                         │   非阻塞地把队列里所有包一次掏空          │
                         │        │                                 │
                         │        ▼                                 │
                         │   pkts[-1]  只取最新的一个用来显示        │
                         │        │                                 │
                         │        ▼                                 │
                         │   render_block() → 打印 / 写 CSV         │
                         └──────────────────────────────────────────┘

为什么要把这两件事拆成两个线程?
    因为 recvfrom() 是「阻塞」的 —— 没有数据时它会一直卡在那里。
    如果主线程直接去 recv, 界面就会卡死、也没法按 Ctrl+C 退出。
    拆开之后: 后台线程专职等待数据, 主线程只管刷新屏幕, 互不干扰。

数据包长什么样 (54 字节, 串口和 WiFi 完全一样)?
    偏移  字段                                        换算 (见 protocol.py)
    0-11  设备ID          "WT5500011408" (12 字节 ASCII)
    12-19 年 月 日 时 分 秒 (年要 +2000!)
    20-25 AX AY AZ        加速度  int16  → /32768*16      (g)
    26-31 GX GY GZ        角速度  int16  → /32768*2000    (°/s)
    32-37 HX HY HZ        磁场    int16  → *100/1024      (µT)
    38-43 Roll Pitch Yaw  姿态角  int16  → /32768*180     (°)
    44-45 温度             int16  → /100                 (℃)
    46-47 电量             uint16 → /100                 (V)
    48-49 RSSI 信号强度                                     (dBm)
    50-51 固件版本号
    52-53 0x0D 0x0A       回车换行, 用来界定包尾

    注意: 数据流里**没有校验和**, 只靠包头 "WT" 和包尾 CRLF 定位。
    所以协议解析里做了「对不上就丢弃 1 字节重新找」的容错 (见 extract_packets)。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# 只是两个用来对齐输出的名字表, 和协议无关
ANGLE = ("Roll  ", "Pitch ", "Yaw   ")
AXIS = ("X", "Y", "Z")


def make_source(args):
    """【数据来源选择】返回 (数据源对象, 人读的描述)。

    无论走串口还是网络, 返回的对象都提供同一套接口:
        src.poll()   -> 非阻塞取走当前已解析好的数据包列表
        src.close()  -> 关闭
    这叫「鸭子类型」: 本文件后面的主循环完全不关心底层是串口还是 socket。

    --- 两个数据源内部各自做了什么 ---

    UdpAttitudeSource(port)
        · 建一个 UDP socket, bind 到本机 0.0.0.0:1399  (即「当服务器」)
        · 把接收缓冲调大到 1MB, 防止刷屏忙时内核丢包
        · 起一个**后台守护线程**执行 _serve(): 死循环 recvfrom() 收包
        → 这台电脑就成了 UDP 服务器, 设备主动往这儿推

    SerialAttitudeSource(port, baud)
        · 用 pyserial 打开 /dev/ttyUSB0 @9600 8N1
        · 拉高 DTR/RTS (有些 CH340 要这样才开始吐数据)
        · 注意: 串口被固件限制在约 2Hz, WiFi 才能到 100/200Hz
    """
    # ---- 选择规则 (按优先级) ----
    #   1) --net             -> 强制走 WiFi
    #   2) --port 指定了串口  -> 必须用串口; 打不开就明确报错退出 (不偷偷换路径)
    #   3) 其它              -> 有 /dev/ttyUSB0 就先试串口, 打不开再退到 WiFi
    #
    # 为什么第 3 条要「打不开就退」?
    # 因为 USB 只插着充电时 /dev/ttyUSB0 照样存在, 但普通用户常常没有 dialout
    # 权限 —— 如果直接抛异常, 整个程序会崩, 而其实 WiFi 是通的。
    if not args.net:
        port = args.port or ("/dev/ttyUSB0" if os.path.exists("/dev/ttyUSB0") else None)
        if port:
            from wt901.serial_source import SerialAttitudeSource
            try:
                # 构造函数里就打开了串口
                return SerialAttitudeSource(port, args.baud), f"串口 {port} @{args.baud}"
            except Exception as e:
                if args.port:
                    # 用户点名要用串口, 那就把原因说清楚再退出
                    _report_serial_failure(args.port, e)
                    raise SystemExit(1)
                print(f"# 串口 {port} 打不开 ({e}), 自动改用 WiFi", file=sys.stderr)

    from wt901.net_source import UdpAttitudeSource
    # 构造函数里就已经 bind 好端口并启动后台线程了
    return UdpAttitudeSource(args.listen_port), f"WiFi UDP :{args.listen_port}"


def _report_serial_failure(port: str, err: Exception) -> None:
    """串口打不开时给人看得懂的提示 (最常见原因是没有 dialout 权限)。"""
    print(f"打不开串口 {port}: {err}", file=sys.stderr)
    if "Permission denied" in str(err):
        print(
            "  原因: 当前用户不在 dialout 组, 或该组权限尚未在本会话生效。\n"
            "  (查询: id -nG | tr ' ' '\\n' | grep dialout)\n"
            "  解决办法任选其一:\n"
            "    sudo usermod -aG dialout $USER   然后注销重新登录 (一劳永逸)\n"
            "    newgrp dialout                    只对当前终端临时生效\n"
            "    python print9.py --net            直接走 WiFi 通路",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。跟数据获取相关的只有 --port/--baud/--net/--listen-port,
    其余都是显示选项 (不影响采数)。"""
    p = argparse.ArgumentParser(description="WT901WIFI 终端数据打印 (含地磁)")
    p.add_argument("--port", help="串口设备 (默认 /dev/ttyUSB0, 不存在则走 WiFi)")
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument("--net", action="store_true", help="强制走 WiFi UDP")
    p.add_argument("--listen-port", type=int, default=1399, help="WiFi 监听端口 (默认 1399)")
    # --hz 只限制「打印频率」, 不限制「采集频率」: 数据一直照收, 只是不都显示
    p.add_argument("--hz", type=float, default=10.0, help="刷新率上限, 默认 10 Hz")
    p.add_argument("--lines", action="store_true", help="每包一行滚动输出")
    p.add_argument("--ang-accel", action="store_true", help="中间那组显示角加速度 (陀螺差分)")
    p.add_argument("--count", type=int, default=0, help="打印 N 条后退出 (0=一直打印)")
    p.add_argument("--csv", metavar="FILE", help="同时写入 CSV")
    return p


def render_block(p, mid, mid_label, mid_unit, rate, printed):
    """把**一个已解析好的数据包**排版成多行文本 (原地刷新用)。

    这里纯粹是字符串拼接 + 对齐, 没有任何 IO, 也不碰设备。
    参数:
        p          : AttitudePacket 对象 (protocol.py 里解析出来的)
        mid        : 中间那一列要显示的三元组 —— 角速度或角加速度
        mid_label  : 该列的表头文字, "Gyro" 或 "AngAcc"
        mid_unit   : 该列的单位, "°/s" 或 "°/s²"
        rate       : 实测收包速率 (Hz), 由主循环统计
        printed    : 已打印条数
    """
    # 从数据包对象里取出三组值。这些属性是 protocol.py 解析时就算好的物理量,
    # 此处直接用, 不需要再做任何换算。
    ang = (p.roll, p.pitch, p.yaw)
    acc = (p.acc_x, p.acc_y, p.acc_z)
    mag = (p.mag_x, p.mag_y, p.mag_z)
    # 磁场模长 |m|: 三轴平方和开根。用来一眼判断磁场是否正常 (~46µT 是地球磁场)。
    mnorm = math.sqrt(mag[0] ** 2 + mag[1] ** 2 + mag[2] ** 2)

    lines = [
        # 表头: 设备ID / 实测速率 / 已打印包数 / 磁场模长
        f"{p.device_id}   {rate:5.1f} Hz   包 {printed}   |m| = {mnorm:7.1f} µT",
        "─" * 70,
        # 三列的列名。{:>16s} 表示右对齐到 16 字符宽
        f"{'':10s}{mid_label + ' (' + mid_unit + ')':>16s}"
        f"{'加速度 (g)':>14s}{'磁场 (µT)':>14s}",
    ]
    # 三行数据, 每行 4 组: 角度 | 角速度(或角加速度) | 加速度 | 磁场
    for i in range(3):
        lines.append(
            f"  {ANGLE[i]} {ang[i]:8.2f}   {AXIS[i]} {mid[i]:8.2f}   "
            f"{AXIS[i]} {acc[i]:+7.3f}   {AXIS[i]} {mag[i]:8.1f}"
        )
    return lines


def main() -> int:
    args = build_parser().parse_args()

    # ① 建立数据源。执行到这一行时, UdpAttitudeSource 已经:
    #    bind 好 1399 端口 + 启动了后台线程开始收包 —— 数据此刻已经在往队列里进了。
    src, how = make_source(args)

    # 是否原地刷新: 只有在交互式终端 (TTY) 且没指定 --lines 时才用 ANSI 光标回退。
    # 重定向到文件时不是 TTY, 就退化成逐块输出, 免得出现乱七八糟的控制字符。
    inplace = (not args.lines) and sys.stdout.isatty()

    # CSV: 与屏幕显示同一份数据, 只是另存一份
    fh = open(args.csv, "w", encoding="utf-8") if args.csv else None
    if fh:
        fh.write("roll,pitch,yaw,gyro_x,gyro_y,gyro_z,"
                 "acc_x,acc_y,acc_z,mag_x,mag_y,mag_z\n")

    # 提示信息走 stderr, 这样 stdout 重定向时不会污染数据
    print(f"# {how}   Ctrl+C 退出", file=sys.stderr)

    # --- 循环里用到的状态变量 ---
    prev_g = None          # 上一次的角速度, 仅 --ang-accel 差分用
    prev_t = None          # 上一次的时间戳, 仅 --ang-accel 差分用
    printed = 0            # 已打印条数
    rate_n = 0             # 本轮统计周期内收到的包数
    rate_t0 = time.time()  # 本轮统计周期的起点
    rate = 0.0             # 实测收包速率 (Hz)
    next_out = 0.0         # 下一次允许打印的时刻
    block_lines = 0        # 上一屏占了几行, 原地刷新时要用它把光标退回开头

    try:
        # ② 主循环。注意这里**完全没有 socket 操作** —— 只是不停地问
        #    "队列里现在有什么?", 有就显示, 没有就睡 5ms 让出 CPU。
        while True:
            # poll() 是「非阻塞掏空队列」: 一次把已解析好的包全部取走并返回列表。
            # 因为有这一步, 队列不会被撑爆; 但也意味着这次拿到的可能是**一批**包
            # (10Hz 打印 + 100Hz 采集 → 每次大约 10 个)。
            pkts = src.poll()
            if not pkts:
                # 队列是空的, 说明 5ms 内设备没发新数据。睡一下, 否则就是死循环空转占满 CPU。
                time.sleep(0.005)
                continue

            # 收包速率统计: 累计「收到的包数」而不是「打印的次数」,
            # 这样统计出来的才是设备真实回传速率 (≈100Hz), 与 --hz 无关。
            rate_n += len(pkts)
            now = time.time()
            if now - rate_t0 >= 0.5:      # 每 0.5 秒算一次, 免得数字乱跳
                rate = rate_n / (now - rate_t0)
                rate_t0, rate_n = now, 0

            # ---- 显示节流 ----
            # 设备 100Hz 地发, 但我们不需要 100Hz 地刷屏 (人眼也看不清)。
            # 没到 next_out 就直接丢弃这一批, 只把时间用在刷屏上。
            # 注意: 丢弃只是不显示, 采集本身一直在进行、速率统计也没受影响。
            if now < next_out:
                continue
            next_out = now + (1.0 / args.hz if args.hz > 0 else 0)

            # 一批包里可能有 10 个, 显示只需要最新的那一个 —— 取最后一个。
            p = pkts[-1]

            # ---- 中间那一列: 角速度 or 角加速度 ----
            g = (p.gyro_x, p.gyro_y, p.gyro_z)
            if args.ang_accel:
                # 设备不输出角加速度, 只能自己对角速度做差分: α ≈ Δω/Δt
                # 这是「后验」的, 且会被陀螺噪声放大, 所以数值会明显更抖。
                # 用实测时间差 dt 而不是标称 1/100 s, 否则时间抖动会变成数值抖动。
                if prev_g is not None and prev_t is not None and now - prev_t > 1e-6:
                    dt = now - prev_t
                    mid = tuple((g[i] - prev_g[i]) / dt for i in range(3))
                else:
                    mid = (0.0, 0.0, 0.0)   # 第一个包没有前值, 先给 0
                prev_g, prev_t = g, now
                mid_label, mid_unit = "AngAcc", "°/s²"
            else:
                # 默认直接显示设备给的角速度原始值
                mid = g
                mid_label, mid_unit = "Gyro", "°/s"

            acc = (p.acc_x, p.acc_y, p.acc_z)
            mag = (p.mag_x, p.mag_y, p.mag_z)

            # ---- 写 CSV ----
            # 注意 CSV 里始终写角速度 g, 而不是 --ang-accel 的差分值,
            # 保证 CSV 内容稳定、可复现。
            if fh:
                fh.write(f"{p.roll:.3f},{p.pitch:.3f},{p.yaw:.3f},"
                         f"{g[0]:.3f},{g[1]:.3f},{g[2]:.3f},"
                         f"{acc[0]:.4f},{acc[1]:.4f},{acc[2]:.4f},"
                         f"{mag[0]:.1f},{mag[1]:.1f},{mag[2]:.1f}\n")
                fh.flush()   # 立刻落盘, 这样 Ctrl+C 或断电时已写的数据不会丢

            printed += 1

            # ---- 输出: 两种模式 ----
            if args.lines:
                # 滚动模式: 一个包一行, 方便重定向到文件/用 grep 过滤
                print(f"{p.device_id}  "
                      f"R={p.roll:8.2f} P={p.pitch:7.2f} Y={p.yaw:8.2f} | "
                      f"G=({mid[0]:8.2f},{mid[1]:8.2f},{mid[2]:8.2f}) | "
                      f"A=({acc[0]:+6.3f},{acc[1]:+6.3f},{acc[2]:+6.3f}) | "
                      f"M=({mag[0]:7.1f},{mag[1]:7.1f},{mag[2]:7.1f})µT")
            else:
                # 表格模式: 原地刷新。原理是 ANSI 转义序列 "\033[<n>A"
                # = 把光标上移 n 行, 然后重新覆盖写同样的行数。
                lines = render_block(p, mid, mid_label, mid_unit, rate, printed)
                if inplace and block_lines:
                    sys.stdout.write(f"\033[{block_lines}A")   # 光标退回上一屏开头
                # {l:<72s} 左对齐补空格到 72 列: 必须补齐, 否则新内容比旧的短时
                # 会残留上一帧的尾巴, 看起来像花屏。
                sys.stdout.write("\n".join(f"{l:<72s}" for l in lines) + "\n")
                sys.stdout.flush()          # 不加这句会攒在缓冲区里, 看不到实时刷新
                block_lines = len(lines)    # 记住这次占了几行, 下次好回退

            if args.count and printed >= args.count:
                break

    except KeyboardInterrupt:
        # Ctrl+C 是正常退出手段, 不当作错误
        pass
    finally:
        # ③ 收尾。close() 会置位 stop 事件 (让后台线程退出循环)
        #    并关闭 socket。线程是 daemon, 所以即使它还没退, 进程也能正常结束。
        try:
            src.close()
        except Exception:
            pass
        if fh:
            fh.close()
        if not args.lines:
            print()   # 原地刷新模式下光标停在表格里, 补一个换行让终端提示符回到新行
    print(f"# 共打印 {printed} 条", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
