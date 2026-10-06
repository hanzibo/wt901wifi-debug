#!/usr/bin/env python3
"""Read-only, single-WT901 long-run IMU recorder (exclusive UDP 1399)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wt901.gui_udp import GuiUdpSource
from wt901.protocol import AttitudePacket


PACKET_FIELDS = tuple(item.name for item in fields(AttitudePacket))
CSV_FIELDS = ("sample", "host_recv_mono_ns", "host_write_wall_ns",
              "elapsed_s", "phase", "device_time") + PACKET_FIELDS
DEFAULT_OUTPUT = ROOT / "OptUSBProj-ubuntu" / "recordings" / "imu_long_run"


def phase_at(elapsed: float, neutral_s: float, movement_s: float) -> str | None:
    if elapsed < 0:
        return None
    if elapsed < neutral_s:
        return "neutral_start"
    if elapsed < neutral_s + movement_s:
        return "movement"
    if elapsed < 2 * neutral_s + movement_s:
        return "neutral_end"
    return None


def clock_text(seconds: float) -> str:
    """Human-readable elapsed/countdown time without rounding away seconds."""
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return (f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
            if hours else f"{minutes:02d}:{seconds_part:02d}")


def progress_text(elapsed: float, neutral_s: float, movement_s: float,
                  packets: int, last_packet_age_s: float, battery_v: float) -> str:
    """Build one progress report; elapsed is measured by a monotonic clock."""
    total = 2 * neutral_s + movement_s
    phase = phase_at(elapsed, neutral_s, movement_s)
    if phase == "neutral_start":
        label, phase_end = "起始中立", neutral_s
    elif phase == "movement":
        label, phase_end = "运动", neutral_s + movement_s
    elif phase == "neutral_end":
        label, phase_end = "结束中立", total
    else:
        label, phase_end = "完成", total
    remaining = math.ceil(max(0.0, phase_end - elapsed))
    link = "在线" if last_packet_age_s < 2.0 else f"断流 {last_packet_age_s:.1f}s"
    return (f"已用 {clock_text(elapsed)} / {clock_text(total)} | {label}"
            f"剩余 {clock_text(remaining)} | 包 {packets} | "
            f"{link} | 电压 {battery_v:.2f} V")


class Recorder:
    def __init__(self, output: Path, device_id: str, start_ns: int,
                 neutral_s: float, movement_s: float, endpoint: tuple[str, int],
                 initial_packet: AttitudePacket) -> None:
        self.path = output / (datetime.now().strftime("%Y%m%d-%H%M%S")
                              + "-" + uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.device_id = device_id
        self.start_ns = start_ns
        self.start_wall_ns = time.time_ns()
        self.neutral_s = neutral_s
        self.movement_s = movement_s
        self.endpoint = endpoint
        self.initial_battery_v = initial_packet.battery_v
        self.initial_battery_percent = initial_packet.battery_percent
        self.counts = {"neutral_start": 0, "movement": 0, "neutral_end": 0}
        self.closed = False
        self.last_received_ns: int | None = None
        self.final_battery_v: float | None = None
        self.file = (self.path / "samples.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self._manifest("recording")

    def _manifest(self, status: str) -> None:
        data = {
            "format": "wt901-imu-long-run-v1",
            "device_id": self.device_id,
            "endpoint_at_start": [self.endpoint[0], self.endpoint[1]],
            "start_mono_ns": self.start_ns,
            "start_wall_ns": self.start_wall_ns,
            "neutral_seconds_each": self.neutral_s,
            "movement_seconds": self.movement_s,
            "phase_counts": self.counts,
            "initial_battery_v": self.initial_battery_v,
            "initial_battery_percent_estimate": self.initial_battery_percent,
            "final_battery_v": self.final_battery_v,
            "last_elapsed_s": None if self.last_received_ns is None else
                (self.last_received_ns - self.start_ns) / 1e9,
            "status": status,
            "notes": ["Read-only; no register commands or device reset were sent.",
                      "Host receive timestamps include Wi-Fi/OS jitter.",
                      "Battery percentage is an approximate voltage-table mapping."],
        }
        (self.path / "manifest.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def append(self, packet: AttitudePacket, received_ns: int) -> None:
        if self.closed or packet.device_id != self.device_id:
            return
        elapsed = (received_ns - self.start_ns) / 1e9
        phase = phase_at(elapsed, self.neutral_s, self.movement_s)
        if phase is None:
            return
        self.writer.writerow({
            "sample": sum(self.counts.values()),
            "host_recv_mono_ns": received_ns,
            "host_write_wall_ns": time.time_ns(),
            "elapsed_s": elapsed,
            "phase": phase,
            "device_time": packet.time_str,
            **asdict(packet),
        })
        self.counts[phase] += 1
        self.last_received_ns = received_ns
        self.final_battery_v = packet.battery_v
        if sum(self.counts.values()) % 100 == 0:
            self.file.flush()

    def close(self, status: str) -> None:
        if self.closed:
            return
        self.closed = True
        self.file.flush()
        self.file.close()
        self._manifest(status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", help="覆盖自动选择；完整 ID 或唯一尾号")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--discover-sec", type=float, default=8.0)
    parser.add_argument("--prepare-sec", type=float, default=10.0)
    parser.add_argument("--neutral-sec", type=float, default=10.0)
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="中间运动时长；两段中立位另加")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.discover_sec, args.neutral_sec, args.minutes) <= 0 or args.prepare_sec < 0:
        parser.error("发现、中立、运动时间须大于零，准备时间不能为负")
    return args


def select_device(packets: dict[str, AttitudePacket], requested: str | None) -> str:
    if not packets:
        raise RuntimeError("未发现 WT 数据；确认电脑热点、设备电源和 UDP 1399")
    if requested:
        matches = [device for device in packets if device == requested or device.endswith(requested)]
        if len(matches) != 1:
            raise RuntimeError(f"指定设备 {requested!r} 未发现或尾号不唯一；在线：{list(packets)}")
        return matches[0]
    return max(packets, key=lambda device: (packets[device].battery_v, device))


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    print("只读实验；请先关闭人体/手腕 GUI（独占 UDP 端口）。", flush=True)
    try:
        source = GuiUdpSource(port=args.listen_port)
    except OSError as exc:
        print(f"无法监听 UDP {args.listen_port}：{exc}", file=sys.stderr)
        return 2
    recorder: Recorder | None = None
    try:
        latest: dict[str, tuple[int, AttitudePacket]] = {}
        deadline = time.monotonic() + args.discover_sec
        print(f"正在自动发现设备 {args.discover_sec:g} 秒…", flush=True)
        while time.monotonic() < deadline:
            for ns, packet in source.poll_timed():
                latest[packet.device_id] = (ns, packet)
            time.sleep(0.01)
        fresh = {device: packet for device, (ns, packet) in latest.items()
                 if time.monotonic_ns() - ns < 3_000_000_000}
        for device, packet in sorted(fresh.items()):
            print(f"  {device}  {packet.battery_v:.2f} V / 约 {packet.battery_percent}%"
                  f"  RSSI {packet.rssi} dBm", flush=True)
        chosen = select_device(fresh, args.device_id)
        selected = fresh[chosen]
        print(f"选择 {chosen}：{selected.battery_v:.2f} V / 约"
              f" {selected.battery_percent}%（电压并不保证续航）", flush=True)
        if selected.battery_v < 3.87:
            print("警告：电量估计低于约 75%，半小时测试可能中断；可先充电。", flush=True)
        print(f"{args.prepare_sec:g} 秒准备；随后中立 {args.neutral_sec:g} 秒，"
              f"运动 {args.minutes:g} 分钟，最后中立 {args.neutral_sec:g} 秒。", flush=True)
        ready_at = time.monotonic() + args.prepare_sec
        while time.monotonic() < ready_at:
            source.poll_timed()
            time.sleep(0.01)
        endpoint = source.seen_endpoints().get(chosen)
        if endpoint is None or source.latest_packet(chosen) is None:
            raise RuntimeError("准备期间选中的设备离线，未开始录制")
        start_ns = time.monotonic_ns()
        recorder = Recorder(output, chosen, start_ns, args.neutral_sec,
                            args.minutes * 60, (endpoint[0], endpoint[1]), selected)
        print(f"\a开始录制：起始中立位。输出：{recorder.path}", flush=True)
        movement_s = args.minutes * 60
        transitions = [(args.neutral_sec, "开始运动")]
        if movement_s > 60:
            transitions.append((args.neutral_sec + movement_s - 60, "还剩 60 秒运动"))
        transitions.append((args.neutral_sec + movement_s, "回到原中立位并保持静止"))
        announced: set[str] = set()
        last_status_sec = -1
        replace_line = sys.stdout.isatty()
        last_online_ns = start_ns
        total = args.neutral_sec * 2 + movement_s
        while True:
            for ns, packet in source.poll_timed():
                if packet.device_id == chosen:
                    last_online_ns = ns
                    recorder.append(packet, ns)
            elapsed = (time.monotonic_ns() - start_ns) / 1e9
            for boundary, message in transitions:
                if boundary > 0 and elapsed >= boundary and message not in announced:
                    announced.add(message)
                    if replace_line and last_status_sec >= 0:
                        print()
                    print(f"\a{elapsed:.0f}s：{message}", flush=True)
            if int(elapsed) != last_status_sec:
                last_status_sec = int(elapsed)
                status = progress_text(
                    elapsed, args.neutral_sec, movement_s,
                    sum(recorder.counts.values()),
                    (time.monotonic_ns() - last_online_ns) / 1e9,
                    recorder.final_battery_v or selected.battery_v)
                if replace_line:
                    print("\r\033[2K" + status, end="", flush=True)
                else:
                    print(status, flush=True)
            if elapsed >= total:
                break
            time.sleep(0.01)
        recorder.close("completed")
        if replace_line:
            print()
        print(f"\a录制完成：{recorder.path}", flush=True)
        return 0
    except KeyboardInterrupt:
        if recorder is not None:
            recorder.close("interrupted")
            if sys.stdout.isatty():
                print()
            print(f"已保存部分数据：{recorder.path}", flush=True)
        return 130
    except (OSError, RuntimeError) as exc:
        if recorder is not None:
            recorder.close(f"error: {exc}")
            if sys.stdout.isatty():
                print()
            print(f"已保存部分数据：{recorder.path}", flush=True)
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
