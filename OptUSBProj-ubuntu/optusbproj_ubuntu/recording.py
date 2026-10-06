"""Lossless-ish CSV packet recording and time-paced offline replay."""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
import time

from wt901.protocol import AttitudePacket


PACKET_FIELDS = tuple(field.name for field in fields(AttitudePacket))
CSV_FIELDS = ("host_time_ns",) + PACKET_FIELDS
INTEGER_FIELDS = {"year", "month", "day", "hour", "minute", "second",
                  "millisecond", "rssi", "version"}


class PacketRecorder:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.count = 0

    def write(self, packet: AttitudePacket, host_time_ns: int | None = None) -> None:
        row = {name: getattr(packet, name) for name in PACKET_FIELDS}
        row["host_time_ns"] = time.time_ns() if host_time_ns is None else host_time_ns
        self.writer.writerow(row)
        self.count += 1
        if self.count % 100 == 0:
            self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


class ReplaySource:
    """GuiUdpSource-compatible, read-only stream from a recorded CSV file."""

    def __init__(self, path: Path, speed: float = 1.0):
        if speed <= 0:
            raise ValueError("回放速度必须大于 0")
        self.path = path
        self.speed = speed
        self.file = path.open("r", newline="", encoding="utf-8")
        self.reader = csv.DictReader(self.file)
        if self.reader.fieldnames != list(CSV_FIELDS):
            self.close()
            raise ValueError("不是当前版本的 WT901 录制 CSV")
        self._next = self._read_next()
        self._first_ns = self._next[0] if self._next else 0
        self._started_at = time.monotonic()
        self._seen: dict[str, float] = {}

    def _read_next(self) -> tuple[int, AttitudePacket] | None:
        row = next(self.reader, None)
        if row is None:
            return None
        values = {}
        for name in PACKET_FIELDS:
            raw = row[name]
            values[name] = raw if name == "device_id" else (
                int(raw) if name in INTEGER_FIELDS else float(raw))
        return int(row["host_time_ns"]), AttitudePacket(**values)

    def poll(self, now: float | None = None) -> list[AttitudePacket]:
        moment = time.monotonic() if now is None else now
        due_ns = self._first_ns + int((moment - self._started_at) * self.speed * 1e9)
        packets = []
        while self._next is not None and self._next[0] <= due_ns and len(packets) < 512:
            _, packet = self._next
            packets.append(packet)
            self._seen[packet.device_id] = moment
            self._next = self._read_next()
        return packets

    def seen_devices(self, max_age: float = 3.0) -> dict[str, tuple[str, float]]:
        moment = time.monotonic()
        return {device_id: ("回放", seen) for device_id, seen in self._seen.items()
                if moment - seen <= max_age}

    def seen_endpoints(self, max_age: float = 3.0) -> dict[str, tuple[str, int, float]]:
        return {device_id: ("回放", 0, seen)
                for device_id, (_, seen) in self.seen_devices(max_age).items()}

    def close(self) -> None:
        self.file.close()

