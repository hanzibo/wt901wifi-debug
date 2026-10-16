"""One UDP socket for live WT901 packets and device register control.

The GUI must not open a second listener on port 1399: two UDP listeners can
split incoming datagrams. This source also binds device IDs to their current
sender IPs, so calibration commands cannot be sent to a stale/unverified IP.
"""

from __future__ import annotations

from collections import deque
import socket
import struct
import sys
import time

from .protocol import AttitudePacket, extract_packets
from .registers import CMD_READ, UNLOCK_KEY, UNLOCK_REG


class GuiUdpSource:
    def __init__(self, port: int = 1399, host: str = "0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self._buffers: dict[tuple[str, int], bytearray] = {}
        self._devices: dict[str, tuple[str, float]] = {}
        self._endpoints: dict[str, tuple[str, int, float]] = {}
        self._latest_packets: dict[str, AttitudePacket] = {}
        self._replies: deque[tuple[str, tuple[int, int, int, int]]] = deque()
        self._diagnostics = None
        self._diagnostic_sinks = []
        self._observer_state = None
        self._kernel_stamp_option = None
        self._kernel_stamp_original = None
        self._diagnostic_sink_errors = 0

    def add_diagnostic_sink(self, sink):
        """Observe the existing socket; never open a second UDP listener."""
        if sink in self._diagnostic_sinks:
            return
        self._diagnostic_sinks.append(sink)
        if self._observer_state is None:
            self._observer_state = {'last_poll':None, 'endpoints':{}, 'devices':{}}
        # Ubuntu 64-bit SO_TIMESTAMPNS_OLD uses native struct timespec.
        if self._kernel_stamp_option is None and sys.platform.startswith('linux') and struct.calcsize('@ll') == 16:
            option = getattr(socket, 'SO_TIMESTAMPNS', 35)
            try:
                original = self.sock.getsockopt(socket.SOL_SOCKET, option)
                self.sock.setsockopt(socket.SOL_SOCKET, option, 1)
                self._kernel_stamp_original = original
                self._kernel_stamp_option = option
            except (OSError, AttributeError):
                pass
        self.diagnostic_event('acquisition_trace_enabled',
                              kernel_timestamp_available=self._kernel_stamp_option is not None)

    def remove_diagnostic_sink(self, sink):
        if sink in self._diagnostic_sinks:
            self._diagnostic_sinks.remove(sink)
        if not self._diagnostic_sinks:
            self._observer_state = None
            if self._kernel_stamp_option is not None:
                try:
                    self.sock.setsockopt(socket.SOL_SOCKET, self._kernel_stamp_option, self._kernel_stamp_original)
                except OSError:
                    pass
                self._kernel_stamp_option = None

    def enable_diagnostics(self, enabled: bool) -> None:
        self._diagnostics = ({'rows': deque(maxlen=20000), 'dropped': 0,
                              'last_poll': None, 'endpoints': {}, 'devices': {}}
                             if enabled else None)

    def diagnostic_event(self, event: str, **values) -> None:
        state = self._diagnostics
        row = {'event': event, 'mono_s': time.monotonic(), **values}
        for sink in tuple(self._diagnostic_sinks):
            try:
                sink(dict(row))
            except Exception:
                # Instrumentation must not interrupt acquisition/control.
                self._diagnostic_sink_errors += 1
        if state is None: return
        if len(state['rows']) == state['rows'].maxlen:
            state['dropped'] += 1
        state['rows'].append(row)

    def take_diagnostics(self) -> list[dict]:
        state = self._diagnostics
        if state is None:
            return []
        rows = list(state['rows']); state['rows'].clear()
        if state['dropped']:
            rows.append({'event': 'diagnostic_overflow', 'mono_s': time.monotonic(),
                         'dropped': state['dropped']})
            state['dropped'] = 0
        return rows

    def poll(self) -> list[AttitudePacket]:
        return [packet for _, packet in self.poll_timed()]

    def poll_timed(self) -> list[tuple[int, AttitudePacket]]:
        """Return packets with the host monotonic timestamp at UDP receipt."""
        packets: list[tuple[int, AttitudePacket]] = []
        started = time.monotonic()
        diag = self._diagnostics if self._diagnostics is not None else self._observer_state
        if diag is not None:
            self.diagnostic_event('udp_poll_begin', interval_ms=None if diag['last_poll'] is None
                                  else (started-diag['last_poll'])*1000)
            diag['last_poll'] = started
        datagrams = 0
        for _ in range(512):
            kernel_received_wall_ns = None
            try:
                if self._kernel_stamp_option is not None:
                    data, ancillary, flags, addr = self.sock.recvmsg(4096, socket.CMSG_SPACE(16))
                    for level,kind,value in ancillary:
                        if level == socket.SOL_SOCKET and kind == self._kernel_stamp_option and len(value) >= 16:
                            seconds,nanos = struct.unpack('@ll',value[:16])
                            kernel_received_wall_ns = seconds*1_000_000_000+nanos
                else:
                    data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            received_ns = time.monotonic_ns()
            read_wall_ns = time.time_ns() if diag is not None else None
            datagrams += 1
            if diag is not None:
                endpoint = f'{addr[0]}:{addr[1]}'
                prior = diag['endpoints'].get(endpoint)
                diag['endpoints'][endpoint] = received_ns
                self.diagnostic_event('udp_datagram', endpoint=endpoint, bytes=len(data),
                                      read_mono_ns=received_ns, read_wall_ns=read_wall_ns,
                                      kernel_received_wall_ns=kernel_received_wall_ns,
                                      socket_queue_ms=None if kernel_received_wall_ns is None else
                                                      (read_wall_ns-kernel_received_wall_ns)/1e6,
                                      gap_ms=None if prior is None else (received_ns-prior)/1e6)
            if len(data) == 11 and data[:2] == b"\x55\x5f" and sum(data[:10]) & 0xff == data[10]:
                for device_id, (ip, seen) in self._devices.items():
                    if ip == addr[0] and time.monotonic() - seen < 3.0:
                        values = tuple(data[i] | data[i + 1] << 8 for i in (2, 4, 6, 8))
                        self._replies.append((device_id, values))
                        break
                continue
            buffer = self._buffers.setdefault(addr, bytearray())
            before = len(buffer)
            buffer.extend(data)
            raws = extract_packets(buffer)
            if diag is not None:
                self.diagnostic_event('udp_parse', endpoint=endpoint, packets=len(raws),
                    buffered_bytes=len(buffer), discarded_bytes=before+len(data)-len(buffer)-sum(map(len,raws)))
            for raw in raws:
                packet = AttitudePacket.parse(raw)
                seen = time.monotonic()
                self._devices[packet.device_id] = (addr[0], seen)
                self._endpoints[packet.device_id] = (addr[0], addr[1], seen)
                self._latest_packets[packet.device_id] = packet
                packets.append((received_ns, packet))
                if diag is not None:
                    prior = diag['devices'].get(packet.device_id)
                    diag['devices'][packet.device_id] = received_ns
                    self.diagnostic_event('imu_parsed', device=packet.device_id, endpoint=endpoint,
                        read_mono_ns=received_ns, parsed_mono_ns=time.monotonic_ns(),
                        gap_ms=None if prior is None else (received_ns-prior)/1e6,
                        sensor_time=packet.time_str, rssi_dbm=packet.rssi)
        if diag is not None:
            self.diagnostic_event('udp_poll_end', duration_ms=(time.monotonic()-started)*1000,
                                  datagrams=datagrams, packets=len(packets), budget_reached=datagrams==512)
        return packets

    def take_replies(self) -> list[tuple[str, tuple[int, int, int, int]]]:
        out = list(self._replies)
        self._replies.clear()
        return out

    def seen_devices(self, max_age: float = 3.0) -> dict[str, tuple[str, float]]:
        """最近收过数据的设备 ID -> (sender IP, monotonic timestamp)。"""
        now = time.monotonic()
        return {device_id: (ip, seen) for device_id, (ip, seen) in self._devices.items()
                if now - seen <= max_age}

    def seen_endpoints(self, max_age: float = 3.0) -> dict[str, tuple[str, int, float]]:
        """最近收到 WT 数据的设备 ID -> (sender IP, source port, monotonic timestamp)。"""
        now = time.monotonic()
        return {device_id: endpoint for device_id, endpoint in self._endpoints.items()
                if now - endpoint[2] <= max_age}

    def latest_packet(self, device_id: str, max_age: float = 3.0) -> AttitudePacket | None:
        """Return a fresh packet without consuming the GUI's data stream."""
        endpoint = self._endpoints.get(device_id)
        if endpoint is None or time.monotonic() - endpoint[2] > max_age:
            return None
        return self._latest_packets.get(device_id)

    def device_ip(self, device_id: str, max_age: float | None = 3.0) -> str:
        ip, seen = self._devices.get(device_id, ("", 0.0))
        if not ip or (max_age is not None and time.monotonic() - seen > max_age):
            raise RuntimeError(f"{device_id} 当前不在线；不会发送配置指令")
        if any(other_id != device_id and other_ip == ip and other_seen > seen
               for other_id, (other_ip, other_seen) in self._devices.items()):
            raise RuntimeError(f"{device_id} 原 IP 已被其他设备使用；不会发送配置指令")
        return ip

    def _send(self, device_id: str, payload: bytes, max_age: float | None = 3.0) -> None:
        self.sock.sendto(payload, (self.device_ip(device_id, max_age), 9250))

    def read_registers(self, device_id: str, address: int) -> None:
        self._send(device_id, bytes((0xff, 0xaa, CMD_READ, address & 0xff, 0)))

    def switch_to_ap_udp(self, device_id: str) -> None:
        """Vendor ASCII command: leave STA and create the device's own AP/UDP hotspot."""
        self._send(device_id, b"WITAPUDP\r\n")

    def write_register(self, device_id: str, address: int, value: int,
                       max_age: float | None = 3.0) -> None:
        self._send(device_id, bytes((0xff, 0xaa, address & 0xff,
                                     value & 0xff, value >> 8 & 0xff)), max_age)

    def unlock(self, device_id: str, max_age: float | None = 3.0) -> None:
        self.write_register(device_id, UNLOCK_REG, UNLOCK_KEY, max_age)

    def close(self) -> None:
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
