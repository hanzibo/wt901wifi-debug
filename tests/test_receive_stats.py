"""Deterministic rates and local loopback only; no IMU or robot commands."""
import socket
import time
import unittest
from wt901.gui_udp import GuiUdpSource
from wt901.receive_stats import ReceiveRateTracker


class ReceiveRateTests(unittest.TestCase):
    def test_warmup_fixed_window_and_independent_devices(self):
        tracker = ReceiveRateTracker()
        for i in range(151):
            tracker.record("fast", i / 50)
            if i % 5 == 0:
                tracker.record("slow", i / 50)
        stats = tracker.snapshot(3)
        self.assertEqual(stats["fast"].rate_hz, 50)
        self.assertEqual(stats["slow"].rate_hz, 10)
        self.assertEqual(stats["fast"].age_ms, 0)
        tracker.record("new", 3)
        self.assertIsNone(tracker.snapshot(3.5)["new"].rate_hz)

    def test_no_new_packets_decay_to_zero_and_age_grows(self):
        tracker = ReceiveRateTracker()
        tracker.record("device", 0)
        tracker.record("device", 0.5)
        self.assertEqual(tracker.snapshot(2)["device"].rate_hz, 0.5)
        stale = tracker.snapshot(3)["device"]
        self.assertEqual(stale.rate_hz, 0)
        self.assertEqual(stale.age_ms, 2500)
        self.assertEqual(len(tracker._samples["device"]), 0)
        tracker.record("device", 4)
        self.assertEqual(tracker.snapshot(4)["device"].rate_hz, 0.5)

    def test_batched_packets_all_count_not_gui_ticks(self):
        tracker = ReceiveRateTracker()
        tracker.record("device", 0)
        for _ in range(100):
            tracker.record("device", 1)
        self.assertEqual(tracker.snapshot(2)["device"].rate_hz, 50)

    def test_empty_and_invalid_window(self):
        self.assertEqual(ReceiveRateTracker().snapshot(10), {})
        with self.assertRaises(ValueError):
            ReceiveRateTracker(0)

    def test_udp_counts_only_parsed_attitude_packets_without_consuming(self):
        source = GuiUdpSource(port=0, host="127.0.0.1")
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        raw = b"WT5500011381" + bytes(40) + b"\r\n"
        try:
            sender.sendto(b"garbage", source.sock.getsockname())
            sender.sendto(raw + raw, source.sock.getsockname())
            sender.sendto(raw, source.sock.getsockname())
            packets = []
            deadline = time.monotonic() + 1
            while len(packets) < 3 and time.monotonic() < deadline:
                packets.extend(source.poll())
            self.assertEqual(len(packets), 3)
            stats = source.receive_stats()["WT5500011381"]
            self.assertEqual(stats.packets_in_window, 3)
            self.assertIsNone(stats.rate_hz)
            self.assertEqual(source.receive_stats()["WT5500011381"].packets_in_window, 3)
            self.assertEqual(source.poll(), [])
            self.assertEqual(source.latest_packet("WT5500011381"), packets[-1])
        finally:
            sender.close()
            source.close()
