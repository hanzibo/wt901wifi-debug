import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from wt901.gui_udp import GuiUdpSource
from wt901.receive_recording import Distribution, ReceiveRecorder, ReceiveSummary


class ReceiveRecordingTests(unittest.TestCase):
    def test_distributions_and_empty_values(self):
        metric = Distribution()
        self.assertIsNone(metric.result()["p99_ms"])
        for value in (0, 20, 100, 150, None, -1, float("nan")):
            metric.add(value)
        result = metric.result()
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["p50_ms"], 60)
        self.assertEqual(result["max_ms"], 150)
        self.assertEqual(result["over_100_ms"], 1)

    def test_summary_distinguishes_kernel_gap_queue_and_read_gap(self):
        stats = ReceiveSummary()
        for i, (kernel, read) in enumerate(((1000000000, 1000000000),
                                           (1020000000, 1150000000))):
            stats.feed(dict(event="udp_poll_begin", mono_s=read/1e9))
            stats.feed(dict(event="udp_datagram", endpoint="127.0.0.1:1",
                read_mono_ns=read, kernel_received_wall_ns=kernel, socket_queue_ms=(read-kernel)/1e6))
            stats.feed(dict(event="imu_parsed", device="imu", read_mono_ns=read,
                kernel_received_wall_ns=kernel, socket_queue_ms=(read-kernel)/1e6))
            stats.feed(dict(event="udp_poll_end", duration_ms=2, datagrams=1, packets=1))
        result = stats.result(1)
        self.assertAlmostEqual(result["poll_gap_ms"]["max_ms"], 150)
        device = result["devices"]["imu"]
        self.assertEqual(device["read_gap_ms"]["max_ms"], 150)
        self.assertEqual(device["kernel_gap_ms"]["max_ms"], 20)
        self.assertEqual(device["socket_queue_ms"]["max_ms"], 130)
        self.assertEqual(device["count"], 2)

    def test_clock_anomaly_not_reported_as_negative_gap(self):
        stats = ReceiveSummary()
        for stamp in (100, 90):
            stats.feed(dict(event="imu_parsed", device="imu", read_mono_ns=200,
                kernel_received_wall_ns=stamp, socket_queue_ms=-1))
        result = stats.result(1)["devices"]["imu"]
        self.assertEqual(result["timestamp_anomalies"], 3)
        self.assertIsNone(result["kernel_gap_ms"]["max_ms"])
        self.assertIsNone(result["socket_queue_ms"]["max_ms"])

    def test_record_and_summary_files_close_and_repeated_stop(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = ReceiveRecorder(root, {"test_label": "路由器-省电关闭"})
            recorder.record(dict(event="imu_parsed", device="imu", read_mono_ns=time.monotonic_ns()))
            recorder.stop("test")
            recorder.stop("should_not_override")
            recorder.record(dict(event="must_not_record"))
            recorder.thread.join(2)
            self.assertTrue(recorder.finished)
            self.assertIsNone(recorder.error)
            rows = [json.loads(line) for line in (recorder.path/"events.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["metadata", "imu_parsed", "recording_stop"])
            summary = json.loads((recorder.path/"summary.json").read_text())
            self.assertEqual(summary["stop_reason"], "test")
            self.assertEqual(summary["devices"]["imu"]["count"], 1)
            self.assertIn("不能直接测端到端延迟", (recorder.path/"summary.md").read_text())
            self.assertEqual(list(Path(root).iterdir()), [recorder.path])

    def test_full_queue_drops_instrumentation_not_acquisition(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("wt901.receive_recording.threading.Thread.start"):
                recorder = ReceiveRecorder(root, queue_size=1)
            recorder.record({"event": "test"})
            self.assertEqual(recorder.dropped, 1)
            recorder.stop()
            recorder._write()
            self.assertEqual(recorder.summary["dropped_events"], 1)

    def test_disk_failure_reported_without_raising_to_caller(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("wt901.receive_recording.threading.Thread.start"):
                recorder = ReceiveRecorder(root)
            with patch.object(Path, "open", side_effect=OSError("disk full")):
                recorder._write()
            self.assertIn("disk full", recorder.error)
            self.assertTrue(recorder._stop.is_set())

    def test_loopback_same_socket_and_preserves_other_observers(self):
        source = GuiUdpSource(port=0, host="127.0.0.1")
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        other = []
        source.add_diagnostic_sink(other.append)
        try:
            with tempfile.TemporaryDirectory() as root:
                recorder = ReceiveRecorder(root)
                sink = recorder.record
                sock = source.sock
                source.add_diagnostic_sink(sink)
                raw = b"WT5500011381" + bytes(40) + b"\r\n"
                sender.sendto(raw, sock.getsockname())
                packets = []
                end = time.monotonic()+1
                while not packets and time.monotonic() < end:
                    packets.extend(source.poll_timed())
                self.assertEqual(len(packets), 1)
                self.assertIs(source.sock, sock)
                source.remove_diagnostic_sink(sink)
                recorder.stop()
                recorder.thread.join(2)
                self.assertTrue(recorder.finished)
                self.assertIsNone(recorder.error)
                self.assertEqual(recorder.summary["devices"]["WT5500011381"]["count"], 1)
                self.assertIsNotNone(source._observer_state)
                source.diagnostic_event("remaining_observer")
                self.assertEqual(other[-1]["event"], "remaining_observer")
                source.remove_diagnostic_sink(other.append)
                self.assertIsNone(source._observer_state)
        finally:
            sender.close()
            source.close()
