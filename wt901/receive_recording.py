"""Non-blocking observer for the existing UDP source, with offline summaries."""
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from uuid import uuid4


class Distribution:
    """Bound percentile memory; keep full-run counts, means and maxima."""
    def __init__(self):
        self.values = deque(maxlen=50000)
        self.count = self.over_100 = 0
        self.total = 0.0
        self.maximum = None

    def add(self, value):
        if value is None or not math.isfinite(value) or value < 0:
            return
        self.values.append(value)
        self.count += 1
        self.total += value
        self.over_100 += value > 100
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def result(self):
        values = sorted(self.values)
        def percentile(p):
            if not values:
                return None
            index = (len(values) - 1) * p
            low = int(index)
            high = min(low + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (index - low)
        return dict(count=self.count, mean_ms=self.total / self.count if self.count else None,
                    p50_ms=percentile(.50), p95_ms=percentile(.95), p99_ms=percentile(.99),
                    max_ms=self.maximum, over_100_ms=self.over_100,
                    percentile_samples=len(values), percentiles_truncated=self.count > len(values))


class ReceiveSummary:
    def __init__(self):
        self.poll_gaps = Distribution()
        self.poll_durations = Distribution()
        self.last_poll = None
        self.batches = self.max_batch_datagrams = self.max_batch_packets = 0
        self.budget_reached = 0
        self.devices = {}
        self.endpoints = {}
        self.kernel_timestamp_available = False

    def feed(self, row):
        event = row.get("event")
        if event == "acquisition_trace_enabled":
            self.kernel_timestamp_available = bool(row.get("kernel_timestamp_available"))
        elif event == "udp_poll_begin":
            now = row["mono_s"]
            if self.last_poll is not None:
                self.poll_gaps.add((now - self.last_poll) * 1000)
            self.last_poll = now
        elif event == "udp_poll_end":
            self.batches += 1
            self.poll_durations.add(row.get("duration_ms"))
            self.max_batch_datagrams = max(self.max_batch_datagrams, row.get("datagrams", 0))
            self.max_batch_packets = max(self.max_batch_packets, row.get("packets", 0))
            self.budget_reached += bool(row.get("budget_reached"))
        elif event in ("udp_datagram", "imu_parsed"):
            is_device = event == "imu_parsed"
            groups = self.devices if is_device else self.endpoints
            key = row["device"] if is_device else row["endpoint"]
            state = groups.setdefault(key, dict(count=0, first=None, last=None,
                last_kernel=None, read_gaps=Distribution(), kernel_gaps=Distribution(),
                queue=Distribution(), timestamp_anomalies=0, kernel_samples=0))
            now = row["read_mono_ns"]
            if state["last"] is not None:
                state["read_gaps"].add((now - state["last"]) / 1e6)
            if state["first"] is None:
                state["first"] = now
            state["last"] = now
            state["count"] += 1
            kernel = row.get("kernel_received_wall_ns")
            if kernel is not None:
                state["kernel_samples"] += 1
                if state["last_kernel"] is not None:
                    gap = (kernel - state["last_kernel"]) / 1e6
                    if gap < 0:
                        state["timestamp_anomalies"] += 1
                    else:
                        state["kernel_gaps"].add(gap)
                state["last_kernel"] = kernel
            queue_ms = row.get("socket_queue_ms")
            if queue_ms is not None and queue_ms < 0:
                state["timestamp_anomalies"] += 1
            state["queue"].add(queue_ms)

    def result(self, duration_s):
        def groups(states):
            return {key: dict(count=state["count"],
                rate_over_session_hz=state["count"] / duration_s if duration_s > 0 else None,
                read_gap_ms=state["read_gaps"].result(),
                kernel_gap_ms=state["kernel_gaps"].result(),
                socket_queue_ms=state["queue"].result(),
                kernel_timestamp_samples=state["kernel_samples"],
                timestamp_anomalies=state["timestamp_anomalies"])
                for key, state in states.items()}
        return dict(duration_s=duration_s, kernel_timestamp_available=self.kernel_timestamp_available,
            poll_gap_ms=self.poll_gaps.result(), poll_duration_ms=self.poll_durations.result(),
            batches=self.batches, max_batch_datagrams=self.max_batch_datagrams,
            max_batch_packets=self.max_batch_packets, budget_reached=self.budget_reached,
            devices=groups(self.devices), endpoints=groups(self.endpoints))


def summary_markdown(summary):
    def number(value):
        return "不可用" if value is None else f"{value:.2f}"
    lines = ["# 手腕 UDP 收包诊断", "",
        f"测试标签：{summary['metadata'].get('test_label') or '未填写'}。",
        f"录制时长：{summary['duration_s']:.1f} 秒；丢弃诊断事件：{summary['dropped_events']}。",
        f"内核时间戳功能启用：{summary['kernel_timestamp_available']}（实际采到数量见 summary.json）。", "",
        "所有间隔单位为 ms。超过 100 ms 的次数不是丢包数量。",
        "内核到包时间不是 IMU 采样时刻，不能直接测端到端延迟。", "",
        "| 对象 / 指标 | p50 | p95 | p99 | 最大 | >100 ms 次数 |",
        "| --- | --- | --- | --- | --- | --- |"]
    rows = [("GUI/接收轮询间隔", summary["poll_gap_ms"]),
            ("接收轮询耗时", summary["poll_duration_ms"])]
    for key, state in summary["devices"].items():
        rows.extend([(f"{key} 程序读取间隔", state["read_gap_ms"]),
                     (f"{key} 内核到包间隔", state["kernel_gap_ms"]),
                     (f"{key} 本机 socket 排队", state["socket_queue_ms"])])
    for label, values in rows:
        lines.append(f"| {label} | {number(values['p50_ms'])} | {number(values['p95_ms'])} | "
                     f"{number(values['p99_ms'])} | {number(values['max_ms'])} | {values['over_100_ms']} |")
    lines += ["", f"单批最多读取数据报：{summary['max_batch_datagrams']}；"
              f"单批最多有效姿态包：{summary['max_batch_packets']}。", "",
        "## 如何解读", "",
        "- 内核间隔出现尖峰：问题在内核时间戳之前，仍可能来自 IMU、空口、路由器或驱动。",
        "- 内核间隔平稳、socket 排队与轮询间隔升高：优先检查 GUI 忙碌和本机调度。",
        "- 两类指标同时升高：可能同时存在多个因素，不能只归因于无线。",
        "- 无内核时间戳、诊断事件丢弃或写盘错误时，结论受限。",
        "- 内核时间戳使用系统墙钟，调时或时间异常也会影响其间隔。",
        "- 分位数最多保留最近 50000 个值；总次数、平均值和最大值覆盖全程。",
        "- 同一数据报包含多个姿态包时，其内核间隔可为零；可对照 summary.json 的 endpoints。",
        "- 本记录不测无线丢包率，也不证明机器人控制安全。", ""]
    return "\n".join(lines)


class ReceiveRecorder:
    """Queue events quickly; write files and compute summaries off the GUI thread."""
    def __init__(self, root, metadata=None, queue_size=20000):
        self.path = Path(root) / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.stopped = None
        self.reason = "user"
        self.error = None
        self.dropped = 0
        self.summary = None
        self._queue = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.metadata = dict(metadata or {})
        self.record(dict(event="metadata", mono_s=self.started,
                         wall_time=datetime.now().astimezone().isoformat(),
                         schema_version=1, **self.metadata))
        self.thread = threading.Thread(target=self._write, name="wrist-udp-diagnostics", daemon=True)
        self.thread.start()

    def record(self, row):
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(dict(row))
        except Full:
            self.dropped += 1

    def stop(self, reason="user"):
        if not self._stop.is_set():
            self.stopped = time.monotonic()
            self.reason = reason
            self._stop.set()

    @property
    def finished(self):
        return not self.thread.is_alive()

    def _write(self):
        stats = ReceiveSummary()
        try:
            with (self.path / "events.jsonl").open("x", encoding="utf-8") as stream:
                last_flush = time.monotonic()
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        row = self._queue.get(timeout=.1)
                    except Empty:
                        row = None
                    if row is not None:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stats.feed(row)
                    if time.monotonic() - last_flush >= .5:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.write(json.dumps(dict(event="recording_stop", mono_s=self.stopped,
                    reason=self.reason, dropped_events=self.dropped), ensure_ascii=False) + "\n")
            self.summary = dict(stats.result(max(0, self.stopped - self.started)),
                                dropped_events=self.dropped, stop_reason=self.reason,
                                metadata=self.metadata)
            (self.path / "summary.json").write_text(
                json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            (self.path / "summary.md").write_text(summary_markdown(self.summary), encoding="utf-8")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop("write_error")
