"""Bounded asynchronous diagnostic JSONL writer; never performs serial IO."""
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time


class Recorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('x', encoding='utf-8')
        self.queue = queue.Queue(maxsize=4096)
        self.error = None
        self.closed = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.flush_interval = 0.  # PC default: every row; board may batch telemetry.
        self.thread.start()

    def record(self, event, **data):
        if self.error or self.closed:return
        item = dict(event=event, wall_time=datetime.now().astimezone().isoformat(),
                    monotonic_ns=time.monotonic_ns(), **data)
        try:self.queue.put_nowait(item)
        except queue.Full:self.error = '日志队列已满，后续记录停用；记录不完整'

    def _run(self):
        flushed = time.monotonic()
        critical = {'error', 'engage', 'paused', 'input_gap_started', 'input_gap_recovered',
                    'input_timeout', 'auto_start_cancelled', 'initial_ik_started',
                    'initial_ik_finished', 'arm_start_rejected', 'connected'}
        try:
            while not self.done.is_set() or not self.queue.empty():
                try:item = self.queue.get(timeout=.1)
                except queue.Empty:
                    if time.monotonic()-flushed >= self.flush_interval:
                        self.file.flush()
                        flushed = time.monotonic()
                    continue
                self.file.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                now = time.monotonic()
                if now-flushed >= self.flush_interval or item['event'] in critical:
                    self.file.flush()
                    flushed = now
        except Exception as exc:
            self.error = f'日志写入失败：{exc}；记录不完整'
        finally:
            self.file.close()

    def close(self):
        self.closed = True
        self.done.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():self.error = '日志仍在后台排空；尚未确认完整保存'
