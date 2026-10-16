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
        self.thread.start()

    def record(self, event, **data):
        if self.error or self.closed:return
        item = dict(event=event, wall_time=datetime.now().astimezone().isoformat(),
                    monotonic_ns=time.monotonic_ns(), **data)
        try:self.queue.put_nowait(item)
        except queue.Full:self.error = '日志队列已满，后续记录停用；记录不完整'

    def _run(self):
        try:
            while not self.done.is_set() or not self.queue.empty():
                try:item = self.queue.get(timeout=.1)
                except queue.Empty:continue
                self.file.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                self.file.flush()
        except Exception as exc:
            self.error = f'日志写入失败：{exc}；记录不完整'
        finally:
            self.file.close()

    def close(self):
        self.closed = True
        self.done.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():self.error = '日志仍在后台排空；尚未确认完整保存'
