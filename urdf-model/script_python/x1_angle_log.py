"""Optional asynchronous ROS-preview diagnostics, not physical robot feedback."""
import json
from pathlib import Path
import queue
import threading
import time
from datetime import datetime
from uuid import uuid4


class AngleLogger:
    def __init__(self, root):
        self.root = Path(root)
        self.path = None
        self.error = ''
        self.dropped = 0
        self.thread = None
        self.inbox = None

    def status(self):
        return dict(supported=True, enabled=self.thread is not None and not self.error,
                    path=str(self.path) if self.path else None,
                    error=self.error, dropped=self.dropped)

    def start(self, names):
        if self.thread is not None:
            return
        self.error, self.dropped = '', 0
        try:
            folder = self.root / (datetime.now().strftime('%Y%m%d-%H%M%S-')+uuid4().hex[:6])
            folder.mkdir(parents=True, exist_ok=False)
            self.path = folder / 'angles.jsonl'
            stream = self.path.open('w', encoding='utf-8', buffering=1)
        except OSError as exc:
            self.error = str(exc)
            return
        self.inbox = queue.Queue(maxsize=10000)
        def write():
            try:
                with stream:
                    while True:
                        event = self.inbox.get()
                        if event is None:
                            break
                        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            except (OSError, ValueError) as exc:
                self.error = str(exc)
        self.thread = threading.Thread(target=write, daemon=True)
        self.thread.start()
        self.emit('start', joint_names=list(names), units='radians',
                  scope='ROS JointState preview only; no physical feedback', version=1)

    def emit(self, event, **data):
        if self.thread is None or self.error:
            return
        # Snapshot before the ROS thread mutates positions or gate dictionaries.
        item = json.loads(json.dumps(dict(event=event, mono_ns=time.monotonic_ns(),
                                         wall_ns=time.time_ns(), **data)))
        try:
            self.inbox.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self.thread is not None:
            self.emit('stop', dropped=self.dropped)
            while self.thread.is_alive():
                try:
                    self.inbox.put(None, timeout=.1)
                    break
                except queue.Full:
                    continue
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.error = '日志线程未及时结束'
                return
            self.thread = None
