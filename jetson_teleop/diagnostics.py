"""Bounded, read-only JSONL diagnostics. Never opens a device or grants output.

Producers enqueue immutable snapshots; JSON conversion and IO run off acquisition.
Queue overflow/disk errors explicitly mark the recording incomplete, not control.
"""
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from uuid import uuid4
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, bytes): return {'bytes_hex':value.hex()}
    raise TypeError(type(value).__name__)


class DiagnosticLog:
    def __init__(self, root, metadata, seconds=60., capacity=4096):
        directory=Path(root)/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6])
        directory.mkdir(parents=True,exist_ok=False)
        self.path=directory/'diagnostic.jsonl'
        self.file=self.path.open('x',encoding='utf-8')
        self.queue=queue.Queue(maxsize=capacity)
        self.done=threading.Event()
        self.deadline=time.monotonic()+seconds
        self.dropped=0; self.error=None; self.closed=False
        self.thread=threading.Thread(target=self._run,name='wt901-diagnostic-writer',daemon=True)
        self.record('metadata',schema_version=1,read_only=True,settings=metadata)
        self.thread.start()

    def record(self,event,**data):
        if self.closed or self.error or time.monotonic()>self.deadline:return
        row=dict(event=event,monotonic_ns=time.monotonic_ns(),**data)
        try:self.queue.put_nowait(row)
        except queue.Full:self.dropped+=1

    def _write(self,row):
        self.file.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_value)+'\n')

    def _run(self):
        flushed=time.monotonic()
        try:
            while not self.done.is_set() or not self.queue.empty():
                try:row=self.queue.get(timeout=.1)
                except queue.Empty:row=None
                if row is not None:self._write(row)
                if time.monotonic()-flushed>=.25:
                    self.file.flush();flushed=time.monotonic()
            self._write(dict(event='recording_end',dropped=self.dropped,complete=self.dropped==0))
        except Exception as exc:self.error=f'诊断记录不完整：{exc}'
        finally:
            try:self.file.close()
            except Exception as exc:self.error=f'诊断文件关闭失败，记录不完整：{exc}'

    def close(self):
        self.closed=True;self.done.set();self.thread.join(timeout=2)
        if self.thread.is_alive():self.error='诊断记录尚未排空，不能确认完整'
