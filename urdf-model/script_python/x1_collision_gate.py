"""stdlib-only, asynchronous fail-closed gate owned by the ROS publisher."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


def default_python():
    base = Path(sys.prefix).parents[1]
    return os.environ.get('X1_COLLISION_PYTHON', str(base / 'envs/OptUSBProj-yolo/bin/python'))


class CollisionGate:
    def __init__(self, python=None):
        self.python = python or default_python()
        self.process = None
        self.inbox = queue.SimpleQueue()
        self.enabled = False
        self.state = '关闭'
        self.reason = ''
        self.model_version = None
        self.accepted = None
        self.pending = None
        self.serial = 0
        self.revision = 0
        self.last_result = {}
        self.boot_at = 0.
        self.ready = False
        self.validated = False

    def configure(self, enabled, positions):
        self.close()
        self.enabled = enabled
        self.accepted = dict(positions)
        self.validated = False
        self.last_result = {}
        self.revision += 1
        self.reason = ''
        if not enabled:
            self.state = '关闭'
            return
        self.state = '初始化'
        self.boot_at = time.monotonic()
        try:
            worker = Path(__file__).with_name('x1_collision_worker.py')
            env = os.environ.copy()
            # ROS Python paths must not contaminate the independent env.
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            self.process = subprocess.Popen([self.python, '-u', str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1, env=env)
            self.inbox = queue.SimpleQueue()
            inbox, process = self.inbox, self.process
            def read():
                try:
                    for line in process.stdout:
                        inbox.put(json.loads(line))
                except Exception as exc:
                    inbox.put(dict(type='fatal', reason=str(exc)))
            self.reader = threading.Thread(target=read, daemon=True)
            self.reader.start()
        except OSError as exc:
            self.fail(str(exc))

    def fail(self, reason):
        self.state, self.reason = '故障', reason
        self.pending = None

    def submit(self, goal, metadata=None):
        if not self.enabled or not self.ready or self.pending is not None or self.state == '故障':
            return False
        self.serial += 1
        request = dict(id=self.serial, start=self.accepted, goal=dict(goal))
        self.pending = dict(id=self.serial, goal=dict(goal), metadata=metadata,
                            at=time.monotonic(), initial=self.state == '初始化')
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
            self.process.stdin.flush()
            return True
        except (OSError, ValueError) as exc:
            self.fail(str(exc))
            return False

    def poll(self):
        """Return only a matching, validated commit; never a receipt ACK."""
        commit = None
        while not self.inbox.empty():
            data = self.inbox.get()
            if not self.enabled or self.state == '故障':
                continue
            if data.get('type') == 'fatal':
                self.fail(data.get('reason', 'Worker failed'))
            elif data.get('type') == 'ready':
                self.ready = True
                self.model_version = data['model_version']
                self.submit(self.accepted)
            elif data.get('type') == 'result' and self.pending and data.get('id') == self.pending['id']:
                pending, self.pending = self.pending, None
                self.last_result = data
                self.reason = data.get('reason', '')
                if time.monotonic()-pending['at'] > .3:
                    self.fail('Collision worker response timeout')
                elif data.get('model_version') != self.model_version:
                    self.fail('Model version mismatch')
                elif data.get('safe') is True:
                    alpha = data.get('alpha', 1.)
                    if type(alpha) not in (float, int) or not 0 < alpha <= 1:
                        self.fail('Invalid accepted step fraction')
                        continue
                    self.accepted = {k:v+alpha*(pending['goal'][k]-v)
                                     for k,v in self.accepted.items()}
                    self.validated = True
                    self.revision += 1
                    self.state = '正常' if alpha == 1 else '截短'
                    commit = (self.accepted.copy(), pending['metadata'])
                elif pending['initial']:
                    self.fail('初始构型未通过：'+self.reason)
                else:
                    self.state = '已拦截'
        now = time.monotonic()
        if self.enabled and self.process is not None and self.state != '故障':
            if self.process.poll() is not None:
                self.fail('Collision worker exited')
            elif not self.ready and now-self.boot_at > 15:
                self.fail('Collision worker initialization timeout')
            elif self.pending and now-self.pending['at'] > .3:
                self.fail('Collision worker response timeout')
        return commit

    def status(self):
        return dict(supported=True, enabled=self.enabled, state=self.state,
                    reason=self.reason, revision=self.revision,
                    model_version=self.model_version,
                    accepted=self.accepted if self.enabled and self.validated else None,
                    result=self.last_result)

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
        self.process = None
        self.pending = None
        self.ready = False
        self.model_version = None
