#!/usr/bin/env python3
"""LAN gripper input with opt-in loopback RViz preview; never motor commands."""
from __future__ import annotations

import argparse
import json
import math
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class ControlState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.owner = None
        self.sequence = -1
        self.enabled = False
        self.target = self.value = 0.0
        self.received = None
        self.updated = clock()
        self.joint = '57'
        self.preview = False
        self.reverse = False
        self.ros_feedback = None

    def _tick(self):
        now = self.clock()
        dt = max(0.0, min(now - self.updated, 0.1))
        self.updated = now
        if self.received is None or now - self.received > 0.7:
            self.enabled = False
            self.owner = None
            self.target = self.value  # Freeze, never close automatically.
        if self.enabled:
            self.value += max(-dt, min(dt, self.target - self.value))

    def command(self, data):
        if not isinstance(data, dict) or data.get('version') != 1:
            raise ValueError('协议版本错误')
        client, seq = data.get('client'), data.get('seq')
        enabled, value = data.get('enabled'), data.get('closure')
        joint = data.get('joint', '57')
        preview, reverse = data.get('preview', False), data.get('reverse', False)
        if (not isinstance(client, str) or not 8 <= len(client) <= 100 or
                type(seq) is not int or seq < 0 or type(enabled) is not bool or
                type(value) not in (float, int) or not math.isfinite(value) or
                not 0 <= value <= 1 or joint not in ('7', '57') or
                type(preview) is not bool or type(reverse) is not bool):
            raise ValueError('无效控制数据')
        with self.lock:
            self._tick()
            if self.owner is not None and self.owner != client:
                raise ValueError('已有另一部手机控制；请先停用或等待超时')
            if self.owner == client and seq <= self.sequence:
                raise ValueError('过期序号')
            self.owner = client if enabled else None
            self.sequence = seq
            self.enabled = enabled
            self.received = self.clock()
            self.target = float(value) if enabled else self.value
            self.joint, self.preview, self.reverse = joint, preview, reverse

    def stop(self):
        with self.lock:
            self._tick()
            self.enabled = False
            self.owner = None
            self.target = self.value

    def snapshot(self):
        with self.lock:
            self._tick()
            return dict(enabled=self.enabled, closure=self.value, target=self.target,
                        age_s=None if self.received is None else self.clock()-self.received,
                        joint=self.joint, preview=self.preview, reverse=self.reverse,
                        ros_feedback=self.ros_feedback,
                        ros_feedback_age_s=(None if self.ros_feedback is None else
                            (time.monotonic_ns()-self.ros_feedback['sent_mono_ns'])/1e9),
                        purpose='rviz_only')


def forward_preview(server, stopped, port=13991):
    """Only forward fresh, explicitly enabled input. Silence freezes ROS state."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.setblocking(False)
        seq = 0
        while not stopped.wait(.05):
            while True:
                try:
                    raw, peer = sender.recvfrom(4096)
                except BlockingIOError:
                    break
                try:
                    feedback = json.loads(raw)
                    if (peer == ('127.0.0.1', port) and feedback.get('type') == 'gripper_status'
                            and type(feedback.get('sent_mono_ns')) is int
                            and isinstance(feedback.get('joints'), dict)):
                        with server.state.lock:
                            server.state.ros_feedback = feedback
                except (ValueError, AttributeError):
                    continue
            state = server.state.snapshot()
            if not (state['enabled'] and state['preview']):
                continue
            seq += 1
            closure = state['closure']
            frame = dict(version=1, type='gripper', seq=seq,
                         sent_mono_ns=time.monotonic_ns(), joint=state['joint'],
                         position=closure if state['reverse'] else 1.-closure)
            sender.sendto(json.dumps(frame).encode(), ('127.0.0.1', port))


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token=None):
        self.token = token or secrets.token_urlsafe(18)
        self.state = ControlState()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Do not log session tokens or identifiers.

    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def authorized(self):
        query = parse_qs(urlsplit(self.path).query)
        supplied = query.get('token', [''])[0]
        return secrets.compare_digest(supplied, self.server.token)

    def do_GET(self):
        if not self.authorized():
            return self.reply(403, {'error': '请使用终端给出的完整链接'})
        path = urlsplit(self.path).path
        if path == '/':
            return self.reply(200, Path(__file__).with_name('lan_gripper.html').read_bytes(),
                              'text/html; charset=utf-8')
        if path == '/state':
            return self.reply(200, self.server.state.snapshot())
        self.reply(404, {'error': 'not found'})

    def do_POST(self):
        if not self.authorized():
            return self.reply(403, {'error': '无效会话'})
        # Browser writes must be same-origin. No CORS or external resources.
        origin = self.headers.get('Origin')
        if origin != 'http://' + self.headers.get('Host', ''):
            return self.reply(403, {'error': '来源不匹配'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 2048:
                raise ValueError('请求大小错误')
            self.connection.settimeout(2)
            data = json.loads(self.rfile.read(size))
            path = urlsplit(self.path).path
            if path == '/command':
                self.server.state.command(data)
            elif path == '/stop':
                self.server.state.stop()
            else:
                return self.reply(404, {'error': 'not found'})
            self.reply(200, self.server.state.snapshot())
        except (ValueError, TypeError, TimeoutError):
            self.reply(400, {'error': '控制请求无效、重复或已有其他控制端'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='10.42.0.1')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('端口须为 1024–65535')
    server = Server((args.host, args.port))
    stopped = threading.Event()
    worker = threading.Thread(target=forward_preview, args=(server, stopped), daemon=True)
    worker.start()
    url = f'http://{args.host}:{server.server_port}/?token={server.token}'
    print(f'手机控制：{url}\n电脑观测：{url}&monitor=1\n勾选发送至 RViz 后使用本机 UDP 13991；不控制真机。Ctrl+C 停止。', flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.state.stop()
        stopped.set()
        worker.join(1)
        server.server_close()


if __name__ == '__main__':
    main()
