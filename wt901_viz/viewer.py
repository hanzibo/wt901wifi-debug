#!/usr/bin/env python3
"""WT901WIFI 姿态 3D 可视化: 在屏幕上绘制随传感器转动的三条轴线。

三条轴线的颜色约定:
    红 = X 轴  绿 = Y 轴  蓝 = Z 轴
中间半透明立方体代表传感器本体, 灰白网格是静止的世界参考系。

用法:
    python viewer.py                        # 读 /dev/ttyUSB0 @9600
    python viewer.py --port /dev/ttyUSB1
    python viewer.py --demo                 # 无硬件, 用合成姿态演示
    python viewer.py --screenshot out.png   # 渲染一帧存 PNG 后退出

操作:
    鼠标拖动 = 旋转视角, 滚轮 = 缩放, 中键拖动 = 平移
    R = 复位视角, C = 截图, Esc/Q = 退出
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

# 允许直接以脚本方式运行, 复用上一级的 wt901 解析包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wt901 import SerialAttitudeSource, TcpAttitudeSource, UdpAttitudeSource  # noqa: E402
from wt901.serial_source import DEFAULT_BAUD, DEFAULT_PORT  # noqa: E402
from wt901_viz.orientation import rotation_matrix  # noqa: E402

AXIS_COLORS = {
    "x": (1.0, 0.30, 0.30, 1.0),
    "y": (0.30, 1.0, 0.40, 1.0),
    "z": (0.40, 0.60, 1.0, 1.0),
}


# --------------------------------------------------------------------------
# 网格构建
# --------------------------------------------------------------------------
def _ring(n: int, radius: float, z: float) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, z)], axis=1)


def arrow_mesh(length: float = 1.0, shaft_r: float = 0.022,
               head_len: float = 0.26, head_r: float = 0.065,
               n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """沿 +Z 的箭头 (圆柱杆 + 圆锥头)。返回 (vertexes, faces)。"""
    z1 = length - head_len
    V = np.vstack([
        [[0.0, 0.0, 0.0]],          # 0        底心
        _ring(n, shaft_r, 0.0),     # 1..n     底环
        _ring(n, shaft_r, z1),      # n+1..2n  杆顶环
        _ring(n, head_r, z1),       # 2n+1..3n 锥底环
        [[0.0, 0.0, length]],       # 3n+1     锥尖
    ]).astype(np.float32)

    faces = []
    for i in range(n):
        j = (i + 1) % n
        b_i, b_j = 1 + i, 1 + j              # 底环
        t_i, t_j = n + 1 + i, n + 1 + j      # 杆顶环
        h_i, h_j = 2 * n + 1 + i, 2 * n + 1 + j  # 锥底环
        apex = 3 * n + 1
        faces += [
            [0, b_i, b_j],                   # 底盖
            [b_i, b_j, t_j], [b_i, t_j, t_i],        # 杆侧
            [t_i, t_j, h_j], [t_i, h_j, h_i],        # 锥底环
            [h_i, h_j, apex],                # 锥侧
        ]
    return V, np.array(faces, dtype=np.uint32)


def cube_mesh(size: float = 0.42) -> tuple[np.ndarray, np.ndarray]:
    """以原点为中心、边长 size 的立方体。"""
    h = size / 2.0
    V = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float32)
    F = np.array([
        [0, 1, 2], [0, 2, 3],   # -Z
        [4, 6, 5], [4, 7, 6],   # +Z
        [0, 4, 5], [0, 5, 1],   # -Y
        [3, 2, 6], [3, 6, 7],   # +Y
        [0, 3, 7], [0, 7, 4],   # -X
        [1, 5, 6], [1, 6, 2],   # +X
    ], dtype=np.uint32)
    return V, F


def axis_base_rotation(axis: str) -> np.ndarray:
    """把沿 +Z 的箭头转到 +X / +Y / +Z 方向。"""
    if axis == "x":      # +Z -> +X
        return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    if axis == "y":      # +Z -> +Y
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return np.eye(3)     # +Z


def to_qmatrix(R: np.ndarray) -> QtGui.QMatrix4x4:
    """3x3 旋转矩阵 -> QMatrix4x4 (行主序)。"""
    m = QtGui.QMatrix4x4(
        float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), 0.0,
        float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), 0.0,
        float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    return m


def lerp_angle(a: float, b: float, t: float) -> float:
    """角度插值, 走最短路径 (处理 ±180° 回绕)。"""
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + d * t


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------
class OrientationViewer(QtWidgets.QMainWindow):
    def __init__(self, source, order: str = "ZYX", interval_ms: int = 16,
                 title: str = "WT901WIFI 姿态", jitter_delay: float = 0.08):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(960, 720)
        self.source = source
        self.order = order
        # 抖动缓冲: 按固定延迟回放, 吸收设备/网络的到达抖动
        self._buf: deque[tuple[float, float, float, float]] = deque()
        self._delay = jitter_delay

        # --- 顶部信息栏 ---
        self.info = QtWidgets.QLabel("等待数据 ...")
        self.info.setStyleSheet(
            "font-family: monospace; font-size: 14px; padding: 6px;"
            "background:#111; color:#eee;"
        )
        self.info.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        # 固定高度 + 忽略宽度建议: 文本变化不会引起布局重排,
        # 否则 GL 视口会被反复 resize, 造成间歇性卡顿
        self.info.setFixedHeight(30)
        self.info.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                QtWidgets.QSizePolicy.Policy.Fixed)
        self.info.setTextFormat(QtCore.Qt.TextFormat.PlainText)

        # --- 3D 视图 ---
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((18, 18, 22))
        self.view.setCameraPosition(distance=3.2, elevation=22, azimuth=-60)
        self.view.opts["fov"] = 50

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        grid.setColor((110, 110, 120, 90))
        self.view.addItem(grid)

        # 世界参考轴 (暗色细线, 不随传感器转动)
        world_len = 1.6
        for vec, col in (((world_len, 0, 0), (0.5, 0.25, 0.25, 0.55)),
                         ((0, world_len, 0), (0.25, 0.5, 0.25, 0.55)),
                         ((0, 0, world_len), (0.25, 0.35, 0.5, 0.55))):
            self.view.addItem(gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], vec], dtype=np.float32),
                color=col, width=1.0, antialias=True))

        # 传感器本体立方体 (半透明, 需要 translucent 才会混合透明度)
        cv, cf = cube_mesh(0.36)
        self.body = gl.GLMeshItem(vertexes=cv, faces=cf, smooth=False,
                                  color=(0.35, 0.45, 0.80, 0.16),
                                  glOptions="translucent",
                                  drawEdges=True, edgeColor=(0.55, 0.62, 0.95, 0.45))
        self.view.addItem(self.body)

        # 三条机体轴线
        av, af = arrow_mesh(length=1.0)
        self.axes = {}
        for axis in ("x", "y", "z"):
            R_base = axis_base_rotation(axis)
            V = av @ R_base.T
            item = gl.GLMeshItem(vertexes=V.astype(np.float32), faces=af,
                                 smooth=True, color=AXIS_COLORS[axis],
                                 shader="shaded", glOptions="opaque")
            self.view.addItem(item)
            self.axes[axis] = item

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.info)
        lay.addWidget(self.view, 1)
        self.setCentralWidget(central)

        # --- 状态 ---
        self._count = 0
        self._rate = 0.0
        self._rate_t0 = time.time()
        self._rate_n = 0
        self._last = None
        self._screenshot = None
        self._screenshot_after = 0

        # 文字刷新限流 + 定时器抖动统计
        self._last_info = 0.0
        self._info_interval = 0.1
        self._tick_last = None
        self._tick_gaps: list[float] = []
        self._underruns = 0
        self.stats = False
        self._stats_t0 = time.time()
        self._stats_count = 0
        self._stats_underruns = 0

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # -- 每帧轮询数据源并更新场景 -----------------------------------------
    def _tick(self):
        now = time.time()
        if self._tick_last is not None:
            self._tick_gaps.append(now - self._tick_last)
        self._tick_last = now

        try:
            packets = self.source.poll()
        except Exception as e:
            self.timer.stop()
            QtWidgets.QMessageBox.critical(self, "读取失败", str(e))
            return

        # 有新数据就入缓冲 (记录到达时刻)
        if packets:
            for pkt in packets:
                self._buf.append((now, pkt.roll, pkt.pitch, pkt.yaw))
            self._last = packets[-1]
            self._count += len(packets)
            self._rate_n += len(packets)

        # 每帧都按固定延迟回放 -> 数据到达抖动时画面依然连续
        self._playback(now)

        if now - self._rate_t0 >= 0.5:
            self._rate = self._rate_n / (now - self._rate_t0)
            self._rate_t0 = now
            self._rate_n = 0

        # 文字单独限流, 不与渲染同频
        if self._last is not None and now - self._last_info >= self._info_interval:
            self._last_info = now
            self._update_info(self._last)

        if self.stats:
            self._report_stats(now)

        if self._screenshot and self._count >= self._screenshot_after:
            self._save_screenshot()

    def _playback(self, now: float):
        """抖动缓冲回放: 播放 now-delay 时刻的姿态, 中间做插值。

        设备(AP 模式)会间歇性停摆数百毫秒, 直接显示最新包会一顿一顿。
        这里固定滞后 self._delay 秒回放, 只要缓冲里还有数据, 画面就是
        匀速连续的; 缓冲耗尽时停在最后一个姿态(欠载), 不会乱跳。
        """
        if not self._buf:
            return
        # 丢弃过旧样本, 限制内存
        while len(self._buf) > 1 and self._buf[1][0] < now - self._delay - 2.0:
            self._buf.popleft()

        target = now - self._delay
        prev = None
        nxt = None
        for s in self._buf:
            if s[0] <= target:
                prev = s
            else:
                nxt = s
                break

        if prev is None:                      # 缓冲还没攒够
            prev = self._buf[0]
        if nxt is None:                       # 欠载: 停在最新姿态
            self._underruns += 1
            self._apply_pose(prev[1], prev[2], prev[3])
            return

        span = nxt[0] - prev[0]
        t = 0.0 if span <= 1e-9 else (target - prev[0]) / span
        self._apply_pose(
            lerp_angle(prev[1], nxt[1], t),
            lerp_angle(prev[2], nxt[2], t),
            lerp_angle(prev[3], nxt[3], t),
        )

    def _apply(self, pkt):
        self._apply_pose(pkt.roll, pkt.pitch, pkt.yaw)

    def _apply_pose(self, roll: float, pitch: float, yaw: float):
        """只更新 3D 变换 (每帧都要快)。"""
        m = to_qmatrix(rotation_matrix(roll, pitch, yaw, self.order))
        for item in self.axes.values():
            item.setTransform(m)
        self.body.setTransform(m)

    def _update_info(self, pkt):
        self.info.setText(
            f"{pkt.device_id}   "
            f"Roll {pkt.roll:8.2f}°   Pitch {pkt.pitch:8.2f}°   Yaw {pkt.yaw:8.2f}°    "
            f"| Acc ({pkt.acc_x:6.2f},{pkt.acc_y:6.2f},{pkt.acc_z:6.2f}) g    "
            f"| {self._rate:5.1f} Hz    | 包 {self._count}"
        )

    def _report_stats(self, now):
        if now - self._stats_t0 < 1.0:
            return
        dt = now - self._stats_t0
        polled = self._count - self._stats_count
        self._stats_count = self._count
        gaps = sorted(self._tick_gaps)
        self._tick_gaps = []
        if gaps:
            p50 = gaps[len(gaps) // 2] * 1000
            p99 = gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))] * 1000
            mx = gaps[-1] * 1000
        else:
            p50 = p99 = mx = 0.0
        qsize = getattr(getattr(self.source, "_queue", None), "qsize", lambda: -1)()
        under = self._underruns - self._stats_underruns
        self._stats_underruns = self._underruns
        frames = max(1, int(dt * 1000 / max(p50, 1e-9)))
        print(f"[stats] 收包 {polled / dt:6.1f} Hz | 定时器 "
              f"p50={p50:5.1f}ms p99={p99:6.1f}ms max={mx:6.1f}ms | "
              f"缓冲欠载 {under:4d}/{frames} ({under / frames * 100:4.1f}%) | 队列={qsize}",
              file=sys.stderr, flush=True)
        self._stats_t0 = now

    # -- 截图 --------------------------------------------------------------
    def _save_screenshot(self):
        path = self._screenshot
        self._screenshot = None
        try:
            # 主动激活 GL 上下文, 否则 grabFramebuffer 里 PyOpenGL 取不到 context
            self.view.makeCurrent()
            img = self.view.grabFramebuffer()
            ok = img.save(path)
            print(f"[截图] {path} {'OK' if ok else '失败'}", file=sys.stderr)
        except Exception as e:
            print(f"[截图失败] {e}", file=sys.stderr)
        QtCore.QTimer.singleShot(200, self.close)

    # -- 键盘 --------------------------------------------------------------
    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            self.close()
        elif k == QtCore.Qt.Key.Key_R:
            self.view.setCameraPosition(distance=3.2, elevation=22, azimuth=-60)
        elif k == QtCore.Qt.Key.Key_C:
            path = f"wt901_view_{time.strftime('%Y%m%d_%H%M%S')}.png"
            self.view.grabFramebuffer().save(path)
            print(f"[截图] {path}", file=sys.stderr)
        else:
            super().keyPressEvent(ev)


# --------------------------------------------------------------------------
# 合成姿态源 (无硬件演示 / 渲染自检)
# --------------------------------------------------------------------------
class DemoSource:
    """不接硬件, 生成缓慢摆动的姿态, 便于预览界面。"""

    def __init__(self, hz: float = 50.0):
        self._t0 = time.time()
        self._hz = hz
        self._next = 0.0

    def poll(self):
        now = time.time() - self._t0
        out = []
        while self._next <= now:
            t = self._next
            out.append(_DemoPacket(
                roll=35.0 * math.sin(2 * math.pi * 0.13 * t),
                pitch=25.0 * math.sin(2 * math.pi * 0.19 * t + 1.0),
                yaw=180.0 * math.sin(2 * math.pi * 0.07 * t),
            ))
            self._next += 1.0 / self._hz
        return out


class _DemoPacket:
    def __init__(self, roll, pitch, yaw):
        self.device_id = "DEMO"
        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self.acc_x = self.acc_y = self.acc_z = 0.0


# --------------------------------------------------------------------------
def _ensure_working_qt_platform() -> None:
    """规避 PyOpenGL 在 Qt6 xcb 后端下取不到 GL 上下文的问题。

    Qt6 在 X11 下走 EGL, 而 PyOpenGL 默认按 GLX 找上下文, 会持续报
    "Attempt to retrieve context when no valid context" 并画不出任何东西。
    若当前是 Wayland 会话但环境把 QT_QPA_PLATFORM 设成了 xcb, 自动切回。
    """
    if (os.environ.get("WAYLAND_DISPLAY")
            and os.environ.get("QT_QPA_PLATFORM", "").startswith("xcb")):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        print("[提示] QT_QPA_PLATFORM=xcb 会导致 PyOpenGL 无法获取 GL 上下文, "
              "已自动切换到 wayland", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="WT901WIFI 姿态 3D 可视化")
    ap.add_argument("--port", default=DEFAULT_PORT,
                    help=f"串口设备 (默认 {DEFAULT_PORT})")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--net", choices=("tcp", "udp"),
                    help="改用 WiFi 接收 (设备主动连到本机 1399 端口)")
    ap.add_argument("--listen-port", type=int, default=1399,
                    help="WiFi 模式下的监听端口 (默认 1399)")
    ap.add_argument("--order", default="ZYX", choices=("ZYX", "XYZ"),
                    help="欧拉角旋转顺序 (默认 ZYX, 已用加速度计验证)")
    ap.add_argument("--demo", action="store_true", help="不接硬件, 用合成姿态演示")
    ap.add_argument("--stats", action="store_true",
                    help="每秒向 stderr 打印收包率与定时器抖动 (调优用)")
    ap.add_argument("--delay", type=float, default=80.0,
                    help="抖动缓冲延迟 ms (默认 80; 越大越平滑但越滞后)")
    ap.add_argument("--screenshot", metavar="PNG", help="渲染后存图并退出")
    ap.add_argument("--after", type=int, default=30,
                    help="--screenshot 时, 收到多少帧后截图 (默认 30)")
    args = ap.parse_args()

    _ensure_working_qt_platform()
    app = QtWidgets.QApplication(sys.argv)

    if args.demo:
        source = DemoSource()
        title = "WT901WIFI 姿态 (演示模式)"
    elif args.net:
        cls = TcpAttitudeSource if args.net == "tcp" else UdpAttitudeSource
        try:
            source = cls(args.listen_port)
        except Exception as e:
            print(f"启动 {args.net.upper()} 服务器失败: {e}", file=sys.stderr)
            return 1
        title = f"WT901WIFI 姿态 (WiFi {args.net.upper()} :{args.listen_port})"
    else:
        try:
            source = SerialAttitudeSource(args.port, args.baud)
        except Exception as e:
            print(f"打开串口失败: {e}", file=sys.stderr)
            print("提示: 用 --demo 可先预览界面", file=sys.stderr)
            return 1
        title = f"WT901WIFI 姿态 ({args.port} @ {args.baud})"

    win = OrientationViewer(source, order=args.order, title=title,
                            jitter_delay=args.delay / 1000.0)
    win.stats = args.stats
    if args.screenshot:
        win._screenshot = args.screenshot
        win._screenshot_after = args.after
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
