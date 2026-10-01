#!/usr/bin/python3
"""Record two Wayland monitors and the front camera into three video files."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib
except (ImportError, ValueError) as exc:
    print(
        "错误：需要 Ubuntu 系统 Python 的 PyGObject。请用 /usr/bin/python3 运行本程序。\n"
        f"详细信息：{exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)


PORTAL_NAME = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"


@dataclass(frozen=True)
class Monitor:
    name: str
    width: int
    height: int
    x: int
    y: int
    primary: bool = False


@dataclass
class ChildProcess:
    label: str
    path: Path
    process: subprocess.Popen[bytes]
    error_tail: deque[str]
    reader: threading.Thread


def _deep_unpack(value: Any) -> Any:
    if isinstance(value, GLib.Variant):
        return _deep_unpack(value.unpack())
    if isinstance(value, dict):
        return {key: _deep_unpack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_deep_unpack(item) for item in value)
    return value


def detect_monitors() -> list[Monitor]:
    """Read the active XWayland monitor geometry exposed by GNOME."""
    try:
        output = subprocess.run(
            ["xrandr", "--query"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []

    pattern = re.compile(
        r"^(?P<name>\S+) connected(?P<primary> primary)? "
        r"(?P<w>\d+)x(?P<h>\d+)\+(?P<x>-?\d+)\+(?P<y>-?\d+)"
    )
    monitors: list[Monitor] = []
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            monitors.append(
                Monitor(
                    name=match.group("name"),
                    width=int(match.group("w")),
                    height=int(match.group("h")),
                    x=int(match.group("x")),
                    y=int(match.group("y")),
                    primary=bool(match.group("primary")),
                )
            )
    return sorted(monitors, key=lambda item: (item.x, item.y))


def detect_camera() -> tuple[Path | None, str | None]:
    for sys_path in sorted(Path("/sys/class/video4linux").glob("video*")):
        device = Path("/dev") / sys_path.name
        if not device.exists():
            continue
        try:
            caps = (sys_path / "device/capabilities").read_text().strip()
        except OSError:
            caps = ""
        # Metadata-only nodes normally contain "metadata" in this file.
        if caps and "capture" not in caps.lower():
            continue
        try:
            name = (sys_path / "name").read_text().strip()
        except OSError:
            name = device.name
        return device, name
    # Some drivers do not expose the capabilities sysfs file.
    fallback = Path("/dev/video0")
    return (fallback, "Integrated Camera") if fallback.exists() else (None, None)


def scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width // 2 * 2, height // 2 * 2
    ratio = max_width / width
    return max_width // 2 * 2, max(2, round(height * ratio) // 2 * 2)


class ScreenCastPortal:
    def __init__(self) -> None:
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        sender = self.bus.get_unique_name().lstrip(":").replace(".", "_")
        self.request_prefix = f"/org/freedesktop/portal/desktop/request/{sender}"
        self.session_path: str | None = None

    @staticmethod
    def _token(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def _request(
        self, method: str, signature: str, values: tuple[Any, ...], token: str
    ) -> dict[str, Any]:
        request_path = f"{self.request_prefix}/{token}"
        loop = GLib.MainLoop()
        response: dict[str, Any] = {}

        def on_response(
            _connection: Gio.DBusConnection,
            _sender: str,
            _path: str,
            _interface: str,
            _signal: str,
            parameters: GLib.Variant,
            _data: Any,
        ) -> None:
            code, results = _deep_unpack(parameters)
            response["code"] = code
            response["results"] = results
            loop.quit()

        subscription = self.bus.signal_subscribe(
            PORTAL_NAME,
            REQUEST_IFACE,
            "Response",
            request_path,
            None,
            Gio.DBusSignalFlags.NONE,
            on_response,
            None,
        )
        try:
            reply = self.bus.call_sync(
                PORTAL_NAME,
                PORTAL_PATH,
                SCREENCAST_IFACE,
                method,
                GLib.Variant(signature, values),
                GLib.VariantType.new("(o)"),
                Gio.DBusCallFlags.NONE,
                10_000,
                None,
            )
            returned_path = reply.unpack()[0]
            if returned_path != request_path:
                raise RuntimeError(
                    f"屏幕共享门户返回了意外的请求路径：{returned_path}"
                )
            timeout_id = GLib.timeout_add_seconds(120, loop.quit)
            loop.run()
            GLib.source_remove(timeout_id)
        finally:
            self.bus.signal_unsubscribe(subscription)

        if "code" not in response:
            raise TimeoutError("等待屏幕共享授权超时")
        if response["code"] != 0:
            raise RuntimeError("用户取消了屏幕共享，录制未开始")
        return response["results"]

    def start(self) -> list[tuple[int, dict[str, Any]]]:
        create_token = self._token("create")
        session_token = self._token("session")
        created = self._request(
            "CreateSession",
            "(a{sv})",
            (
                {
                    "handle_token": GLib.Variant("s", create_token),
                    "session_handle_token": GLib.Variant("s", session_token),
                },
            ),
            create_token,
        )
        self.session_path = str(created["session_handle"])

        select_token = self._token("select")
        self._request(
            "SelectSources",
            "(oa{sv})",
            (
                self.session_path,
                {
                    "handle_token": GLib.Variant("s", select_token),
                    "types": GLib.Variant("u", 1),  # MONITOR
                    "multiple": GLib.Variant("b", True),
                    "cursor_mode": GLib.Variant("u", 2),  # embedded
                },
            ),
            select_token,
        )

        start_token = self._token("start")
        started = self._request(
            "Start",
            "(osa{sv})",
            (
                self.session_path,
                "",
                {"handle_token": GLib.Variant("s", start_token)},
            ),
            start_token,
        )
        streams = started.get("streams", [])
        return [(int(node), dict(properties)) for node, properties in streams]

    def open_pipewire_remote(self) -> int:
        if not self.session_path:
            raise RuntimeError("屏幕共享会话尚未创建")
        reply, fd_list = self.bus.call_with_unix_fd_list_sync(
            PORTAL_NAME,
            PORTAL_PATH,
            SCREENCAST_IFACE,
            "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (self.session_path, {})),
            GLib.VariantType.new("(h)"),
            Gio.DBusCallFlags.NONE,
            10_000,
            None,
            None,
        )
        return fd_list.get(reply.unpack()[0])

    def close(self) -> None:
        if not self.session_path:
            return
        try:
            self.bus.call_sync(
                PORTAL_NAME,
                self.session_path,
                SESSION_IFACE,
                "Close",
                None,
                None,
                Gio.DBusCallFlags.NONE,
                2_000,
                None,
            )
        except GLib.Error:
            pass
        self.session_path = None


def _gst_available(element: str) -> bool:
    return subprocess.run(
        ["gst-inspect-1.0", element],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def validate_system() -> tuple[list[Monitor], Path, str]:
    missing = [
        command
        for command in (
            "xrandr",
            "gst-launch-1.0",
            "gst-inspect-1.0",
            "ffmpeg",
            "ffprobe",
        )
        if shutil.which(command) is None
    ]
    if missing:
        raise RuntimeError("缺少系统命令：" + ", ".join(missing))
    required_elements = [
        "pipewiresrc",
        "v4l2src",
        "jpegdec",
        "videoconvert",
        "videoscale",
        "videorate",
        "x264enc",
        "matroskamux",
    ]
    missing_elements = [item for item in required_elements if not _gst_available(item)]
    if missing_elements:
        raise RuntimeError("缺少 GStreamer 元件：" + ", ".join(missing_elements))

    monitors = detect_monitors()
    if len(monitors) != 2:
        raise RuntimeError(f"需要恰好两台已连接显示器，当前检测到 {len(monitors)} 台")
    camera, camera_name = detect_camera()
    if camera is None or camera_name is None:
        raise RuntimeError("未检测到可用摄像头")
    if not os.access(camera, os.R_OK | os.W_OK):
        raise RuntimeError(f"当前用户无权访问摄像头 {camera}")
    return monitors, camera, camera_name


def _screen_property(stream: tuple[int, dict[str, Any]], key: str, default: Any) -> Any:
    return _deep_unpack(stream[1].get(key, default))


def assign_streams(
    streams: list[tuple[int, dict[str, Any]]], monitors: list[Monitor]
) -> list[tuple[tuple[int, dict[str, Any]], Monitor]]:
    if len(streams) != 2:
        raise RuntimeError(
            f"请选择两台显示器；屏幕共享门户只返回了 {len(streams)} 路画面"
        )

    remaining = monitors.copy()
    assigned: list[tuple[tuple[int, dict[str, Any]], Monitor]] = []
    ordered = sorted(
        streams,
        key=lambda stream: tuple(_screen_property(stream, "position", (10**9, 10**9))),
    )
    for stream in ordered:
        position = tuple(_screen_property(stream, "position", ()))
        match = next(
            (item for item in remaining if position == (item.x, item.y)),
            remaining[0],
        )
        remaining.remove(match)
        assigned.append((stream, match))
    return sorted(assigned, key=lambda item: (item[1].x, item[1].y))


def _spawn(
    label: str, path: Path, command: list[str], pass_fds: tuple[int, ...] = ()
) -> ChildProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        pass_fds=pass_fds,
    )
    tail: deque[str] = deque(maxlen=30)

    def drain() -> None:
        assert process.stderr is not None
        for raw_line in process.stderr:
            tail.append(raw_line.decode("utf-8", errors="replace").rstrip())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    return ChildProcess(label, path, process, tail, reader)


def screen_command(
    node: int,
    fd: int,
    path: Path,
    width: int,
    height: int,
    fps: int,
    bitrate: int,
) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "-e",
        "pipewiresrc",
        f"fd={fd}",
        f"path={node}",
        "do-timestamp=true",
        "!",
        "queue",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        "videorate",
        "!",
        f"video/x-raw,format=I420,width={width},height={height},framerate={fps}/1",
        "!",
        "x264enc",
        "speed-preset=ultrafast",
        "tune=zerolatency",
        f"bitrate={bitrate}",
        f"key-int-max={fps * 2}",
        "!",
        "matroskamux",
        "!",
        "filesink",
        f"location={path}",
    ]


def camera_command(camera: Path, path: Path, fps: int, bitrate: int) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "-e",
        "v4l2src",
        f"device={camera}",
        "do-timestamp=true",
        "!",
        "image/jpeg,width=1280,height=720,framerate=30/1",
        "!",
        "jpegdec",
        "!",
        "videoconvert",
        "!",
        "videorate",
        "!",
        f"video/x-raw,format=I420,framerate={fps}/1",
        "!",
        "x264enc",
        "speed-preset=ultrafast",
        "tune=zerolatency",
        f"bitrate={bitrate}",
        f"key-int-max={fps * 2}",
        "!",
        "matroskamux",
        "!",
        "filesink",
        f"location={path}",
    ]


def stop_children(children: list[ChildProcess]) -> None:
    for child in children:
        if child.process.poll() is None:
            child.process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 8
    for child in children:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            child.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.process.terminate()
    for child in children:
        try:
            child.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.process.kill()
            child.process.wait()
        child.reader.join(timeout=1)


def video_time_span(path: Path) -> tuple[float, float]:
    """Return the first packet timestamp and total video span without decoding."""
    process = subprocess.Popen(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    first_pts: float | None = None
    last_end: float | None = None
    for line in process.stdout:
        fields = line.strip().split(",")
        try:
            pts = float(fields[0])
            packet_duration = float(fields[1]) if len(fields) > 1 else 0.0
        except (ValueError, IndexError):
            continue
        if first_pts is None:
            first_pts = pts
        last_end = pts + packet_duration
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code or first_pts is None or last_end is None:
        raise RuntimeError(f"无法读取视频时长 {path.name}：{stderr.strip()}")
    return first_pts, last_end - first_pts


def trim_encoder_args() -> list[str]:
    """Prefer the installed NVIDIA encoder, falling back to software x264."""
    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=256x256:d=0.05",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]


def trim_recordings(paths: list[Path], seconds: float = 3.0) -> None:
    """Transactionally remove the start and end of every recording."""
    jobs: list[tuple[Path, Path, float, float]] = []
    for path in paths:
        first_pts, duration = video_time_span(path)
        keep_duration = duration - seconds * 2
        if keep_duration <= 0:
            raise RuntimeError(
                f"{path.name} 只有 {duration:.1f} 秒，无法同时删除首尾各 {seconds:g} 秒"
            )
        temporary = path.with_name(f".{path.stem}.trimmed{path.suffix}")
        jobs.append((path, temporary, first_pts + seconds, keep_duration))

    encoder = trim_encoder_args()
    try:
        for index, (source, temporary, start, duration) in enumerate(jobs, 1):
            print(f"正在裁剪 {index}/{len(jobs)}：{source.name}")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.6f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{duration:.6f}",
                    "-map",
                    "0:v:0",
                    "-an",
                    *encoder,
                    str(temporary),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0 or not temporary.exists() or not temporary.stat().st_size:
                raise RuntimeError(
                    f"裁剪 {source.name} 失败：{result.stderr.strip() or '没有生成有效视频'}"
                )
        for source, temporary, _start, _duration in jobs:
            os.replace(temporary, source)
    finally:
        for _source, temporary, _start, _duration in jobs:
            if temporary.exists():
                temporary.unlink()


def print_check(monitors: list[Monitor], camera: Path, camera_name: str) -> None:
    print("系统检查通过")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print("会话: " + os.environ.get("XDG_SESSION_TYPE", "未知"))
    for index, monitor in enumerate(monitors, 1):
        suffix = "，主显示器" if monitor.primary else ""
        print(
            f"显示器 {index}: {monitor.name}, {monitor.width}x{monitor.height}, "
            f"位置 +{monitor.x}+{monitor.y}{suffix}"
        )
    print(f"摄像头: {camera_name} ({camera})，1280x720 @ 30fps")
    print("编码器: GStreamer x264enc（ultrafast）")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="同时录制两台显示器和前置摄像头，删除首尾各 3 秒后输出三份 MKV。"
    )
    parser.add_argument("--check", action="store_true", help="只检查设备和依赖")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Videos",
        help="时间目录的上级目录（默认：~/Videos）",
    )
    parser.add_argument(
        "--duration", type=float, default=0, help="自动停止秒数；0 表示按 Ctrl+C 停止"
    )
    parser.add_argument("--fps", type=int, default=30, choices=(10, 15, 20, 25, 30))
    parser.add_argument(
        "--max-screen-width",
        type=int,
        default=1920,
        help="屏幕视频的最大宽度；0 表示原始分辨率",
    )
    args = parser.parse_args()

    if 0 < args.duration <= 6:
        parser.error("--duration 必须大于 6 秒，才能删除首尾各 3 秒")

    try:
        monitors, camera, camera_name = validate_system()
    except RuntimeError as exc:
        print(f"检查失败：{exc}", file=sys.stderr)
        return 2
    print_check(monitors, camera, camera_name)
    if args.check:
        return 0
    if os.environ.get("XDG_SESSION_TYPE") != "wayland":
        print("警告：当前不是 Wayland 会话，但仍将尝试使用桌面共享门户。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    portal = ScreenCastPortal()
    children: list[ChildProcess] = []
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        print("\n请在弹出的系统窗口中同时选择两台显示器，然后确认共享。")
        streams = portal.start()
        assignments = assign_streams(streams, monitors)

        for index, (stream, monitor) in enumerate(assignments, 1):
            node, properties = stream
            source_size = tuple(_screen_property(stream, "size", (monitor.width, monitor.height)))
            if len(source_size) != 2:
                source_size = (monitor.width, monitor.height)
            width, height = scaled_size(
                int(source_size[0]), int(source_size[1]), args.max_screen_width
            )
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", monitor.name)
            path = output_dir / f"screen_{index}_{safe_name}.mkv"
            fd = portal.open_pipewire_remote()
            try:
                command = screen_command(
                    node, fd, path, width, height, args.fps, bitrate=8_000
                )
                children.append(_spawn(f"屏幕 {index} ({monitor.name})", path, command, (fd,)))
            finally:
                os.close(fd)
            print(
                f"已启动屏幕 {index}: {monitor.name} -> {width}x{height} "
                f"({properties.get('id', 'portal')})"
            )

        camera_path = output_dir / "camera.mkv"
        children.append(
            _spawn(
                "前置摄像头",
                camera_path,
                camera_command(camera, camera_path, args.fps, bitrate=4_000),
            )
        )
        print(f"已启动摄像头: {camera_name} -> 1280x720")
        print("\n三路录制中。按 Ctrl+C 可安全停止并封装视频。")

        start_time = time.monotonic()
        while not stop_requested:
            failed = [child for child in children if child.process.poll() is not None]
            if failed:
                labels = "、".join(child.label for child in failed)
                raise RuntimeError(f"录制进程意外退出：{labels}")
            if args.duration > 0 and time.monotonic() - start_time >= args.duration:
                break
            time.sleep(0.2)
    except (GLib.Error, RuntimeError, TimeoutError) as exc:
        print(f"\n录制失败：{exc}", file=sys.stderr)
        for child in children:
            if child.error_tail:
                print(f"\n[{child.label} 最近日志]", file=sys.stderr)
                print("\n".join(child.error_tail), file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        if children:
            print("\n正在安全停止三路录制……")
            stop_children(children)
        portal.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    valid_files = [child.path for child in children if child.path.exists() and child.path.stat().st_size]
    if return_code == 0 and len(valid_files) == 3:
        print("\n正在删除三份视频开头和结尾的各 3 秒……")
        try:
            trim_recordings(valid_files)
        except RuntimeError as exc:
            print(f"裁剪失败，已保留未经裁剪的原视频：{exc}", file=sys.stderr)
            return_code = 1
    if valid_files:
        print("\n已保存：")
        for path in valid_files:
            print(f"  {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
