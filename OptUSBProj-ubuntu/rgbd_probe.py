#!/usr/bin/env python3
"""Headless formal RGB-D + WT901 recording smoke test; no register writes."""
from pathlib import Path
import argparse
import sys
import time

PROJECT = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT), str(PROJECT.parent)]


def main():
    parser = argparse.ArgumentParser(description="Gemini RGB-D + WT901 短时联合采集验收")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--warmup", type=float, default=2)
    parser.add_argument("--orbbec-python", type=Path, default=Path.home()/"miniconda3/envs/orbbec336l/bin/python")
    parser.add_argument("--record-parent", type=Path, default=PROJECT/"recordings"/"rgbd")
    parser.add_argument("--listen-port", type=int, default=1399)
    parser.add_argument("--no-imu", action="store_true", help="不占用 UDP；仅验证 RGB-D")
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or not 0 <= args.warmup <= 30 or args.seconds <= args.warmup + 3:
        parser.error("时长必须大于预热+末尾3秒，帧率1～30，预热0～30秒")
    from PySide6 import QtCore
    from wt901.gui_udp import GuiUdpSource
    from optusbproj_ubuntu.orbbec_worker import OrbbecWorker
    from optusbproj_ubuntu.fusion_recording import FusionSessionRecorder
    app = QtCore.QCoreApplication([])
    try:
        source = None if args.no_imu else GuiUdpSource(args.listen_port)
    except OSError as exc:
        parser.error(f"UDP端口被占用或不可用，请先关闭其他GUI：{exc}")
    recorder = FusionSessionRecorder(args.record_parent, max_pending_frames=16)
    worker = OrbbecWorker(args.orbbec_python, fps=args.fps, warmup=args.warmup)
    errors = []
    started = time.monotonic()
    next_report = 5
    def fail(message):
        errors.append(message)
        print(message, file=sys.stderr, flush=True)
        app.quit()
    worker.error.connect(fail)
    def poll():
        nonlocal next_report
        try:
            if source:
                for stamp, packet in source.poll_timed():
                    recorder.write_imu(stamp, packet)
            observation = worker.take_latest()
            if observation is not None:
                recorder.write_camera(observation)
            elapsed = time.monotonic()-started
            if elapsed >= next_report:
                print(f"{elapsed:.1f}/{args.seconds:g}s | RGB-D 已写 {recorder.frame_count} 帧 | IMU {recorder.imu_count} 包 | GUI/消费者跳过 {worker.consumer_skipped}", flush=True)
                next_report += 5
            if elapsed >= args.seconds:
                app.quit()
        except Exception as exc:
            fail(str(exc))
    timer = QtCore.QTimer()
    timer.setInterval(20)
    timer.timeout.connect(poll)
    timer.start()
    worker.start()
    print(f"采集开始；前 {args.warmup:g} 秒预热，末尾 3 秒剔除；目录 {recorder.path}", flush=True)
    try:
        app.exec()
    finally:
        timer.stop()
        worker.stop()
        if source:
            source.close()
        recorder.camera_bridge_skipped = worker.consumer_skipped
        recorder.error = recorder.error or (errors[0] if errors else None)
        try:
            recorder.close()
        except OSError as exc:
            errors.append(str(exc))
        finally:
            if not worker.wait(5000):
                worker._terminate_owned()
                worker.wait()
    print(f"完成：RGB-D {recorder.rgbd_count} 帧，IMU {recorder.imu_count} 包，写盘丢帧 {recorder.dropped_frames}；{recorder.path}", flush=True)
    return 1 if errors or recorder.rgbd_count == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
