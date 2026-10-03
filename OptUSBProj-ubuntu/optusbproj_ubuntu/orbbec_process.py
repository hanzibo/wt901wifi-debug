"""SDK-only subprocess: RGB-D -> private mmap + acknowledged JSON protocol.

Run using the orbbec336l interpreter, NOT the GUI's NumPy-2 environment.
No register/preset/firmware writes; profile selection only affects this stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import signal
import sys
import time

import cv2
import numpy as np

if __package__:
    from .rgbd_rate import PhaseRateLimiter
else:
    from rgbd_rate import PhaseRateLimiter

PREFIX = "@RGBD@"


def send(value):
    print(PREFIX + json.dumps(value, allow_nan=False), flush=True)


def intrinsics(value):
    return {key: getattr(value, key) for key in ("width", "height", "fx", "fy", "cx", "cy")}


def distortion(value):
    return {"model": str(value.model), **{k: getattr(value, k) for k in
            ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")}}


def frame_rgb(frame, obformat):
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    h, w = frame.get_height(), frame.get_width()
    fmt = frame.get_format()
    if fmt == obformat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("MJPG 彩色解码失败")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == obformat.RGB:
        return data.reshape(h, w, 3).copy()
    if fmt == obformat.BGR:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    if fmt == obformat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    raise ValueError(f"不支持彩色格式 {fmt}")


def run(args):
    from pyorbbecsdk import (Pipeline, Config, AlignFilter, OBSensorType,
                            OBStreamType, OBFrameAggregateOutputMode, OBFormat, get_version)
    p = Pipeline()
    config = Config()
    profiles = {}
    for key, sensor in (("color", OBSensorType.COLOR_SENSOR), ("depth", OBSensorType.DEPTH_SENSOR)):
        profile = p.get_stream_profile_list(sensor).get_default_video_stream_profile()
        config.enable_stream(profile)
        profiles[key] = {"width": profile.get_width(), "height": profile.get_height(),
                         "fps": profile.get_fps(), "format": str(profile.get_format())}
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    p.enable_frame_sync()
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    mapping = None
    capture_count = missing = sent = 0
    last_indices = None
    gaps = {"color": 0, "depth": 0}
    p.start(config)
    try:
        cam = p.get_camera_param()
        info = p.get_device().get_device_info()
        calibration = {
            "sdk_version": get_version(), "device": info.get_name(),
            "serial": info.get_serial_number(), "firmware": info.get_firmware_version(),
            "connection": info.get_connection_type(), "profiles": profiles,
            "color_intrinsic": intrinsics(cam.rgb_intrinsic),
            "color_distortion": distortion(cam.rgb_distortion),
            "depth_intrinsic": intrinsics(cam.depth_intrinsic),
            "depth_distortion": distortion(cam.depth_distortion),
            "depth_to_color": {"rotation": np.asarray(cam.transform.rot, dtype=float).reshape(3, 3).tolist(),
                               "translation_mm": np.asarray(cam.transform.transform, dtype=float).reshape(3).tolist()},
            "alignment": "software_depth_to_color", "frame_sync": True,
            "output_fps_limit": args.fps, "warmup_seconds": args.warmup,
            "timestamp_semantics": "captured_mono_ns is host frameset receipt, NOT exposure time; device/system/global timestamps retained separately",
        }
        started = time.monotonic()
        output_limiter = PhaseRateLimiter(args.fps, started + args.warmup)
        last_good = started
        while not stopped:
            frames = p.wait_for_frames(1000)
            received_ns = time.monotonic_ns()
            now = received_ns / 1e9
            if not frames or not frames.get_color_frame() or not frames.get_depth_frame():
                missing += 1
                if now - last_good > 5:
                    raise RuntimeError("相机超过 5 秒未返回完整 RGB-D 帧组")
                continue
            last_good = now
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            capture_count += 1
            indices = (cf.get_index(), df.get_index())
            if last_indices is not None:
                for key, new, old in zip(("color", "depth"), indices, last_indices):
                    if new <= old:
                        raise RuntimeError("相机帧号倒退或重复；停止采集以避免混合参考")
                    gaps[key] += max(0, new - old - 1)
            last_indices = indices
            if now < started + args.warmup or not output_limiter.accept(now):
                continue
            aligned = align.process(frames).as_frame_set()
            ad = aligned.get_depth_frame()
            rgb = frame_rgb(aligned.get_color_frame(), OBFormat)
            raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
            metric = np.frombuffer(ad.get_data(), dtype=np.uint16).reshape(ad.get_height(), ad.get_width()).astype(np.float32) * ad.get_depth_scale()
            if metric.shape != rgb.shape[:2]:
                raise RuntimeError("RGB-D 配准尺寸不一致")
            arrays = (rgb, raw, metric)
            if mapping is None:
                aligned_profile = ad.get_stream_profile().as_video_stream_profile()
                calibration["aligned_depth_intrinsic"] = intrinsics(aligned_profile.get_intrinsic())
                calibration["aligned_depth_distortion"] = distortion(aligned_profile.get_distortion())
                offset, layouts = 0, []
                for array in arrays:
                    layouts.append({"shape": list(array.shape), "dtype": array.dtype.str, "offset": offset})
                    offset += array.nbytes
                path = Path(args.buffer)
                with path.open("wb") as f:
                    f.truncate(offset)
                mapping = np.memmap(path, mode="r+", dtype=np.uint8, shape=(offset,))
                send({"type": "ready", "layouts": layouts, "nbytes": offset, "calibration": calibration})
            for layout, array in zip(layouts, arrays):
                view = np.ndarray(tuple(layout["shape"]), dtype=layout["dtype"], buffer=mapping, offset=layout["offset"])
                view[:] = array
            timing = {"host_received_mono_ns": received_ns,
                      "color_index": indices[0], "depth_index": indices[1],
                      "raw_depth_scale_mm": df.get_depth_scale(), "aligned_depth_scale_mm": ad.get_depth_scale(),
                      "aligned_device_us": ad.get_timestamp_us(),
                      "capture_count": capture_count, "output_count": sent + 1,
                      "source_index_gaps": gaps.copy(), "missing_framesets": missing,
                      "valid_depth_fraction": float((metric > 0).mean())}
            for key, frame in (("color", cf), ("depth", df)):
                for label, method in (("device_us", "get_timestamp_us"), ("system_us", "get_system_timestamp_us"), ("global_us", "get_global_timestamp_us")):
                    timing[f"{key}_{label}"] = getattr(frame, method)()
            send({"type": "frame", "timing": timing})
            # A slot cannot be overwritten until its consumer has copied it.
            while not stopped:
                if select.select([sys.stdin], [], [], .5)[0]:
                    response = sys.stdin.readline().strip()
                    if response != "ACK":
                        stopped = True
                    break
            sent += 1
        send({"type": "stopped", "capture_count": capture_count, "output_count": sent,
              "source_index_gaps": gaps, "missing_framesets": missing})
    finally:
        p.stop()
        if mapping is not None:
            mapping._mmap.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--warmup", type=float, default=2)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or not 0 <= args.warmup <= 30:
        parser.error("帧率需为 1～30，预热需为 0～30 秒")
    try:
        run(args)
    except Exception as exc:
        send({"type": "error", "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
