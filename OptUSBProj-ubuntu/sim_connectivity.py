#!/usr/bin/env python3
"""Verify CoppeliaSim ZeroMQ read/write calls without a robot or IMUs.

The probe creates one temporary dummy in a stopped scene, moves it by 10 cm,
reads the position back, and removes it in a finally block. It never saves a
scene or starts/stops a simulation.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time


def probe(sim: object) -> tuple[list[float], list[float]]:
    """Exercise one read and one reversible write against a sim API object."""
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("仿真正运行中；请停止仿真后再测试，避免修改活动场景")

    handle = sim.createDummy(0.025)
    try:
        start = [0.0, 0.0, 0.4]
        target = [0.1, 0.0, 0.4]
        sim.setObjectPosition(handle, start)
        before = sim.getObjectPosition(handle)
        sim.setObjectPosition(handle, target)
        after = sim.getObjectPosition(handle)
        if any(abs(actual - expected) > 1e-5
               for actual, expected in zip(after, target)):
            raise RuntimeError(f"CoppeliaSim 未返回目标位置：{after} != {target}")
        if len(after) != 3 or len(before) != 3:
            raise RuntimeError("CoppeliaSim 返回的位置维度不正确")
        return list(before), list(after)
    finally:
        sim.removeObjects([handle])


def visual_demo(sim: object, seconds: float) -> None:
    """Move a visible dummy back and forth in a stopped GUI scene."""
    if seconds <= 0:
        raise ValueError("可视演示时长必须大于 0")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("仿真正运行中；请先停止仿真再做可视演示")
    handle = sim.createDummy(0.25)
    try:
        sim.setObjectAlias(handle, "WT901_ZMQ_demo_dummy")
        started = time.monotonic()
        last_report = -2.0
        print(f"可视演示开始：dummy 将在 X=-0.6～+0.6 m 往返 {seconds:g} 秒。", flush=True)
        while (elapsed := time.monotonic() - started) < seconds:
            target = [0.6 * math.sin(2 * math.pi * elapsed / 5.0), 0.0, 0.5]
            sim.setObjectPosition(handle, target)
            actual = sim.getObjectPosition(handle)
            if len(actual) != 3 or any(abs(a - b) > 1e-4
                                       for a, b in zip(actual, target)):
                raise RuntimeError(f"可视演示位置回读异常：{actual} != {target}")
            if elapsed - last_report >= 2.0:
                print(f"  {elapsed:5.1f}s  X 指令/读回 {target[0]:+.3f}/{actual[0]:+.3f} m",
                      flush=True)
                last_report = elapsed
            time.sleep(0.08)
        print("可视演示完成；正在删除临时 dummy。", flush=True)
    finally:
        sim.removeObjects([handle])


def _load_client(coppelia_dir: Path | None):
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        return RemoteAPIClient
    except ModuleNotFoundError as exc:
        if exc.name not in {"coppeliasim_zmqremoteapi_client", "zmq", "cbor", "cbor2"}:
            raise
        if exc.name != "coppeliasim_zmqremoteapi_client":
            raise RuntimeError(
                f"缺少 Python 依赖 {exc.name}；请在 OptUSBProj 环境安装 pyzmq cbor") from exc
        if coppelia_dir is None:
            raise RuntimeError(
                "未安装 CoppeliaSim Python 客户端；请传 --coppelia-dir 指向本机安装目录") from exc
        client_src = (coppelia_dir / "programming" / "zmqRemoteApi" /
                      "clients" / "python" / "src")
        if not (client_src / "coppeliasim_zmqremoteapi_client" / "__init__.py").is_file():
            raise RuntimeError(f"找不到安装包自带的 Python 客户端：{client_src}") from exc
        sys.path.insert(0, str(client_src))
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ModuleNotFoundError as dependency:
            raise RuntimeError(
                f"缺少 Python 依赖 {dependency.name}；请在 OptUSBProj 环境安装 pyzmq cbor") from dependency
        return RemoteAPIClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CoppeliaSim ZeroMQ 最小连通与指令回读测试")
    parser.add_argument("--coppelia-dir", type=Path,
                        default=Path(os.environ["COPPELIASIM_ROOT"])
                        if "COPPELIASIM_ROOT" in os.environ else None,
                        help="CoppeliaSim 安装目录；用于导入其自带的 Python 客户端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--visual-demo-seconds", type=float, default=0.0,
                        help="在可见 GUI 中让较大 dummy 左右往返的时长；默认只做快速连通测试")
    args = parser.parse_args(argv)
    try:
        client_type = _load_client(args.coppelia_dir)
        client = client_type(host=args.host, port=args.port)
        sim = client.require("sim")
        print(f"已连接 CoppeliaSim ZeroMQ：{args.host}:{args.port}", flush=True)
        if args.visual_demo_seconds:
            visual_demo(sim, args.visual_demo_seconds)
        else:
            before, after = probe(sim)
    except Exception as exc:
        parser.exit(1, f"连通测试失败：{exc}\n")
    if not args.visual_demo_seconds:
        print(f"位置回读：{before} → {after} 米；已删除临时 dummy。", flush=True)
    print("PASS：连接、发送指令、读取结果均成功。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
