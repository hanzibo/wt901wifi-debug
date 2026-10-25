#!/usr/bin/env python3
"""Jetson GUI, offline smoke and read-only IMU probe. Never auto-opens UART."""
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'OptUSBProj-ubuntu')]


def probe(seconds, host, port):
    from wt901.gui_udp import GuiUdpSource
    source = GuiUdpSource(port, host)
    counts, gaps, last = Counter(), {}, {}
    start = time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            for stamp, packet in source.poll_timed():
                now = stamp / 1e9
                key = packet.device_id
                counts[key] += 1
                if key in last:
                    gaps.setdefault(key, []).append((now - last[key]) * 1000)
                last[key] = now
            time.sleep(.002)
        elapsed = time.monotonic() - start
        result = {key: {'packets': count, 'hz': round(count / elapsed, 2),
                        'max_read_gap_ms': round(max(gaps.get(key, [0])), 2)}
                  for key, count in counts.items()}
        print(json.dumps({'seconds': elapsed, 'listen': [host, port], 'devices': result,
                          'note': 'application read timing; not physical latency'}, ensure_ascii=False, indent=2))
        return 0 if counts else 2
    finally:
        source.close()


def smoke():
    from optusbproj_ubuntu.real_robot_arm import ArmLimits
    from optusbproj_ubuntu.real_robot_protocol import encode, decode
    from optusbproj_ubuntu.real_robot_shared import SharedArmIK, SharedArmControl
    from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT
    from optusbproj_ubuntu.skeleton import PART_BY_KEY
    import numpy as np
    engine = PoseEngine()
    transforms = engine.transforms(include_natural=False)
    points = [transforms[PART_BY_KEY[f'left_{p}']][:3, 3] for p in ('upper_arm', 'forearm', 'hand')]
    human = {'directions': [(points[1]-points[0])/np.linalg.norm(points[1]-points[0]),
                            (points[2]-points[1])/np.linalg.norm(points[2]-points[1])],
             'points': points, 'hand': transforms[PART_BY_KEY['left_hand']][:3, :3],
             'rotations': {p: transforms[PART_BY_KEY[f'left_{p}']][:3, :3]
                           for p in ('upper_arm', 'forearm')}}
    limits = ArmLimits(enabled=(True,)*6)
    frame = encode([90.]*15, [0]*10)
    solver = SharedArmIK(frame, limits)
    started = time.monotonic()
    solved = solver.solve(human)
    control = SharedArmControl(frame, limits, solved['target'])
    output = control.step()
    assert len(output) == 54 and decode(output)[1] == [0]*10
    print(json.dumps({'offline': True, 'uart_opened': False, 'imu_algorithm': GYRO_TILT,
                      'ik_and_step_ms': (time.monotonic()-started)*1000,
                      'target': solved['target'], 'frame_bytes': len(output)}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('runtime.local.json'))
    parser.add_argument('--listen-ip', default='0.0.0.0')
    parser.add_argument('--listen-port', type=int, default=1399)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--probe', type=float, metavar='SECONDS')
    modes.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.probe is not None:
        if not 1 <= args.probe <= 600:
            parser.error('probe seconds must be 1–600')
        return probe(args.probe, args.listen_ip, args.listen_port)
    if args.self_test:
        return smoke()
    from jetson_teleop.config import load
    from jetson_teleop.window import Window
    from wt901.gui_udp import GuiUdpSource
    from PySide6.QtWidgets import QApplication
    config = load(args.config)
    app = QApplication(sys.argv[:1])
    source = GuiUdpSource(args.listen_port, args.listen_ip)
    try:
        window = Window(source, config)
        window.show()
        return app.exec()
    finally:
        source.close()


if __name__ == '__main__':
    raise SystemExit(main())
