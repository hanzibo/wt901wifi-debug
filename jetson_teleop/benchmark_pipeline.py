"""Fixed-input ablation: IK / +recorded IMU pose / +Qt / +JSONL, NO IO devices.

Recorded packets are already decoded: this does NOT benchmark kernel UDP parsing,
UART, servo execution or the live watchdog. Late benchmark inputs are NOT dropped
so every stage solves the identical sequence; schedule lag is reported explicitly.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'OptUSBProj-ubuntu'))
from .replay_ik import human, solver

STAGES = ('ik', 'pose', 'gui', 'logging')
_APP = None  # Keep QApplication alive across successive test groups.


def stats(values):
    if not values:
        return None
    return dict(n=len(values), **dict(zip(('median','p95','max'),
                map(float,np.percentile(values,[50,95,100])))))


def frequencies():
    result = {}
    for policy in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
        try:
            result[policy.name] = dict(khz=int((policy/'scaling_cur_freq').read_text()),
                                      governor=(policy/'scaling_governor').read_text().strip())
        except (OSError, ValueError):
            pass
    return result


def load_recording(path, control=None, max_frames=200):
    settings = None; begins = []; packets = []; complete = False; snapshots = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'metadata':
                if row.get('schema_version') != 1:
                    raise ValueError('不支持的诊断格式')
                settings = row['settings']
            elif row['event'] == 'ik_begin':
                begins.append(row)
            elif row['event'] == 'imu_packet':
                packets.append((row['read_mono_ns']/1e9, row['packet']))
            elif row['event'] == 'pose_snapshot':
                snapshots.append(row)
            elif row['event'] == 'recording_end':
                complete = bool(row.get('complete'))
    if not settings or not begins or not packets:
        raise ValueError('需要包含实时 ik_begin 和 imu_packet 的录制')
    first = begins[0]['monotonic_ns']/1e9
    initial = None; seed_note = 'neutral seed (no matching initial control target supplied)'
    if control:
        # Match the SAME session by its last transition BEFORE the first IK.
        with Path(control).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['event'] == 'shared_transition' and 0 <= first-row['monotonic_ns']/1e9 < 5:
                    initial = row['target']
        if initial is None or len(initial) != 12:
            raise ValueError('控制日志与录制首次IK不匹配，无法恢复初值')
        seed_note = 'matching initial transition command from control log'
    begins = begins[:max_frames]
    end = begins[-1]['monotonic_ns']/1e9 + 1
    packets = [(stamp,raw) for stamp,raw in packets if stamp <= end]
    # Preserve current neutral geometry for workload fidelity, not just feed cost.
    seed = next((r['entries'] for r in reversed(snapshots)
                 if r['monotonic_ns']/1e9 <= first and all(r['entries'].get(s) for s in settings['limits'])),None)
    return dict(settings=settings, begins=begins, packets=packets, first=first,
                initial=initial, complete=complete, seed_note=seed_note, pose_seed=seed)


class RecordedSource:
    """No socket implementation: only a paced iterator over decoded packets."""
    def __init__(self, packets, origin):
        from wt901.protocol import AttitudePacket
        self.packets = [(stamp-origin, AttitudePacket(**raw)) for stamp,raw in packets if stamp>=origin]
        self.index = 0; self.started = None; self.sinks = []
        self._latest_packets = {}; self.endpoints = {}; self.gaps = []
        self.sock = SimpleNamespace(getsockname=lambda:('recorded-file',0))

    def add_diagnostic_sink(self, sink): self.sinks.append(sink)
    def remove_diagnostic_sink(self, sink): self.sinks.remove(sink)
    def receive_stats(self): return {}
    def seen_endpoints(self, max_age=3.):
        now = time.monotonic()
        return {k:v for k,v in self.endpoints.items() if now-v[2]<=max_age}
    def seen_devices(self, max_age=3.):
        return {k:(v[0],v[2]) for k,v in self.seen_endpoints(max_age).items()}

    def poll_timed(self):
        if self.started is None: return []
        now = time.monotonic(); out = []
        while self.index < len(self.packets):
            offset,packet = self.packets[self.index]
            at = self.started+offset
            if at>now: break
            self.index += 1
            self._latest_packets[packet.device_id] = packet
            self.endpoints[packet.device_id] = ('recorded-file',0,at)
            out.append((int(at*1e9),packet))
        return out

    def switch_to_ap_udp(self, *args):
        raise RuntimeError('离线回放禁止设备指令')

    def close(self): pass


def pose_engine(data):
    from optusbproj_ubuntu.pose import PoseEngine, GYRO_TILT
    settings = data['settings']; config = settings['config']
    engine = PoseEngine(config['assignments'], config['axis_mappings'],
                        {k:GYRO_TILT for k in config['assignments']})
    from wt901.protocol import AttitudePacket
    # Initialize filters OUTSIDE the measured interval; never replay a burst of
    # nine seconds of old packets concurrently with the first timed IK frame.
    for stamp,raw in data['packets']:
        if stamp>=data['first']:break
        engine.feed(AttitudePacket(**raw),stamp)
    # Ordinary zero matrices are recorded in the sample signature as bytes.
    for entry in (data.get('pose_seed') or {}).values():
        if entry is None: continue
        for key,device,mode,mapping,neutral,_ in entry[1]:
            if isinstance(neutral,dict) and 'bytes_hex' in neutral:
                rotation = np.frombuffer(bytes.fromhex(neutral['bytes_hex']),dtype=np.float64).reshape(3,3).copy()
                engine.mode_neutral.setdefault(device,{})[mode] = rotation
                engine.neutral[device] = rotation.copy()
    return engine


def run_stage(data, stage):
    global _APP
    from .backend import PoseService
    arms = solver(data['settings'], True)  # JIT/setup excluded from timing.
    if data['initial'] is not None:
        for side,arm in arms.items():
            arm.previous = arm.limits.to_sim(data['initial'][:6] if side=='right' else data['initial'][6:])
    source = service = window = log = None
    rows = []; outputs = []; failure = []; clock_samples = []
    stop = threading.Event(); done = threading.Event(); latest = {}
    temp = tempfile.TemporaryDirectory(prefix='wt901-ablation-')
    if stage != 'ik':
        from PySide6.QtWidgets import QApplication
        _APP = QApplication.instance() or QApplication([])
        app = _APP
        source = RecordedSource(data['packets'],data['first'])
        if stage in ('gui','logging'):
            from .window import Window
            config = deepcopy(data['settings']['config'])
            config['mode'] = data['settings']['mode']
            config['checkpoints'] = {k:Path(v) for k,v in config['checkpoints'].items()}
            window = Window(source,config,background=False)
            window.setWindowTitle('OFFLINE benchmark — no robot / UDP')
            window.setEnabled(False)
            window.engine = pose_engine(data)
            window.pose_service = PoseService(source,window.engine)
            window.source = window.pose_service.proxy
            window.timer.setInterval(100);window.control.timer.setInterval(100)
            service = window.pose_service
            window.show();window.control.setEnabled(False);window.control.show()
            app.processEvents()
        else:
            service = PoseService(source,pose_engine(data))
        if stage == 'logging':
            from .diagnostics import DiagnosticLog
            log = DiagnosticLog(temp.name,data['settings'],seconds=180)
            service.diagnostic = log
    origin = time.monotonic()+.1
    if source is not None:
        source.started = origin
        service.start()

    def work():
        try:
            for frame in data['begins']:
                due = origin+(frame['monotonic_ns']/1e9-data['first'])
                if stop.wait(max(0,due-time.monotonic())):return
                started = time.monotonic_ns();cpu = time.thread_time_ns()
                result = {s:a.solve(human(frame['payload'][s])) for s,a in arms.items()}
                wall = (time.monotonic_ns()-started)/1e6
                metric = dict(sequence=frame['sequence'],wall_ms=wall,
                              thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,
                              start_lag_ms=max(0,(started/1e9-due)*1000),
                              arms={s:dict(a.last_solve_timing) for s,a in arms.items()})
                rows.append(metric)
                target = result.get('right',{}).get('target',[90.]*6)+result.get('left',{}).get('target',[90.]*6)
                outputs.append(target);latest['target'] = target
                if log is not None:log.record('benchmark_ik',**metric,target=target)
        except Exception as exc:failure.append(repr(exc))
        finally:done.set()

    worker = threading.Thread(target=work,name='offline-fixed-ik')
    worker.start();next_frequency = 0.;begin = time.monotonic()
    try:
        while not done.is_set():
            if time.monotonic()-begin>180:raise TimeoutError('单组测试超过180秒')
            if window is not None:
                if 'target' in latest:
                    from optusbproj_ubuntu.real_robot_protocol import encode
                    window.control.last_frame = encode(latest['target']+[90.]*3,[0]*10)
                app.processEvents()
            if time.monotonic()>=next_frequency:
                clock_samples.append(frequencies());next_frequency=time.monotonic()+.5
            done.wait(.002)
    finally:
        stop.set();worker.join(timeout=5)
        if worker.is_alive():raise RuntimeError('IK线程未退出；本次测试不完整')
        if service is not None:service.close()
        if log is not None:log.close()
        if window is not None:
            window.timer.stop();window.control.timer.stop()
            window.close();app.processEvents();window.deleteLater();app.processEvents()
        temp.cleanup()
    if failure:raise RuntimeError(failure[0])
    if service is not None and service.error:raise RuntimeError(service.error)
    if log is not None and (log.error or log.dropped):raise RuntimeError(log.error or '诊断丢弃')
    return dict(stage=stage,frames=len(rows),wall_ms=stats([r['wall_ms'] for r in rows]),
                thread_cpu_ms=stats([r['thread_cpu_ms'] for r in rows]),
                wall_minus_thread_cpu_ms=stats([r['wall_ms']-r['thread_cpu_ms'] for r in rows]),
                start_lag_ms=stats([r['start_lag_ms'] for r in rows]),
                arms={s:{key:stats([r['arms'][s][key] for r in rows])
                         for key in ('wall_ms','thread_cpu_ms')} for s in arms},
                frequency_samples=clock_samples, replayed_packets=0 if source is None else source.index,
                clock_note='CPU time excludes time not scheduled; difference alone does not prove GIL',
                raw_timings=rows),np.asarray(outputs)


def benchmark(data, stages=STAGES, repeats=1):
    runs=[];baseline=None
    for repeat in range(repeats):
        for stage in (stages if repeat%2==0 else tuple(reversed(stages))):
            print(f'[{repeat+1}/{repeats}] {stage}: running fixed-input replay',file=sys.stderr,flush=True)
            result,angles=run_stage(data,stage)
            if baseline is None:baseline=angles
            result['max_joint_difference_deg']=float(np.max(np.abs(angles-baseline)))
            result['repeat']=repeat+1;runs.append(result)
            print(f'  wall median {result["wall_ms"]["median"]:.1f} ms; '
                  f'thread CPU {result["thread_cpu_ms"]["median"]:.1f} ms',file=sys.stderr,flush=True)
    return dict(schema_version=1,read_only=True,uart_opened=False,udp_opened=False,
                complete_recording=data['complete'],seed=data['seed_note'],runs=runs,
                limitations=['Decoded packet replay, not network/UART/servo timing',
                             'Offscreen Qt; not an on-screen compositor/GPU benchmark',
                             'Fixed input schedule; lagging inputs retained for comparison, NOT production queuing',
                             'Fresh gravity filters; recorded neutral matrices, not exact filter history',
                             'No serial planner or transport watchdog workload in these four groups'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--control',type=Path)
    parser.add_argument('--max-frames',type=int,default=200)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES)
    args=parser.parse_args()
    if not 1<=args.max_frames<=500 or not 1<=args.repeats<=3:parser.error('frames 1..500; repeats 1..3')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from unittest.mock import patch
    # Defense in depth: even an accidental GUI action cannot open an IO device.
    with patch('serial.Serial',side_effect=RuntimeError('offline benchmark forbids UART')),patch(
            'socket.socket',side_effect=RuntimeError('offline benchmark forbids sockets')):
        result=benchmark(load_recording(args.recording,args.control,args.max_frames),tuple(args.stages),args.repeats)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
