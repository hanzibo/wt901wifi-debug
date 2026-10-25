#!/usr/bin/env python3
"""Read-only process sampler: no robot ports, sockets, signals or control imports.

Run using an interpreter with psutil. CPU 100% means one logical CPU, not the
whole machine. MHz is a system-wide observation, NOT the target process clock.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import queue
import statistics
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    def percentile(p):
        at = (len(values)-1)*p
        lo = int(at); hi = min(lo+1, len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(count=len(values), mean=statistics.mean(values),
                p50=percentile(.5), p95=percentile(.95), p99=percentile(.99), max=values[-1])


def summarize(rows):
    fields = ('tree_cpu_percent', 'root_cpu_percent', 'hottest_thread_percent',
              'tree_rss_mib', 'tree_pss_mib', 'system_cpu_percent',
              'system_available_mib', 'swap_used_mib', 'frequency_mean_mhz', 'sample_ms')
    return {field: distribution([r.get(field) for r in rows]) for field in fields}


class Sampler:
    def __init__(self, pid):
        self.root = psutil.Process(pid)
        self.identity = self.root.create_time()
        self.processes = {}
        self.threads = {}
        self.previous = time.monotonic()
        psutil.cpu_percent(None)

    def sample(self):
        started = time.monotonic()
        if not self.root.is_running() or self.root.status() == psutil.STATUS_ZOMBIE:
            raise psutil.NoSuchProcess(self.root.pid)
        if self.root.create_time() != self.identity:
            raise psutil.NoSuchProcess(self.root.pid)
        elapsed = started-self.previous
        # The GUI launches this observer as a child. Do not count the observer
        # itself as control workload (self-observation tests still include root).
        live = [self.root, *[p for p in self.root.children(recursive=True)
                           if p.pid != os.getpid()]]
        active = {}; threads = {}; cpu = rss = pss = 0.; root_cpu = 0.
        top = []; errors = []; pss_complete = True
        for proc in live:
            try:
                key = (proc.pid, proc.create_time())
                known = self.processes.get(key)
                if known is None:
                    known = proc; known.cpu_percent(None)
                    usage = None  # A new process has no previous CPU sample.
                else:
                    usage = known.cpu_percent(None)
                active[key] = known
                if usage is not None:
                    cpu += usage
                    if proc.pid == self.root.pid: root_cpu = usage
                rss += proc.memory_info().rss/2**20
                try:
                    memory = proc.memory_full_info()
                    value = getattr(memory, 'pss', None)
                    if value is None: pss_complete = False
                    else: pss += value/2**20
                except (psutil.Error, OSError):
                    pss_complete = False
                for thread in proc.threads():
                    tid = (key, thread.id)
                    total = thread.user_time+thread.system_time
                    previous = self.threads.get(tid)
                    threads[tid] = total
                    if previous is not None and elapsed > 0:
                        top.append(dict(pid=proc.pid, tid=thread.id,
                                        cpu_percent=max(0., (total-previous)/elapsed*100)))
            except (psutil.Error, OSError) as exc:
                errors.append(type(exc).__name__); pss_complete = False
        self.processes, self.threads, self.previous = active, threads, started
        freq = psutil.cpu_freq(percpu=True) or []
        currents = [f.current for f in freq if f.current > 0]
        memory = psutil.virtual_memory(); swap = psutil.swap_memory()
        top.sort(key=lambda r:r['cpu_percent'], reverse=True)
        return dict(event='sample', wall_time=datetime.now().astimezone().isoformat(),
                    monotonic_s=started, interval_s=elapsed,
                    tree_cpu_percent=cpu, root_cpu_percent=root_cpu,
                    hottest_thread_percent=top[0]['cpu_percent'] if top else None,
                    top_threads=top[:5], process_count=len(active),
                    tree_rss_mib=rss, tree_pss_mib=pss if pss_complete else None,
                    system_cpu_percent=psutil.cpu_percent(None),
                    system_available_mib=memory.available/2**20,
                    swap_used_mib=swap.used/2**20, swap_in_bytes=swap.sin, swap_out_bytes=swap.sout,
                    frequency_mean_mhz=statistics.mean(currents) if currents else None,
                    frequency_min_mhz=min(currents) if currents else None,
                    frequency_max_mhz=max(currents) if currents else None,
                    sample_ms=(time.monotonic()-started)*1000, errors=errors)


def capture(pid, duration, interval, output, stop_event=None, events=None):
    stop_event = stop_event or threading.Event()
    sampler = Sampler(pid)
    metadata = dict(event='metadata', wall_time=datetime.now().astimezone().isoformat(),
                    platform=platform.platform(), python=platform.python_version(),
                    pid=pid, process_created=sampler.identity, command=sampler.root.cmdline(),
                    logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                    memory_total_mib=psutil.virtual_memory().total/2**20,
                    duration_s=duration, interval_s=interval,
                    cpu_convention='100% = one logical CPU; tree RSS may double count shared pages',
                    frequency_convention='system-wide observed MHz; not process-specific or effective IPC',
                    first_sample='priming only, excluded from summary')
    metadata['observer_excluded_from_tree'] = True
    try:
        metadata['cpu_model'] = next(line.split(':',1)[1].strip() for line in
                                     Path('/proc/cpuinfo').read_text().splitlines()
                                     if line.startswith('model name'))
    except (OSError, StopIteration):
        metadata['cpu_model'] = platform.processor() or None
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []; reason = 'duration_complete'
    with (output/'samples.jsonl').open('x', encoding='utf-8') as stream:
        def drain_events():
            if events is not None:
                while True:
                    try: item = events.get_nowait()
                    except queue.Empty: break
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
        stream.write(json.dumps(metadata, ensure_ascii=False)+'\n')
        sampler.sample()  # Prime process CPU and per-thread cumulative clocks.
        end = time.monotonic()+duration
        try:
            while time.monotonic() < end:
                if stop_event.wait(min(interval, max(0., end-time.monotonic()))):
                    reason = 'controller_stop_requested'
                    break
                row = sampler.sample(); rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                drain_events(); stream.flush()
        except KeyboardInterrupt:
            reason = 'observer_interrupted_target_untouched'
        except psutil.NoSuchProcess:
            reason = 'target_exited'
        drain_events()
    result = dict(metadata=metadata, reason=reason, summary=summarize(rows))
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description='只读记录指定 GUI 进程及子进程资源，不连接/停止机器人')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--list', action='store_true', help='列出可能的 WT901 GUI 进程，不自动选择')
    parser.add_argument('--duration', type=float, default=600.)
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gui-control', action='store_true',
                        help='GUI 专用：从标准输入接收上下文；stop 或 EOF 仅停止记录')
    args = parser.parse_args()
    if args.list:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            command = proc.info['cmdline'] or []
            if 'python' in (proc.info['name'] or '').lower() and any(
                    Path(arg).name in ('run.py', 'fusion_run.py', 'wrist_demo.py') for arg in command):
                print(proc.pid, ' '.join(command))
        return
    if args.pid is None or args.pid <= 0: parser.error('请先 --list，再明确指定 --pid')
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 3600:
        parser.error('--duration 须为 1–3600 秒')
    if not math.isfinite(args.interval) or not .5 <= args.interval <= 10:
        parser.error('--interval 须为 0.5–10 秒')
    output = args.output or (ROOT/'OptUSBProj-ubuntu/recordings/runtime_probe'/
                            (datetime.now().strftime('%Y%m%d-%H%M%S')+f'-{args.pid}'))
    print(f'只读观察 PID {args.pid}，输出 {output}；Ctrl+C 仅结束记录器，不停止机器人。', flush=True)
    stop_event = threading.Event(); events = queue.Queue(maxsize=4096)
    if args.gui_control:
        def read_commands():
            try:
                for line in sys.stdin:
                    if line.strip() == 'stop': break
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict) and item.get('event') in ('gui_context', 'gui_state'):
                            events.put_nowait(item)
                    except (ValueError, queue.Full):
                        continue
            finally:
                stop_event.set()
        threading.Thread(target=read_commands, daemon=True).start()
    try:
        result = capture(args.pid, args.duration, args.interval, output, stop_event, events)
    except (psutil.Error, OSError) as exc:
        parser.exit(1, f'记录失败（目标进程未改动）：{exc}\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    print('完成：', output)


if __name__ == '__main__':
    main()
