"""Board runtime. No widget reads, no second UDP socket, no serial IO here.

PoseService owns the mutable source/engine. Readers receive detached copies.
LiveIK owns a single solve stream and publishes through the writer's session gate.
"""
from collections import Counter, deque
from concurrent.futures import Future, TimeoutError
from copy import deepcopy
import queue
import threading
import time
from types import SimpleNamespace

from PySide6.QtCore import QThread, Signal
from optusbproj_ubuntu.scene_robot_state import pose_sample, StaleImuInput
from optusbproj_ubuntu.real_robot_start_diagnostics import reference_deltas


class SessionEnded(RuntimeError):
    pass


class SourceView:
    """Cached inventory; mutations are serialized by the acquisition owner."""
    def __init__(self, owner):
        self.owner = owner
        address = owner.source.sock.getsockname()
        self.sock = SimpleNamespace(getsockname=lambda: address)

    def seen_endpoints(self, max_age=3.):
        data = self.owner.inventory()
        now = time.monotonic()
        return {k:v for k,v in data['endpoints'].items() if now-v[2] <= max_age}

    def seen_devices(self, max_age=3.):
        return {k:(v[0], v[2]) for k,v in self.seen_endpoints(max_age).items()}

    def latest_packet(self, key, max_age=3.):
        if key not in self.seen_endpoints(max_age):
            return None
        return self.owner.inventory()['packets'].get(key)

    def receive_stats(self):
        return self.owner.inventory()['rates']

    def device_ip(self, key):
        endpoints = self.seen_endpoints()
        if key not in endpoints:
            raise RuntimeError('设备离线')
        return endpoints[key][0]

    def switch_to_ap_udp(self, key):
        return self.owner.call(lambda:self.owner.source.switch_to_ap_udp(key))

    def configure_network(self, key, expected_ip, ssid, password, server_ip, port):
        return self.owner.call(lambda:self.owner.source.configure_network(
            key, expected_ip, ssid, password, server_ip, port))

    def add_diagnostic_sink(self, sink):
        self.owner.call(lambda:self.owner.sinks.append(sink) if sink not in self.owner.sinks else None)

    def remove_diagnostic_sink(self, sink):
        if self.owner.stopped.is_set() or self.owner.error:
            return  # Disconnect must still finish after acquisition has failed.
        self.owner.call(lambda:self.owner.sinks.remove(sink) if sink in self.owner.sinks else None)


class PoseService:
    def __init__(self, source, engine):
        self.source, self.engine = source, engine
        self.lock = threading.Lock()
        self.commands = queue.Queue(maxsize=64)
        self.stopped = threading.Event()
        self.error = None
        self.samples = {}
        self.view = deepcopy(engine)
        self.catalog = dict(endpoints={}, packets={}, rates={})
        self.version = 0
        self.generation = 0
        self.listeners = []
        self.sinks = []
        self.diagnostic = None  # Optional read-only recorder, independent of control logs.
        self.trace = deque(maxlen=2000)
        self.trace_overwritten = 0
        self.trace_counts = Counter()
        self.thread = threading.Thread(target=self._run, name='wt901-pose-owner', daemon=True)
        self.proxy = SourceView(self)

    def start(self):
        self.thread.start()

    def call(self, command):
        if self.error or self.stopped.is_set():
            raise RuntimeError(self.error or '采集服务已停止')
        if threading.current_thread() is self.thread:
            return command()
        future = Future()
        try:
            self.commands.put_nowait((future, command))
        except queue.Full:
            raise RuntimeError('采集服务命令队列已满')
        try:
            return future.result(timeout=2.)
        except TimeoutError:
            future.cancel()  # A still-queued command must not execute later.
            raise RuntimeError('采集服务命令超时，请停止并检查后台')

    def inventory(self):
        with self.lock:
            return deepcopy(self.catalog)

    def engine_view(self):
        with self.lock:
            return self.view  # Already detached; the producer never modifies it.

    def sample(self, sides):
        with self.lock:
            error, version, generation = self.error, self.version, self.generation
            entries = {s:self.samples.get(s) for s in sides}
        if error:
            raise RuntimeError(error)
        result, signatures, ages = {}, [], {}
        now = time.monotonic()
        for side, entry in entries.items():
            if entry is None:
                raise ValueError(side+' 尚无有效姿态 / 中立参考')
            sample, signature = entry
            for part, age in sample['device_ages_ms'].items():
                current_age = age/1000 + now-sample['at']
                if not -.1 <= current_age <= .25:
                    raise StaleImuInput(part, part, current_age)
                ages[part] = current_age*1000
            result[side] = sample['sides'][side]
            signatures.extend(signature)
        # Copies prevent a consumer from mutating the acquisition owner's cache.
        return (dict(at=now, sides=deepcopy(result), device_ages_ms=ages),
                tuple(signatures)+(('backend_generation',generation),), version)

    def zero(self, keys):
        def command():
            online = self.source.seen_devices(max_age=2.)
            completed = [k for k in keys if self.engine.assignments[k] in online
                         and self.engine.assignments[k] in self.engine.current]
            for key in completed:
                self.engine.zero(key)
            with self.lock:
                self.generation += 1
            self._publish(view=True)
            return completed
        return self.call(command)

    def subscribe(self, listener):
        with self.lock:
            self.listeners.append(listener)

    def unsubscribe(self, listener):
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)

    def _diagnose(self, event, **values):
        row = dict(values)
        row.setdefault('mono_s', time.monotonic())
        row['event'] = event
        if len(self.trace) == self.trace.maxlen:
            self.trace_overwritten += 1
        self.trace.append(row)
        self.trace_counts[event] += 1
        if event in ('imu_parsed', 'pose_poll', 'pose_feed'):
            self.record_diagnostic(event, **values)

    def record_diagnostic(self, event, **values):
        log = self.diagnostic
        if log is not None:
            log.record(event, **values)

    def _observe(self, row):
        self._diagnose(row['event'], **{k:v for k,v in row.items() if k != 'event'})

    def dump_trace(self, recorder):
        def command():
            recorder.record('acquisition_trace', overwritten=self.trace_overwritten,
                            retained=len(self.trace), rows=list(self.trace))
        if threading.current_thread() is self.thread:
            command()
            return
        if self.stopped.is_set():
            recorder.record('acquisition_trace_unavailable', reason='acquisition_stopped')
            return
        # Nonblocking: don't stall an IK fault while waiting for acquisition.
        try:
            self.commands.put_nowait((Future(), command))
        except queue.Full:
            recorder.record('acquisition_trace_unavailable', reason='command_queue_full')

    def _publish(self, view=False):
        wall, cpu = time.monotonic_ns(), time.thread_time_ns()
        entries = {}
        for side in ('right','left'):
            try:
                entries[side] = pose_sample(self.engine, (side,), (side,), True)
            except StaleImuInput:
                # Keep the last valid timestamp so readers can apply bounded gap.
                entries[side] = self.samples.get(side)
            except ValueError:
                entries[side] = None
        catalog = dict(endpoints=self.source.seen_endpoints(max_age=float('inf')),
                       packets=dict(self.source._latest_packets), rates=self.source.receive_stats())
        detached = deepcopy(self.engine) if view else None
        self.record_diagnostic('pose_snapshot', entries=entries,
                               generation=self.generation, version=self.version+1)
        with self.lock:
            self.samples = entries
            self.catalog = catalog
            self.version += 1
            if view:
                self.view = detached
            listeners = tuple(self.listeners)
        for listener in listeners:
            listener()
        self.record_diagnostic('pose_publish',wall_ms=(time.monotonic_ns()-wall)/1e6,
                               thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,view=view)

    def _run(self):
        next_sample = next_view = next_summary = 0.
        try:
            self.source.add_diagnostic_sink(self._observe)
            while not self.stopped.is_set():
                for _ in range(16):
                    try:
                        future, command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if future.set_running_or_notify_cancel():
                        try:
                            future.set_result(command())
                        except Exception as exc:
                            future.set_exception(exc)
                before = time.monotonic()
                poll_cpu = time.thread_time_ns()
                packets = self.source.poll_timed()
                for stamp, packet in packets:
                    feed_at = time.monotonic()
                    feed_cpu = time.thread_time_ns()
                    self.engine.feed(packet, stamp/1e9)
                    self._diagnose('pose_feed', device=packet.device_id,
                                   read_mono_ns=stamp, feed_ms=(time.monotonic()-feed_at)*1000,
                                   thread_cpu_ms=(time.thread_time_ns()-feed_cpu)/1e6)
                    if self.diagnostic is not None:
                        self.record_diagnostic('imu_packet', read_mono_ns=stamp,
                                               packet=vars(packet))
                now = time.monotonic()
                self._diagnose('pose_poll', packets=len(packets), duration_ms=(now-before)*1000,
                               thread_cpu_ms=(time.thread_time_ns()-poll_cpu)/1e6)
                if now >= next_sample:
                    self._publish(view=now >= next_view)
                    next_sample = now+.02
                    if now >= next_view:
                        next_view = now+.1
                if now >= next_summary:
                    row = dict(event='pose_service_summary', mono_s=now,
                               counts=dict(self.trace_counts), trace_overwritten=self.trace_overwritten)
                    for sink in tuple(self.sinks):
                        sink(dict(row))
                    self.trace_counts.clear()
                    next_summary = now+.1
                self.stopped.wait(.002)
        except Exception as exc:
            with self.lock:
                self.error = '采集 / 姿态服务失败：'+str(exc)
                listeners = tuple(self.listeners)
            for listener in listeners:
                listener()
        finally:
            self.source.remove_diagnostic_sink(self._observe)
            self.stopped.set()
            while not self.commands.empty():
                future, _ = self.commands.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError(self.error or '采集服务已停止'))

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError('采集线程尚未退出，不能关闭 UDP socket')


class LiveIK(QThread):
    """Latest snapshot, one solver, direct guarded delivery; Qt signals display only."""
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, solver, service, writer, sides, capture, tolerance, parent=None):
        super().__init__(parent)
        self.solver, self.service, self.writer = solver, service, writer
        self.sides, self.capture = tuple(sides), deepcopy(capture)
        self.tolerance = tolerance
        self.cancelled = threading.Event()
        self.gate = threading.RLock()
        self.task_id = None
        self.sequence = 0
        self.failure = None
        from .process_ik import ProcessIK
        if isinstance(solver, ProcessIK):solver.cancelled = self.cancelled

    def submit(self, *args):
        pass  # Compatibility only: the GUI is never the live-input producer.

    def shutdown(self):
        with self.gate:
            self.cancelled.set()

    def _fault(self, reason):
        with self.gate:
            if self.cancelled.is_set():
                return
            self.failure = reason
            self.cancelled.set()
            self.writer.pause(reason)
            self.writer.recorder.record('backend_control_fault', task_id=self.task_id, reason=reason)
            self.service.dump_trace(self.writer.recorder)
            self.failed.emit(reason)

    def _read(self):
        with self.writer.lock:
            if (self.task_id != self.writer.task_id or self.writer.pause_reason
                    or self.writer.quit_requested or not self.writer.active):
                raise SessionEnded('控制会话已撤销')
        sample, signature, version = self.service.sample(self.sides)
        if signature != self.capture[2]:
            raise ValueError('后台设备分配 / 中立参考 / 映射变化；需重新接管')
        payload = sample['sides'] if 'rotations' not in self.capture[0] else sample['sides']['left']
        _, state, _ = self.writer.return_snapshot()
        if state is None or state.get('phase') != 'following':
            deltas = reference_deltas((payload,), self.capture)
            if max(deltas.values()) > self.tolerance:
                raise ValueError('后台过渡期间人体超差；需重新接管')
        deadline = sample['at']+.25-max(sample['device_ages_ms'].values())/1000
        return payload, deadline, version

    def monitor(self):
        # Runs on the acquisition thread, independently of both IK and Qt.
        if self.cancelled.is_set():
            return
        try:
            self._read()
        except SessionEnded:
            self.shutdown()
            self.service.dump_trace(self.writer.recorder)
        except StaleImuInput:
            pass  # The existing serial watchdog owns 250 + 250 ms grace.
        except Exception as exc:
            self._fault(str(exc))

    def run(self):
        self.task_id = self.writer.task_id
        self.service.subscribe(self.monitor)
        previous = None
        try:
            while not self.cancelled.is_set():
                read_wall, read_cpu = time.monotonic_ns(), time.thread_time_ns()
                try:
                    payload, deadline, version = self._read()
                except StaleImuInput:
                    self.cancelled.wait(.003)
                    continue
                if version == previous:
                    self.cancelled.wait(.002)
                    continue
                previous = version
                before = time.monotonic()
                read_metrics = dict(input_read_wall_ms=(time.monotonic_ns()-read_wall)/1e6,
                                    input_read_thread_cpu_ms=(time.thread_time_ns()-read_cpu)/1e6)
                if before > deadline:
                    continue
                self.sequence += 1
                self.writer.recorder.record('arm_ik_submit', sequence=self.sequence,
                                            deadline=deadline, backend=True, task_id=self.task_id)
                self.service.record_diagnostic('ik_begin', sequence=self.sequence,
                    task_id=self.task_id, payload=payload, deadline=deadline, version=version)
                solve_cpu = time.thread_time_ns()
                result = self.solver.solve(payload)
                after = time.monotonic()
                measured = getattr(self.solver,'last_solve_timing',None)
                solve_metrics = dict(ik_thread_cpu_ms=(time.thread_time_ns()-solve_cpu)/1e6,
                                     solver_timing=deepcopy(measured) if isinstance(measured,dict) else None)
                if isinstance(measured,dict) and measured.get('backend') == 'spawn_process':
                    solve_metrics['ik_proxy_thread_cpu_ms'] = solve_metrics['ik_thread_cpu_ms']
                    solve_metrics['ik_thread_cpu_ms'] = measured['thread_cpu_ms']
                    solve_metrics['ik_backend'] = 'spawn_process'
                self.service.record_diagnostic('ik_solved', sequence=self.sequence,
                    task_id=self.task_id, ik_ms=(after-before)*1000, deadline=deadline,
                    solve_finished_at=after, **solve_metrics, **read_metrics)
                timing = dict(sequence=self.sequence, submitted_at=before, queue_ms=0.,
                              solve_finished_at=after, backend=True, **solve_metrics, **read_metrics)
                result.update(deadline=deadline, ik_ms=(after-before)*1000, timing=timing)
                with self.gate:
                    if self.cancelled.is_set():
                        return
                    try:
                        validation_wall, validation_cpu = time.monotonic_ns(), time.thread_time_ns()
                        self._read()  # Current input and transition guard, not solve-time freshness.
                    except StaleImuInput:
                        continue
                    timing.update(post_solve_validation_ms=(time.monotonic_ns()-validation_wall)/1e6,
                                  post_solve_validation_thread_cpu_ms=(time.thread_time_ns()-validation_cpu)/1e6)
                    if time.monotonic() > deadline:
                        self.service.record_diagnostic('ik_expired', sequence=self.sequence,
                                                       task_id=self.task_id, deadline=deadline)
                        self.writer.recorder.record('arm_ik_discard', stage='backend_after_solve', **result)
                        continue
                    delivered = time.monotonic()
                    timing['backend_delivery_ms'] = (delivered-after)*1000
                    if not self.writer.update_arm_target(result['target'], deadline, timing,
                                                         task_id=self.task_id):
                        self.cancelled.set()
                        return
                    self.service.record_diagnostic('ik_delivered', sequence=self.sequence,
                        task_id=self.task_id, timing=timing, target=result['target'], deadline=deadline)
                    self.writer.recorder.record('arm_ik', **result)
                self.result.emit(result)  # Optional display; never authorizes output.
        except SessionEnded:
            self.service.dump_trace(self.writer.recorder)
        except Exception as exc:
            self._fault(str(exc))
        finally:
            self.cancelled.set()
            self.service.unsubscribe(self.monitor)
            from .process_ik import ProcessIK
            if isinstance(self.solver, ProcessIK):self.solver.close()
