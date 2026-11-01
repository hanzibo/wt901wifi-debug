"""Spawn-only IK owner. One outstanding request, no target/history queue.

Only detached kinematic data crosses the pipe; no Qt/source/writer is inherited.
The parent retains session, reference and original freshness authority.
"""
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import sys
import threading
import time


def describe(solver):
    """Export the exact geometry/rest/seed, never pickle compiled closures."""
    from optusbproj_ubuntu.real_robot_shared import SharedArmIK
    from optusbproj_ubuntu.real_robot_bilateral import BilateralIK
    if not isinstance(solver, (SharedArmIK, BilateralIK)):
        raise TypeError('独立进程仅支持同源单臂 / 双臂 IK')
    arms = solver.solvers if isinstance(solver, BilateralIK) else {solver.side:solver}
    return dict(bilateral=isinstance(solver, BilateralIK), provenance=deepcopy(solver.provenance),
                arms={s:dict(limits=a.limits, palm=a.palm, offset=deepcopy(a.offset),
                    profile=(dict(q=a.model.reference.copy(),
                                  joints=[t[:3].reshape(-1).copy() for t in a.model.homes],
                                  palm=a.model.tool[:3].reshape(-1).copy()),
                             a.solver.axes.copy(), a.provenance),
                    human_rest=deepcopy(a.solver.human_rest), previous=a.previous.copy())
                      for s,a in arms.items()})


def build(spec):
    from optusbproj_ubuntu.real_robot_shared import SharedArmIK
    from optusbproj_ubuntu.real_robot_protocol import encode
    from .fast_geometry import prepare
    arms = {}
    for side, raw in spec['arms'].items():
        args = {k:v for k,v in raw.items() if k != 'previous'}
        arm = prepare(SharedArmIK(encode([90.]*15,[0]*10), side=side, **args))
        arm.previous = raw['previous'].copy()
        arms[side] = arm
    return arms


def solve_arms(arms, bilateral, payload):
    if not bilateral:
        return next(iter(arms.values())).solve(payload)
    if set(payload) != set(arms):
        raise ValueError('双臂输入侧别与输出许可不一致')
    target, errors = [], {}
    for side in ('right','left'):
        if side not in arms:
            target.extend([90.]*6)
            continue
        result = arms[side].solve(payload[side])
        target.extend(result['target'])
        errors[side] = {k:v for k,v in result.items() if k != 'target'}
    return dict(target=target, errors=errors,
                position_error_mm=max(e['position_error_mm'] for e in errors.values()),
                orientation_error_deg=None)


def _forbid_devices(event, args):
    if event == 'socket.__new__':
        raise RuntimeError('IK process forbids sockets')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if path.startswith(('/dev/tty','/dev/serial')):
            raise RuntimeError('IK process forbids UART')


def _owner(spec, requests, results):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'OptUSBProj-ubuntu'))
    sys.addaudithook(_forbid_devices)
    try:
        arms = build(spec)
        results.send(('ready', None))
        while True:
            request = requests.recv()
            if request is None:break
            sequence, payload = request
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            result = solve_arms(arms, spec['bilateral'], payload)
            timing = dict(wall_ms=(time.monotonic_ns()-wall)/1e6,
                          thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,
                          success=True, arms={s:dict(a.last_solve_timing) for s,a in arms.items()})
            results.send(('result', (sequence,result,timing)))
    except EOFError:
        pass
    except Exception as exc:
        try:results.send(('error', type(exc).__name__+': '+str(exc)))
        except (EOFError,OSError):pass
    finally:
        requests.close();results.close()


class ProcessIK:
    """A synchronous proxy called by LiveIK's latest-snapshot loop, not Qt.

    The input is detached before send. At most one request is in flight. When
    it finishes, LiveIK re-reads the latest snapshot; it never replays a backlog.
    No process result grants motion or refreshes an input deadline by itself.
    """
    def __init__(self, spec, *, startup_timeout=30.):
        self.provenance = deepcopy(spec['provenance'])
        self.cancelled = threading.Event()
        self.closed = False
        self.faulted = False
        self.sequence = 0
        self.last_solve_timing = None
        self.lock = threading.Lock()
        ctx = mp.get_context('spawn')
        request_rx, self.requests = ctx.Pipe(duplex=False)
        self.results, result_tx = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_owner,args=(spec,request_rx,result_tx),
                                   name='wt901-ik-owner',daemon=True)
        try:
            self.process.start()
            request_rx.close();result_tx.close()
            kind, value = self._receive(startup_timeout)
            if kind != 'ready':raise RuntimeError(str(value))
        except BaseException:
            request_rx.close();result_tx.close();self.close()
            raise

    @classmethod
    def from_solver(cls, solver):
        return cls(describe(solver))

    def _receive(self, timeout):
        until = time.monotonic()+timeout
        while True:
            if self.cancelled.is_set():raise RuntimeError('独立 IK 请求已取消')
            if self.results.poll(.01):
                try:return self.results.recv()
                except EOFError:raise RuntimeError('独立 IK 进程通信中断') from None
            if not self.process.is_alive():raise RuntimeError('独立 IK 进程已退出')
            if time.monotonic() >= until:raise TimeoutError('独立 IK 进程响应超时')

    def solve(self, payload):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('独立 IK 不允许并发排队')
        try:
            if self.closed or self.faulted or self.cancelled.is_set():
                raise RuntimeError('独立 IK 已关闭 / 取消 / 故障')
            self.sequence += 1
            started = time.monotonic_ns()
            self.requests.send((self.sequence,deepcopy(payload)))
            kind, value = self._receive(5.)
            if kind != 'result':raise RuntimeError(str(value))
            sequence,result,timing = value
            if sequence != self.sequence:raise RuntimeError('独立 IK 响应版本不匹配')
            self.last_solve_timing = dict(timing, backend='spawn_process',
                roundtrip_ms=(time.monotonic_ns()-started)/1e6)
            return result
        except BaseException:
            # Do not set the caller's cancellation event here: LiveIK must
            # distinguish a child fault from an already-requested manual stop.
            self.faulted = True
            raise
        finally:
            self.lock.release()

    def close(self):
        if self.closed:return
        self.closed = True
        self.cancelled.set()
        if self.process.pid is not None:
            if self.process.is_alive():self.process.terminate()
            self.process.join(timeout=1.)
            if self.process.is_alive():
                self.process.kill();self.process.join(timeout=1.)
        self.requests.close();self.results.close()
