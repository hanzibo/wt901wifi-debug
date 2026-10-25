"""Opt-in wall/thread-CPU measurements; never changes an IK result or gate."""
from functools import wraps
import time


def instrument(solver):
    if getattr(solver, '_solve_instrumented', False):
        return solver
    arms = getattr(solver, 'solvers', None)
    if arms is not None:
        for arm in arms.values():
            instrument(arm)
    original = solver.solve

    @wraps(original)
    def measured(*args, **kwargs):
        wall, cpu = time.monotonic_ns(), time.thread_time_ns()
        ok = False
        try:
            result = original(*args, **kwargs)
            ok = True
            return result
        finally:
            metric = dict(wall_ms=(time.monotonic_ns()-wall)/1e6,
                          thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6, success=ok)
            if arms is not None:
                metric['arms'] = {s:dict(a.last_solve_timing) for s,a in arms.items()
                                  if hasattr(a, 'last_solve_timing')}
            solver.last_solve_timing = metric

    solver.solve = measured
    solver._solve_instrumented = True
    return solver


def observe(event, service_getter):
    """Measure a GUI callback only while a diagnostic recording is active."""
    def decorate(method):
        @wraps(method)
        def measured(self, *args, **kwargs):
            service = service_getter(self)
            if service is None or service.diagnostic is None:
                return method(self,*args,**kwargs)
            wall,cpu=time.monotonic_ns(),time.thread_time_ns()
            try:
                return method(self,*args,**kwargs)
            finally:
                service.record_diagnostic(event,wall_ms=(time.monotonic_ns()-wall)/1e6,
                                          thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6)
        return measured
    return decorate
