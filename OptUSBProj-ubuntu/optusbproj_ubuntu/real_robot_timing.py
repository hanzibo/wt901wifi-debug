"""GUI-thread input observations only; never change freshness or motion gates."""
import math
import time


def acquisition_poll(owner, started, timed_packets=None):
    emit = getattr(owner.source, 'diagnostic_event', None)
    if emit is None: return
    if timed_packets is None:
        previous = getattr(owner,'_acquisition_poll_at',None)
        owner._acquisition_poll_at = started
        emit('main_gui_poll_begin', started_at=started,
             interval_ms=None if previous is None else (started-previous)*1000)
    else:
        emit('main_gui_batch', packets=len(timed_packets),
             poll_read_ms=(time.monotonic()-started)*1000,
             oldest_read_mono_ns=min((stamp for stamp,_ in timed_packets),default=None))


def acquisition_feed(owner, packet, received_ns, before):
    emit = getattr(owner.source, 'diagnostic_event', None)
    if emit is None: return
    now = time.monotonic()
    emit('pose_engine_feed', device=packet.device_id, read_mono_ns=received_ns,
         feed_started_at=before, feed_finished_at=now,
         read_to_feed_ms=(before-received_ns/1e9)*1000,
         feed_ms=(now-before)*1000,
         engine_received_at=owner.engine.received_at.get(packet.device_id))


def observe(dialog, now=None):
    now = time.monotonic() if now is None else now
    previous = getattr(dialog, '_timing_poll_at', None)
    dialog._timing_poll_at = now
    engine = getattr(dialog.viewer, 'engine', None)
    assignments = getattr(engine, 'assignments', {})
    received = getattr(engine, 'received_at', {})
    old = getattr(dialog, '_timing_received', {})
    devices = {}
    try:
        sides = dialog._selected_sides()
    except Exception:
        sides = ()
    current = {}
    for side in sides:
        for segment in ('upper_arm','forearm','hand'):
            part = side+'_'+segment
            device = assignments.get(part)
            at = received.get(device)
            at = float(at) if at is not None and math.isfinite(at) else None
            current[part] = (device, at)
            devices[part] = dict(device=device, received_at=at,
                age_ms=None if at is None else (now-at)*1000,
                changed=old.get(part) != (device, at))
    dialog._timing_received = current
    return dict(sample_observed_at=now,
                gui_poll_gap_ms=None if previous is None else (now-previous)*1000,
                devices=devices)
