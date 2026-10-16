"""One worker owns serial IO; bounded mailbox; no automatic/retried writes."""
import threading
import time
from PySide6.QtCore import QThread, Signal
from .protocol import decode


class SerialWorker(QThread):
    ready = Signal()
    sent = Signal(bytes)
    failed = Signal(str)
    received = Signal(bytes)
    motion_done = Signal()
    recording_failed = Signal(str)

    def __init__(self, port, parent=None, factory=None):
        super().__init__(parent)
        self.port, self.factory = port, factory
        self.lock = threading.Lock()
        self.stopping = False
        self.enabled = False
        self.pending = None
        self.epoch = 0
        self.recorder = None
        self.live_epoch = None
        self.live_target = None

    def record(self, event, **data):
        if self.recorder:self.recorder.record(event, **data)

    def enable(self, enabled):
        with self.lock:
            self.epoch += 1
            self.enabled = bool(enabled) and not self.stopping
            self.pending = None
            self.live_epoch = self.live_target = None

    def submit(self, frame):
        decode(frame)
        with self.lock:
            if not self.enabled or self.stopping:
                raise ValueError('发送未解锁或连接已停止')
            self.pending = (bytes(frame), self.epoch)

    def stop(self):
        with self.lock:
            self.stopping = True
            self.enabled = False
            self.pending = None
            self.epoch += 1
            self.live_epoch = self.live_target = None

    def submit_motion(self, trajectory):
        with self.lock:
            if not self.enabled or self.stopping:
                raise ValueError('发送未解锁或连接已停止')
            self.pending = (trajectory, self.epoch)
            self.live_epoch = self.epoch if getattr(trajectory, 'live', False) else None

    def retarget_live(self, target):
        with self.lock:
            if not self.enabled or self.stopping or self.live_epoch != self.epoch:
                raise ValueError('实时随动未解锁或已停止')
            self.live_target = target # Bounded latest-only mailbox, no FIFO.

    def run(self):
        connection = None
        try:
            if self.factory is None:
                import serial
                if not hasattr(serial, 'Serial'):
                    raise RuntimeError('缺少 pyserial；当前 serial 包不是串口库')
                self.factory = serial.Serial
            # Set handshake lines before opening; no deliberate reset pulse.
            # Hardware/driver may still produce transients; opening is not electrically inert.
            connection = self.factory(port=None, baudrate=115200, bytesize=8,
                                      parity='N', stopbits=1, timeout=0,
                                      write_timeout=.2, xonxoff=False,
                                      rtscts=False, dsrdtr=False, exclusive=True)
            connection.dtr = False
            connection.rts = False
            connection.port = self.port
            with self.lock:
                if self.stopping:
                    return
            connection.open()
            self.record('serial_open', port=self.port, baudrate=115200,
                        angle_feedback_available=False)
            self.ready.emit()
            motion = None
            step = 0
            next_tick = 0.
            previous_frame = None
            previous_write_ns = None
            sequence = 0
            motion_id = 0
            reported_record_error = False
            while True:
                with self.lock:
                    if self.stopping:
                        if motion:self.record('motion_cancelled', motion_id=motion_id, last_step=step,
                                              reason='worker_stop')
                        break
                    job, self.pending = self.pending, None
                    if motion and (not self.enabled or motion[1] != self.epoch):
                        self.record('motion_cancelled', motion_id=motion_id, last_step=step)
                        motion = None
                    if job and not isinstance(job[0], bytes):
                        if self.enabled and job[1] == self.epoch:
                            motion = job
                            step = 0
                            next_tick = time.monotonic() + motion[0].period
                            motion_id += 1
                            previous_frame = motion[0].frame(0)
                            self.record('motion_start', motion_id=motion_id,
                                start=motion[0].start, target=motion[0].target,
                                duration_s=motion[0].duration, steps=motion[0].steps,
                                period_s=motion[0].period, frequency_hz=motion[0].frequency_hz,
                                limits=motion[0].limits)
                            if getattr(motion[0], 'live', False):
                                self.record('live_start', axis=motion[0].axis,
                                    lower=motion[0].lower, upper=motion[0].upper,
                                    warning='software test envelope, not measured mechanical limits')
                        job = None
                    if motion and getattr(motion[0], 'live', False) and self.live_target is not None:
                        motion[0].set_target(self.live_target)
                        self.record('live_target', axis=motion[0].axis, target_deg=self.live_target)
                        self.live_target = None
                    if motion and time.monotonic() >= next_tick:
                        step += 1
                        job = (motion[0].frame(step), motion[1])
                    if job and self.enabled and job[1] == self.epoch:
                        # Guard and write are atomic w.r.t. stop/revoke; bounded by write_timeout.
                        frame = job[0]
                        if motion is None:
                            self.enabled = False  # explicit single-frame mode
                        before_ns = time.monotonic_ns()
                        count = None
                        write_error = None
                        try:
                            count = connection.write(frame)
                        except Exception as exc:
                            write_error = str(exc)
                            raise
                        finally:
                            after_ns = time.monotonic_ns()
                            angles, fingers = decode(frame)
                            previous = decode(previous_frame) if previous_frame else None
                            gap_ms = ((before_ns-previous_write_ns)/1e6
                                      if previous_write_ns is not None else None)
                            deltas = [a-b for a,b in zip(angles,previous[0])] if previous else None
                            finger_deltas = [a-b for a,b in zip(fingers,previous[1])] if previous else None
                            try:out_waiting = connection.out_waiting
                            except Exception:out_waiting = None
                            sequence += 1
                            self.record('write', sequence=sequence, motion_id=motion_id if motion else None,
                                step=step if motion else None, total_steps=motion[0].steps if motion else None,
                                write_start_ns=before_ns, write_end_ns=after_ns,
                                interval_ms=gap_ms, write_duration_ms=(after_ns-before_ns)/1e6,
                                angles_deg=angles, fingers=fingers, angle_delta_deg=deltas,
                                finger_delta=finger_deltas,
                                command_velocity_deg_s=([v*1000/gap_ms for v in deltas]
                                    if deltas is not None and gap_ms and gap_ms > 0 else None),
                                bytes_written=count, write_error=write_error,
                                os_write_complete=count==len(frame) and write_error is None,
                                out_waiting_bytes=out_waiting, frame_hex=frame.hex(' '))
                        if count != len(frame):
                            raise RuntimeError(f'写入不完整 {count}/54；不重试，需重新检查')
                        self.sent.emit(frame)
                        previous_frame = frame
                        previous_write_ns = before_ns
                        if motion:
                            # No wall-clock catch-up after stalls: slow the trajectory instead.
                            next_tick = time.monotonic() + motion[0].period
                            if step == motion[0].steps or getattr(motion[0], 'done', False):
                                self.record('motion_complete', motion_id=motion_id, steps=step,
                                            reason=getattr(motion[0], 'reason', 'target_reached'))
                                motion = None
                                self.enabled = False
                                self.motion_done.emit()
                if connection.in_waiting:
                    self.received.emit(connection.read(min(connection.in_waiting, 4096)))
                if self.recorder and self.recorder.error and not reported_record_error:
                    reported_record_error = True
                    self.recording_failed.emit(self.recorder.error)
                # Wake closer to the next deadline; never catch up with a burst.
                wait_ms = max(1, min(10, int((next_tick-time.monotonic())*1000))) if motion else 10
                self.msleep(wait_ms)
        except Exception as exc:
            self.record('serial_error', error=str(exc), write_result_uncertain=True)
            self.enable(False)
            self.failed.emit(str(exc) + '；写入结果可能不确定，不自动重发')
        finally:
            self.record('serial_close', reason='stop_or_error', actual_position_known=False)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            if self.recorder:
                self.recorder.close()
                if self.recorder.error:self.recording_failed.emit(self.recorder.error)
