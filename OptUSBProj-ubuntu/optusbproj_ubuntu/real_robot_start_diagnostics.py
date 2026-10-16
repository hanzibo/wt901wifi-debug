"""JSON-safe start diagnostics. Observations only; never authorize movement."""
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation


def reference_deltas(sample, capture):
    if sample is None or capture is None:
        return None
    current, reference = sample[0], capture[0]
    if 'rotations' not in current:
        return {side+'.'+key:value for side in current
                for key,value in reference_deltas((current[side],), (reference[side],)).items()}
    pairs = {'upper_arm': (reference['rotations']['upper_arm'], current['rotations']['upper_arm']),
             'forearm': (reference['rotations']['forearm'], current['rotations']['forearm']),
             'hand': (reference['hand'], current['hand'])}
    values = {key:float(np.rad2deg(Rotation.from_matrix(a.T@b).magnitude()))
              for key,(a,b) in pairs.items()}
    if not all(math.isfinite(x) for x in values.values()):
        raise ValueError('参考朝向差不是有限值')
    return values


def snapshot(dialog):
    now = time.monotonic()
    remaining = None if dialog.arm_sample is None else (dialog.arm_sample[1]-now)*1000
    # Preserve a diagnostic failure without breaking the asynchronous JSON writer.
    error, deltas = None, None
    try:
        deltas = reference_deltas(dialog.arm_sample, dialog.arm_capture)
    except Exception as exc:
        error = str(exc)
    return dict(busy=dialog.busy, connected=dialog.connected, serial_worker_present=dialog.worker is not None,
                ik_worker_present=dialog.ik_worker is not None, reference_present=dialog.arm_capture is not None,
                origin_present=dialog.origin is not None, safety_confirmed=dialog.safety.isChecked(),
                sample_present=dialog.arm_sample is not None,
                signature_matches=None if dialog.arm_sample is None or dialog.arm_capture is None else
                                  dialog.arm_sample[2] == dialog.arm_capture[2],
                input_remaining_ms=remaining if remaining is None or math.isfinite(remaining) else None,
                input_age_ms=None if remaining is None or not math.isfinite(remaining) else 250-remaining,
                device_ages_at_snapshot_ms=dialog.arm_ages,
                orientation_delta_deg=deltas, orientation_check_error=error,
                orientation_threshold_deg=dialog._transition_tolerance(), freshness_threshold_ms=250.,
                start_settings=dialog.auto_start.values() if dialog.auto_start is not None else None,
                auto_start_phase=dialog.auto_start.phase if dialog.auto_start is not None else 'idle',
                enabled=[row[0].isChecked() for row in dialog.axis_rows],
                right_enabled=[row[0].isChecked() for row in getattr(dialog,'right_axis_rows',())],
                bilateral=dialog._bilateral_mode() if hasattr(dialog,'_bilateral_mode') else False,
                input_error=dialog.arm_input_error)


def delta_text(data):
    values = data.get('orientation_delta_deg')
    threshold = data.get('orientation_threshold_deg', 2.)
    if values is not None and 'upper_arm' not in values:
        return ' / '.join(f'{key} {value:.2f}°' for key,value in values.items())+f'（阈值 {threshold:g}°）'
    return '' if values is None else ('上臂 %.2f° / 前臂 %.2f° / 手掌 %.2f°' %
                                     (values['upper_arm'], values['forearm'], values['hand']))+f'（阈值 {threshold:g}°）'
