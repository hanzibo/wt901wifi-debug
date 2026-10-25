"""Strict per-machine configuration; no sockets, Qt or serial operations."""
import json
import math
from pathlib import Path

from optusbproj_ubuntu.axis_mapping import normalized_mapping

PARTS = tuple(f'{side}_{part}' for side in ('left', 'right')
              for part in ('upper_arm', 'forearm', 'hand'))
MODES = ('left_shared_ik', 'bilateral_shared_ik')


def load(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    required = {'version', 'mode', 'assignments', 'axis_mappings', 'frequency',
                'signs', 'palm_offsets', 'checkpoints'}
    if not isinstance(raw, dict) or set(raw) != required or type(raw['version']) is not int or raw['version'] != 1:
        raise ValueError('运行配置字段或版本无效')
    if raw['mode'] not in MODES:
        raise ValueError('仅支持左臂或双臂同源 IK')
    for key in ('assignments', 'axis_mappings'):
        if not isinstance(raw[key], dict) or set(raw[key]) != set(PARTS):
            raise ValueError(f'{key} 须包含双臂六个部位')
    ids = list(raw['assignments'].values())
    if any(not isinstance(v, str) or not v.startswith('WT') for v in ids) or len(set(ids)) != 6:
        raise ValueError('六个部位须配置不同的 WT 设备 ID')
    for mapping in raw['axis_mappings'].values():
        normalized_mapping(mapping)
    hz = raw['frequency']
    if type(hz) not in (int, float) or not math.isfinite(hz) or not 1 <= hz <= 100:
        raise ValueError('发送频率须在 1–100 Hz')
    if any(not isinstance(raw[k], dict) or set(raw[k]) != {'left', 'right'}
           for k in ('signs', 'palm_offsets')):
        raise ValueError('双臂方向或掌向配置不完整')
    for side in ('left', 'right'):
        signs = raw['signs'][side]
        if not isinstance(signs, list) or len(signs) != 6 or any(type(v) is not int or v not in (-1, 1) for v in signs):
            raise ValueError('每臂须配置六个 ±1 协议方向')
        offset = raw['palm_offsets'][side]
        if (not isinstance(offset, dict) or set(offset) != {'axis', 'degrees'} or offset['axis'] not in ('x', 'y', 'z')
                or type(offset['degrees']) not in (int, float)
                or not math.isfinite(offset['degrees']) or not -180 <= offset['degrees'] <= 180):
            raise ValueError('掌向偏置轴或角度无效')
    if (not isinstance(raw['checkpoints'], dict) or set(raw['checkpoints']) != set(MODES)
            or any(not isinstance(v, str) or not v for v in raw['checkpoints'].values())):
        raise ValueError('须指定左臂及双臂 checkpoint 路径')
    raw['checkpoints'] = {mode: (path.parent / value).resolve()
                          for mode, value in raw['checkpoints'].items()}
    return raw
