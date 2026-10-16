"""Versioned parameter presets only. No Qt, sensors, serial or runtime state."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import UUID, uuid4

from .real_robot_arm import ArmLimits
from .real_robot_stop_gesture import GestureConfig, gesture_indices, axis_name
from .real_robot_start_settings import DEFAULT_START, START_RANGES

MODES = {'left_shared_ik': ('left',), 'bilateral_shared_ik': ('right', 'left')}
FIELDS = ('lower', 'upper', 'speed', 'acceleration')
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'config' / 'real_robot_checkpoints'
HOLD_RANGE = (.1, 120.)
DEFAULT_STOP = dict(enabled=False, seconds=7., kind='lateral_raise')
DEFAULT_RETURN = dict(order=[1, 3, 5, 6, 4, 2], speed=10., acceleration=20., dwell=.5)


def validate_options(stop, returning):
    keys(stop, ('enabled', 'seconds', 'kind'), '停止信号')
    gesture_indices(stop['kind'])
    if type(stop['enabled']) is not bool:
        raise ValueError('停止信号开关必须是布尔值')
    if not HOLD_RANGE[0] <= number(stop['seconds']) <= HOLD_RANGE[1]:
        raise ValueError('停止信号持续时间须为 0.1–120 秒')
    keys(returning, DEFAULT_RETURN, '回位设置')
    order = returning['order']
    if (not isinstance(order, list) or len(order) != 6
            or any(type(i) is not int for i in order) or set(order) != set(range(1, 7))):
        raise ValueError('每侧回位顺序必须包含 1–6 各一次')
    for field in ('speed', 'acceleration', 'dwell'):
        lo, hi = (0., 10.) if field == 'dwell' else (.1, 250.)
        if not lo <= number(returning[field]) <= hi:
            raise ValueError(f'回位 {field} 须在 {lo}–{hi} 范围内')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def axes(mode):
    if mode not in MODES:
        raise ValueError('不支持的 Checkpoint 模式')
    return tuple(f'{side[0].upper()}{i}' for side in MODES[mode] for i in range(1, 7))


def gesture_axes(mode, kind='lateral_raise'):
    axes(mode)
    return tuple(map(axis_name, gesture_indices(kind, bilateral=mode == 'bilateral_shared_ik')))


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label} 字段缺失或含未知字段')


def number(value):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('参数必须是有限数值，不接受布尔值或字符串')
    if abs(value-round(value, 2)) > 1e-8:
        raise ValueError('参数最多保留两位小数，避免加载时静默取整')
    return float(value)


def validate(data, mode=None):
    base = ('schema_version', 'mode', 'id', 'name', 'notes', 'created_at',
            'updated_at', 'joints', 'gesture')
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] not in (1, 2, 3, 4):
        raise ValueError('不支持的 Checkpoint 格式版本')
    expected = base if data['schema_version'] == 1 else (*base, 'stop_signal', 'return_settings')
    if data['schema_version'] >= 3: expected = (*expected, 'start_settings')
    keys(data, expected, 'Checkpoint')
    data = deepcopy(data)
    if data['schema_version'] == 1:
        data.update(schema_version=2, stop_signal=dict(enabled=False, seconds=7.),
                    return_settings=deepcopy(DEFAULT_RETURN))
    if data['schema_version'] == 2:
        data.update(schema_version=3, start_settings=deepcopy(DEFAULT_START))
    if data['schema_version'] == 3:
        keys(data['stop_signal'], ('enabled', 'seconds'), '旧版停止信号')
        data['stop_signal']['kind'] = 'lateral_raise'
        data['schema_version'] = 4
    keys(data['start_settings'], DEFAULT_START, '接管设置')
    for key, (lo, hi) in START_RANGES.items():
        if not lo <= number(data['start_settings'][key]) <= hi:
            raise ValueError(f'接管 {key} 须在 {lo}–{hi} 范围内')
    validate_options(data['stop_signal'], data['return_settings'])
    current = data['mode']
    if not isinstance(current, str) or current not in MODES or (mode is not None and current != mode):
        raise ValueError('Checkpoint 与当前控制模式不匹配')
    try:
        if str(UUID(data['id'])) != data['id']:
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise ValueError('Checkpoint ID 无效') from None
    for field, length in (('name', 80), ('notes', 2000)):
        if not isinstance(data[field], str) or len(data[field]) > length:
            raise ValueError(f'{field} 文本无效或过长')
    if not data['name'].strip():
        raise ValueError('名称不能为空')
    for field in ('created_at', 'updated_at'):
        try:
            if datetime.fromisoformat(data[field]).tzinfo is None:
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(f'{field} 时间戳无效') from None
    keys(data['joints'], axes(current), '关节')
    for axis, values in data['joints'].items():
        keys(values, FIELDS, axis)
        for value in values.values(): number(value)
    for side in MODES[current]:
        ArmLimits(side=side, **{
            field: tuple(data['joints'][f'{side[0].upper()}{i}'][field] for i in range(1, 7))
            for field in FIELDS})
    monitored = gesture_axes(current, data['stop_signal']['kind'])
    keys(data['gesture'], monitored, '结束手势')
    for axis, values in data['gesture'].items():
        keys(values, ('angle', 'tolerance'), axis+' 手势')
        angle, tolerance = number(values['angle']), number(values['tolerance'])
        if not data['joints'][axis]['lower'] <= angle <= data['joints'][axis]['upper']:
            raise ValueError(f'{axis} 停止姿态参考角不在预设关节范围内')
        if not .1 <= tolerance <= 30.:
            raise ValueError(f'{axis} 容差须为 0.1–30°')
    GestureConfig(tuple(data['gesture'][a]['angle'] for a in monitored),
                  tuple(data['gesture'][a]['tolerance'] for a in monitored),
                  indices=gesture_indices(data['stop_signal']['kind'], bilateral=current == 'bilateral_shared_ik'),
                  kind=data['stop_signal']['kind'], seconds=data['stop_signal']['seconds'])
    return deepcopy(data)


def draft(mode, joints, gesture, name='新预设', notes='', *, stop_signal=None, return_settings=None,
          start_settings=None):
    """Unsaved editor document; may contain values the user must correct."""
    axes(mode)
    now = stamp()
    return deepcopy(dict(schema_version=4, mode=mode, id=str(uuid4()), name=name,
                         notes=notes, created_at=now, updated_at=now, joints=joints, gesture=gesture,
                         stop_signal=DEFAULT_STOP if stop_signal is None else stop_signal,
                         return_settings=DEFAULT_RETURN if return_settings is None else return_settings,
                         start_settings=DEFAULT_START if start_settings is None else start_settings))


def create(mode, joints, gesture, name='新预设', notes='', **options):
    return validate(draft(mode, joints, gesture, name, notes, **options))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON 字段重复：{key}')
        result[key] = value
    return result


class CheckpointStore:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)

    def path(self, mode, ident):
        axes(mode)
        try:
            if str(UUID(ident)) != ident: raise ValueError()
        except (TypeError, ValueError, AttributeError):
            raise ValueError('Checkpoint ID 无效') from None
        return self.root / mode / (ident+'.json')

    def load(self, mode, ident):
        path = self.path(mode, ident)
        if path.is_symlink() or path.stat().st_size > 1024*1024:
            raise ValueError('不接受符号链接或过大的预设文件')
        data = validate(json.loads(path.read_text(encoding='utf-8'),
                                   object_pairs_hook=unique_object), mode)
        if data['id'] != ident:
            raise ValueError('文件名与 Checkpoint ID 不一致')
        return data

    def list(self, mode):
        axes(mode)
        rows, errors = [], []
        for path in sorted((self.root/mode).glob('*.json')):
            try: rows.append(self.load(mode, path.stem))
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f'{path.name}: {exc}')
        return sorted(rows, key=lambda d: d['updated_at'], reverse=True), errors

    def save(self, data, *, overwrite=False):
        data = validate(data)
        path = self.path(data['mode'], data['id'])
        if path.exists() or path.is_symlink():
            if not overwrite: raise ValueError('预设已存在，请另存或明确保存修改')
            previous = self.load(data['mode'], data['id'])
            if previous['updated_at'] != data['updated_at']:
                raise ValueError('文件已被其他窗口修改，请重新选择后编辑')
            data['created_at'] = previous['created_at']
        elif overwrite:
            raise ValueError('原预设已不存在，请另存为新预设')
        data['updated_at'] = stamp()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix='.checkpoint-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()
        return data

    def delete(self, mode, ident):
        self.load(mode, ident)
        trash = self.root/'trash'/mode
        trash.mkdir(parents=True, exist_ok=True)
        destination = trash/(ident+'-'+str(uuid4())+'.json')
        self.path(mode, ident).replace(destination)
        return destination
