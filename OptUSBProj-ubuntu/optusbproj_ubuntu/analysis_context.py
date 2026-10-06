"""Per-run offline data cache. No persisted fit results or live state writes."""
import csv
import json
from pathlib import Path


class AnalysisContext:
    def __init__(self, session):
        self.session = Path(session).resolve()
        self._files = {}
        self._runtime_by_device = None
        self._imu_index = None
        self.live_series = {}
        self.camera_validation = {}

    def check_session(self, session):
        if Path(session).resolve() != self.session:
            raise ValueError('分析数据上下文与录制目录不一致')
        for name, (identity, _) in self._files.items():
            if self._identity(name) != identity:
                raise ValueError('分析过程中原始录制发生变化，请重新计算：' + name)

    def _identity(self, name):
        stat = (self.session / name).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _read(self, name, loader):
        if name not in self._files:
            before = self._identity(name)
            with (self.session / name).open(encoding='utf-8', newline='') as stream:
                data = loader(stream)
            if self._identity(name) != before:
                raise ValueError('分析读取期间录制发生变化：' + name)
            self._files[name] = before, data
        return self._files[name][1]

    @property
    def manifest(self):return self._read('manifest.json', json.load)

    @property
    def guide(self):
        if not (self.session / 'camera_guide.json').exists():return {}
        return self._read('camera_guide.json', json.load)

    @property
    def runtime(self):
        return self._read('runtime_pose.jsonl', lambda f: [json.loads(line) for line in f])

    @property
    def vision(self):
        return self._read('vision.jsonl', lambda f: [json.loads(line) for line in f])

    @property
    def imu(self):return self._read('imu.csv', lambda f: list(csv.DictReader(f)))

    def runtime_for(self, device):
        if self._runtime_by_device is None:
            self._runtime_by_device = {}
            for row in self.runtime:
                self._runtime_by_device.setdefault(row['device_id'], []).append(row)
        return self._runtime_by_device.get(device, [])

    @property
    def imu_index(self):
        if self._imu_index is None:
            self._imu_index = {(int(r['host_recv_mono_ns']), r['device_id']): r for r in self.imu}
        return self._imu_index


class MemoizedOrientationSeries:
    """Reuse the exact original scalar SLERP result, not a different interpolator."""
    def __init__(self, series):
        self.series = series
        self._cache = {}

    def __getattr__(self, name):return getattr(self.series, name)

    def at(self, camera_ns, *, lag_ms=0.0):
        key = (int(camera_ns), lag_ms)
        if key not in self._cache:
            self._cache[key] = self.series.at(camera_ns, lag_ms=lag_ms)
        value = self._cache[key]
        return None if value is None else value.copy()
