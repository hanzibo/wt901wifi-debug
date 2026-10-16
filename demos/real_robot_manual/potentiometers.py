"""POT5 parsing and calibrated mappings; no hardware IO."""
from dataclasses import dataclass


def parse_line(line):
    fields = line.strip().split(b',')
    if len(fields) != 8 or fields[0] != b'POT5':
        return None
    try:
        seq, uptime, *values = map(int, fields[1:])
    except ValueError:
        return None
    if not 0 <= seq <= 0xffffffff or not 0 <= uptime <= 0xffffffff or any(not 0 <= v <= 1023 for v in values):
        return None
    return seq, uptime, tuple(values)


def parse_a4(line):
    """Extract only A4 from installed POT5 firmware; ignore A0-A3."""
    fields = line.strip().split(b',')
    if len(fields) != 8 or fields[0] != b'POT5':
        return None
    try:
        seq, uptime, a4 = (int(fields[i]) for i in (1, 2, 7))
    except ValueError:
        return None
    if not 0 <= seq <= 0xffffffff or not 0 <= uptime <= 0xffffffff or not 0 <= a4 <= 1023:
        return None
    return seq, uptime, (0, 0, 0, 0, a4)


@dataclass(frozen=True)
class FingerMap:
    channel: int
    opened: int
    closed: int
    output_open: int = 0
    output_closed: int = 200
    enabled: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('手指许可必须为布尔值')
        for value, lo, hi in ((self.channel,0,4),(self.opened,0,1023),(self.closed,0,1023),
                              (self.output_open,0,4095),(self.output_closed,0,4095)):
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError('通道或标定数值无效')
        if abs(self.closed-self.opened) < 20:
            raise ValueError('张开/闭合 ADC 端点至少相差 20；须实际标定')

    def target(self, values):
        if len(values) != 5 or any(type(v) is not int or not 0 <= v <= 1023 for v in values):
            raise ValueError('五路 ADC 数据无效')
        ratio = min(1.,max(0.,(values[self.channel]-self.opened)/(self.closed-self.opened)))
        return round(self.output_open+ratio*(self.output_closed-self.output_open))


def targets(maps, values):
    if len(maps) != 5 or len({m.channel for m in maps}) != 5:
        raise ValueError('五指须使用五个不同输入通道')
    return [m.target(values) for m in maps]
