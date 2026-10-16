"""Persisted startup parameters, never persisted motion authorization."""
DEFAULT_START = dict(tolerance=2., countdown=10., settle=2.)
START_RANGES = dict(tolerance=(.1, 30.), countdown=(1., 120.), settle=(0., 30.))
START_LABELS = dict(tolerance='过渡姿态容差 °', countdown='启动倒计时 秒',
                    settle='指令完成后等待 秒（不是到位反馈）')
