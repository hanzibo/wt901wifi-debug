# 双屏 + 前置摄像头三路录制

这个小程序在 GNOME Wayland 下同时录制：

1. 外接显示器（单独一份 MKV）；
2. 内置显示器（单独一份 MKV）；
3. 前置摄像头（单独一份 MKV）。

停止录制后，程序会自动删除三份视频开头和结尾的各 3 秒，只保留中间部分。处理时先生成临时结果，三份全部成功后才替换原视频。

## 使用

先做无弹窗检查：

```bash
./triple_recorder/start_recording.sh --check
```

开始录制：

```bash
./triple_recorder/start_recording.sh
```

系统会弹出屏幕共享窗口。请**同时选择两台显示器**并确认。终端出现“三路录制中”后，按 `Ctrl+C` 可安全停止。程序会在 `~/Videos` 下创建以开始时间命名的目录，三份视频统一保存在其中，例如：

```text
~/Videos/20260928_153045/
├── screen_1_HDMI-5.mkv
├── screen_2_eDP-1.mkv
└── camera.mkv
```

也可以指定时间目录的上级目录，或设置自动录制时长：

```bash
./triple_recorder/start_recording.sh --output-dir /path/to/videos
./triple_recorder/start_recording.sh --duration 60
```

指定 `--output-dir` 后，程序同样会在该目录下创建一个新的时间子目录。
由于要删除首尾各 3 秒，`--duration` 必须大于 6 秒；手动录制也应超过 6 秒。若裁剪失败，程序会保留未经裁剪的三份原视频。

两块屏幕的原始分辨率很高，程序默认等比缩放到最大 1920 像素宽，以保证三路同时录制时的稳定性。若要保留原始分辨率，可使用：

```bash
./triple_recorder/start_recording.sh --max-screen-width 0
```

## 当前机器的实现说明

- 使用桌面门户 + PipeWire 捕获 Wayland 屏幕，不使用容易黑屏的 `x11grab`。
- 使用 GStreamer 分别编码三路 H.264，并封装为较耐意外中断的 MKV。
- 使用 `/usr/bin/python3`；它已经具备所需的 PyGObject。现有 conda 环境均缺少这一绑定，因此无需修改任何 conda 环境。
- 只录制画面，不录制麦克风或系统声音。
