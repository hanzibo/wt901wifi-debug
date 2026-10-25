# Jetson 轻量真机遥操作

本目录是独立入口，不替换 PC 程序。姿态积分、重力纠倾斜、手动轴映射、同源 IK、
轨迹规划、断流保护、停止手势和回位仍复用原项目代码。因此必须保留完整 Git 仓库，
不能只复制此目录。没有 RGB-D、穿戴标定、CoppeliaSim、VTK、ROS 或手指控制。
人体预览是固定骨长骨架，机器人预览是发送指令的 FK，不是真机反馈或碰撞检测。
没有外部航向校正，长期航偏仍会累积。

## 安装与运行

已验证目标：Ubuntu 20.04 / aarch64 / L4T 35.4.1。
保留系统 Python 3.8，使用 uv 安装 Python 3.11；不要替换 `/usr/bin/python3`。
仓库根目录下的 `.venv` 为本板专用，不提交或从 PC 复制。

```bash
~/.local/bin/uv python install 3.11
~/.local/bin/uv venv --python 3.11 .venv
CMAKE_BUILD_PARALLEL_LEVEL=2 ~/.local/bin/uv pip install --python .venv/bin/python -r jetson_teleop/requirements.txt
.venv/bin/python jetson_teleop/run.py --self-test
.venv/bin/python -m unittest discover -s jetson_teleop/tests
.venv/bin/python jetson_teleop/run.py --probe 10
.venv/bin/python jetson_teleop/run.py
```

Qt 只安装 PySide6-Essentials（Core/Gui/Widgets），无需 Addons。
Ruckig 0.19.4 在 ARM 上需要源码编译及较新 CMake（≥3.26），不可擅自换规划算法。
部署时可临时设置 `http_proxy` / `https_proxy`，不改系统网络配置。
`--probe` 只接收并统计 UDP，不加载 Qt、不打开串口；与其他程序同时占用 1399 会失败。
`--self-test` 只离线检查 IK、轨迹和零手指协议帧。

## 每机配置（不进 Git）

`runtime.local.json` 保存六个设备 ID、六套轴映射、发送 Hz、左右各六个协议方向、
掌向局部偏置轴/角，以及左臂和双臂的默认 checkpoint 路径。
可在 PC 使用 `prepare_config.py --left <JSON> --bilateral <JSON>` 打印配置草稿，
再明确保存为 `runtime.local.json`。它默认 20 Hz、方向均 +1、掌向 Z +90°；
这与近期运行记录一致，但更换机器人/穿戴方式后必须重新核对，不能当机械标定。
`checkpoints/` 保存单独通过 SSH/SCP 传来的个人预设；两个目录都被 Git 忽略。
加载 checkpoint 仍不恢复启动许可、归零参考、停止姿态确认或自动启动。
手动映射必须启用；不接受依赖穿戴标定的预设。

控制面板的 Checkpoint 窗口也使用本目录的 `checkpoints/`；启动默认预设以
`runtime.local.json` 指定的文件为准。不修改 PC 的预设目录。

## 首次验证顺序

1. 不连接机器人串口，先 `--self-test` 和 `--probe 10`。IMU 目标地址须是板子的
   有线 IP（当前 `192.168.3.8:1399`），不是 PC 的 `.7`。探测程序不改设备配置。
2. 板子本地桌面启动 GUI，核对设备部位、轴映射、协议方向与掌向偏置，人体自然下垂归零。
   左臂模式只要求左臂三个设备在线；双臂模式要求六个设备。允许更换模式后重新归零。
3. 真机重新上电并人工确认全部关节 90°、手指量 0；选择串口，连接本身不发送角度帧。
4. 核对 Checkpoint、仅勾选需要的输出轴，确认独立断电手段、安全空间后，沿用原启动流程。
   先单臂小幅测试，再双臂；延时锁定与自动跟随仍需要二次确认。
5. 停止手势、手动暂停、顺序回位沿用原方案；断流超过保护期限后锁停，不自动恢复。

本软件无位置反馈，安全确认和“到位”都不是编码器实测。退出会请求原控制线程制动、
关闭串口后再退出，但软件暂停/制动不等于独立急停。此次自动化部署不执行真实运动测试。
