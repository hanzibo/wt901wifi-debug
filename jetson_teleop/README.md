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

也可运行 `bash jetson_teleop/launch.sh`。本目录附带 `WT901-Jetson.desktop`，
其中路径针对当前板子用户 `mic-711on`；更换用户/仓库位置时须修改，不用于 PC 启动。

## 独立 IMU 接入与管理

桌面入口 `WT901-Jetson-IMU.desktop` 或 `bash jetson_teleop/launch.sh --imu-only`
只打开管理窗口，显示板子所有网卡 IPv4、实际监听端口、在线 / 离线设备 ID、分配部位、
设备发送端点、两秒平均收包 Hz、距末包 ms、电量估计和 RSSI。不启动机器人输出。
两种桌面入口使用仅同用户可访问的本地 IPC，只负责打开窗口，不传递启动或运动指令；
管理与控制始终共用一个进程里的单个 UDP 1399 socket。重复点击不会新增接收者或抢包。
若已运行，第二次指定的配置路径不会重新加载；更换运行配置须安全退出后重启。
关闭控制主窗口但保留管理窗口时，会先按原逻辑制动、关闭串口，然后继续收包。

PC 辅助配置步骤：

1. 原电脑热点下找到在线 IMU，确认没有真机输出后切回 AP/UDP；然后 PC 连接该设备 AP。
2. 重启 PC 手腕程序以加载新版，进入“接入新设备”，选择
   **外部 Wi-Fi / 路由器（其他设备 / Jetson 接收）**。
3. 填路由器 SSID（如 `HUAWEI-509`）及密码，UDP 目标填板子有线 IPv4
   （当前 `192.168.3.8`），端口 `1399`；不要使用当前 `bjut_wifi` 的自动地址或路由器 `.1`。
4. 确认写入后，在板子的 IMU 管理窗口验证设备 ID、Hz 和距末包时间。
   PC 显示“配置已发送，远端未验证”，不会因为自己未收到数据而误报失败或接入成功。
5. 多台设备逐一重复；板子 / PC DHCP 地址变化后须重新核对接收目标。

管理窗口的“将选中的在线 IMU 切回 AP / UDP”仅操作选中设备，默认确认选项为“否”。
任何真机串口连接（即使暂停）、IK / 启动 / 回位期间都不允许执行；确认后再次核对状态和
设备新鲜度，再复用原 `GuiUdpSource.switch_to_ap_udp`。UDP 没有设备确认回执，
收包消失也不能证明成功，必须到 PC 的 Wi-Fi 列表验证对应 ID 热点；失败时按设备说明恢复 AP。

Qt 只安装 PySide6-Essentials（Core/Gui/Widgets），无需 Addons。
实际 X11 桌面还需 Ubuntu 的 `libxcb-cursor0`（离屏测试不会检查此依赖）。
有管理员权限时安装该系统包；无管理员权限时，可把官方 arm64 deb 解包到
`jetson_teleop/.runtime/`，`launch.sh` 会仅为本程序加载其中的运行库。
Ruckig 0.19.4 在 ARM 上需要源码编译（CMake ≥3.15、C++ 编译器和 Ninja）；
该板已有的 GCC 9 / CMake 3.16 已满足构建要求，不可擅自换规划算法。
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
2. 板子本地桌面启动 GUI，核对设备部位、轴映射、协议方向与掌向偏置，分组采集中立参考。
   先点击上臂 / 前臂采集，在倒计时结束前自然下垂；再让手背处于所需中立姿态，单独点击
   手背采集。两组倒计时独立设置，默认均为 3 秒，可设 0–60 秒，可取消。
   左臂模式采集左侧 2 个臂段与 1 个手背，双臂模式采集 4 个臂段与 2 个手背。
   每组只改本组参考，采集时要求本组距末包 ≤250 ms；两组完成才允许连接串口。
   重新采集任一组会暂时撤销连接许可；更换模式或取消配置核对会清除两组完成标记。
3. 真机重新上电并人工确认全部关节 90°、手指量 0；选择串口，连接本身不发送角度帧。
4. 核对 Checkpoint、仅勾选需要的输出轴，确认独立断电手段、安全空间后，沿用原启动流程。
   先单臂小幅测试，再双臂；延时锁定与自动跟随仍需要二次确认。
5. 停止手势、手动暂停、顺序回位沿用原方案；断流超过保护期限后锁停，不自动恢复。

本软件无位置反馈，安全确认和“到位”都不是编码器实测。退出会请求原控制线程制动、
关闭串口后再退出，但软件暂停/制动不等于独立急停。此次自动化部署不执行真实运动测试。

## 私有仓库同步

代码在 PC 开发、提交并推送。当前板子没有 GitHub 凭据，因此首次部署使用
`git bundle` 经 SSH 同步同一提交，没有复制 GitHub 凭据或临时公开仓库。
后续可继续增量 bundle，或另行配置板子的只读部署密钥后使用普通 `git pull --ff-only`。
个人 checkpoint 与运行配置始终通过 SCP 单独复制，覆盖已有配置前应先备份。
