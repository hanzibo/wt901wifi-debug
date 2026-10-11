# WT901WIFI：人体姿态、RGB-D 纠偏与 X1 随动预览

本仓库在 Ubuntu 上实现 WT901WIFI 多设备采集、人体姿态映射、穿戴标定、RGB-D 辅助航向纠偏，以及 X1 机器人双臂的 **RViz 运动学预览**。当前主路线位于 [OptUSBProj Ubuntu 版](OptUSBProj-ubuntu/README.md)；根目录的 [双手手腕控制台](wrist_demo.py) 保留为独立调试入口。说明更新于 2026-10-06。

主要数据链路为：WT901 原始数据 → 姿态估计 → 穿戴轴映射与中立参考 → STL 人体模型 → 肘腕目标及 IK → X1 `/joint_states`。Gemini 336L 的彩色图像经二维关键点检测后，结合真实配准深度得到骨段方向，为四个臂段提供低频航向纠偏；高频传播仍由 IMU 完成。

两个 GUI 共用 [`wt901/`](wt901/) 通信模块，实时模式均监听 UDP 1399，**不要同时启动**。相机也不能被 OrbbecViewer 或另一个采集程序同时占用。旧版 Windows 项目 `OptUSBProj/` 是本机未跟踪的参考资料，不参与 Ubuntu 程序运行。

| 想做什么 | 入口 | 环境 |
| --- | --- | --- |
| 双手手腕演示、设备联网和参数设置 | `wrist_demo.py` | `wt901` |
| 左右手四 IMU 夹爪离散开合（仅 RViz，可在人体GUI中配置） | GUI“左右手 IMU 夹爪配置…”／`OptUSBProj-ubuntu/gripper_gesture_demo.py` | `OptUSBProj-yolo`；见[操作说明](OptUSBProj-ubuntu/docs/GRIPPER_GESTURE_DEMO.md) |
| 手机局域网滑杆控制 X1 左／右夹爪（仅 RViz） | `OptUSBProj-ubuntu/lan_gripper_demo.py` | Python 标准库＋现有 ROS 桥；见[操作说明](OptUSBProj-ubuntu/docs/LAN_GRIPPER_DEMO.md) |
| 全身 IMU 人体模型 | `OptUSBProj-ubuntu/run.py` | `OptUSBProj` |
| 单目相机姿态原型 | `OptUSBProj-ubuntu/camera_run.py` | 见 [相机环境说明](OptUSBProj-ubuntu/docs/CAMERA_SETUP.md) |
| Gemini RGB-D＋IMU、双臂一体标定与纠偏 | `OptUSBProj-ubuntu/fusion_run.py --camera-source orbbec` | 推荐已配置的 `OptUSBProj-yolo`；SDK 独立环境见下文 |
| X1 双臂／手掌 RViz 随动 | 人体 GUI 内的“X1 双臂实时预览…” | ROS 接收端使用 `x1_ros2_py` |
| 新机器人双臂／手掌 CoppeliaSim 随动 | 人体 GUI 内的“新机器人双臂实时预览…” | 本机 ZeroMQ API；仅停止状态预览；见[说明](docs/COPPELIASIM_LIVE_RETARGET.md) |

## 当前能力与边界

- 15 个 STL 人体部件可独立分配设备、归零和设置轴映射；主要穿戴实测集中于双臂及手部，并非所有部位都已完成实机验收。
- 人体 GUI 保留厂商 R/P/Y，可逐部位选择电脑端“陀螺＋加计”；后者按设备片内时间积分，学习陀螺零偏，并用重力纠正倾斜。两者都无法仅靠六轴 IMU 消除长期航向漂移。
- Gemini RGB-D GUI 支持 RTMW / YOLO26 二维姿态后端和本地 YOLO 权重选择。三维点使用相机真实深度，而不是模型预测深度；表面深度不等于解剖关节中心。
- 双臂一体标定用一套动作分别计算穿戴映射和相机标定，四部位独立验收、选择应用。相机采用斜前举等可见动作，降低正面视角前平举遮挡的影响；原 IMU 穿戴拟合仍使用标准前举／侧举。
- 左右上臂、前臂可启用“陀螺＋加计＋RGB-D”航向纠偏。用户已在多次长于 5 分钟的运行中观察到纠偏有效，但这不是有 GT 的精度测量，也不保证遮挡、长轴扭转或任意动作下均可观测。
- 默认关闭的“双人体实时对比”并列显示 IMU-only 与 RGB-D 融合姿态，仅影响显示，不改变实际算法或 X1 输出。相机结果应用后仍须明确启用对应部位融合，否则纠偏待机。
- X1 支持双臂位置随动、可选双手朝向、统一／分段长度比例，以及数值最小二乘／PoE 雅可比两种 IK 后端。完整末端位姿优先模式保留用于对比，可能产生较大关节转动。
- 手部自然掌面参考、视觉估计 C0、历史标定快速恢复、视觉辅助穿戴标定及不限累计纠偏角等仍属实验功能。手腕尚无与四臂段同等的长期 RGB-D 航向纠偏。

**X1 链路只发布 RViz 预览关节状态，不是实体机器人控制方案。** 现提供默认关闭的实验性自碰撞检查与拦截；它依赖 CAD 几何、接触排除及预览状态，不能作为真机安全保证。当前未完成真机安全限位、执行器闭环或急停。

## 主路线快速启动

以下命令复用本机已配置环境，不会自动安装依赖或下载模型。首次部署见 [RGB-D 采集](OptUSBProj-ubuntu/docs/RGBD_STAGE_A.md)、[融合环境](OptUSBProj-ubuntu/docs/FUSION_SETUP.md) 和 [YOLO 后端](OptUSBProj-ubuntu/docs/YOLO_RGBD_BACKEND.md)。相机 SDK 在独立 `orbbec336l` 环境中，由采集子进程调用；二维视觉模型依赖在 GUI 环境中。

从仓库根目录运行 RGB-D 主界面：

```bash
cd /home/hzb/Work/WT901
conda run --no-capture-output -n OptUSBProj-yolo python -u OptUSBProj-ubuntu/fusion_run.py \
  --camera-source orbbec --rgbd-fps 10 --rgbd-warmup 2 \
  --rgbd-pose --pose-backend yolo26 --pose-imgsz 640
```

默认 YOLO 权重为本地 `no-improtent/yolo-models/yolo26m-pose.pt`；可用 `--yolo-model /path/to/yolo26l-pose.pt` 或 GUI 的“YOLO权重…”选择其他姿态权重。RTMW 后端可用 `--pose-backend rtmw`。更换后端、权重或输入尺寸后应重新完成相机标定，不能直接复用旧视觉会话结果。

也可使用本机桌面“WT901-RGBD-YOLO对比”启动项，或：

```bash
bash OptUSBProj-ubuntu/launch_rgbd_yolo.sh
```

纯 IMU 人体模型无需相机：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/run.py
```

桌面文件和部分模型默认路径针对本机；迁移电脑时应修改路径并重新检查依赖。不要通过同时开启多个 GUI 来比较算法，应使用同一 RGB-D 窗口的双人体对比功能。

### 推荐操作顺序

1. 开启电脑 `WT901-LAN` 2.4 GHz 热点，设备以 STA/UDP 接入；固定绑带和相机，按实际佩戴位置分配不同设备。
2. 四个臂段选择“陀螺＋加计”，支撑双臂并等待静止零偏学习完成；相机画面只保留目标使用者，确认肩肘腕和深度有效。
3. 打开“双臂一体标定 · 穿戴＋相机（一次动作）”，按倒计时提示动作。首次和回到中立位均保持同样的放松姿态；不要首次展背、结束时才放松。
4. 检查每个部位的穿戴／相机独立结果和回放，明确应用所选合格结果。随后可一键启用本轮已应用且仍有效部位的 RGB-D 融合，未通过部位保留原模式并显示原因。
5. 先验收人体模型，再在 X1 弹窗启动 RViz 模型及随动。GUI 可启动 ROS 接收端，无需额外终端；同一时刻只允许一个 `/joint_states` 发布者。
6. 要重新录制标定时，先停止 X1 随动，点击一体窗口“一键切回双臂陀螺＋加计”，再开始录制。该按钮不归零、不重置积分器，也不删除已应用标定。

相机／绑带移动、设备参数或参考系变化会使历史结果不再可靠。不要为了强行通过而放宽质量门槛；融合失效提示、独立复测和配对中立参考都是流程的一部分。

## 文档导航

| 主题 | 文档 |
| --- | --- |
| GUI 入口与功能 | [Ubuntu 版说明](OptUSBProj-ubuntu/README.md) |
| 一次动作完成穿戴＋相机标定 | [一体标定流程及诊断](OptUSBProj-ubuntu/docs/COMBINED_ARM_CAMERA_CALIBRATION.md) |
| 实时航向纠偏及失效规则 | [RGB-D 实时融合](OptUSBProj-ubuntu/docs/RGBD_LIVE_FUSION.md) |
| 后端选择与同录像比较 | [RTMW / YOLO26](OptUSBProj-ubuntu/docs/YOLO_RGBD_BACKEND.md) |
| 多人画面的目标选择、丢失保护与 OSNet 外观验证恢复 | [人物选择与锁定](OptUSBProj-ubuntu/docs/RGBD_PERSON_TARGET.md) |
| 标准穿戴与视觉辅助穿戴 | [穿戴标定](OptUSBProj-ubuntu/docs/WEARING_CALIBRATION.md)、[视觉辅助](OptUSBProj-ubuntu/docs/VISUAL_ASSISTED_WEARING.md) |
| 设备姿态、归零和运动换基 | [SO(3) 与李代数](docs/SO3_IMU_MODEL.md) |
| 连续映射 B、弱先验与拟合验收 | [连续穿戴标定数学](docs/CONTINUOUS_ARM_CALIBRATION.md) |
| 陀螺积分、重力反馈 | [重力纠倾斜推导](docs/GYRO_GRAVITY_TILT_MATH.md) |
| 相机 Q、骨轴及航向纠偏 | [相机标定与融合数学](docs/RGBD_CAMERA_CALIBRATION_FUSION_MATH.md) |
| 人体 → 机器人目标、优化与 FK | [重定向数学](docs/HUMAN_TO_X1_RETARGETING_MATH.md)、[URDF / PoE 等价关系](docs/URDF_POE_EQUIVALENCE.md) |
| X1 启动、诊断及 IK 对比 | [双臂实时预览](urdf-model/docs/IMU双臂实时预览.md)、[IK 后端](OptUSBProj-ubuntu/docs/X1_IK_BACKENDS.md) |
| X1 可开关自碰撞检查与拦截 | [实验性保护说明](docs/X1_COLLISION_PREVIEW.md) |
| 自然掌面与历史复用 | [自然掌面数学](docs/NATURAL_PALM_C0_MATH.md)、[快速恢复](OptUSBProj-ubuntu/docs/CALIBRATION_REUSE.md) |
| 项目总览 | [总览 HTML](docs/PROJECT_PIPELINE_OVERVIEW.html)、[图像讲解源文件及复现](docs/project-explainer/README.md) |

子文档中的阶段 A/B 或早期重构说明描述对应阶段边界；当前整套系统的能力以实际入口代码和本文概览为准。

## 双手手腕姿态控制台

基于 WT901WIFI 惯性测量单元（IMU）的双手手腕转角演示程序。电脑通过 Wi-Fi 接收设备数据，以各只手的当前姿态建立中立位，在 3D 界面中显示相对转角和随动手掌。项目还提供设备接入、分配、参数设置与校准入口，适合验证手腕动作映射到人形模型的可行性。

> 当前演示关注**姿态/转角**，不是空间位置追踪，也尚未直接控制机械臂。单靠 IMU 加速度积分无法稳定获得长期准确的位置坐标。

## 主要功能

- 同时接收多台 WT901WIFI 的 UDP 数据，按设备唯一 ID 区分；默认 `WT5500011381` 控制左手、`WT5500011408` 控制右手。
- 分别显示两只手相对各自中立位的 X/Y/Z 转角，以及固定前臂、随动手掌的 3D 模型。
- 在界面中查看设备数据、电压与估算电量；读取和设置 6/9 轴算法、带宽、回传频率、陀螺仪/加速度计量程及安装方向。
- 提供磁场、加速度计校准，以及设备参考角和界面中立位归零操作。
- 每只手可切换厂商 RPY、俯仰保护、全程陀螺 SO(3)、陀螺＋加计纠倾斜四种电脑端姿态来源，便于对比；保护区范围可设置，但不保证消除保护区外的航向漂移。
- 通过“设备分配”将热点中在线的 WT 设备绑定到手部；通过“热点设备”查看设备 ID 与网络端点，并可将选中设备从 STA/UDP 切回 AP/UDP。
- 通过“接入新设备”向导，把处于 AP 模式的新设备配置为连接电脑热点的 STA/UDP 模式。

界面右上角的“帮助 / Help”按主题说明各项参数、校准和接入操作。

## 环境要求

- Python 3.11（其他 Python 版本未在本项目中验证）；推荐使用 Conda 环境 `wt901`。
- 支持 OpenGL 的桌面环境；GUI 使用 PySide6、pyqtgraph 和 PyOpenGL。
- 需要至少一台 WT901WIFI 设备。使用 GUI 的热点管理、设备发现和接入向导时，电脑需运行 Linux/NetworkManager，并具备可开启 **2.4 GHz Wi-Fi 热点**的无线网卡；相关操作会调用 `nmcli`、`iw`、`ip`。
- 设备应配置为向电脑的 UDP 监听地址发送数据；默认监听端口为 `1399`。

## 安装与启动

在项目根目录执行：

```bash
conda create -n wt901 python=3.11
conda activate wt901
python -m pip install -r wt901_viz/requirements.txt
python wrist_demo.py
```

已经创建过 `wt901` 环境时，不必重复创建。也可以不激活环境，直接运行：

```bash
conda run --no-capture-output -n wt901 python -u wrist_demo.py
```

默认监听所有本机网卡的 UDP `1399` 端口，因此电脑在热点与设备 AP 之间切换时，程序可以保持运行。需要改设备分配或监听端口时，例如：

```bash
python wrist_demo.py --left-id WT5500011381 --right-id WT5500011408 --listen-port 1399
```

运行 `python wrist_demo.py --help` 可查看全部启动参数。程序也支持 `--net tcp` 接收姿态数据，但**设备发现、接入和寄存器控制功能按 UDP 工作**；使用完整控制台功能请保持默认 UDP 模式。一个监听端口即可接收多台设备，无需为每台设备单独启动程序。

手腕模型默认显示随手掌转动的局部坐标轴：红 X、绿 Y、蓝 Z；顶部复选框可隐藏。左右手各自的“轴映射”按钮可指定设备正 X/Y/Z 轴对应模型的哪根轴，程序会拒绝重复轴或镜像坐标系。映射只影响当前 GUI 的模型和角度显示，不改设备寄存器；重启后恢复默认映射。

想观察从 GUI 零位到当前姿态的 (SO(3)) 旋转向量，可单独运行 [旋转向量实验窗口](so3_rotvec_demo.py)：

```bash
conda run --no-capture-output -n wt901 python -u so3_rotvec_demo.py --device-id WT5500011361
```

收到数据后点击“以当前姿态作为模型零位”，再旋转设备。黄箭头是模型坐标下的旋转向量（方向为旋转轴，长度为转角，单位弧度），青线是端点轨迹；右侧同时显示转角、旋转轴及旋转矩阵。默认采用单位轴映射，也可用 `--mapping +X,-Z,+Y` 指定合法映射。该窗口只监听 UDP 1399，不写设备寄存器；**不要与手腕控制台或人体模型程序同时运行**。图中显示旋转对数的主值，转角跨过 180° 时可能出现分支跳变，而物理姿态并未跳变。

仓库中的 [桌面启动器](WT901-手腕控制台.desktop) 和 [启动脚本](launch_wt901.sh) 是本机便捷入口，目前写有原作者电脑上的项目路径和 Conda 路径。换电脑后须先修改这些路径；上面的终端启动命令不依赖这些固定路径。

## 设备联网与首次使用

设备有两种常用 Wi-Fi 工作方式：

| 模式 | 连接方式 | 用途 |
| --- | --- | --- |
| AP | 电脑连接设备发出的 `WT55…` Wi-Fi | 初次配置、单机调试 |
| STA/UDP | 设备连接电脑的 2.4 GHz 热点，并向电脑发送 UDP 数据 | 多设备同时接入与本项目的主要使用方式 |

接入向导需要电脑上**预先存在**名为 `WT901-LAN` 的 NetworkManager 热点配置，并要求它是带固定 IPv4 地址、`ipv4.method=shared` 的 AP 配置；向导不会从零创建热点配置。具体接入步骤：

1. 启动控制台，点击“接入新设备”。如电脑热点正在运行，先在向导中关闭；此时已接入的设备会暂时离线。
2. 让电脑连接新设备的 `WT55…` Wi-Fi，并在向导中刷新状态。
3. 核对电脑热点名称，输入**电脑热点密码**，点击“写入 STA + UDP 配置”。向导会写入热点信息及电脑的接收地址。
4. 重新启动电脑热点，等待向导检测到该设备 ID 的实时数据。接入后可在“设备分配”中选择由哪只手使用它；此分配只在本次程序运行中生效。

一次接入一台设备。修改网络工作模式会使设备断开当前 Wi-Fi；若要改回设备自身热点，可在“热点设备”窗口选中已确认在线的 WT 设备，使用“将选中设备切回 AP/UDP”，随后用另一台电脑连接其 `WT55…` Wi-Fi 验证。设备无法联网时，参照[操作手册](手册中文/WT901WIFI%20操作手册.md)中的恢复 AP 说明。

## 界面操作

| 操作 | 作用 |
| --- | --- |
| 空格 / `1` / `2` | 分别将双手 / 左手 / 右手的**界面中立位**设为当前姿态，不写设备寄存器 |
| `R` | 重置 3D 视角 |
| `Esc` / `Q` | 关闭窗口 |
| “读取设备参数” | 先读取当前设备的设置，再调整算法、带宽、速率、量程等参数 |
| “设备分配” | 从当前热点中有有效 WT 数据的设备里选择控制各部位的设备 |
| “热点设备” | 查看已关联终端、设备唯一 ID、设备端点及电脑接收端点 |

设置设备参数会发送命令、保存并回读确认；离线设备不能配置。更改融合算法或安装方向后，界面会用下一包数据重建该手的中立位。磁场校准时应远离磁铁、磁吸配件和大块金属，并按界面指引充分绕三个轴旋转。

## 其他程序与目录

| 路径 | 用途 |
| --- | --- |
| [wrist_demo.py](wrist_demo.py) | 双手手腕姿态 GUI 主程序 |
| [wt901/](wt901/) | WT 数据包解析、网络接收与设备寄存器通信 |
| [wt901_viz/](wt901_viz/) | 3D 显示、联网向导及独立姿态查看器；详见其 [README](wt901_viz/README.md) |
| [OptUSBProj-ubuntu/](OptUSBProj-ubuntu/) | 人体 GUI、RGB-D 采集／回放、穿戴与相机标定、实时融合；业务代码在 `optusbproj_ubuntu/` |
| [urdf-model/](urdf-model/) | X1 URDF、ROS 2 可视化、FK/IK、离线回放和实时预览工具 |
| [docs/](docs/) | 数学推导、项目总览与图像讲解源文件 |
| [multi9.py](multi9.py)、[print9.py](print9.py) | 命令行接收、检查传感器数据 |
| [magcal.py](magcal.py)、[configure.py](configure.py)、[setsta.py](setsta.py) | 命令行校准、配置与 STA 接入工具 |
| [pose9.py](pose9.py) | 基于 IMU 的位置估计实验；积分漂移明显，不适合当作可靠定位结果 |
| [手册中文/](手册中文/) | 设备操作与通信协议资料 |
| [tests/](tests/) | 自动化测试 |

本地归档目录 `no-improtent/` 已被 `.gitignore` 排除，不会随仓库提交；其中的原版资料或示例 SDK 不是运行主程序所必需的。

人体 GUI 的 `OptUSBProj-ubuntu/recordings/`、`calibrations/` 和 `local_config.json` 也只保留本地；模型下载权重、ROS `build/install/log` 和生成的私人讲解素材不入库。记录可能含人脸、房间环境、设备标识和动作数据，分享前应检查并匿名化。

## 手机局域网夹爪输入

手机局域网夹爪标量输入（无需 App、不占用 IMU 端口）：

```bash
python3 OptUSBProj-ubuntu/lan_gripper_demo.py
```

手机和电脑分别打开终端输出的控制／观测链接。可选择左／右夹爪并勾选发送至 RViz，由现有 ROS 桥统一合并双臂和夹爪状态；不发送电机指令。详见 [使用说明](OptUSBProj-ubuntu/docs/LAN_GRIPPER_DEMO.md)。

## 测试

统一测试入口可从任意工作目录调用。使用当前机器已配置的两套环境，运行旧手腕核心测试与人体/RGB-D 测试：

```bash
python /home/hzb/Work/WT901/scripts/run_tests.py --core-env wt901 --app-env OptUSBProj-yolo
```

入口自动设置导入路径，默认使用无头 Qt，不安装依赖，不运行真实仿真联调脚本。
已有激活环境时可省略 `--core-env` / `--app-env`；此时使用启动入口的 Python。
快速检查新机器人链路：

```bash
conda run --no-capture-output -n OptUSBProj-yolo python /home/hzb/Work/WT901/scripts/run_tests.py --suite scene
```

用 `--suite core|app` 选择测试组，`--pattern test_pose.py` 选择文件，`--verbose` 显示具体用例。
重构基线、固定数值样本与验收边界见 [阶段 0 基线说明](docs/REFACTOR_PHASE0_BASELINE.md)。以下原始命令仍保留兼容。

```bash
conda run --no-capture-output -n wt901 python -m unittest discover -s tests -v
```

人体／RGB-D 功能的测试是另一套测试，使用已配置依赖的环境：

```bash
cd /home/hzb/Work/WT901/OptUSBProj-ubuntu
QT_QPA_PLATFORM=offscreen PYTHONPATH=/home/hzb/Work/WT901:/home/hzb/Work/WT901/OptUSBProj-ubuntu \
  conda run --no-capture-output -n OptUSBProj-yolo python -m unittest discover -s tests
```

2026-10-06 合并至 `main` 后，上述人体／RGB-D 测试共 322 项通过。这是该版本的验收记录，不是未来测试数量或传感器精度保证；无头 GUI 测试中的 OpenGL 提示也不能替代真实桌面渲染检查。

自动化测试不代替真实设备验证。首次运行时，请先确认设备在线、电脑监听端口未被其他程序占用，并在界面检查接收到的设备 ID 与预期一致。

## 已知限制与注意事项

- 该模型用于演示手腕附近的相对转角，前臂固定；它不追踪手臂位置，也没有人体关节约束或机械臂安全控制。
- 6 轴模式不依赖磁力计修正航向，短时间可避开磁干扰，但长期航向可能漂移；9 轴模式可约束航向漂移，但会受磁场环境和校准质量影响。
- 电量百分比是依据手册电压档位的**粗略估算**，不能视为精确剩余续航。
- 切换设备 Wi-Fi 模式、写入热点凭据和修改寄存器都会改变设备状态；操作前确认选中的设备 ID。首次接入时不要在日志或公开仓库中保存真实热点密码。

硬件细节与指令依据：[WT901WIFI 操作手册](手册中文/WT901WIFI%20操作手册.md)、[WT901WIFI 通信协议](手册中文/WT901WIFI%20通信协议.md)。
