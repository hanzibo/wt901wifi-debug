# OptUSBProj → Ubuntu 重构计划

状态：探索与方案阶段。`OptUSBProj/` 为原始参考，保持不修改；本目录用于 Ubuntu 版实现。当前工作分支：`optusbproj-ubuntu`。

## 1. 已核实的旧项目行为

- 真正的上身演示入口是 `OptUSBProj/IMUDataShow.py`；`main.py` 仍是 PyCharm 默认示例。主界面通过 PySide6 加载 `UI/IMUDataVisUI.ui`，用 PyVista/QtInteractor 加载 15 个 STL 部件构成人体模型。
- 传感器主链路实际为 **UDP 1399**，而非直接读取 USB IMU。师兄当时用路由器把多台设备接入同一网络；`UdpService` 解析 54 字节 WT901WIFI 报文，并把 7 个固定设备 ID 分别映射到节点 1、3、4、5、6、7、8（躯干及双臂）；其余节点保留在 15 节点模型中。界面上的“连接USB”按钮命名与实际行为不符。
- 按 `Config/CfgDataShow.xml` 的层级、模型偏移和传感器校准偏移，计算相邻身体部位的旋转与全局 4×4 变换矩阵。程序支持中立位/校准、滤波、实时模型、CSV 录制与回放相关工具。
- 正运动学模式把人体相对旋转近似映射到双臂 12 个机器人关节；逆运动学模式先构造双手末端位姿，再求关节角。另有碰撞检测、串口实体机器人控制及 CoppeliaSim 仿真接口；这些应分别验证，不能因 GUI 能启动就视为已迁移完成。

## 2. 已确认的迁移障碍

| 项目 | 旧代码现状 | Ubuntu 处理方向 |
| --- | --- | --- |
| Windows DLL | `remoteApi.dll`、`RobotFkIk.dll`、`GJKCcd.dll`，且部分 `ctypes.CDLL` 写死 `D:\...` 路径；模块导入时立即加载 | 新项目不直接加载 DLL；优先用 Python 运动学实现及已有 SAT/Numba 碰撞逻辑，必要时另建可选 Linux 原生扩展 |
| CoppeliaSim 接口 | `sim.py` 使用旧版 `simx*` API，依赖 Windows `remoteApi.dll`、端口 19997、旧场景对象名 | 优先改为本机 CoppeliaSim 4.10 提供的 ZeroMQ Remote API；建立单独适配器，先测试连接与简单关节，再接完整模型 |
| 仿真场景 | 原目录未发现 `.ttt`/`.ttm` 场景，旧代码引用 `jointR1…6`、`jointL1…6`、手指关节和碰撞盒对象 | 请补原场景/机器人模型；若找不到，先构建简化可验证场景，再逐步补足关节、手指和碰撞对象。未获场景前不能保证原仿真外观及对象对应关系 |
| 设备绑定 | 7 个旧设备 ID 写死在 `UdpService`，与你当前设备 ID 不同 | 改为配置式“设备 ID → 身体节点”映射，复用本仓库 `wt901` 的报文解析与多设备接收能力 |
| 网络拓扑 | 旧实验依靠外部路由器提供局域网 | 新版默认使用电脑 `WT901-LAN` 2.4 GHz 热点：各 WT901 设为 STA/UDP，目标为电脑热点地址和端口 1399；同一 UDP 监听器接收所有设备，无需路由器、虚拟网卡或每设备单独端口 |
| 路径与串口 | 相对当前目录读取 UI/STL/XML，机器人串口为 `COM8` | 用 `pathlib` 从包位置解析资源；串口参数化为 `/dev/serial/by-id/...` 等 Linux 路径 |
| 实时稳定性 | UDP 缓冲与状态为类级共享变量；停止时 `recvfrom()` 阻塞；设备关闭按钮仍访问不存在的 USB `device` 属性 | 拆分采集、状态、计算、渲染线程；增加超时、退出与异常处理；按设备 ID 管理时间戳和掉线状态 |
| 算法可验证性 | 姿态轴系、欧拉顺序、偏置与 FK/IK 分散在 GUI 文件；无针对性测试 | 将 4×4 变换、坐标系、关节限位、FK/IK 独立为纯计算模块，建立零位、单轴旋转、左右镜像和 FK→IK 回环测试 |

### WT901 通信模块的复用边界

- **直接复用**仓库根目录的 `wt901.protocol.AttitudePacket`、`extract_packets()` 和 `wt901.gui_udp.GuiUdpSource`。前者负责 54 字节报文定界、字段换算、电压等；后者以一个 `0.0.0.0:1399` UDP 套接字接收所有设备，并提供 `poll()`、`seen_devices()`、`seen_endpoints()` 与安全的设备寄存器通信。新项目不复制旧版 `DeviceModel`/`UdpService`，也不重写一份解析协议。
- 主 GUI 定时器从**同一个** `GuiUdpSource.poll()` 取包，以 `packet.device_id` 为键保存最新样本，再经“设备 ID → 身体节点”配置送入姿态链。数据包模型直接使用 `roll/pitch/yaw`、`battery_v`、`rssi` 等字段；人体层只负责坐标系换算、安装偏置、中立位和运动学。
- 热点设备发现与 AP→STA 接入可借用 `wt901_viz.device_list`、`wt901_viz.onboarding` 的现有逻辑；界面应按全身节点重新组织，不把手腕模型的左右手状态类当作全身通用数据结构。
- **不能同时运行** `wrist_demo.py` 与新程序来分别监听 UDP 1399；两进程占用同一端口可能分走数据。需要同时查看手腕和全身时，应在一个进程内复用单一接收源并分发样本。

## 3. 本机环境调查

- Conda 环境 `OptUSBProj`：Python 3.11.16。已验证可导入 `numpy`、`scipy`、`PySide6`、`pyvista`、`pyvistaqt`、`vtk`、`transforms3d`、`roboticstoolbox`、`spatialmath`、`numba`、`pyserial`、`openpyxl` 和 `matplotlib`。
- `hid` 未安装，但当前演示的主传感器链路是 UDP；新项目可去掉无用的 HID 强制导入。暂不要求安装。
- `coppeliasim-zmqremoteapi-client` / `pyzmq` 未安装。进入仿真联调阶段前，需在 `OptUSBProj` 环境安装并验证：`conda run -n OptUSBProj python -m pip install coppeliasim-zmqremoteapi-client`。CoppeliaSim 本机附带的 ZeroMQ API 手册和 Python 客户端示例也给出该安装方式。
- CoppeliaSim 位于 `/home/hzb/Software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/`：启动脚本可执行，`libsimZMQ.so` 存在，主程序的静态链接检查未见缺失项。这只证明安装文件和动态库看起来齐全；尚未实际启动 GUI、加载场景或测试远程连接。安装包名标注 Ubuntu 24.04，当前系统为 Ubuntu 26.04，运行兼容性仍需实测。
- 目前**没有确认必须补装的系统依赖包**。若后续 Qt/VTK/CoppeliaSim 实际启动报缺库，再针对具体错误补装。

## 4. 建议实施顺序与验收点

1. **建立新项目骨架与最小可运行 GUI**：将配置、资源、程序入口、传感器接口、运动学和仿真接口分包；保留 15 节点人体结构和旧模型资源。使用离线示例数据启动 GUI、显示人体，并让单节点旋转正确。交付明确的环境文件与启动说明。
2. **接入电脑热点与实时多设备数据**：复用手腕控制台已验证的 `WT901-LAN`/NetworkManager 接入流程及 WT901 UDP 解码。热点开启时，所有设备以 STA/UDP 加入电脑热点，向电脑热点 IP 的 1399 端口发送数据；监听地址默认 `0.0.0.0`。按配置绑定 7 个身体节点，并提供热点终端/设备 ID 列表、新设备 AP→STA 接入、掉线提示；未接满 7 台时允许局部演示。再完成逐节点中立位、轴向安装偏置及录制/回放，用真实数据核对姿态方向和延迟。接入向导切换电脑 Wi-Fi 模式时，已在线设备会暂时离线，热点恢复后应自动重连。
3. **重建并验证姿态链与模型随动**：把原有 15 节点拓扑、STL 单位（mm→m）和旋转约定写成可测试算法；与旧代码保存的数据/已知动作对照。重点验证肩、肘、腕及左右镜像，避免只凭画面“看着像”判定正确。
4. **迁移 FK/IK 与碰撞约束**：先实现纯 Python 可运行的 FK/IK、关节范围、求解失败回退；再迁移可跨平台的碰撞检测。Windows DLL 不作为新程序启动前提。用离线轨迹和边界姿态做回归测试。
5. **接入 CoppeliaSim**：先用 ZeroMQ API 验证程序连接、读取对象、发送单关节角；取得原场景后匹配全部关节和碰撞盒。若无原场景，明确建立替代场景并记录与原项目的差异。仿真成功后才考虑实体机器人串口控制。
6. **联调与交付**：在**没有外部路由器**的条件下验证电脑热点接收 1/2/7 台设备、设备 ID 分流、掉线恢复、热点重启、GUI 关闭、长时间运行、回放一致性，以及“传感器→人体 GUI→FK/IK→CoppeliaSim”端到端链路；补 README、配置样例、测试和故障排查。

## 5. 需要用户确认或提供

- 最关键：师兄的 CoppeliaSim 场景文件（`.ttt`/`.ttm`）或机器人模型/对象树是否还能取得？原目录仅有人体 STL，没有仿真场景。
- 当前可用于上半身的 WT901WIFI 有几台、各自设备 ID 与计划佩戴部位是什么？旧代码的 7 个 ID 是师兄设备的，不应直接沿用；未分配的设备可以先接入电脑热点，之后在 GUI 中按 ID 绑定身体部位。
- 是否需要“真实机器人串口控制”作为本轮目标？此功能涉及硬件风险，建议先完成 GUI 与 CoppeliaSim 仿真，再单独启用并验证限位与急停。

## 6. 版本管理边界

原始 `OptUSBProj/` 目前是未跟踪目录，含旧 DLL、个人 IDE 配置和历史数据；不要直接 `git add -A` 提交它。新项目只按需复制 STL、UI、配置或经重构的算法。是否把原始项目作为归档纳入仓库，需要单独决定。
