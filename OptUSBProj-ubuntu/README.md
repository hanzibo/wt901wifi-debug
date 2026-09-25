# OptUSBProj Ubuntu 版（进行中）

当前里程碑：可用 1～3 台（后续扩至 7 台）WT901WIFI，通过电脑 `WT901-LAN` 热点向本机 UDP 1399 发数据，在 15 部件人体模型中驱动已绑定的上身部位。此版本**尚未包含**旧项目的机器人 FK/IK、碰撞检测和 CoppeliaSim 映射；进度和后续验收见 [重构计划](REFACTOR_PLAN.md)。

## 运行

从仓库根目录：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/run.py
```

也可双击桌面上的“WT901 上身姿态映射”启动器。仓库内的 [`WT901-上身姿态映射.desktop`](WT901-上身姿态映射.desktop) 调用 [`launch.sh`](launch.sh)，使用同一个 `OptUSBProj` 环境。若启动失败，会弹出提示，并将详细错误写到 `~/.local/state/wt901/optusbproj-ubuntu.log`。桌面文件含本机绝对路径，项目迁移到其他电脑时需同步修改 `Exec`、`Path`、`Icon` 及脚本中的 Conda 路径。

电脑需先启动 `WT901-LAN` 2.4 GHz 热点；设备应设置为 STA/UDP，并向电脑热点 IP 的 1399 端口发送数据。与手腕程序共用一个接收端口，**请先关闭 `wrist_demo.py`**。程序启动时没有设备也能显示静态模型；上线设备会出现在“设备 ID”列表，按实际佩戴位置绑定即可。只连接 3 台时，其他部位跟随父部位或保持静态。首次数据包自动建立该设备的界面中立位，可用“归零”调整；分配保存到本地 `local_config.json`，不会提交到 Git。

界面中的“热点设备”和“接入新设备”沿用现有手腕项目的电脑热点工具。按空格可将已绑定设备全部归零，按 `R` 重置视角。设备断流时保留最后姿态并显示数据龄。

穿戴设备不方便立即操作时，可在主界面选择 3～120 秒延时，点击“定时全部归零”；倒计时会显示在按钮下方，也可随时取消。时间到后仅对仍在接收数据的已绑定设备建立中立位，断流设备会被跳过并在界面提示。直接点击“全部归零”或按空格仍是立即执行，同时取消未完成的倒计时。

点击主界面的“设备参数”，在弹窗中选择当前在线的设备。三个标签页分别展示实时传感器数据、融合算法/安装方向/带宽/回传频率/陀螺与加计量程，以及磁场和加速度计校准、设备角度参考归零。参数在读取成功前不会开放写入；每次设置都会先确认，再解锁、写入、保存和回读验证。磁场校准先做三轴覆盖预检，进入正式校准后务必点击“结束磁场校准并保存”。弹窗与人体界面共用同一个 UDP 接收端口，不会重复抢占 1399。录制回放模式下设备参数入口禁用。

默认在 15 个模型部件的关节原点显示局部坐标轴：红色 X、绿色 Y、蓝色 Z，方向随对应部件姿态变化。可用右侧复选框或 `A` 键显示/隐藏。它们表示**模型部件轴**，不是传感器外壳上印的 X/Y/Z，也不是固定不动的世界轴。

每个可绑定部位的设备 ID 左侧有“轴映射”按钮。弹窗中分别指定**归零姿势下**设备的 +X、+Y、+Z 轴指向模型的哪一根正/负轴，例如设备 +Y → 模型 +Z（上）。默认是 X→X、Y→Y、Z→Z；不能重复使用同一根模型轴，也不能形成镜像（左手坐标系）。映射按部位保存到 `local_config.json`，更换绑法后需要重新核对。计算先求出姿态在**设备归零时自身坐标系**中的变化，再用所选映射转换为模型姿态；不再沿用把 WT 世界竖直 Z 误当成模型前后 X 的固定矩阵。

简单桌面验证：设备 Z 轴竖直向上，左手轴映射保持默认并点击“归零”，然后绕竖直 Z 轴转动设备，模型左手应绕蓝色 Z 轴转。若在弹窗中手动选了“设备 Z → 模型 X”，同样动作就会绕红色 X 轴转；这是所选规则的预期结果。桌面姿势与真实绑在手上的中立姿势不同，正式佩戴后请重新归零并核对映射。

需要留存动作样本时，点击“开始录制 CSV”，选定文件后再点击“停止录制 CSV”。回放录制文件（不占用 1399 端口）可用：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/run.py --replay /path/to/samples.csv
```

首次回放前先在界面分配文件中的设备 ID。录制只保存 WT 原始数据和电脑接收时间，不包含机器人控制命令。

## CoppeliaSim 最小连通测试

此测试不依赖 WT901、机器人模型或旧版 Windows DLL。先在 `OptUSBProj` 环境安装 `pyzmq` 和 `cbor`；Python 客户端代码直接使用本机 CoppeliaSim 4.10 安装包自带的 2.0.4 版本，无需另装客户端包。打开 CoppeliaSim 并保持仿真**停止**，再从仓库根目录运行：

```bash
conda run --no-capture-output -n OptUSBProj python -m pip install pyzmq cbor
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/sim_connectivity.py \
  --coppelia-dir /home/hzb/Software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04
```

探针连接默认 ZeroMQ 端口 23000，创建一个临时 dummy，发送两次位置命令，读回移动后的坐标并删除 dummy。输出 `PASS` 才说明“客户端连接 → 发指令 → 仿真端执行 → 读回”已贯通。探针不会启动仿真、控制机器人或保存场景；仿真正在运行时会拒绝执行。只启动 CoppeliaSim 而没有收到 `PASS` 不算完成测试。

2026-09-25 已在本机 CoppeliaSim 4.10 的独立空场景完成实测：连接 `127.0.0.1:23000`，dummy 位置读回 `[0.0, 0.0, 0.4] → [0.1, 0.0, 0.4]` 米，结果 `PASS`；测试后关闭了该仿真实例。测试使用 `-h` 无窗口模式和虚拟显示器，因此不会在当前桌面显示 CoppeliaSim 窗口；手动去掉 `-h` 并在桌面会话启动才会显示 GUI。此结果只验证远程 API 通信和普通对象位置命令，**尚未**验证机器人关节、FK/IK 或旧项目场景。

若要肉眼观察指令生效，先在真实桌面启动 `coppeliaSim.sh`（不带 `-h`，并保持仿真停止），再运行：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/sim_connectivity.py \
  --coppelia-dir /home/hzb/Software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04 \
  --visual-demo-seconds 60
```

这会创建较大的 `WT901_ZMQ_demo_dummy`，沿世界 X 轴左右往返，并逐次读回位置；60 秒后自动删除临时对象，不保存场景。GUI 保持打开，直到用户自行关闭。

## 实现边界

- 接收层直接使用仓库已有的 `wt901.gui_udp.GuiUdpSource` 和 `wt901.protocol.AttitudePacket`，不运行旧目录的 `UdpService` 或 Windows DLL。
- 人体位置来自旧模型的固定肢段长度/偏移；当前不估计 IMU 在空间中的绝对位移。当前支持逐部位设备轴映射和中立位，但安装偏置的实际方向、各设备航向一致性仍待实机验收。
- 原项目里 15 个 STL 中当前选用的模型已复制到本目录 `assets/models/`；原始 `OptUSBProj/` 不参与运行。

## 测试

```bash
conda run --no-capture-output -n OptUSBProj python -m unittest discover -s OptUSBProj-ubuntu/tests -v
```

离线测试覆盖姿态链、设备绑定、CSV 回放和 CoppeliaSim 连通探针的清理逻辑。GUI/OpenGL 已在虚拟显示器下通过启动与渲染测试；当前热点上的 `WT5500011381`、`WT5500011408`、`WT5500011624` 各接收约 50 包/秒。三台设备的实际佩戴轴向和人体动作映射仍需现场验证，不能仅凭数据到达认定动作方向正确。
