# OptUSBProj Ubuntu 版（进行中）

当前里程碑：可用 1～3 台（后续扩至 7 台）WT901WIFI，通过电脑 `WT901-LAN` 热点向本机 UDP 1399 发数据，在 15 部件人体模型中驱动已绑定的上身部位。此版本**尚未包含**旧项目的机器人 FK/IK、碰撞检测和 CoppeliaSim 映射；进度和后续验收见 [重构计划](REFACTOR_PLAN.md)。

## 运行

从仓库根目录：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/run.py
```

电脑需先启动 `WT901-LAN` 2.4 GHz 热点；设备应设置为 STA/UDP，并向电脑热点 IP 的 1399 端口发送数据。与手腕程序共用一个接收端口，**请先关闭 `wrist_demo.py`**。程序启动时没有设备也能显示静态模型；上线设备会出现在“设备 ID”列表，按实际佩戴位置绑定即可。只连接 3 台时，其他部位跟随父部位或保持静态。首次数据包自动建立该设备的界面中立位，可用“归零”调整；分配保存到本地 `local_config.json`，不会提交到 Git。

界面中的“热点设备”和“接入新设备”沿用现有手腕项目的电脑热点工具。按空格可将已绑定设备全部归零，按 `R` 重置视角。设备断流时保留最后姿态并显示数据龄。

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

## 实现边界

- 接收层直接使用仓库已有的 `wt901.gui_udp.GuiUdpSource` 和 `wt901.protocol.AttitudePacket`，不运行旧目录的 `UdpService` 或 Windows DLL。
- 人体位置来自旧模型的固定肢段长度/偏移；当前不估计 IMU 在空间中的绝对位移。当前支持逐部位设备轴映射和中立位，但安装偏置的实际方向、各设备航向一致性仍待实机验收。
- 原项目里 15 个 STL 中当前选用的模型已复制到本目录 `assets/models/`；原始 `OptUSBProj/` 不参与运行。

## 测试

```bash
conda run --no-capture-output -n OptUSBProj python -m unittest discover -s OptUSBProj-ubuntu/tests -v
```

离线测试覆盖姿态链、设备绑定、CSV 回放和 CoppeliaSim 连通探针的清理逻辑。GUI/OpenGL 已在虚拟显示器下通过启动与渲染测试；当前热点上的 `WT5500011381`、`WT5500011408`、`WT5500011624` 各接收约 50 包/秒。三台设备的实际佩戴轴向和人体动作映射仍需现场验证，不能仅凭数据到达认定动作方向正确。
