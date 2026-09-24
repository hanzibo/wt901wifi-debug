# OptUSBProj Ubuntu 版（进行中）

当前里程碑：可用 1～3 台（后续扩至 7 台）WT901WIFI，通过电脑 `WT901-LAN` 热点向本机 UDP 1399 发数据，在 15 部件人体模型中驱动已绑定的上身部位。此版本**尚未包含**旧项目的机器人 FK/IK、碰撞检测和 CoppeliaSim 映射；进度和后续验收见 [重构计划](REFACTOR_PLAN.md)。

## 运行

从仓库根目录：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/run.py
```

电脑需先启动 `WT901-LAN` 2.4 GHz 热点；设备应设置为 STA/UDP，并向电脑热点 IP 的 1399 端口发送数据。与手腕程序共用一个接收端口，**请先关闭 `wrist_demo.py`**。程序启动时没有设备也能显示静态模型；上线设备会出现在“设备 ID”列表，按实际佩戴位置绑定即可。只连接 3 台时，其他部位跟随父部位或保持静态。首次数据包自动建立该设备的界面中立位，可用“归零”调整；分配保存到本地 `local_config.json`，不会提交到 Git。

界面中的“热点设备”和“接入新设备”沿用现有手腕项目的电脑热点工具。按空格可将已绑定设备全部归零，按 `R` 重置视角。设备断流时保留最后姿态并显示数据龄。

默认在 15 个模型部件的关节原点显示局部坐标轴：红色 X、绿色 Y、蓝色 Z，方向随对应部件姿态变化。可用右侧复选框或 `A` 键显示/隐藏。它们表示**模型部件轴**，不是传感器外壳上印的 X/Y/Z，也不是固定不动的世界轴；当前姿态映射仍保留原版本算法，便于对照轴向问题。

需要留存动作样本时，点击“开始录制 CSV”，选定文件后再点击“停止录制 CSV”。回放录制文件（不占用 1399 端口）可用：

```bash
conda run --no-capture-output -n OptUSBProj python -u OptUSBProj-ubuntu/run.py --replay /path/to/samples.csv
```

首次回放前先在界面分配文件中的设备 ID。录制只保存 WT 原始数据和电脑接收时间，不包含机器人控制命令。

## 实现边界

- 接收层直接使用仓库已有的 `wt901.gui_udp.GuiUdpSource` 和 `wt901.protocol.AttitudePacket`，不运行旧目录的 `UdpService` 或 Windows DLL。
- 人体位置来自旧模型的固定肢段长度/偏移；当前不估计 IMU 在空间中的绝对位移。当前仅支持每设备的中立位，安装偏置/轴向校准仍待实机验收。
- 原项目里 15 个 STL 中当前选用的模型已复制到本目录 `assets/models/`；原始 `OptUSBProj/` 不参与运行。

## 测试

```bash
conda run --no-capture-output -n OptUSBProj python -m unittest discover -s OptUSBProj-ubuntu/tests -v
```

离线测试覆盖姿态链、设备绑定和 CSV 回放。GUI/OpenGL 已在虚拟显示器下通过启动与渲染测试；当前热点上的 `WT5500011381`、`WT5500011408`、`WT5500011624` 各接收约 50 包/秒。三台设备的实际佩戴轴向和人体动作映射仍需现场验证，不能仅凭数据到达认定动作方向正确。
