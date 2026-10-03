# WT901 + 单目相机：联合采集与时间对齐基线

此入口把 WT901 数据和摄像头识别结果放进同一个窗口，并录制可回放的会话。
**默认仍由原IMU算法驱动；右臂可选实时RGB-D航向纠偏已接入。** 需重新做绑定实时参考系的相机引导标定并明确应用，见 [实时纠偏](RGBD_LIVE_FUSION.md)。独立离线对比见 [离线实验说明](RGBD_OFFLINE_FUSION.md)；不自动启用融合或机器人。
已提供离线动作延迟检查与按相机帧时间插值 IMU 姿态，作为后续标定和融合的输入。

Gemini 336L 已新增独立 SDK 进程、RGB-D 共享内存接口、深度无损联合录制及回放。
运行加 `--camera-source orbbec`；仍然只采集，不进行视觉纠偏。环境、时间戳语义与
操作步骤见 [RGB-D 阶段 A](RGBD_STAGE_A.md)。

上半身真实深度观测已提供可选开关：`--rgbd-pose`。诊断、质量门控、记录及试验流程见 [阶段 B](RGBD_STAGE_B.md)。当前不做视觉/IMU 融合。

## 运行

在仓库根目录执行：

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_run.py
```

默认使用 RTMW3D、摄像头 0 和 UDP 1399。电脑热点与 WT901 的设置沿用
`run.py`；启动前关闭手腕控制台及其他占用 UDP 1399 或摄像头的程序。
若想先用 MediaPipe，可加 `--backend mediapipe`，前提是当前环境有
MediaPipe 和已下载的 `.task` 模型。原纯 IMU `run.py` 和纯相机
`camera_run.py` 保持独立，方便对照。

不方便在穿戴后操作电脑时，可设置「延迟」和「录制」时长，点击「定时录制」。
默认等待 10 秒后开始，录制 30 秒后自动结束；倒计时期间可取消预约，
不会创建录制文件。手动结束和自动结束采用相同规则：**末尾 3 秒的
IMU 包及相机帧均不保存**，为走回电脑和操作鼠标预留缓冲。少于 3 秒的
手动录制可能没有有效数据，建议录制时长明显长于 3 秒。

窗口右侧可沿用原有设备分配、轴映射和归零操作。相机预览会显示识别线；
录制的帧则是未绘线的原始图像。录制文件保存在
`OptUSBProj-ubuntu/recordings/<时间>-<短ID>/`：

- `imu.csv`：WT901 包字段、UDP 到达时的主机单调时钟时间戳。
- `camera.jsonl`：相机取帧时间、识别后端、原始像素坐标中的 2D 关键点及置信度。
- `frames/*.jpg`：未绘线的视频帧，一帧一个 JPEG。
- `manifest.json`：会话时钟锚点、计数和丢帧数。

`manifest.json` 也记录实际停止时刻、保留截止时刻和尾部剔除的包/帧数。
最后 3 秒先暂存于内存，结束时直接丢弃，不会先写到磁盘再删除。

JPEG 包含真实摄像头画面，可能拍到人脸；`recordings/` 已在 `.gitignore`
中，不要误将其上传。UDP 时间戳代表电脑收到数据的时刻，不是设备内部
采样时刻；相机时间戳取于 `VideoCapture.read()` 返回后，也不是摄像头
硬件曝光时间。因此本阶段得到的是两路运动信号在**记录时间轴**上的相对滞后，
不是传感器内部采样时刻的硬件同步测量。

## 第二阶段：离线时间检查与旋转插值

录制文件目前没有保存设备与部位的绑定快照，分析时须显式指定本次录制的
设备 ID。例如右上臂 `WT5500011572`、右前臂 `WT5500011408`：

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_sync.py \
  OptUSBProj-ubuntu/recordings/20260926-101048-b89911 \
  --torso-id WT5500011608 \
  --right-upper-arm-id WT5500011572 \
  --right-forearm-id WT5500011408
```

脚本提取肩—肘及肘—腕在图像中的 2D 运动速度，与相应设备的三轴角速度模长
作互相关。正延迟表示记录时间轴上 IMU 动作晚于相机动作。静止、遮挡、动作
过小、相关性弱或最佳值落在搜索边界时，结果会标为不可估计或低可信。
由于视觉投影、肢段滑动和帧率会影响峰值，不应把单次结果当作精确系统延迟。

可选 `--aligned-output /tmp/某文件.jsonl`，另外输出每一相机帧对应的 IMU
旋转矩阵。默认偏移为 **0 ms**；确认延迟稳定后才考虑加
`--apply-estimated-lags`。插值在 SO(3) 上进行，避免欧拉角跨 ±180° 时跳变；
超出样本范围或相邻 IMU 包间隔超过 100 ms 时写入 `null`，不进行外推。
输出保留相机时间戳、设备 ID 及实际采用的延迟，不修改原始录制目录。

这一步只对齐**传感器方向**，尚未把它转换为骨段方向，也未进行传感器到身体段
标定、相机投影约束或视觉纠漂。躯干可直接参与逐帧旋转插值，但因为缺乏
稳定的 2D 躯干旋转观测，本脚本暂不为躯干估计动作延迟。

## 引导式穿戴标定（推荐）

当前 GUI 已提供“穿戴标定…”独立窗口：倒计时引导躯干、左右上臂、左右前臂和左手腕完成规定动作，保存 IMU 原始数据及动作时间标记，自动检查静止与独立复测质量，再通过双模型回放比较手动映射和连续候选矩阵。通过后可临时应用到实时人体模型；重新启动或重新穿戴需重新验证。完整步骤见 [穿戴标定说明](WEARING_CALIBRATION.md)。

以下命令行三窗口标定仍保留，适用于分析已经录好的旧会话；它不会自动应用到实时模型。

## 第三阶段：右臂穿戴方向的候选标定

新增 `fusion_calibrate.py`。它利用**中立位、直臂前举、直臂侧举**三个静止姿势，
为上臂/前臂求连续的传感器到模型坐标旋转矩阵 `B`，比只能选 XYZ 整轴的手动
轴映射细。默认 `--method bone` 利用中立位的竖直方向与前举/侧举时的**骨段
指向**来求安装方向；`--method axis` 用两次完整姿态旋转轴拟合，要求两次
动作近似绕互相正交的轴。人体肩部动作常有轴向旋转，故后者主要用于诊断。
单靠中立位无法确定绕肢段长轴的安装角。结果先作为**候选**另存，不自动
改写现有设置。

建议先用现在的右上臂和右前臂两台设备测试；躯干那台只用来监测上身是否
明显跟着动。绑定设备后，保持绑带固定、肘部伸直、手腕不单独转动、站姿
面向相机，执行以下顺序：

1. 双臂自然下垂，身体不动，保持 4–5 秒（中立位）。
2. 右臂伸直向**正前方**缓慢抬到约肩高，保持 3–4 秒。
3. 缓慢放回下垂姿势，停 2–3 秒。
4. 右臂伸直向**身体右侧**缓慢抬到约肩高，保持 3–4 秒。
5. 放下手臂，等待录制自动结束。不要在静止窗口中耸肩或扭转躯干。

可在 `fusion_run.py` 中设定延迟 10 秒、录制 33 秒，点击「定时录制」后到
镜头前准备。录制最后 3 秒会被自动裁掉。实际动作不必卡死在上述秒数；
录制后根据视频和陀螺数据再确定三个静止窗口。**旧录制没有保存设备绑定
快照**，从新版开始 `manifest.json` 增加 `assignments_at_start`；即便有快照，
仍应在标定前核对绑带上的实际设备 ID。

选择窗口后的命令示例（秒数仅示意，不要照抄到其他录制）：

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_calibrate.py \
  OptUSBProj-ubuntu/recordings/<本次会话目录> \
  --part right_upper_arm --device-id <右上臂设备ID> \
  --neutral 2:4 --forward 8:10 --lateral 21:23 \
  --output /tmp/right-upper-arm-calibration.json
```

前臂将 `--part` 换成 `right_forearm`、设备 ID 换成右前臂，其余窗口保持一致。
`B` 只在**设备绑法未改变**时可复用。骨段方法假设中立位手臂自然下垂，
且 WT901 姿态的世界 Z 与重力竖直方向一致；前举/侧举应尽量到水平。
候选可通过 `fusion_run.py --replay ... --calibration ...` **仅在回放中临时预览**：
同一窗口并排显示两个人体模型，左边使用手动 XYZ 轴映射，右边使用候选
连续标定；两者读入完全相同的 IMU 包、共享视角，并分别标注来源。相机画面
只作参照，不参与姿态计算。会话和设备 ID 必须匹配；窗口会锁定设备分配与
手动轴映射，关闭窗口后不会保存到 `local_config.json`。例如：

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_run.py \
  --replay OptUSBProj-ubuntu/recordings/<本次会话目录> \
  --calibration /tmp/right-upper-arm-calibration.json \
  --calibration /tmp/right-forearm-calibration.json
```

从新录制起，`manifest.json` 会保存 `axis_mappings_at_start`，回放对比优先
使用录制时映射快照。旧录制 `20260926-103707-35fa31` 只有设备分配快照，
其左侧模型使用**当前** `local_config.json` 中的手动映射；若录制后改过该
配置，左侧就不一定等于录制当时所见。

先用真实录制验证其稳定性与动作方向，合格后再考虑用于实时模型。这里估计的是安装
坐标映射，不包含摄像头外参、肘关节中心、绝对位置或视觉纠漂。

## 回放

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_run.py \
  --replay OptUSBProj-ubuntu/recordings/<会话目录>
```

回放不打开摄像头或 UDP 端口；可加 `--speed 0.5` 等参数。回放窗口的
相机预览显示保存的原始图像，人体模型仍按录制的 IMU 数据运动。

## 第一轮穿戴检查

先连接躯干、左上臂和左前臂三台设备即可，在窗口中绑定实际设备 ID。
相机尽量拍到双肩、左肘、左腕和部分腰部，保持绑带稳固。分别录制静止
约 15 秒、缓慢抬臂、快速抬臂和前臂短时遮挡。结束录制后用回放命令
检查是否同时存在 `imu.csv` 记录、相机画面和关键点；如丢帧数非零，
先排查存储速度和 GUI/推理负载。
