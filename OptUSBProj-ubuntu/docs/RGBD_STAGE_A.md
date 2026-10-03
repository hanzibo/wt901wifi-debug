# Gemini 336L：正式 RGB-D / WT901 联合采集（阶段 A）

本阶段只采集、保存和回放数据，**不估计人体关键点、不纠正 IMU 航向，也不让相机驱动人体或机器人**。原普通摄像头入口保留。

## 1. 环境和启动

目前已验证两个环境，不需要再安装依赖：

- `orbbec336l`：Python 3.11、pyorbbecsdk2 2.1.2、NumPy 1.26、OpenCV 4.11；SDK 独立进程。
- `OptUSBProj-fusion`：现有人体 GUI 和联合记录；NumPy 2.x，不直接导入 SDK。

在仓库根目录启动：

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_run.py \
  --camera-source orbbec --rgbd-fps 10 --rgbd-warmup 2
```

相机需直连 USB 3.x，并关闭 OrbbecViewer。关闭手腕 GUI、纯 IMU GUI 或其他占用 UDP 1399 的程序。

本机桌面新增 `WT901-RGBD联合采集.desktop`，双击等价于上述命令（10 FPS 输出上限、2 秒预热）。
仓库模板为 `OptUSBProj-ubuntu/WT901-RGBD联合采集.desktop`，启动脚本为 `launch_rgbd.sh`。
旧的纯 IMU 桌面启动项保持不变；换电脑后须更新 desktop 中的绝对路径。
启动日志位于 `~/.local/state/wt901/rgbd-capture.log`（设置了 `XDG_STATE_HOME` 时使用对应位置）。

解释器路径默认是当前用户的 `~/miniconda3/envs/orbbec336l/bin/python`，可用 `--orbbec-python /absolute/path/to/python` 指定。相机使用 SDK 默认彩色与深度 profile；已在 Gemini 336L 上实测默认 1280×720 彩色、848×480 深度约 30 FPS。默认预热 2 秒，预热帧不输出。

“相机来源”可选择普通摄像头或 Gemini；先停止相机再切换或修改输出帧率。RGB-D 帧率控件是**输出/录制上限**，不是写入设备回传频率，实际输出可能略低于上限。默认 10 FPS，范围 1～30。原普通摄像头仍由 `--backend` 选择 RTMW3D/MediaPipe；阶段 A 的 Orbbec 不需要这些模型。

右侧完整控制栏支持滚动，较矮屏幕上可向下滚动查看设备分配与其他按钮，不会挤压相机预览或重叠控件。

## 2. GUI 操作

1. 启动后自动打开相机，等待预热和首帧。
2. 状态栏显示 RGB-D 时间差、深度有效率、消费者跳过帧数。
3. 可勾选“预览配准深度”；伪彩色只用于显示。
4. 按原方式分配 WT901、选择姿态算法、穿戴标定与归零。
5. 点击“开始联合录制”，或设置延迟/时长后点击“定时录制”。
6. 手动或定时结束都剔除末尾 3 秒，RGB、两种深度、帧元数据和 IMU 同步按主机时标截尾。

Orbbec 尚未输出有效首帧或最近观测过期时，禁止开始 RGB-D 录制。相机异常会结束当前录制并在 manifest 留下错误；手动停止 Orbbec 相机也会先结束录制。不会自动替换成普通摄像头或覆盖缺失深度。

## 3. 跨环境接口与实时性

```text
orbbec336l SDK 子进程
  ├─ 原生 RGB / Y16 采流，帧同步
  ├─ 软件 Depth-to-Color 配准、单位转换
  └─ 私有共享内存槽 + JSON 元数据 + ACK
                ↓ 复制完成后才允许覆盖同一槽
GUI 环境 QThread
  └─ 只保存最新一帧的邮箱（覆盖计数，不无限排队）
                ↓ GUI 定时取最新帧
录制器
  ├─ 末尾 3 秒待定缓冲
  └─ 有界后台写盘队列（16 帧；溢出计数）
```

共享内存优先放在 `/dev/shm` 的私有临时目录。SDK 日志和 stderr 也在临时目录；停止后释放进程和共享槽。不使用图像 UDP 分包，不在 Qt 主线程中解码/配准/写入深度。

这不是硬实时系统。共享内存复制、GUI 忙碌和磁盘吞吐都可能减少保留帧数；计数应随记录检查。默认全分辨率无损深度约需数十 MB/s，10 FPS 的一分钟会话可能达到约 2～3 GB。长时录制先检查磁盘容量，必要时降低输出帧率。不要为“高频 IMU”强行把视觉记录也提高到同频。

## 4. 录制目录和数据契约

GUI 默认记录到 `OptUSBProj-ubuntu/recordings/<时间-ID>/`，已被 Git 忽略。

```text
manifest.json                  格式 wt901-fusion-session-v2，计数与配置
imu.csv                        原设备数据 + 主机收包/写入时间
camera.jsonl                   每帧文件引用、全部时间戳、帧号和采集统计
camera_calibration.json        活动 profile、原始/配准内参、畸变与 D2C 外参
frames/00000001.jpg            RGB 编码为 JPEG，无叠加标注（有损）
depth/00000001_raw.npy          原始 uint16 深度，无损
depth/00000001_aligned_mm.npy   配准到彩色的 float32 毫米深度，无损
```

原始深度需要乘该帧 `raw_depth_scale_mm` 才是毫米；配准文件已乘 `aligned_depth_scale_mm`，不能再乘一次。零值表示没有有效深度，不能当作物体在相机原点。保存的深度不是伪彩色，不使用 JPEG。

每帧元数据包括：

- `color_index`、`depth_index`。
- 彩色/深度的 `device_us`、`system_us`、`global_us`。
- `aligned_device_us`，检查配准是否保留来源时标。
- `host_received_mono_ns`：SDK 返回原始完整帧组时的主机单调时间。
- `host_bridge_mono_ns`：GUI 环境复制共享槽完成的主机单调时间。
- 深度尺度、有效率、源帧号跳跃/不完整帧组计数、采流和输出计数。

为兼容原格式，`captured_mono_ns` 取 `host_received_mono_ns`。**字段名字不代表曝光时间，也不能直接当作严格同步的采样时刻。**设备时钟、global 时钟和 WT901 时钟尚未建立偏置/漂移映射；这是后续时间标定阶段的工作。当前回放只按主机收帧/收包时间排序。

输出限频丢弃的源帧是主动采样，不等同于 USB 丢帧。共享邮箱覆盖、写盘队列溢出、末尾截断分别计数。`camera_last_timing` 是最后提交观测（可能位于剔除尾段），保留数据统计应以 `camera.jsonl` 为准。

当前记录设备分配、轴映射和姿态算法选择，但没有完整序列化陀螺滤波器的内部状态。回放可恢复相机测量和原始 IMU 数据，不应宣称可逐比特复现实时滤波姿态或完整穿戴标定状态。

## 5. 回放

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/fusion_run.py \
  --replay OptUSBProj-ubuntu/recordings/<时间-ID>
```

不打开相机、不占用 UDP。可切换 RGB/配准深度显示。原 v1 单目录制仍可回放；v2 恢复无损深度及标定/时间元数据。阶段 A 的关键点和置信度为 null，不能直接拿现有依赖 2D 关键点的单目分析工具来完成 RGB-D 姿态分析。

## 6. 无 GUI 联合验收

```bash
conda run --no-capture-output -n OptUSBProj-fusion python -u OptUSBProj-ubuntu/rgbd_probe.py \
  --seconds 15 --fps 10
```

默认同时监听 UDP 1399，输出到 `recordings/rgbd/`，每 5 秒报进度。只测相机时加 `--no-imu`；指定临时结果目录时加 `--record-parent /tmp/rgbd-test`。时长必须大于预热加 3 秒尾段。

它只读取设备，不写 IMU 寄存器。没有打开 WT901 时仍能验证相机，但 IMU 包数为零，不能据此宣称完成双设备时间同步验收。

## 7. 实现和后续范围

- `orbbec_process.py`：SDK 子进程、配准和共享槽生产者。
- `orbbec_worker.py`：Qt 进程桥接、最新帧邮箱、进程释放。
- `fusion_capture.py`：RGB-D 数据契约。
- `fusion_recording.py`、`fusion_replay.py`：联合记录与回放。
- `fusion_gui.py`、`fusion_run.py`：现有 UI 和 CLI 集成。

下一步是阶段 B：可信 2D 关键点 + 配准深度形成 3D 观测；随后建立时间/坐标标定，再离线验证航向纠偏。SDK 的帧同步不能替代 RGB-D 与 WT901 的跨设备同步。
