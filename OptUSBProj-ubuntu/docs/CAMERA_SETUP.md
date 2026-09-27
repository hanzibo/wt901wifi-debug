# 单目前置摄像头：上身 7 部位原型准备

目标是先用普通 RGB 摄像头估计躯干、左右上臂、左右前臂、左右手的姿态，驱动现有 PyVista 人体模型。首版不与 WT901 融合，也不控制 CoppeliaSim。模型仍使用固定肢段长度；相机输出的髋部相对关键点不作为精确的房间坐标。

## 1. 独立 Conda 环境

在仓库根目录执行：

```bash
conda create -n OptUSBProj-camera --clone OptUSBProj
conda run --no-capture-output -n OptUSBProj-camera python -m pip install mediapipe
conda run --no-capture-output -n OptUSBProj-camera python -m pip check
conda run --no-capture-output -n OptUSBProj-camera python -c 'import mediapipe as mp, cv2, PySide6, pyvista; print("mediapipe", mp.__version__, "opencv", cv2.__version__)'
```

如最后一步仅因缺少 `cv2` 失败，再执行 `conda run --no-capture-output -n OptUSBProj-camera python -m pip install opencv-python-headless` 并重试。已有的 PySide6 窗口负责显示，不需要 OpenCV 的 `imshow`。不要在同一环境主动混装多个 OpenCV wheel。

## 2. 下载官方模型

仅下载 Pose Landmarker lite 模型文件，无需克隆 MediaPipe 仓库：

```bash
mkdir -p OptUSBProj-ubuntu/assets/mediapipe
curl -fL --retry 3 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task' -o OptUSBProj-ubuntu/assets/mediapipe/pose_landmarker_lite.task
test -s OptUSBProj-ubuntu/assets/mediapipe/pose_landmarker_lite.task
```

`.task` 为本机下载资源，已加入仓库 `.gitignore`。

## 3. 摄像头自检

在用户的 Ubuntu 桌面终端执行；相机可能不会出现在远程或沙箱会话中：

```bash
ls -l /dev/video*
conda run --no-capture-output -n OptUSBProj-camera python -c 'import cv2; c=cv2.VideoCapture(0); ok, frame=c.read(); print("opened", c.isOpened(), "frame", frame.shape if ok else None); c.release()'
```

如果 `opened` 为 `False`，先关闭占用摄像头的应用、检查系统相机权限与设备编号，再试 `VideoCapture(1)`。有些 `/dev/video*` 是辅助节点，须以实际读到视频帧为准。

## 4. 运行纯相机原型

在仓库根目录执行：

```bash
conda run --no-capture-output -n OptUSBProj-camera python -u OptUSBProj-ubuntu/camera_run.py
```

默认使用摄像头 `0` 和上面下载的 lite 模型；其他摄像头可加 `--camera-index 1`。程序不监听 UDP 1399，因此可以与 WT901 接收程序并存，不过它们会分别打开两个模型窗口。

左侧渲染原有 15 个 STL 部件，右侧预览摄像头并显示 7 个上身部位的识别状态。启动后面向摄像头、双臂自然下垂，再点“当前姿势归零”（或按空格），随后分别抬起左右手臂观察是否对应。需要离开电脑摆姿势时，可设置 3～120 秒延时并点击“定时归零”；倒计时可取消。到时仅对摄像头当前新近识别到的部位建立中立位；若未识别到人或相机已停止，原中立位不变。`R` 重置视角；“镜像预览”只改变画面显示，不改变模型控制方向。

若腰/髋关键点不入镜，躯干只能从肩部估计粗略朝向，俯仰/弯腰会不准确；建议尽量拍到肩、双臂和腰部。双手指尖关键点太小或被遮挡时，模型手掌会跟随前臂，不能指望准确估计手腕轴向旋转。首版不估计人体整体空间平移，也不把单目结果用于机器人关节控制。

官方资料：[Python 使用指南](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)、[模型下载页](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)、[PyPI 包](https://pypi.org/project/mediapipe/)。

## 5. MediaPipe / RTMW3D 同窗口对比

GUI 右侧的「姿态后端」下拉框可在 MediaPipe 和 RTMW3D（GPU）之间切换。切换会先停止旧摄像头线程，再启动新后端；中立位会清空，请在相同的站姿下重新归零，才能公平比较动作。RTMW3D 首次加载会下载 YOLOX 和 RTMW3D 模型；已缓存后可复用。建议每次比较时保持相机位置、人与相机的距离、光照及动作尽可能相同。

RTMW3D 下方的「RTMW3D 防抖」默认开启，可随时关闭做同模型 A/B 对比。防抖会缓慢更新肩宽尺度，对 X/Y 和深度 Z 使用不同强度的速度自适应时间滤波，拒绝单帧大跳点，并在关键点短暂低置信度时保持最近可靠值约 0.18 秒。快速动作仍应能跟上，但滤波会带来一定延迟；切换开关后建议等待一秒，再观察静止抖动和挥手响应。这个开关不改变 MediaPipe 路径。

两个后端必须位于同一 Python 环境才能在 GUI 内切换。如果你已经建好 `OptUSBProj-rtmw3d` 环境并完成 RTMW3D GPU 自检，可在该环境补装现有 GUI 与 MediaPipe 依赖：

```bash
conda activate OptUSBProj-rtmw3d
python -m pip install PySide6 pyvista pyvistaqt scipy mediapipe
python -c 'import onnxruntime as ort; ort.preload_dlls(directory=""); print(ort.get_available_providers())'
python -u OptUSBProj-ubuntu/camera_run.py --backend rtmw3d
```

最后一条命令需在仓库根目录运行；也可不加 `--backend`，默认先打开 MediaPipe。若切换时报缺少依赖，先确认终端仍处于 `OptUSBProj-rtmw3d` 环境。模型 `.task` 文件仍须按第 2 节准备好。不要同时运行 `rtmw3d_probe.py` 或其他占用同一摄像头的程序。

RTMW3D 的 133 点不是可直接驱动 STL 模型的关节旋转。当前实现选取躯干、双臂及手部关键点，把裁剪图像中的 X/Y 以肩宽近似归一化，再保留模型深度 Z，供现有 7 部位姿态映射使用。因此它适合比较相对动作，但不是绝对位置或严格的米制三维重建；若转身、手被遮挡或仅有上半身入镜，两种后端都可能失准。
