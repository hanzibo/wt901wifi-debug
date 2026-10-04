# RGB-D 二维模型对比：RTMW / YOLO26

只替换二维关键点来源，复用Gemini真实深度、穿戴标定、相机标定和航向纠偏。只接受单人画面；可用率不是准确率，没有GT关节中心真值。

## 启动

新图标「WT901-RGBD-YOLO对比」使用OptUSBProj-yolo。原RGB-D图标也在检测到完整YOLO环境后优先使用OptUSBProj-yolo，否则回退OptUSBProj-fusion；默认二维模型仍为RTMW。纯IMU入口不变。不能同时占用相机或UDP1399。

```bash
cd /home/hzb/Work/WT901
conda run --no-capture-output -n OptUSBProj-yolo python -u OptUSBProj-ubuntu/fusion_run.py --camera-source orbbec --rgbd-pose --pose-backend yolo26 --pose-imgsz 640
```

控制栏选RTMW/YOLO26、YOLO输入640/960；「YOLO权重…」选本地姿态.pt。默认权重是 `/home/hzb/Work/WT901/no-improtent/yolo-models/yolo26m-pose.pt`。不自动下载、不回退CPU。普通摄像头路线不变。

## 切换安全

- 录制/预约、引导标定/分析、IMU穿戴标定窗口、X1随动期间禁止切换。
- 切换结束旧工作线程，清除旧结果/深度时序状态，只处理最新帧。
- 保留穿戴映射；旧相机纠偏停用，已有修正冻结保留，IMU继续传播，不突然归零。
- 更换权重/输入尺寸也需重做相机标定。报告绑定权重SHA256、参数签名、视觉epoch。切回同名模型不能恢复旧会话标定。
- 重做相机标定后，切回陀螺＋加计并明确应用，再逐部位启用RGB-D。不自动启动机器人。
- 无视觉配置的旧报告只允许本次一直保持默认RTMW的会话按旧规则应用；YOLO或已切换会话拒绝。
- 回放显示原模型/观测，不在线重新推理。

## 诊断与接口

「三维 / 可靠性详情」新增二维可用列。二维可用要求坐标有限、在原彩色图内、分数达门限；三维失败原因包括深度支持、混合边缘、骨长/比例、跳变。

YOLO17点适配现有133点数组，其余NaN/零分数，不伪造手指/面部。肩肘腕髋索引仍为5–12。坐标还原到原始彩色像素再查真实深度；RTMW预测三维坐标继续丢弃。两种模型NumPy输入均为BGR。

## 同帧离线对比

原录制需保留彩色、原始/配准深度、camera.jsonl和camera_calibration.json。输出只有JSONL、报告、最多10张截图，不复制深度、不覆盖原录制。输出目录必须尚不存在：

```bash
cd /home/hzb/Work/WT901
conda activate OptUSBProj-yolo
python OptUSBProj-ubuntu/rgbd_analyze.py OptUSBProj-ubuntu/recordings/20261003-174504-0ab79b --output OptUSBProj-ubuntu/recordings/analysis_yolo/rtmw-mytest --pose-backend rtmw --every 5 --max-frames 160
python OptUSBProj-ubuntu/rgbd_analyze.py OptUSBProj-ubuntu/recordings/20261003-174504-0ab79b --output OptUSBProj-ubuntu/recordings/analysis_yolo/yolo-mytest --pose-backend yolo26 --pose-imgsz 640 --every 5 --max-frames 160
python OptUSBProj-ubuntu/rgbd_compare.py OptUSBProj-ubuntu/recordings/analysis_yolo/rtmw-mytest OptUSBProj-ubuntu/recordings/analysis_yolo/yolo-mytest
```

原录制已删除时换新目录。每5帧取1帧；全部分析用`--every 1 --max-frames 0`。GPU分析顺序执行，避免争用污染耗时。

报告统计全帧和引导侧举(lateral/lateral_repeat hold)窗口的逐点二维/三维可用率、连续失败帧数/采样首尾跨度、失败原因。分母是处理帧数，不是全部采集帧；抽帧无法推断未处理帧或精确遮挡时长，两个侧举窗口不串联失败。耗时中位/P95包含颜色转换、推理、坐标传输、深度处理；不含加载/磁盘读取，前5帧排除耗时但仍计可用率。比较器要求完全相同源帧/顺序/门限。不同模型置信度未必同尺度，不能单独降低YOLO门限宣称更优。

## 验收

固定相机、绑带、灯光，确保两腕侧举完整入镜。先看二维叠加，再查三维详情；录制中立→前举→T字侧举→屈肘→中立。同帧比较后确定后端，重做相机标定并至少运行5分钟。先验收人体模型，再启用RViz。不包括模型重训、手指姿态、精确关节中心恢复或真机控制。
