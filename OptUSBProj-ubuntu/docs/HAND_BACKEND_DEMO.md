# 手部后端检测独立 Demo

入口：`OptUSBProj-ubuntu/hand_backend_demo.py`。仅做二维手框/21点对比，不使用IMU、人体腕点裁剪、真实深度验收、穿戴标定或X1。Gemini仍复用现有RGB-D采集桥，但本Demo只使用其中原始RGB图，不估计三维。

## 环境与下载

已检查本机 `OptUSBProj-yolo`：Python3.11、MediaPipe、rtmlib、ONNX Runtime均已安装；ORT列出了CUDA/CPU提供者，但列出CUDA不代表模型会话已经成功使用CUDA。CPU可先用于验收，Demo会显示实际会话提供者；选择CUDA但会话未使用CUDA会报告失败。

模型由用户手动下载，程序不会联网。RTM两个地址来自[rtmlib Hand源码](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/hand.py)。不是人体17点模型，也不是RTMW3D：检测器为RTMDet-nano Hand（320×320），姿态为RTMPose-m Hand5（256×256、21点）。

```bash
mkdir -p /home/hzb/Work/WT901/no-improtent/hand-models/rtm-det
mkdir -p /home/hzb/Work/WT901/no-improtent/hand-models/rtm-pose
cd /home/hzb/Work/WT901/no-improtent/hand-models
wget -c -O rtmdet-hand.zip 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip'
wget -c -O rtmpose-hand.zip 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip'
unzip -n rtmdet-hand.zip -d rtm-det
unzip -n rtmpose-hand.zip -d rtm-pose
```

若OpenMMLab服务器SSL/网络失败，替换为项目镜像：

```bash
cd /home/hzb/Work/WT901/no-improtent/hand-models
wget -c -O rtmdet-hand.zip 'https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip'
wget -c -O rtmpose-hand.zip 'https://huggingface.co/Tau-J/RTMPose/resolve/main/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip'
unzip -n rtmdet-hand.zip -d rtm-det
unzip -n rtmpose-hand.zip -d rtm-pose
```

GUI默认搜索各目录下唯一的 `.onnx`；若压缩包有多个ONNX，请点击“选择”直接指定对应模型。不能把ZIP作为模型输入。MediaPipe使用已下载的 `hand_landmarker.task`。

本机不用重复安装依赖。其他环境缺少rtmlib时，可参考[项目安装说明](https://github.com/Tau-J/rtmlib)，不要把CPU版onnxruntime和GPU版同时装进现有环境。

## 启动与操作

```bash
cd /home/hzb/Work/WT901
source /home/hzb/miniconda3/etc/profile.d/conda.sh
conda activate OptUSBProj-yolo
python OptUSBProj-ubuntu/hand_backend_demo.py
```

1. 先在其他GUI中停止Gemini相机；同一相机不能由两个采集进程同时占用。
2. 默认MediaPipe，检测/存在阈值0.5，始终CPU、IMAGE逐帧模式。点击启动，手掌展开朝相机，无需任何标定。
3. 点击停止，等后端完全退出、控件恢复可用，再选RTM并启动；默认检测阈值0.3、关键点显示阈值0.3。RTM两个阶段均输入BGR，显示RGB。
4. 两后端都默认全图检测，不依赖身体模型。中央50%区域选项用于小目标对照，请把整个手掌放入黄色框；区域坐标会还原到原始图像系。
5. 界面显示手框数、可显示手数、点数、推理时间、帧龄和累计检出占比。RTM一只手至少5点通过显示阈值才绘制；这不是标定验收标准。MediaPipe不提供逐关键点置信度，因此点显示阈值不作用于它，左右标签也不是身体关联验收。
6. 可选择图片离线测试同一张图：选择图片→启动→停止→换后端→再启动。仅推理一次，不占相机。点击“切回Gemini”回到实机输入。

注意：RTMDet未检出手框时不调用RTMPose，防止RTMPose空框回退到全图预测“手”。两模型阈值/分数含义不同，累计可显示占比不是识别精度，没有GT。建议固定距离/光照，每个后端相同姿态各保持10秒，观察正掌、斜掌、侧掌和转腕，分别测试全图/中央区域。

线程只保留最新待推理帧，避免积压；不保存视频或截图，不修改现有人体GUI标定/纠偏。已验证合成输入、空框防护、ROI坐标、RGB/BGR转换、窗口和MediaPipe本地加载；RTM权重推理及Gemini实机仍需下载后测试，不预先宣称效果优于MediaPipe。
