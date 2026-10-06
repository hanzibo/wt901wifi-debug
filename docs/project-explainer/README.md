# 图像版项目讲解

打开 [index.html](index.html)。HTML、`assets/` 必须放在一起，图片与公式离线可用。
主页面由 answer-me-with-html 生成；可探索架构图复用 archify 上一版已验收产物。
MathML、响应式图像对照和放大操作是主渲染器没有内建组件的少量 HTML 扩展。

## 素材性质

- 真实观测：`20261004-135948-f649c2` 的少量 RGB、真实配准深度、原关键点。
- 新推理：同记录侧举帧用本地 YOLO26-l / 960 推理，未重新标定。
- 计算重建：该次穿戴 B 和配对 gyro_tilt 中立 N、同期原始姿态，经现有 PoseEngine 渲染 STL。
- X1：当前 LiveRetargeter 模仿模式、统一长度比例、20 Hz 热启动；同 URDF 和 FK，PyVista 无头渲染。不是历史关节输出，也不是 RViz 截屏。手部跟随关闭。
- 历史 RViz：9 月 28 日录屏中约 2 秒位置提取，与上述记录不同时。
- 原理示意：同倾斜、不同航向的合成向量，不代表实测漂移。
- 验收表：`20261004-144730-de9b33-0ab2c0` 报告，原始图像已删除；训练残差不当作 GT。
- 曲线：`20261004-103131-25da07-guided-b88d89/comparison.jsonl`，同数据短记录离线残差，包含训练片段，不作为独立长期验收。

所有原始输入只读，无监听端口、相机启动、标定状态写入或实体机器人控制。
图片含人脸和房间环境，仅保存本地。未做匿名化或上传。

## 复现

```bash
cd /home/hzb/Work/WT901
QT_QPA_PLATFORM=offscreen /home/hzb/miniconda3/envs/OptUSBProj-yolo/bin/python docs/project-explainer/build_assets.py
node /home/hzb/.codex/skills/answer-me-with-html/scripts/am.mjs render docs/project-explainer/index.am.md -o docs/project-explainer/index.html --no-open
```

脚本需要现有录制目录及依赖，不会自动下载。重建依赖当前工作树，未来修改算法可能改变输出。
素材哈希与来源见 `assets/provenance.json`。
历史截屏和架构图是另外提取／复制的资产；脚本重建其他图片，不会删除它们。

## 数学记号

R：设备本体系 S 相对于固定软件参考系 W 的姿态。
Q：每个设备 W→相机 C 的固定对齐；k_C=Qk_W。
b_S：设备本体系中的固定骨段方向；不等于陀螺零偏 b_gyro。
B：中立设备系→人体模型系的运动换基。
N：与 R 同模式、同参考系的中立／归零姿态。
R_f：左乘固定系航向补偿后的输出；不覆盖原积分器 R。

方向观测不足以恢复完整姿态或骨段长轴扭转。模型合法性、拟合通过和真值精度是不同概念。
