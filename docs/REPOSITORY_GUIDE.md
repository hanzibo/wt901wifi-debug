# 仓库与开发工具导航

## 当前目录边界

| 目录 | 内容 | 管理方式 |
| --- | --- | --- |
| `wt901/` | 设备协议、数据源与配置通信 | 两条 GUI 共用，不单独抢占端口 |
| `wt901_viz/` | 手腕 GUI、姿态滤波及数学工具 | 保留独立调试入口 |
| `OptUSBProj-ubuntu/optusbproj_ubuntu/` | 人体、标定、融合、机器人应用 | 业务代码；`core/` 是共享数学 |
| `OptUSBProj-ubuntu/*.py` | GUI、分析和实验兼容入口 | 保留原命令，不一次性搬迁 |
| `scripts/` | 统一启动、检查及测试 | 默认不接触硬件；启动 GUI 必须明确执行入口 |
| `requirements/` | 分层依赖声明 | 不是整机锁定文件，也不包含权重 |
| `tests/` / `OptUSBProj-ubuntu/tests/` | 核心 / 应用离线测试 | 统一测试器分进程运行 |
| `docs/` / `OptUSBProj-ubuntu/docs/` | 数学、操作与验收 | 根 README 及本页作为导航，不机械合并历史文档 |
| `urdf-model/` | X1 模型、ROS 桥与相关说明 | 保留资源路径；RViz 不等于硬件控制 |
| `robot-urdf-model-new/` | 新机器人场景与手姿配置 | 大场景按当前本地忽略规则保存，不自动上传 |
| `demos/` / `experiments/` | 独立验证 / 研究实验 | 不接入默认启动，不为了清洁而删掉 |
| `OptUSBProj/` / `no-improtent/` | 本机历史参考及模型等 | 已忽略，不参与默认开发工具打包 |

录制、标定、本机设备配置、权重及工具缓存不应混入源码提交。
大文件检查只报告，不删除或自动加入忽略规则。

## 从任意工作目录使用

无需安装，直接运行绝对路径：

```bash
python /home/hzb/Work/WT901/scripts/run_project.py --list
python /home/hzb/Work/WT901/scripts/run_project.py --dry-run --env OptUSBProj-yolo fusion -- --camera-source orbbec --rgbd-pose
python /home/hzb/Work/WT901/scripts/check_repo.py
python /home/hzb/Work/WT901/scripts/run_tests.py --core-env wt901 --app-env OptUSBProj-yolo
conda run --no-capture-output -n OptUSBProj-yolo python /home/hzb/Work/WT901/scripts/run_smoke.py
```

实际打开融合 GUI（先关闭占用相机、UDP 1399 的其他程序）：

```bash
python /home/hzb/Work/WT901/scripts/run_project.py --env OptUSBProj-yolo fusion -- --camera-source orbbec --rgbd-pose
```

`--env` 必须放在入口名字之前；`--` 后的参数原样交给旧入口。
默认使用当前 Python。相对 `--replay`、`--settings` 路径仍相对于调用者目录；
工具不会为了自己定位仓库而悄悄改变其含义。入口本身使用绝对路径。
旧启动器和旧命令仍有效，不要求更换桌面图标。

## 可选的开发工具安装

仅在你希望获得短命令时安装，不必为此次验收改动当前环境：

```bash
python -m pip install --no-deps -e /home/hzb/Work/WT901
wt901-project --list
wt901-check
wt901-smoke
wt901-tests --core-env wt901 --app-env OptUSBProj-yolo
```

`pyproject.toml` 当前仅将 `scripts/` 开发工具打包为 `wt901_tools`（避免安装一个通用名称 `scripts` 的包），明确不打包业务包、STL、URDF、
TTT、录像及权重。使用 editable 安装才能让检查/测试短命令定位源码仓库；
这不是把完整应用发布成可独立安装的 wheel。非 editable 工具的启动器可显式
使用 `--repo`，但完整测试仍应在源码检出中运行。暂不承诺 pip 安装即能跑全链路。

## 依赖分层

`requirements/core.txt` 支持协议、数学及 FK/IK 冒烟测试；`gui.txt` 增加 Qt/VTK
人体界面；`scene.txt` 增加远程客户端；`collision.txt` 属于暂停中的可选实验。
这些文件是依赖范围声明，不是所有模块都测试过的完整最低版本兼容矩阵。
安装它们可能更换已有依赖，因此已可用的 Conda 环境不要因整理仓库而重新安装。

本次本机读取到的版本为 NumPy 2.4.6、SciPy 1.17.1、PySide6 6.11.2、
PyVista 0.49.0、PyVistaQt 0.13.1、CoppeliaSim 远程客户端 2.0.4；
这是当前环境证据，不是完整锁定文件或对其他版本的保证。

视觉后端/权重、ONNX CPU/GPU、Torch/CUDA、Orbbec SDK、ROS 和 CoppeliaSim
程序分别按现有专题文档部署，不混入无 GPU 的基础测试依赖：

- [相机环境](../OptUSBProj-ubuntu/docs/CAMERA_SETUP.md)
- [Orbbec RGB-D](../OptUSBProj-ubuntu/docs/RGBD_STAGE_A.md)
- [YOLO 后端](../OptUSBProj-ubuntu/docs/YOLO_RGBD_BACKEND.md)
- [新机器人链路](REFACTOR_PHASE1_ACCEPTANCE.md)

## 检查与 CI 的边界

`check_repo` 对 Git 管理及未忽略的新 Python 文件做 AST 语法检查，检查入口存在
和当前 diff 空白错误，报告大资源。它不是完整静态类型检查，也不检查 STL 装配。
Ruff 初期仅启用致命语法/错误规则，对 `scripts/` 和工具测试设门禁，不批量格式化
历史业务代码；后续可逐模块扩大，不把数千条历史风格问题混进功能提交。

GitHub workflow 使用 Python 3.11 和 `.[test]`，运行检查、Ruff 与离线冒烟套件；
不安装 Qt、视觉模型、ROS，不启动仿真器。CI 不能替代本机全量测试及手动验收。
本机实现不等于远程 Actions 已成功，须推送后看其运行结果。

## 重构历史

| 阶段 | 验收说明 | 已验收提交 |
| --- | --- | --- |
| 0 | [基线与统一测试](REFACTOR_PHASE0_BASELINE.md) | `deab1c2` |
| 1 | [新机器人运行服务](REFACTOR_PHASE1_ACCEPTANCE.md) | `c55f392` |
| 2 | [融合协调器](REFACTOR_PHASE2_ACCEPTANCE.md) | `1f77e42` |
| 3 | [人体窗口与穿戴流程](REFACTOR_PHASE3_ACCEPTANCE.md) | `def3645` |
| 4 | [共享数学核心](REFACTOR_PHASE4_ACCEPTANCE.md) | `ad1c1b1` |
| 5 | [仓库与工具](REFACTOR_PHASE5_ACCEPTANCE.md) | 2026-10-12 已手动验收，本次提交 |

阶段 0–5 是这一轮低风险整理，不意味着所有巨型模块均已拆完。
后续可继续对 `pose`、标定实现、视觉跟踪等逐块重构，必须新增针对性回归证据；
未完成的碰撞精度工作仍冻结，不作为本轮自动恢复的开发任务。
