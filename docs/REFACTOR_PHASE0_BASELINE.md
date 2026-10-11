# 重构阶段 0：测试入口与行为基线

阶段 0 只建立回归保护，不改变遥操作算法、限速、坐标偏置、标定阈值或通信恢复策略。
基线源版本为 `d994958`，实施日期为 2026-10-11。

## 1. 统一入口

```bash
python /home/hzb/Work/WT901/scripts/run_tests.py --core-env wt901 --app-env OptUSBProj-yolo
```

该命令分别在两个子进程运行根目录 `tests/` 与 `OptUSBProj-ubuntu/tests/`。
入口自动设置仓库导入路径和子进程工作目录；没有预设 Qt 平台时使用 `offscreen`。
运行某一组失败后仍会运行其他组，最后以非零退出码报告失败。
指定的环境或依赖不存在时报告失败，不自动安装、不悄悄跳过。

如果使用同一套完整 Python 环境，可省略两个环境选项。
旧手腕测试需要 pyqtgraph；本机 `wt901` 已配置它，但 `OptUSBProj-yolo` 没有。
这也是测试分组必须能够选择环境的原因。

快速检查与单文件检查：

```bash
conda run --no-capture-output -n OptUSBProj-yolo python /home/hzb/Work/WT901/scripts/run_tests.py --suite scene
conda run --no-capture-output -n OptUSBProj-yolo python /home/hzb/Work/WT901/scripts/run_tests.py --suite app --pattern test_fusion_capture.py --verbose
```

`--suite core|app|scene|all` 选择范围，默认 `all`。
`scene` 是 app 的子集，不应再次加到完整测试总数中。
`--verbose` 展示具体用例。
本入口不发现 `demos/` 中的真实仿真联调脚本；它们需用户单独授权执行。
现有 app 测试含本地模型加载，因此完整套件可能需要 GPU、模型缓存和一到两分钟。
这里尚未把快测与模型集成测试全面拆开；后续工程整理阶段再处理。

## 2. 测试夹具修复

原 `test_udp_source_timestamp_keeps_legacy_poll_compatible` 用 `object.__new__` 绕过构造函数，手工填写几个字段。
诊断功能添加 `_diagnostics` 后，此夹具不完整，导致 AttributeError。

现在只替换 socket 工厂，再调用真实 `GuiUdpSource` 构造函数。
测试检查绑定参数、非阻塞设置，并注册资源清理。
这样新增构造字段不会再次要求测试复制生产对象的内部布局。
生产 UDP 接收代码没有改变，测试也不绑定真实的 1399 端口。

## 3. 冻结数值样本

- 文件：`OptUSBProj-ubuntu/tests/fixtures/scene_retarget_baseline.json`。
- 用例：`OptUSBProj-ubuntu/tests/test_scene_baseline.py`。
- 类型：合成双臂几何的特征化测试，不是真实机器人场景，也不是真实 IMU 回放。
- 输入：固定关节几何、初始角、肘腕位置、手掌与骨段朝向、固定步长。
- 覆盖：左右两侧、`staged` 与 `joint` 两种求解方式，各四帧，共 16 次求解。
- 动作顺序：中立位、臂部动作、相同臂部目标下转腕、返回中立目标。

每帧保存旧版本输出角、骨段端点误差和手掌朝向误差。
测试按顺序使用上一帧求解结果作为下一帧初值。
同时检查单步角度边界。它不是收敛到目标的长时间运动测试，也不是网络回放。

数值容差：角度 `1e-5 rad`，端点误差 `1e-6 m`，朝向误差 `1e-3 deg`，相对容差为零。
这些容差用于检测同一模型/输入的实现变化，不是传感器或机器人精度指标。
不同 SciPy 版本可能产生不同局部优化结果；升级依赖时必须显式审查，不能自动刷新期望值。

本次冻结环境：Python 3.11.16、NumPy 2.4.6、SciPy 1.17.1、PySide6 6.11.2、PyVista 0.49.0。
Coppelia Python 客户端为 2.0.4；数值样本本身不连接 CoppeliaSim。

## 4. 沿用的生命周期保护

现有 `test_scene_robot.py` 与 `test_scene_hand.py` 已覆盖：

- 求解期间撤销授权，丢弃旧结果；过期快照不发送。
- 短暂 IMU 断流等待与恢复，长期断流停止，参考改变后不自动恢复。
- RPC 超时后不重放结果不确定的写入，恢复时检查场景身份。
- 手腕动作不反过来改变分阶段求解的肩肘结果。
- 手指输入租约、双侧独立授权、停止保持与配置确认。

阶段 1 必须保留这些用例，不应因模块拆分而减少覆盖。
线程、消息类型或日志器改变时，应增加新接口测试，再迁移旧测试。

## 5. 验收与边界

本阶段的自动化验收结果见下方记录。另已从 `/tmp` 调用绝对路径入口，验证不依赖启动目录。
`git diff --check` 用于检查补丁空白问题。

2026-10-11 本机验收：

| 测试组 | 环境 | 结果 |
|---|---|---|
| 根目录核心/旧手腕 | wt901 | 39 项通过，无跳过 |
| 人体/RGB-D 全套 | OptUSBProj-yolo | 442 项通过，无跳过，测试约 104 秒 |
| 新机器人快测（上述全套子集） | OptUSBProj-yolo | 40 项通过，测试约 1.2 秒 |
| 修复的融合采集测试（全套子集） | OptUSBProj-yolo | 5 项通过 |
| 统一入口自身测试（核心组子集） | wt901 | 5 项通过 |

完整两组共 481 项，通过统一入口执行；后面的子集不能重复计数。
未指定环境时，在 `OptUSBProj-yolo` 中执行旧手腕测试会因缺少 pyqtgraph 失败。
此环境差异已通过显式分组选环境解决，没有新增依赖或把失败改成跳过。

自动化基线不代表真实设备精度、实时帧率或桌面渲染通过。
没有启动真实 CoppeliaSim、ROS 或硬件控制。
真实遥操作录制的黄金回放尚未建立；后续改动应用协调层时，仍需同一场景与配置下的人工演示回归。
本次保持已有 `hand_poses.json` 用户修改，不将其作为自动化测试样本。

不要在重构后批量重新生成冻结输出，让新实现给自己生成期望值。
出现差异应先定位坐标、初值、任务权重、限速、授权时序或依赖版本。
