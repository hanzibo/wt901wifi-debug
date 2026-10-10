# 新机器人 CoppeliaSim 关节验证 Demo

独立于 X1 GUI、ROS、IMU 与碰撞模块。原 `.ttt` 只加载，不保存。

先启动 CoppeliaSim 并打开场景，保持仿真停止：

```bash
/home/hzb/Software/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/coppeliaSim.sh /home/hzb/Work/WT901/robot-urdf-model-new/NowRobotWithHand1.ttt
```

然后启动滑杆窗口：

```bash
cd /home/hzb/Work/WT901
conda run --no-capture-output -n OptUSBProj-yolo python demos/coppeliasim_joint_demo.py
```

默认连接 `127.0.0.1:23000`，支持 `--host`、`--port`。客户端依赖为官方 `coppeliasim-zmqremoteapi-client` 和 PySide6。

- 自动读取旋转关节，按双臂、左手、右手分为三个页签；当前场景为 12＋15＋15 个关节。
- 初始目标采用读取到的角度，不自动归零。勾选“允许写入”后才能使用滑杆和数值输入。
- 显示目标角与实际读回角；滑杆精度 0.1°，数值输入精度 0.01°。
- “恢复连接时角度”恢复此次连接时的姿态，不是强制全零。恢复操作只修改内存中的场景，不保存文件。
- 从属关节不可独立写入；场景配置区间不是实机安全限位。
- 仿真开始运行、网络错误或对象路径失效时停止控制，取消写入；停止仿真后重新连接。
- 后台线程处理通信，写入调度上限 30 Hz，实际角读回 5 Hz。采用有界最新值邮箱，不积压过时拖动事件。
- 每批通过 `sim.executeLuaCode` 在仿真端一次完成停止态检查、对象检查、全批参数检查与写入；不增添持久场景脚本。全批验证通过后才开始写角度。意外运行时写入错误不提供事务回滚。
- 调用耗时超过调度周期时自动降低有效频率，不补发过时帧。窗口显示近期实际写入频率与最近 RPC 耗时；闲置时写入频率为零，不等于渲染帧率。
- 网络调用超时约 2.5 秒。取消勾选会清空最新值邮箱，但不能撤回已经发送的批次。
- 禁用写入不保证撤回已在执行的网络调用；关闭窗口不会自动还原角度或关闭 CoppeliaSim。

本版直接调用 `sim.setJointPosition`，仅验证运动学摆姿。当前场景关节均为动态模式，但在停止状态下摆姿不验证动态控制器、接触或力矩。请勿运行仿真时同时用此 Demo 写角度。后续动态控制需要审查脚本与控制模式，再使用目标位置接口。

验证命令：

```bash
conda run --no-capture-output -n OptUSBProj-yolo python -m unittest discover -s demos -p 'test_coppeliasim_joint_demo.py'
conda run --no-capture-output -n OptUSBProj-yolo python demos/coppeliasim_joint_demo.py --smoke-test
QT_QPA_PLATFORM=offscreen conda run --no-capture-output -n OptUSBProj-yolo python demos/coppeliasim_joint_demo.py --gui-smoke-test
conda run --no-capture-output -n OptUSBProj-yolo python demos/coppeliasim_joint_demo.py --benchmark-test
```

实际接口测试会暂时将左肘 `jointL4` 增加 5°，在 finally 中恢复；请保持场景停止、不与其他控制程序共用。若仿真在测试中途开始，保护会拒绝还原，应人工停止后检查姿态。GUI 测试读取关节但不写入。

本版本没有新机器人 IK、人体映射、动力学遥操作或碰撞保证。

性能测试使用 60 帧全部独立关节批量写入，写回当前值以避免改变姿态，每六帧读回并校验。一次本机短测为 24.0 Hz，RPC 平均 21.1 ms、P95 61.9 ms；当时同时执行 GUI 读取测试，结果不是稳定帧率保证，也不是渲染 FPS。场景复杂度、撤销历史和渲染负载均可能影响通信周期。
