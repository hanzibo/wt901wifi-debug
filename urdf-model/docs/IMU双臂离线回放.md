# WT901 → X1 双臂同步离线回放

本阶段只发布 `/joint_states` 供 RViz 预览；不连接实体 X1、串口或 `ros2_control`。右臂的单独入口与轨迹仍然保留，见 [右臂说明](IMU右臂离线回放.md)。

## 数据与限制

默认同一场标定采集 `OptUSBProj-ubuntu/calibrations/20260926-174220-4304e9/` 中有四个 IMU：左上臂 `WT5500011624`、左前臂 `WT5500011433`、右上臂 `WT5500011572`、右前臂 `WT5500011408`。程序逐份核对设备 ID、标定状态与采集目录，拒绝混用另一场穿戴标定的矩阵。两臂在同一电脑接收时间轴上计算；一侧断流只保持该侧，不拖停另一侧。

本次采集没有有效躯干或手背数据，故假定躯干固定、机器人腕部两个关节接近中立。左前臂标定虽为“通过”，其侧举误差为 **17.6°**，超过标定界面的建议值 15°；躯干是否保持固定也无法独立核验。CSV 中的 IK 误差只衡量机器人对**推算目标**的拟合，并不能消除这些输入不确定性。

## 生成双臂轨迹

在 `/home/hzb/Work/WT901`：

```bash
conda run --no-capture-output -n OptUSBProj-fusion \
  python -u OptUSBProj-ubuntu/x1_dual_offline.py
```

输出在同一采集目录内：`x1_dual_trajectory.csv` 与 `x1_dual_trajectory.json`。CSV 含同一时间戳下的左臂 `1`～`6`、右臂 `51`～`56`、各侧目标/实际肘腕坐标、骨段夹角、误差与断流标记；JSON 汇总每侧质量和标定注意事项。目录是本机采集数据，已被 Git 忽略。

## RViz 预览

先在终端 A 启动模型：

```bash
cd /home/hzb/Work/WT901/urdf-model
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py use_gui:=false
```

再在终端 B 回放：

```bash
cd /home/hzb/Work/WT901/urdf-model
source /home/hzb/miniconda3/etc/profile.d/conda.sh
conda activate x1_ros2_py
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python script_python/x1_imu_replay.py --dual
```

只校验文件可加 `--dry-run`，半速观察可加 `--speed 0.5`。默认不加 `--dual` 时仍回放原右臂轨迹；也可以显式传入任意右臂或双臂轨迹 CSV 路径。回放前关闭 X1 关节 Qt 面板和卖家滑块 GUI，避免多个持续 `/joint_states` 发布者争夺 RViz 姿态。

## 已量化的第一轮结果

同一场 25.47 秒记录生成 764 帧。静止窗口内，左臂两次前举的肘关节 `4` 约为 **0.9°、0.8°**，侧举约 **2.3°**；右臂 `54` 仍约 **1°**。左臂前举检查时刻的手腕目标误差约 **1.5–1.6 cm**，侧举检查时刻约 **0.6 cm**；右臂与单侧版本几乎一致。两侧手腕关节中心的最小距离约 **24 cm**，但此值不是完整的网格自碰撞检测。左臂侧举视觉方向应以 RViz 人工检查为准，尤其留意左前臂标定的 17.6° 提示。

之后若要在线接设备，应增加真机所需的独立关节限位、速率/加速度限制、断流处理、碰撞检查和硬件反馈链路；本 CSV 与 RViz 发布器都不是实体机器人命令接口。
