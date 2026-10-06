# X1 机器人模型与 ROS 2 可视化控制

本项目已在 Ubuntu 26.04、ROS 2 Lyrical 和 Wayland 桌面上跑通以下链路：启动 RViz 加载 X1 的 URDF 模型，使用 Conda 环境中的 Python/PySide6 控制面板，通过 `rclpy` 发布 16 个关节角度，在 RViz 中观察姿态变化。

当前实现的是**关节姿态可视化**。`/joint_states` 直接驱动 TF 与模型显示；Gazebo 动力学、MoveIt 轨迹规划和真实机器人电机控制尚未接入这条链路。

## 快速启动

首次使用或修改 `pkg_robot_model/` 后，先用系统 Python 构建描述包：

```bash
cd /home/hzb/Work/urdf-model
source /opt/ros/lyrical/setup.bash
colcon build --packages-select pkg_robot_model
```

终端 1：启动 RViz。`use_gui:=false` 关闭卖家的滑块发布器，避免与 Python 面板同时发布关节状态。

```bash
cd /home/hzb/Work/urdf-model
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py use_gui:=false
```

终端 2：启动 Qt 控制面板。

```bash
cd /home/hzb/Work/urdf-model
source ~/miniconda3/etc/profile.d/conda.sh
conda activate x1_ros2_py
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python script_python/check_environment.py
python script_python/x1_joint_qt.py
```

现有 `x1_ros2_py` 环境已验证可导入 PySide6、`rclpy` 与 `sensor_msgs`。新机器可参考 [Python 控制说明](script_python/README.md) 创建环境。

WT901 离线数据到 X1 关节轨迹、再到 RViz 回放的实验见 [IMU 右臂离线回放](docs/IMU右臂离线回放.md)和 [IMU 双臂同步离线回放](docs/IMU双臂离线回放.md)。两者只预览机器人姿态，不调用真机控制接口。

复用人体 STL 主界面的在线标定状态驱动双臂 RViz 预览，见 [IMU 双臂实时预览](docs/IMU双臂实时预览.md)；同样不接真机。

## 数据流

```text
PySide6 控制面板 → rclpy → /joint_states → robot_state_publisher → /tf → RViz RobotModel
                                     URDF ────────┘                     ↑
                                     /robot_description ───────────────┘
```

启动文件还会在 RViz 就绪后发布一次全部关节的零位状态，让模型在 Qt 面板启动前也能完整显示；随后该临时发布器退出。

## 项目目录

| 目录 | 用途 |
| --- | --- |
| `pkg_robot_model/` | URDF、STL 网格、RViz 配置和当前可用的显示启动文件 |
| `script_python/` | PySide6 面板、`rclpy` 发布器、动作示例和 Conda 环境定义 |
| `pkg_robot_model_moveit_config/` | 卖家提供的 MoveIt 配置，目前尚未在本机验证运行 |
| `xiaobei_hardware/` | 卖家提供的舵机与硬件接口，目前未适配本机 ROS 2 版本 |
| `docs/` | 项目现况和环境排障记录 |
| `build/`、`install/`、`log/` | `colcon` 生成目录；修改源码后按需重新构建 |

详细现况与开发入口见 [项目现况与开发指南](docs/项目现况与开发指南.md)，Wayland/RViz 的历史排障记录见 [环境审查与 RViz 排障指南](docs/环境审查与RViz排障指南.md)。
