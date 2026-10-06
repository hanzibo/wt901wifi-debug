# X1 Python 关节可视化控制

这里的 PySide6 面板通过 `rclpy` 发布 `/joint_states`，改变 RViz 中的 16 个关节姿态。当前用户已在 `x1_ros2_py` Conda 环境中跑通这条链路。项目总览见 [根目录 README](../README.md)，结构和开发入口见 [项目现况与开发指南](../docs/项目现况与开发指南.md)。

## 当前推荐运行方式

终端 1 启动 RViz：

```bash
cd /home/hzb/Work/urdf-model
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py use_gui:=false
```

终端 2 启动 Qt 面板：

```bash
cd /home/hzb/Work/urdf-model
source ~/miniconda3/etc/profile.d/conda.sh
conda activate x1_ros2_py
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python script_python/check_environment.py
python script_python/x1_joint_qt.py
```

`use_gui:=false` 关闭卖家的滑块发布器。launch 会先发布一次零位关节状态，让 RViz 在 Qt 面板启动前也能显示完整模型；Qt 面板启动后约以 30 Hz 持续发布关节状态。

## 文件

| 文件 | 用途 |
| --- | --- |
| `x1_joint_qt.py` | 推荐使用的 PySide6 面板：关节滑块、0.1° 角度输入、预设姿态、自动演示 |
| `x1_common.py` | 关节名称、界面范围与预设姿态 |
| `x1_motion_demo.py` | 无界面关节动作示例 |
| `check_environment.py` | 检查当前 Python、ROS 模块、Tk 和 PySide6 |
| `environment.yml` | 新建 Conda 环境的依赖定义 |
| `x1_joint_gui.py` | 旧版 Tk 面板，保留作为备用 |

无界面动作演示可以替代 Qt 面板运行，避免同时开启两个持续发布器：

```bash
python script_python/x1_motion_demo.py --duration 15 --rate 30 --amplitude-deg 20
```

## 新机器的 Conda 环境

本机 ROS 2 Lyrical 的 `rclpy` 是针对 Python 3.14 构建的，因此 Conda 环境也要使用 Python 3.14。新建环境后，激活 Conda，再加载 ROS 和工作区：

```bash
cd /home/hzb/Work/urdf-model
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f script_python/environment.yml
conda activate x1_ros2_py
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python script_python/check_environment.py
```

如果已有 `x1_ros2_py`，但检查提示缺少 PySide6，请手动安装：

```bash
conda install -n x1_ros2_py -c conda-forge pyside6
```

若检查提示缺少 `lark`，请手动安装：

```bash
conda install -n x1_ros2_py -c conda-forge lark
```

只构建 ROS 描述包时，使用系统 Python 所在的终端；不要在激活 Conda 的终端直接运行 `colcon build`：

```bash
cd /home/hzb/Work/urdf-model
source /opt/ros/lyrical/setup.bash
colcon build --packages-select pkg_robot_model
```

## 开发约定与边界

- 面板以度显示角度，发布 `/joint_states` 前换算成弧度。界面范围仅供可视化，不代表真机机械限位。
- 改关节标签、范围或预设姿态，编辑 `x1_common.py`；改界面排版与字体，编辑 `x1_joint_qt.py`。当前字体是 Noto Sans CJK SC 13 磅。
- Qt 面板运行时，用 `ros2 topic info /joint_states --verbose` 检查持续发布者是否只有 `x1_python_joint_qt`。多个持续发布者会让模型来回跳动。
- 此实现不运行 `ros2_control`，不访问串口，不向舵机发送指令，也不模拟重力、碰撞或动力学。
- 将来接入 Gazebo 或真机时，控制指令应交给相应控制器；`/joint_states` 应由仿真或硬件反馈产生。
