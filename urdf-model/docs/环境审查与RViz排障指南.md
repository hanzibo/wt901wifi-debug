# X1 机器人资料包环境审查与 RViz 排障指南

当前可运行链路和开发入口请先看 [项目现况与开发指南](项目现况与开发指南.md)；本文主要保留环境审查与历史排障过程。

> 最后验证日期：2026-09-28  
> 工作目录：`/home/hzb/Work/urdf-model`  
> 适用机器：Ubuntu 26.04、ROS 2 Lyrical、GNOME Wayland、双屏混合缩放环境

本文记录 X1 机器人资料包的目录结构、ROS 环境审查结果、当前可实现的功能，以及 RViz 模型不可见、3D 面板闪烁、字体太小和帧率不足等问题的处理经验。

卖家提供的 Word 教程属于待分析资料，其中的 `sudo apt install ros-humble-*` 等命令仅代表卖家原始环境，不是本文针对当前系统推荐直接执行的命令。

## 1. 结论速览

当前已验证：

- URDF/Xacro 能正确解析。
- RViz 能显示完整机器人模型。
- `joint_state_publisher_gui` 能通过滑块改变全部 16 个可动关节的可视姿态。
- Wayland 下的 RViz 3D 面板闪烁已通过固定 Qt 缩放解决。
- RViz 字体已单独放大，不会改变 Ogre 渲染面的像素比例。
- 已恢复 Intel Mesa 硬件 OpenGL，RViz 目标帧率为 30 FPS。

当前尚未完成：

- 现有 Gazebo 启动代码面向 Gazebo Classic，不能直接用于本机的现代 Gazebo。
- MoveIt 尚未安装。
- 真机硬件插件面向较旧 ROS 2 API，在 Lyrical 上不能直接编译。
- 现有轨迹控制器只覆盖双臂与夹爪共 14 轴，不包含头部 `101`、`102` 两轴。
- 真机校准值和方向包含推测值，未完成安全校准前不要连接机器人执行动作。

## 2. 快速启动 RViz

每次打开新终端后执行：

```bash
cd /home/hzb/Work/urdf-model
conda deactivate
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py
```

如果源码或 RViz 配置有修改，先重新构建描述包：

```bash
cd /home/hzb/Work/urdf-model
conda deactivate
source /opt/ros/lyrical/setup.bash
colcon build --packages-select pkg_robot_model
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py
```

不要直接运行不带 `--packages-select` 的完整构建来验证 RViz，因为当前 `xiaobei_hardware` 包在 ROS 2 Lyrical 下仍会构建失败。

## 3. 资料包目录说明

```text
urdf-model/
├── pkg_robot_model/                 # 机器人描述、网格、RViz/Gazebo 启动文件
│   ├── config/
│   │   ├── display.rviz             # 当前使用的 RViz 显示、相机和帧率配置
│   │   ├── my_controllers.yaml      # ros2_control 控制器配置
│   │   └── joint_names_*.yaml
│   ├── launch/
│   │   ├── display.launch.py        # 当前可用的 RViz 启动入口
│   │   ├── gazebo.launch.py         # Gazebo Classic 启动入口，当前系统不可直接用
│   │   └── real_robot.launch.py     # 真机入口，未经校准不要运行
│   ├── meshes/                      # STL 外观和碰撞网格
│   ├── urdf/                        # URDF、Xacro、SolidWorks 导出 CSV
│   └── worlds/clean_room.world      # Gazebo Classic 世界
├── pkg_robot_model_moveit_config/   # MoveIt 规划与控制器映射
├── xiaobei_hardware/                # 串口总线舵机 SDK 和硬件接口
├── docs/                            # 项目经验文档
└── 第 3 章 ROS 中URDF与数字孪生.docx # 卖家教程，目标环境为 Humble
```

辅助文件：

- `pkg_robot_model/export.log` 是 SW2URDF 导出日志，不参与运行。
- `pkg_robot_model/urdf/pkg_robot_model.csv` 保存质量、惯量和关节导出数据。
- `pkg_robot_model/urdf/red_box.urdf` 是简单测试物体，不是机器人主体。
- `pkg_robot_model/textures/` 当前为空。
- `build/`、`install/`、`log/` 是 `colcon` 生成目录。

## 4. 机器人模型审查结果

模型统计：

| 项目 | 数量/说明 |
| --- | --- |
| Link | 23 |
| Joint | 22 |
| 可动关节 | 16 |
| 左臂 | `1`～`6` |
| 左夹爪 | `7` |
| 右臂 | `51`～`56` |
| 右夹爪 | `57` |
| 头部 | `101`、`102` |

所谓“7 轴”是指每侧 6 个手臂关节加 1 个夹爪关节。整机模型还包含两个头部自由度。

已执行并通过：

```bash
xacro pkg_robot_model/urdf/pkg_robot_model.urdf.xacro use_sim:=true
xacro pkg_robot_model/urdf/pkg_robot_model.urdf.xacro use_sim:=false
check_urdf <展开后的 URDF>
```

所有实体 Link 均有 `visual`、`collision` 和 `inertial` 数据，STL 文件齐全。

### 4.1 RViz 滑块控制的含义

`display.launch.py` 启动：

- `robot_state_publisher`
- `joint_state_publisher_gui`
- `rviz2`

滑块改变的是 `/joint_states`，随后由 `robot_state_publisher` 计算 TF。这属于运动学姿态预览，不包含重力、碰撞响应、执行器动力学或轨迹控制，不能等同于 Gazebo 物理仿真。

## 5. 当前系统环境审查

检测时的环境：

| 项目 | 当前值 |
| --- | --- |
| 操作系统 | Ubuntu 26.04.1 LTS |
| 内核 | Linux 7.0.0-34-generic |
| 桌面会话 | GNOME Wayland |
| ROS | ROS 2 Lyrical |
| CPU 图形 | Intel Raptor Lake-S UHD，Mesa 26.0.8 |
| 独立显卡 | NVIDIA RTX 4060 Laptop，驱动 595.91.07 |
| OpenGL | 4.6，硬件直接渲染可用 |
| 内存 | 14 GiB |
| 磁盘 | 检测时剩余约 13 GiB |

显示器配置是本次 RViz 问题的重要背景：

| 屏幕 | 缩放 | 刷新率 |
| --- | ---: | ---: |
| 外接屏 HDMI-5 | 100% | 约 60 Hz |
| 笔记本屏 eDP-1 | 133.33% | 约 165 Hz |

GNOME 同时启用了：

```text
scale-monitor-framebuffer
xwayland-native-scaling
```

XWayland 向程序报告的 `Xft.dpi` 为 192。不同屏幕缩放、不同刷新率、Qt 高 DPI 和 Ogre 子窗口共同构成了闪烁触发条件。

### 5.1 已安装和缺失的 ROS 组件

已安装并验证：

- RViz2
- `robot_state_publisher`
- `joint_state_publisher_gui`
- Xacro
- `controller_manager`
- `joint_trajectory_controller`
- 现代 Gazebo 的 `ros_gz_sim`

缺失或与资料包不匹配：

- MoveIt 相关包
- Gazebo Classic 的 `gazebo_ros`
- Gazebo Classic 的 `gazebo_ros2_control`
- 完整的 ROS 2 Lyrical 开发依赖

卖家教程使用 Ubuntu 22.04、ROS 2 Humble 和 Gazebo Classic。当前系统是 Ubuntu 26.04、ROS 2 Lyrical 和现代 Gazebo，不能原样照搬教程依赖。

### 5.2 Miniconda 对构建的影响

当前默认 `python3` 可能指向 Miniconda Python，其中没有 `catkin_pkg` 和 `yaml`。这会导致 `ament_cmake` 配置阶段报错。

构建前优先执行：

```bash
conda deactivate
```

必要时可显式使用系统 Python：

```bash
env PATH=/opt/ros/lyrical/bin:/usr/bin:/bin \
  colcon build --packages-select pkg_robot_model \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## 6. 模型加载成功但画面中看不到机器人

典型现象：

- RViz 中 `Fixed Frame: world` 为 OK。
- `RobotModel: Status OK`。
- `/joint_states` 和 TF 都正常。
- 中央画面只有网格，看不到模型。

原因是 RViz 默认相机距离约 10 米，而该机器人模型不足 1 米，模型小到几乎不可见。

项目新增了 `pkg_robot_model/config/display.rviz`，主要设置为：

```yaml
Global Options:
  Fixed Frame: world
  Frame Rate: 30

Views:
  Current:
    Distance: 1.35
    Focal Point:
      X: 0
      Y: 0
      Z: 0.34
```

`display.launch.py` 会自动通过 `rviz2 -d .../display.rviz` 加载该配置，无需手动添加 RobotModel。

## 7. Wayland 下 3D 面板闪烁

### 7.1 现象和排除过程

当时的现象是：

- 左侧 RViz 属性树稳定。
- 只有中央 Ogre 3D 面板快速闪烁。
- RobotModel、TF 和模型文件均为 OK。
- 改用软件 OpenGL 后仍闪烁。

因此问题不是 URDF、TF、网格缺失，也不只是显卡驱动问题，而是 Qt 主窗口和嵌入式 Ogre 原生子窗口在混合 DPI 环境下获得了不同的像素尺寸。

RViz 上游已有相同类别的问题：

- [rviz2 screen flickering when enable HiDPI · ros2/rviz#1052](https://github.com/ros2/rviz/issues/1052)
- [rviz2 major screen flickering/tearing · ros2/rviz#1079](https://github.com/ros2/rviz/issues/1079)
- [Wayland Support · ros2/rviz#847](https://github.com/ros2/rviz/issues/847)
- [Fix compile with wayland · ros2/rviz#1569](https://github.com/ros2/rviz/pull/1569)

截至本文验证时，原生 Wayland 支持 PR 仍未合并。

### 7.2 为什么仍使用 `xcb`

Ubuntu 26.04 不再提供 GNOME X11 桌面会话，不代表 XWayland 兼容层不存在。本机安装了 XWayland，Qt 的 `xcb` 后端可以通过它运行。

实际测试原生 Wayland：

```bash
QT_QPA_PLATFORM=wayland rviz2
```

当前 ROS 2 Lyrical 二进制会报：

```text
Invalid parentWindowHandle (wrong server or screen)
GLXWindow::create
Unable to create the rendering window after 100 tries
```

这说明当前 `rviz_ogre_vendor` 仍使用 GLX，不能把 `QT_QPA_PLATFORM` 直接切换成 `wayland`。

### 7.3 当前有效的解决方案

`display.launch.py` 只对 RViz 子进程设置以下环境变量：

```python
additional_env={
    'QT_QPA_PLATFORM': 'xcb',
    'QT_X11_NO_MITSHM': '1',
    'QT_AUTO_SCREEN_SCALE_FACTOR': '0',
    'QT_ENABLE_HIGHDPI_SCALING': '0',
    'QT_SCALE_FACTOR': '1',
    'QT_SCREEN_SCALE_FACTORS': '1;1',
    'QT_FONT_DPI': '144',
}
```

核心原则是：

1. 通过 XWayland 运行当前 GLX 版 Ogre。
2. 禁止 Qt 自动读取混合 DPI。
3. 两块屏幕都按 1 倍像素比例创建 Qt/Ogre 窗口。
4. 单独提高字体 DPI，而不改变渲染面的像素比例。

`QT_SCREEN_SCALE_FACTORS='1;1'` 对应当前两块屏幕。如果显示器数量变化，应按实际屏幕数调整，例如单屏使用 `1`，三屏使用 `1;1;1`。

### 7.4 仍然闪烁时的系统级回退方案

先尝试：

- 让 RViz 窗口完整停留在一块屏幕内，不要跨屏。
- 暂时把两块屏幕都改为 100% 缩放。
- 暂时把两块屏幕都设为 60 Hz，以排除混合刷新率影响。

最后的系统级方案是关闭 GNOME 的实验性 XWayland 原生缩放：

```bash
gsettings set org.gnome.mutter experimental-features "['scale-monitor-framebuffer']"
```

该设置需要注销并重新登录。副作用是部分 XWayland 程序可能变得模糊，因此当前项目已优先采用只影响 RViz 的环境变量方案，不主动修改桌面全局设置。

## 8. 字体大小调节

不要通过增大 `QT_SCALE_FACTOR` 来放大界面，因为它可能重新触发 Ogre 面板闪烁。

只修改 `pkg_robot_model/launch/display.launch.py` 中的：

```python
'QT_FONT_DPI': '144',
```

建议值：

| DPI | 效果 |
| ---: | --- |
| 120 | 略大 |
| 144 | 当前默认，约为 150% 字体大小 |
| 168 | 较大 |
| 192 | 约为 200% 字体大小 |

修改后重新构建并重启 RViz：

```bash
colcon build --packages-select pkg_robot_model
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py
```

## 9. 帧率和 GPU 调节

### 9.1 RViz 目标帧率

目标帧率位于 `pkg_robot_model/config/display.rviz`：

```yaml
Global Options:
  Frame Rate: 30
```

24～30 FPS 足以进行 URDF 姿态预览。继续调高只会增加 GPU/CPU 占用，不会提升关节状态发布频率。

### 9.2 为什么之前只有约 10 FPS

排查闪烁期间曾临时设置：

```text
LIBGL_ALWAYS_SOFTWARE=1
```

这会让 RViz 使用 `llvmpipe` 软件光栅器。本模型包含较多高面数 STL，软件渲染只能达到约 10 FPS。

最终配置已删除该变量，恢复：

```text
OpenGL renderer: Mesa Intel(R) Graphics (RPL-S)
direct rendering: Yes
```

因此不要在 `.bashrc` 或启动文件中重新加入 `LIBGL_ALWAYS_SOFTWARE=1`，除非只是排查驱动问题。

检查实际渲染器：

```bash
glxinfo -B | grep -E 'direct rendering|OpenGL vendor|OpenGL renderer|OpenGL version'
```

### 9.3 必要时使用 RTX 4060

如果 Intel 核显仍无法稳定达到 24 FPS，可只让 RViz 使用 NVIDIA 独显。在 `rviz_node` 的 `additional_env` 中加入：

```python
'__NV_PRIME_RENDER_OFFLOAD': '1',
'__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
```

先用以下命令验证 NVIDIA PRIME：

```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
glxinfo -B
```

只有当输出的 OpenGL renderer 确实是 RTX 4060 时再写入启动文件。混合缩放相关的 Qt 环境变量仍需保留。

### 9.4 其他性能建议

- 保持 `Collision Enabled: false`，只显示 Visual 模型。
- 不需要时关闭 TF、PointCloud、Image 等额外 Display。
- 不要同时打开多个 RViz 窗口。
- 如果未来仍不足 24 FPS，可为 RViz 生成低面数 STL；物理仿真则应另外使用简化碰撞几何。

## 10. 控制器与仿真限制

当前 `my_controllers.yaml` 定义：

- `right_arm_controller`：`51`～`56`
- `right_gripper_controller`：`57`
- `left_arm_controller`：`1`～`6`
- `left_gripper_controller`：`7`

头部 `101`、`102` 没有轨迹控制器。因此：

- RViz 滑块可以显示全部 16 个可动关节。
- ros2_control/MoveIt 当前只配置了 14 个双臂和夹爪关节。
- 若要求仿真控制全部关节，需要增加头部控制器，并同步修改 MoveIt SRDF 和 controller mapping。

现有 Gazebo 文件使用：

```text
gazebo_ros
gazebo_ros2_control/GazeboSystem
libgazebo_ros2_control.so
libgazebo_ros_camera.so
```

这些属于 Gazebo Classic 技术栈。当前系统安装的是现代 Gazebo 与 `ros_gz_sim`，需要迁移到 `gz_ros2_control`，不能通过简单修改包名解决。

## 11. 真机安全注意事项

在完成硬件适配与校准前，不要运行：

```bash
ros2 launch pkg_robot_model real_robot.launch.py
```

主要风险：

- 串口硬编码为 `/dev/ttyACM0`。
- 波特率硬编码为 `1000000`。
- 舵机零点全部初始化为 2048。
- 源码注释明确说明部分方向是推测值。
- 激活硬件接口会立即给全部舵机上扭矩。
- 初始命令为零弧度，可能使机器人突然回中位。
- 大多数手臂关节在 URDF 中被声明为 `continuous`，没有真实机械限位。
- 写入端只限制 0～4095 脉冲，没有逐关节软限位、速度限制或急停逻辑。

真机调试应先完成：

1. 单舵机断载测试。
2. 逐关节确认 ID、正方向、零点和机械极限。
3. 将校准数据从源码移入独立配置文件。
4. 增加逐关节位置、速度、加速度和电流限制。
5. 上电时用反馈位置初始化命令，避免激活瞬间跳变。
6. 准备独立物理急停和断电手段。

## 12. 常用诊断命令

检查 ROS 节点和话题：

```bash
ros2 node list
ros2 topic list -t
ros2 topic echo /joint_states --once
```

检查 TF：

```bash
ros2 run tf2_ros tf2_echo world base_link_0
```

检查桌面与缩放：

```bash
echo "$XDG_SESSION_TYPE"
gsettings get org.gnome.mutter experimental-features
xrdb -query | grep -i dpi
xrandr --current
```

检查当前 RViz 使用的环境变量：

```bash
pid=$(pgrep -n rviz2)
tr '\0' '\n' < /proc/$pid/environ | \
  grep -E '^(QT|DISPLAY|WAYLAND|XDG_SESSION_TYPE|LIBGL|MESA)'
```

检查依赖：

```bash
rosdep check --from-paths . --ignore-src
```

检查最近 RViz 日志：

```bash
find ~/.ros/log -type f -mmin -30 -print
grep -RniE 'error|warn|render|ogre|glx|egl|wayland' ~/.ros/log
```

## 13. 推荐后续路线

若目标只是学习 URDF、TF 和关节姿态：继续使用当前 Lyrical + RViz 配置即可。

若目标是尽快运行卖家原版 Gazebo/MoveIt：准备 Ubuntu 22.04 + ROS 2 Humble 环境，兼容成本最低。

若必须长期使用 Ubuntu 26.04 + ROS 2 Lyrical：建议按以下顺序迁移：

1. 修正 Miniconda 与系统 Python 隔离。
2. 安装 Lyrical MoveIt。
3. 把 Gazebo Classic 启动和插件迁移到 `ros_gz_sim`、`gz_ros2_control`。
4. 更新 `xiaobei_hardware` 的 CMake 和新版 `hardware_interface` API。
5. 增加头部控制器和 MoveIt 规划组。
6. 为所有真实关节补充安全限位和可靠校准数据。

## 14. RViz 初始姿态与 Qt 控制面板

使用 `use_gui:=false` 启动时，卖家的关节滑块发布器不会运行。此前，在 Python 控制面板首次发布 `/joint_states` 前，活动关节缺少 TF，RViz 可能只显示底座和躯干，RobotModel 显示红色状态。现在 `display.launch.py` 会启动一次性零位发布器：等 `robot_state_publisher` 和 RViz 的 TF 订阅出现后，给 URDF 中的 16 个活动关节发布一次零位状态，然后退出。若已有控制器发布关节状态，它会跳过。RViz 的 `/robot_description` 订阅改为 Transient Local，以接收启动时发布的模型描述。

旧 Tk 面板在当前 Wayland/XWayland 桌面上使用的默认字体太小。推荐的 `script_python/x1_joint_qt.py` 使用 PySide6、Noto Sans CJK SC 13 磅字体，并提供关节滑块、精确角度输入、预设姿态和自动演示。它仍只发布 `/joint_states`，不会控制真实机器人。

已有 Conda 环境需手动安装：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate x1_ros2_py
conda install -c conda-forge pyside6
```

启动前先退出旧 RViz，并使用系统 Python 构建一次 ROS 包：

```bash
cd /home/hzb/Work/urdf-model
conda deactivate
source /opt/ros/lyrical/setup.bash
colcon build --packages-select pkg_robot_model
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py use_gui:=false
```

在另一个终端启动 Qt 面板：

```bash
cd /home/hzb/Work/urdf-model
source ~/miniconda3/etc/profile.d/conda.sh
conda activate x1_ros2_py
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python script_python/x1_joint_qt.py
```
