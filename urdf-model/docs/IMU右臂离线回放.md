# WT901 → X1 右臂离线回放（RViz-only）

这是第一阶段的**运动学预览**，不连接串口、不启动 `ros2_control`，不发送真机轨迹。X1 右臂使用 URDF 关节 `51`～`56`；夹爪 `57`、左臂和头部固定在零位。不要将生成的 CSV 当作真机安全轨迹使用。

## 输入与边界

默认使用 `OptUSBProj-ubuntu/calibrations/20260926-174220-4304e9/`。该目录含左右上/前臂 IMU 数据，以及与**同一场采集**匹配、状态为“通过”的右上臂和右前臂标定结果。离线程序会拒绝设备 ID 或采集目录不匹配的标定，不会把后来某次穿戴的矩阵套到旧录制上。

这组数据没有躯干和右手 IMU：当前假设躯干固定；可以计算上臂、肘和手腕位置，但**不能观测完整手掌姿态**。因此 IK 优先匹配手腕位置、其次肘部位置；机器人手腕两个关节保持近零。人体模型和 X1 都按 X 前、Y 左、Z 上处理，围绕各自中立位映射位移并按臂长缩放。此方案适合验证方向、连续性和可达性，尚不是最终人体模仿精度评估。

## 1. 生成离线轨迹

在项目根目录 `/home/hzb/Work/WT901`：

```bash
conda run --no-capture-output -n OptUSBProj-fusion \
  python -u OptUSBProj-ubuntu/x1_right_offline.py
```

默认输出到该标定目录下的 `x1_right_trajectory.csv` 和 `x1_right_trajectory.json`（本机数据目录已被 Git 忽略）。CSV 每帧含时间、6 个关节弧度、目标/实际肘和腕坐标、人体/机器人骨段夹角及误差。JSON 汇总丢失数据、保持帧和位置误差。可用 `--session`、`--urdf`、`--output`、`--rate` 改输入与采样率；新录制必须同时保存并通过右上/前臂穿戴标定，才符合当前输入格式。

## 2. 在 RViz 回放

终端 A 先启动模型（在 `urdf-model/` 工作目录）：

```bash
cd /home/hzb/Work/WT901/urdf-model
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ros2 launch pkg_robot_model display.launch.py use_gui:=false
```

终端 B 启动回放：

```bash
cd /home/hzb/Work/WT901/urdf-model
source /home/hzb/miniconda3/etc/profile.d/conda.sh
conda activate x1_ros2_py
source /opt/ros/lyrical/setup.bash
source install/setup.bash
python script_python/x1_imu_replay.py
```

先检查 CSV 而不接触 ROS：

```bash
conda run --no-capture-output -n x1_ros2_py \
  python urdf-model/script_python/x1_imu_replay.py --dry-run
```

回放前**关闭** `x1_joint_qt.py`、卖家 `joint_state_publisher_gui` 等持续发布 `/joint_states` 的程序，防止姿态跳动。`seed_joint_state.py` 只发布一次零位，可以保留。回放默认原速；可用 `--speed 0.5` 放慢。回放到末帧后维持 2 秒并退出。

## 算法概要

每台设备的当前姿态与穿戴标定共同中立位组成 `ΔR = R₀ᵀR(t)`，再通过连续轴映射 `B ΔR Bᵀ` 转入人体模型系。人体肩、肘、腕的位置由固定骨长和层级运动学得到，**不对加速度二次积分**。求出人体上臂、前臂相对中立位的位移；以统一的肩—腕可达长度比，把肘/腕的位移叠加到 X1 URDF 的零位目标点。通过 SciPy 有界最小二乘求右臂关节角，同时抑制突变与无观测的腕关节。

另外，计算人体上臂与前臂的三维方向夹角：人体接近伸直时，对 X1 的肘关节 `54` 加接近中立的软约束；人体弯肘较多时逐渐减弱它。该约束不是把 `54` 硬设零，也不是宣称机器人零位几何上完全伸直。X1 肩部各旋转轴存在偏置，其肩关节原点到肘关节原点的距离会随姿态变化，因此**不能简单假定 URDF 上臂为一根固定的肩—肘线段、再按人体两段长度分别缩放**。我们实际试过这种映射：前举略有改善，侧举却产生约 3 cm 的手腕目标误差，故正式回放保留任务空间位移目标与完整 URDF FK/IK。求解失败、数据断流或手腕误差超过 8 cm 时保持上一帧。

在 `20260926-174220-4304e9` 的两次前平举静止区间，人体模型上/前臂方向差约 5–6°。修改前 X1 关节 `54` 约 33°；修改后约 1°，机器人肩—肘—腕关节中心线夹角从约 36–37° 降至约 25°（零位本身约 25°）。代价是前举手腕目标平均误差从约 0.6 cm 增到约 1.5 cm；侧举约 0.9 cm。这些是 RViz 运动学内部量，不是实体机器人实测精度。前举时右肩 `51` 约 94–95°，该值仍须与真机真实限位核对。

这里的优化边界只用于 RViz 防止离谱姿态，**不是舵机物理限位**。模型中大部分关节在 URDF 声明为 `continuous`，真实限位、零点、转向、速度和急停尚未逐轴验证；在完成独立安全工程前不要接真机。

## 可参考的开源路线

- [OpenArm `mimic`](https://github.com/Manas-arumalla/openarm-control)：其 README 描述以肩—肘、肘—腕方向做人体到机器人映射，并用热启动 IK、平滑和限速保持连续。值得借鉴目标表示和测试方式；但其机器人、相机输入及仿真栈与本项目不同。
- [Humanoid Arm Retarget](https://github.com/Hao-Starrr/humanoid-arm-retarget)：使用手和前臂观测、机器人正运动学与 relaxed IK，值得参考“肘部可观测性”和不可达目标的处理。输入是 Vision Pro，且面向其他 7-DOF 手臂，不能直接接 WT901/X1。
- [GMR](https://github.com/YanjieZe/GMR) 的 [IK 配置说明](https://github.com/YanjieZe/GMR/blob/master/DOC.md)：将人体与机器人身体部位匹配、位置与方向设不同权重，并允许固定偏移；是更通用的设计参考，但现在移植整套模型与依赖的成本高于维护这条小型 RViz 验证链路。

这些是方法对照，当前代码没有克隆或复制上述项目实现，也没有安装其依赖。

## 下一轮录制建议

验证这版回放方向后，再穿戴躯干、右上臂、右前臂，最好再加右手背。在同一次穿戴中进行标定和动作录制；保存当时生效的标定矩阵及共同中立位。右手腕目前没有正式穿戴标定流程，因此在它完成前也不要把右手 IMU 姿态当成已验证的末端朝向。
