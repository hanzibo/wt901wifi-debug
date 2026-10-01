#!/usr/bin/env bash
# RViz + the vendor's joint_state_publisher_gui, for inspecting X1 joint IDs.
set -euo pipefail

workspace=/home/hzb/Work/WT901/urdf-model
ros_setup=/opt/ros/lyrical/setup.bash
workspace_setup="$workspace/install/setup.bash"

if [[ ! -f "$ros_setup" || ! -f "$workspace_setup" ]]; then
    printf '缺少 ROS 2 或 X1 工作空间的 setup.bash。\n' >&2
    read -r -n 1 -p '按任意键关闭终端…' _ || true
    exit 1
fi

# A desktop session can inherit Conda/Qt paths from other applications.
# Keep RViz and its GUI on the system ROS/Qt environment.
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH PYTHONHOME VIRTUAL_ENV
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

cd "$workspace"
# ROS 2 setup scripts legitimately reference optional variables before they
# are defined, so nounset must be disabled only while sourcing them.
set +u
source "$ros_setup"
source "$workspace_setup"
set -u

printf '启动 X1 RViz + 关节滑块。请先关闭其他 X1 /joint_states 发布器。\n'
printf '关闭 RViz 或在终端按 Ctrl+C 可结束本次启动。\n'
ros2 launch pkg_robot_model display.launch.py use_gui:=true || {
    status=$?
    printf '\n启动失败，退出码：%s\n' "$status" >&2
    read -r -n 1 -p '按任意键关闭终端…' _ || true
    exit "$status"
}
