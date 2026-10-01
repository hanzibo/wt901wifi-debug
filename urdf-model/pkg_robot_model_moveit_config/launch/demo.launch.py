from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 加载 MoveIt 配置
    moveit_config = MoveItConfigsBuilder("pkg_robot_model", package_name="pkg_robot_model_moveit_config").to_moveit_configs()

    # 【核心修改】定义仿真时间参数
    use_sim_time = {"use_sim_time": True}

    # 1. 启动 MoveGroup (大脑)
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            use_sim_time,  # <--- 强制使用仿真时间
        ],
    )

    # 2. 启动 RViz (可视化)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", str(moveit_config.package_path / "config/moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            use_sim_time,  # <--- 强制使用仿真时间
        ],
    )

    # 注意：这里没有启动 robot_state_publisher，因为你的 gazebo.launch.py 已经启动了它。
    # 如果两边都启动，会导致 TF 冲突 (机器人会在 RViz 里疯狂闪烁)。
    
    return LaunchDescription([
        run_move_group_node,
        rviz_node,
    ])