import os
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, IncludeLaunchDescription, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 1. 获取包路径
    pkg_robot_model = FindPackageShare("pkg_robot_model")
    pkg_moveit_config = FindPackageShare("pkg_robot_model_moveit_config")

    # 2. 解析 URDF，并传入 use_sim:=false (切换到真机模式)
    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            PathJoinSubstitution([pkg_robot_model, "urdf", "pkg_robot_model.urdf.xacro"]),
            " ",
            "use_sim:=false", 
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # 3. 获取控制器配置文件路径
    robot_controllers = PathJoinSubstitution(
        [pkg_robot_model, "config", "my_controllers.yaml"]
    )

    # 4. 启动 ros2_control_node (这是真机控制的核心，负责加载 C++ 硬件接口)
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
    )

    # 5. 机器人状态发布 (TF)
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # 6. 定义要启动的控制器列表 (名字必须和 my_controllers.yaml 里的一致!)
    controllers_to_spawn = [
        "joint_state_broadcaster",
        "right_arm_controller",
        "right_gripper_controller",
        "left_arm_controller",
        "left_gripper_controller"
    ]

    # 7. 循环创建 Spawner 节点
    # 注意：我们使用 TimerAction 稍微延时一点，或者通过事件链，确保 controller_manager 准备好后再加载
    spawn_controllers = []
    for controller in controllers_to_spawn:
        spawn_controllers.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--controller-manager", "/controller_manager"],
                output="screen",
            )
        )

    # 8. 启动 MoveIt (RViz + MoveGroup)
    # 传入参数告诉 MoveIt 不要使用仿真时间
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_moveit_config, "launch", "move_group.launch.py"])
        ),
        launch_arguments={'allow_trajectory_execution': 'true',
                          'moveit_manage_controllers': 'true',
                          'use_sim_time': 'false'}.items(),
    )

    moveit_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_moveit_config, "launch", "moveit_rviz.launch.py"])
        ),
        launch_arguments={'use_sim_time': 'false'}.items(),
    )

    # 9. 组装启动项
    # 为了稳妥，我们可以让控制器在 control_node 启动 3 秒后再开始加载
    # 实际生产中常用 RegisterEventHandler 监听 controller_manager 的 socket，这里用延时简单有效
    delayed_spawners = TimerAction(
        period=3.0,
        actions=spawn_controllers
    )

    # MoveIt 最好在控制器加载之后再启动，或者并行启动也可以
    # 这里我们让 MoveIt 稍微晚一点点，等 TF 树稳定
    delayed_moveit = TimerAction(
        period=5.0,
        actions=[move_group_launch, moveit_rviz_launch]
    )

    nodes_to_start = [
        control_node,
        robot_state_pub_node,
        delayed_spawners,
        delayed_moveit,
    ]

    return LaunchDescription(nodes_to_start)