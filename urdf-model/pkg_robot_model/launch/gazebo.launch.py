import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'pkg_robot_model'
    pkg_share = get_package_share_directory(pkg_name)
    pkg_prefix = get_package_prefix(pkg_name)

    # 1. 设置 Gazebo 模型路径
    model_path = os.path.join(pkg_prefix, 'share')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        model_path += os.pathsep + os.environ['GAZEBO_MODEL_PATH']

    env_vars = [
        SetEnvironmentVariable(name='GAZEBO_MODEL_PATH', value=model_path),
        # 屏蔽自动下载模型，防止卡顿
        SetEnvironmentVariable(name='GAZEBO_MODEL_DATABASE_URI', value='')
    ]

    # 2. 读取 URDF 并处理路径
    urdf_file = os.path.join(pkg_share, 'urdf', 'pkg_robot_model.urdf')
    
    print(f"Loading URDF from: {urdf_file}")
    
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # 【核心修复】: 手动替换 URDF 中的 $(find pkg_name) 为实际绝对路径
    robot_desc = robot_desc.replace('$(find pkg_robot_model)', pkg_share)

    # ====================================================
    # 指定 World 文件路径 (加载 clean_room.world)
    # ====================================================
    world_file_path = os.path.join(pkg_share, 'worlds', 'clean_room.world')

    # 3. 启动 Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={
            'verbose': 'true', 
            'world': world_file_path
        }.items()
    )

    # 4. 机器人状态发布
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
    )

    # 5. 生成机器人
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'my_robot',
                   '-z', '0.1'],
        output='screen'
    )

    # ====================================================
    # 6. 加载控制器 (已修改为适配 MoveIt 的4个分组控制器)
    # ====================================================
    
    # 6.1 关节状态广播器 (必须)
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    # 6.2 右臂轨迹控制器
    load_right_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller"],
        output="screen",
    )

    # 6.3 右手夹爪控制器
    load_right_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_controller"],
        output="screen",
    )

    # 6.4 左臂轨迹控制器
    load_left_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller"],
        output="screen",
    )

    # 6.5 左手夹爪控制器
    load_left_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_controller"],
        output="screen",
    )

    return LaunchDescription(env_vars + [
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        # 在机器人生成后，按顺序启动所有控制器
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[
                    load_joint_state_broadcaster,
                    load_right_arm,
                    load_right_gripper,
                    load_left_arm,
                    load_left_gripper
                ],
            )
        ),
    ])