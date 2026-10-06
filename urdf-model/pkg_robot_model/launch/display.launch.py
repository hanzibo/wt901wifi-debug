import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('pkg_robot_model')

    # 声明 launch 参数 'model'
    model_path_arg = DeclareLaunchArgument(
        name='model',
        default_value=os.path.join(
            get_package_share_directory('pkg_robot_model'),
            'urdf',
            'pkg_robot_model.urdf'
        ),
        description='Absolute path to robot urdf file'
    )

    use_gui_arg = DeclareLaunchArgument(
        name='use_gui',
        default_value='true',
        description='Start joint_state_publisher_gui. Set false when Python publishes /joint_states.'
    )

    # 启动 robot_state_publisher 节点
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(os.path.join(
                get_package_share_directory('pkg_robot_model'),
                'urdf',
                'pkg_robot_model.urdf'
            )).read()
        }]
    )

    # 启动 joint_state_publisher_gui 节点
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('use_gui')),
    )

    # Without the vendor GUI, movable links have no TF until an external
    # controller publishes its first JointState. Seed the zero pose once,
    # after RViz and robot_state_publisher have subscribed.
    initial_joint_state_node = Node(
        package='pkg_robot_model',
        executable='seed_joint_state.py',
        name='x1_initial_joint_state',
        condition=UnlessCondition(LaunchConfiguration('use_gui')),
        output='screen',
    )

    # 启动 rviz2 节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            os.path.join(pkg_share, 'config', 'display.rviz')
        ],
        # The packaged RViz/Ogre build cannot create its GLX child window on
        # native Wayland. Run it through XWayland and pin Qt's pixel scaling so
        # the embedded 3D surface does not flicker on mixed-DPI monitors.
        additional_env={
            'QT_QPA_PLATFORM': 'xcb',
            'QT_X11_NO_MITSHM': '1',
            # RViz embeds an Ogre QWindow inside a Qt widget.  Mixed-DPI
            # monitors can make the two surfaces use different pixel sizes,
            # which presents as rapid flashing of only the 3D viewport.
            'QT_AUTO_SCREEN_SCALE_FACTOR': '0',
            'QT_ENABLE_HIGHDPI_SCALING': '0',
            'QT_SCALE_FACTOR': '1',
            'QT_SCREEN_SCALE_FACTORS': '1;1',
            # Enlarge text without changing the Ogre viewport's pixel scale.
            'QT_FONT_DPI': '144',
        },
    )

    return LaunchDescription([
        model_path_arg,
        use_gui_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
        initial_joint_state_node,
    ])
