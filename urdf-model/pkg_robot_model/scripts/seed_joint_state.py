#!/usr/bin/python3
"""Publish one zero pose once RViz and robot_state_publisher are ready."""

import time
from pathlib import Path
from xml.etree import ElementTree

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState


def movable_joint_names() -> list[str]:
    model = Path(get_package_share_directory('pkg_robot_model')) / 'urdf' / 'pkg_robot_model.urdf'
    robot = ElementTree.parse(model).getroot()
    return [joint.attrib['name'] for joint in robot.findall('joint')
            if joint.attrib.get('type') != 'fixed']


def main() -> None:
    rclpy.init()
    node = Node('x1_initial_joint_state')
    publisher = node.create_publisher(JointState, '/joint_states', 10)
    try:
        names = movable_joint_names()
        if not names:
            raise RuntimeError('URDF has no movable joints')

        deadline = time.monotonic() + 15.0
        state_publisher_ready = False
        rviz_ready = False
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            other_publishers = [info for info in node.get_publishers_info_by_topic('/joint_states')
                                if info.node_name != node.get_name()]
            if other_publishers:
                node.get_logger().info('Another joint controller is active; skipping zero pose')
                return

            state_publisher_ready = any(
                info.node_name == 'robot_state_publisher'
                for info in node.get_subscriptions_info_by_topic('/joint_states')
            )
            rviz_ready = (
                'rviz2' in node.get_node_names()
                and bool(node.get_subscriptions_info_by_topic('/tf'))
            )
            if state_publisher_ready and rviz_ready:
                break

        if not state_publisher_ready:
            node.get_logger().warning('robot_state_publisher did not subscribe; no zero pose sent')
            return

        if not rviz_ready:
            node.get_logger().warning('RViz TF subscription was not observed; sending zero pose anyway')

        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.name = names
        message.position = [0.0] * len(names)
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.3)
        node.get_logger().info(f'Published initial zero pose for {len(names)} joints')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
