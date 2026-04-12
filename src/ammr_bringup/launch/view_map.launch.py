import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    map_file = os.path.join(pkg, 'maps', 'simple_room.yaml')

    return LaunchDescription([
        Node(
            package='ammr_bringup',
            executable='map_publisher',
            arguments=[map_file],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
        ),
    ])
