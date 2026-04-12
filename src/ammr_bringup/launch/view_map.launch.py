import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    map_file = os.path.join(pkg, 'maps', 'simple_room.yaml')

    return LaunchDescription([

        LifecycleNode(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            namespace='',
            parameters=[{'yaml_filename': map_file, 'use_sim_time': False}],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
            }],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
        ),
    ])
