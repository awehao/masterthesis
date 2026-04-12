import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node, LifecycleNode


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    map_file = os.path.join(pkg, 'maps', 'simple_room.yaml')

    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        parameters=[{'yaml_filename': map_file, 'use_sim_time': False}],
    )

    # 延遲 2 秒再 configure + activate，確保節點已完全啟動
    configure_cmd = TimerAction(
        period=2.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'lifecycle', 'set', '/map_server', 'configure'],
            output='screen',
        )]
    )

    activate_cmd = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'lifecycle', 'set', '/map_server', 'activate'],
            output='screen',
        )]
    )

    return LaunchDescription([
        map_server,
        configure_cmd,
        activate_cmd,
        Node(
            package='rviz2',
            executable='rviz2',
        ),
    ])
