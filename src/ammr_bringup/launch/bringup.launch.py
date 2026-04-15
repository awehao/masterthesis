import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    sim_pkg  = get_package_share_directory('ammr_simulation')
    nav_pkg  = get_package_share_directory('ammr_navigation')
    bringup_pkg = get_package_share_directory('ammr_bringup')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_pkg, 'launch', 'gazebo.launch.py')
        )
    )

    # Nav2 延遲 5 秒，等 Gazebo 和機器人完全起來
    nav2 = TimerAction(
        period=5.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav_pkg, 'launch', 'nav2.launch.py')
            )
        )]
    )

    # RViz 延遲 6 秒
    rviz = TimerAction(
        period=6.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(bringup_pkg, 'rviz', 'default.rviz')],
            parameters=[{'use_sim_time': True}],
        )]
    )

    # 動態障礙物控制節點，延遲 8 秒等 Gazebo 完全啟動
    dynamic_obs = TimerAction(
        period=8.0,
        actions=[Node(
            package='ammr_bringup',
            executable='dynamic_obstacle_mover',
        )]
    )

    return LaunchDescription([gazebo, nav2, rviz, dynamic_obs])
