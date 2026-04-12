import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    urdf_file = os.path.join(pkg, 'urdf', 'ammr_base.urdf.xacro')
    map_file  = os.path.join(pkg, 'maps', 'simple_room.yaml')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str
    )

    return LaunchDescription([

        # 1. Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': True}],
        ),

        # 2. Nav2 (localization + navigation) with static map
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_pkg, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map': map_file,
                'use_sim_time': 'true',
                'params_file': os.path.join(pkg, 'config', 'nav2_params.yaml'),
            }.items(),
        ),

        # 3. RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg, 'rviz', 'nav2.rviz')]
                if os.path.exists(os.path.join(pkg, 'rviz', 'nav2.rviz'))
                else [],
        ),
    ])
