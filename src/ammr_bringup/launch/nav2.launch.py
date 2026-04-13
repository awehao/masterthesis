import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg      = get_package_share_directory('ammr_bringup')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    map_file    = os.path.join(pkg, 'maps',   'random_room.yaml')
    params_file = os.path.join(pkg, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_pkg, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map':          map_file,
                'params_file':  params_file,
                'use_sim_time': 'true',
                'autostart':    'true',
            }.items(),
        ),
    ])
