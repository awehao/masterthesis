"""Stand-alone GMPC test launch.

Brings up everything needed to exercise the GMPC controller WITHOUT Nav2:

  - static identity TF  map -> odom   (so the omni base's odom-frame pose
    can also serve as its map-frame pose; OK for a no-localisation test)
  - the GMPC controller node (loaded with config/gmpc_params.yaml)
  - the test path publisher (configurable shape via the `shape` launch arg)

Assumes Gazebo + the omni chassis are already running, e.g.:

    ros2 launch ammr_bringup gazebo.launch.py

Then in a second shell:

    ros2 launch ammr_wholebody_mpc gmpc_test.launch.py shape:=line  length:=3.0
    ros2 launch ammr_wholebody_mpc gmpc_test.launch.py shape:=arc   length:=1.5
    ros2 launch ammr_wholebody_mpc gmpc_test.launch.py shape:=square length:=2.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch                       import LaunchDescription
from launch.actions               import DeclareLaunchArgument
from launch.substitutions         import LaunchConfiguration
from launch_ros.actions           import Node


def generate_launch_description():
    pkg     = get_package_share_directory('ammr_wholebody_mpc')
    params  = os.path.join(pkg, 'config', 'gmpc_params.yaml')

    shape   = LaunchConfiguration('shape')
    length  = LaunchConfiguration('length')

    return LaunchDescription([
        DeclareLaunchArgument('shape',  default_value='line',
            description='Test path shape: line | square | arc'),
        DeclareLaunchArgument('length', default_value='3.0',
            description='Trajectory characteristic length in metres'),

        # map -> odom as identity (no AMCL in this test)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_identity',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[{'use_sim_time': True}],
        ),

        # GMPC controller
        Node(
            package='ammr_wholebody_mpc',
            executable='gmpc_node',
            name='gmpc_controller',
            output='screen',
            parameters=[params],
        ),

        # Test path publisher
        Node(
            package='ammr_wholebody_mpc',
            executable='test_path_publisher',
            name='gmpc_test_path_publisher',
            output='screen',
            parameters=[{
                'use_sim_time':   True,
                'shape':          shape,
                'length':         length,
                'spacing':        0.05,
                'frame_id':       'map',
                'topic':          '/plan',
                'publish_period': 1.0,
            }],
        ),
    ])
