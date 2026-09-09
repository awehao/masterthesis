"""Everything gazebo_dynamic.launch.py starts EXCEPT Gazebo itself.

The simulator is replaced by evaluation/isaac_bench_sim.py, which supplies
/clock, /scan_raw, /cmd_vel (in) and the /model/<dyn>/{pose,cmd_vel} pair --
the same interface the ros_gz bridge exposed. Every node below is byte-for-byte
the one the Gazebo trials used, so a difference in results is a difference in
the simulator.

Deliberately NOT started here (started by the Isaac process instead):
    gz sim, ros_gz_bridge parameter_bridge, ros_gz_sim create
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue



# Scenario selection. Both launches read the SAME two environment variables so
# Gazebo and Isaac cannot silently run different scenarios: AMMR_TRAJ_FILE
# picks the trajectory config and AMMR_OBSTACLE_MODE picks legacy (feedback
# ping-pong, historical, not reproducible) or scheduled (position computed from
# simulation time with /case_start as phase zero). Defaults reproduce the
# historical behaviour exactly.
def _scenario(pkg):
    traj = os.environ.get('AMMR_TRAJ_FILE') or os.path.join(
        pkg, 'config', 'dynamic_trajectories.yaml')
    mode = os.environ.get('AMMR_OBSTACLE_MODE', 'legacy')
    return traj, mode


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    urdf_file = os.path.join(pkg, 'urdf', 'ammr_base.urdf.xacro')
    map_file = os.path.join(pkg, 'maps', 'random_room.yaml')
    traj_file, obstacle_mode = _scenario(pkg)

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'dynamic', default_value='true',
            description='If false, dynamic_obstacle_driver is NOT launched.'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description,
                          'use_sim_time': True}]),

        Node(package='ammr_bringup', executable='map_publisher',
             arguments=[map_file]),

        # dead-reckons /odom from /cmd_vel and publishes odom -> base_footprint.
        # Simulator-independent by construction, so the odometry path is
        # identical to the Gazebo trials rather than re-derived from Isaac.
        Node(package='ammr_bringup', executable='omni_drive_controller',
             parameters=[{'use_sim_time': True}], output='screen'),

        # Kept exactly as in gazebo_dynamic.launch.py, including the fact that
        # its 0.19 disagrees with the URDF's lidar_joint 0.11. Identical on both
        # sides, so it cannot bias the comparison; noted, not silently fixed.
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['0', '0', '0.19', '0', '0', '0',
                        'base_link', 'lidar_link'],
             parameters=[{'use_sim_time': True}]),

        Node(package='ammr_bringup', executable='scan_relay',
             parameters=[{'use_sim_time': True}], output='screen'),

        TimerAction(
            period=8.0,
            condition=IfCondition(LaunchConfiguration('dynamic')),
            actions=[Node(package='ammr_bringup',
                          executable='dynamic_obstacle_driver',
                          parameters=[{'use_sim_time': True},
                                      {'trajectories_file': traj_file},
                                      {'mode': obstacle_mode}],
                          output='screen')]),
    ])
