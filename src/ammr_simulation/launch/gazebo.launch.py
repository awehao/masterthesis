import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    world_file = os.path.join(pkg, 'worlds', 'random_room.sdf')
    urdf_file  = os.path.join(pkg, 'urdf', 'ammr_base.urdf.xacro')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str
    )

    return LaunchDescription([

        # 1. 啟動 Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen',
        ),

        # 2. ROS2 <-> Gazebo bridge（tf、clock、cmd_vel、odom）
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            ],
            output='screen',
        ),

        # 3. Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': True}],
        ),

        # odom -> ammr_base/base_footprint TF（從 /odom 發布，確保時間戳一致）
        Node(
            package='ammr_bringup',
            executable='odom_tf_broadcaster',
            parameters=[{'use_sim_time': True}],
        ),

        # Bridge: ammr_base/base_footprint (Gazebo) -> base_footprint (URDF)
        # Connects robot_state_publisher's kinematic tree to the Nav2/Gazebo tree
        # Gazebo already publishes ammr_base/base_footprint -> ammr_base/base_footprint/lidar
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'ammr_base/base_footprint', 'base_footprint'],
            parameters=[{'use_sim_time': True}],
        ),

        # 5. Spawn robot（延遲 3 秒等 Gazebo 啟動）
        TimerAction(
            period=3.0,
            actions=[Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'ammr_base',
                    '-topic', 'robot_description',
                    '-x', '-8.5', '-y', '-8.5', '-z', '0.1',
                ],
                output='screen',
            )],
        ),

    ])
