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
    map_file   = os.path.join(pkg, 'maps', 'random_room.yaml')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str
    )

    return LaunchDescription([

        # 1. 啟動 Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen',
        ),

        # 2. ROS2 <-> Gazebo bridge
        #    - /cmd_vel        : ROS -> GZ (VelocityControl 接收)
        #    - /scan_raw       : GZ -> ROS
        #    - /clock          : GZ -> ROS
        #    - /gz/*_wheel_vel : ROS -> GZ (JointController 接收 std_msgs/Float64 <-> gz.msgs.Double)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/gz/left_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
                '/gz/right_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
                '/gz/front_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
                '/gz/back_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
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

        # 4. 地圖發布
        Node(
            package='ammr_bringup',
            executable='map_publisher',
            arguments=[map_file],
        ),

        # Omni-drive controller：訂閱 /cmd_vel → 發布每輪速度 + /odom + TF
        Node(
            package='ammr_bringup',
            executable='omni_drive_controller',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        # static TF: base_link -> lidar_link (connects URDF chain to scan frame)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.19', '0', '0', '0', 'base_link', 'lidar_link'],
            parameters=[{'use_sim_time': True}],
        ),

        # 5. Spawn robot（延遲 6 秒等 Gazebo 啟動 + robot_description 就緒）
        TimerAction(
            period=6.0,
            actions=[Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'ammr_base',
                    '-topic', 'robot_description',
                    '-x', '0.0', '-y', '0.0', '-z', '0.0',
                ],
                output='screen',
            )],
        ),

        # scan_raw → /scan relay（修正 frame_id 並統一 topic 名稱）
        Node(
            package='ammr_bringup',
            executable='scan_relay',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

    ])
