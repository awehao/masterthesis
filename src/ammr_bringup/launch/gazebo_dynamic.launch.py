"""
Gazebo + 動態障礙物 launch（衝刺 1A）

與 gazebo.launch.py 的差異：
  - 世界檔改為 worlds/random_room_dynamic.sdf
  - 從 config/dynamic_trajectories.yaml 讀取動態障礙物名單，
    動態為每一個產生 /model/<name>/cmd_vel bridge
  - 啟動 dynamic_obstacle_driver 節點驅動圓柱 ping-pong 移動
"""
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_dyn_names(traj_file):
    """從 dynamic_trajectories.yaml 取出全部 dynamic obstacle 名稱。"""
    try:
        with open(traj_file, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return []
    return [d['name'] for d in cfg.get('dynamic_obstacles', [])]


def generate_launch_description():
    pkg = get_package_share_directory('ammr_bringup')
    world_file = os.path.join(pkg, 'worlds', 'random_room_dynamic.sdf')
    urdf_file  = os.path.join(pkg, 'urdf',   'ammr_base.urdf.xacro')
    map_file   = os.path.join(pkg, 'maps',   'random_room.yaml')
    traj_file  = os.path.join(pkg, 'config', 'dynamic_trajectories.yaml')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str
    )

    # 機器人 + 標準 bridge
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/gz/left_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
        '/gz/right_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
        '/gz/front_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
        '/gz/back_wheel_vel@std_msgs/msg/Float64]gz.msgs.Double',
    ]

    # 動態障礙物 bridge：ROS → GZ cmd_vel
    dyn_names = _load_dyn_names(traj_file)
    for name in dyn_names:
        bridge_args.append(
            f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'
        )

    return LaunchDescription([

        # 1. Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen',
        ),

        # 2. ros_gz_bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=bridge_args,
            output='screen',
        ),

        # 3. Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': True}],
        ),

        # 4. 靜態地圖 (只有 static obstacles)
        Node(
            package='ammr_bringup',
            executable='map_publisher',
            arguments=[map_file],
        ),

        # 5. Omni drive controller
        Node(
            package='ammr_bringup',
            executable='omni_drive_controller',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        # 6. base_link → lidar_link static TF
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.19', '0', '0', '0',
                       'base_link', 'lidar_link'],
            parameters=[{'use_sim_time': True}],
        ),

        # 7. Spawn robot（延遲 6 秒等 Gazebo + robot_description）
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

        # 8. /scan_raw → /scan relay (frame_id 修正)
        Node(
            package='ammr_bringup',
            executable='scan_relay',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        # 9. 動態障礙物驅動節點（延遲 8 秒等 Gazebo 完全 spawn dyn_obs_X）
        TimerAction(
            period=8.0,
            actions=[Node(
                package='ammr_bringup',
                executable='dynamic_obstacle_driver',
                parameters=[
                    {'use_sim_time': True},
                    {'trajectories_file': traj_file},
                ],
                output='screen',
            )],
        ),
    ])
