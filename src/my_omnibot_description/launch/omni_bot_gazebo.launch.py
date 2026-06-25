import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc_pkg = get_package_share_directory('my_omnibot_description')
    bringup  = get_package_share_directory('ammr_bringup')
    realsense = get_package_share_directory('realsense2_description')

    urdf_file  = os.path.join(desc_pkg, 'urdf', 'omni_bot.urdf.xacro')
    world_file = os.path.join(bringup, 'worlds', 'random_room.sdf')
    map_file   = os.path.join(bringup, 'maps', 'random_room.yaml')

    use_camera = LaunchConfiguration('use_camera')
    with_map   = LaunchConfiguration('with_map')
    gui        = LaunchConfiguration('gui')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_camera:=', use_camera]),
        value_type=str,
    )

    # Let gz resolve package:// mesh URIs (chassis/roller + realsense D435 meshes).
    # GZ searches each path for "<pkg>/meshes/...", so we add the *parents* of the
    # package share dirs.
    resource_paths = os.pathsep.join([
        os.path.dirname(desc_pkg),    # .../install/my_omnibot_description/share
        os.path.dirname(realsense),   # .../opt/ros/jazzy/share
    ])
    prev_rp = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if prev_rp:
        resource_paths = resource_paths + os.pathsep + prev_rp

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_paths),
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='Spawn the RealSense D435i RGBD camera'),
        DeclareLaunchArgument('with_map', default_value='true',
                              description='Publish /map via ammr_bringup map_publisher. '
                                          'Set false when a Nav2 map_server provides /map.'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Run gz with GUI. Set false for headless batch '
                                          '(gz sim -s) so trials can run in the background.'),

        # 1. Gazebo Sim (Harmonic) — GUI when gui:=true, headless (-s) otherwise
        ExecuteProcess(cmd=['gz', 'sim', '-r', world_file], output='screen',
                       condition=IfCondition(gui)),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', world_file], output='screen',
                       condition=UnlessCondition(gui)),

        # 2. ROS2 <-> gz bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom_raw@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                # RealSense D435i (gz rgbd_camera, topic prefix "base_camera")
                '/base_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/base_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/base_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/base_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            output='screen',
        ),

        # 3. Robot state publisher (publishes base_link->lidar_link, camera frames, ...)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        ),

        # 4. Map publisher (reused from ammr_bringup) — off when Nav2 owns /map
        Node(package='ammr_bringup', executable='map_publisher', arguments=[map_file],
             condition=IfCondition(with_map)),

        # 5. Ground-truth odom: /odom_raw (gz, bridged) -> /odom + odom->base TF
        #    (replaces the dead-reckoning omni_drive_controller; reused from ammr_bringup)
        Node(package='ammr_bringup', executable='odom_tf_broadcaster',
             parameters=[{'use_sim_time': True}], output='screen'),

        # 6. scan_raw -> /scan relay (reused from ammr_bringup)
        Node(package='ammr_bringup', executable='scan_relay',
             parameters=[{'use_sim_time': True}], output='screen'),

        # 7. Spawn robot (wait for gz + robot_description)
        TimerAction(period=6.0, actions=[Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-name', 'omni_bot', '-topic', 'robot_description',
                       '-x', '0.0', '-y', '0.0', '-z', '0.05'],
            output='screen',
        )]),
    ])
