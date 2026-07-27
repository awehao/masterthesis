"""Phase 2 bring-up: omni_bot WITH the UFACTORY Lite 6 arm, in gz Harmonic.

Spawns the combined robot (base + mast + 6-DOF arm), starts the ros2_control
controllers for the arm, and (optionally) RViz. The base is still driven by gz
VelocityControl on /cmd_vel exactly as in the validated navigation stack -- only
the arm goes through ros2_control -- so nothing about the 98% navigation result
changes.

    ros2 launch my_omnibot_description omni_bot_arm.launch.py

Arguments
    gui:=false        headless (gz sim -s)
    rviz:=false       do not start RViz
    world:=<file>     use a different world (default: the benchmark room)

Check that the arm did not blind the LiDAR:
    ros2 topic hz /scan          # should still be ~28 Hz
    ros2 topic echo /joint_states --once

Move the arm:
    ros2 action send_goal /lite6_traj_controller/follow_joint_trajectory \\
      control_msgs/action/FollowJointTrajectory  "{...}"   (see arm_demo.py)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            RegisterEventHandler, SetEnvironmentVariable,
                            TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc_pkg = get_package_share_directory('my_omnibot_description')
    bringup = get_package_share_directory('ammr_bringup')
    xarm_desc = get_package_share_directory('xarm_description')
    realsense = get_package_share_directory('realsense2_description')

    urdf_file = os.path.join(desc_pkg, 'urdf', 'omni_bot.urdf.xacro')
    default_world = os.path.join(bringup, 'worlds', 'random_room.sdf')
    map_file = os.path.join(bringup, 'maps', 'random_room.yaml')

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    world = LaunchConfiguration('world')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_arm:=true use_camera:=true']),
        value_type=str,
    )

    # gz resolves mesh URIs by searching these roots. xarm_description is added
    # for the Lite 6 meshes.
    resource_paths = os.pathsep.join([
        os.path.dirname(desc_pkg),
        os.path.dirname(xarm_desc),
        os.path.dirname(realsense),
    ])
    prev = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if prev:
        resource_paths += os.pathsep + prev

    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', 'omni_bot', '-topic', 'robot_description',
                   '-x', '0.0', '-y', '0.0', '-z', '0.05'],
        output='screen',
    )

    # controllers must be spawned AFTER the model exists in gz, otherwise the
    # controller_manager the plugin starts is not up yet.
    load_jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager',
                   '/controller_manager'],
        output='screen',
    )
    load_traj = Node(
        package='controller_manager', executable='spawner',
        arguments=['lite6_traj_controller', '--controller-manager',
                   '/controller_manager'],
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_paths),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('world', default_value=default_world),

        ExecuteProcess(cmd=['gz', 'sim', '-r', world], output='screen',
                       condition=IfCondition(gui)),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', world], output='screen',
                       condition=UnlessCondition(gui)),

        Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom_raw@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/base_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/base_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/base_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/base_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            output='screen',
        ),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description,
                          'use_sim_time': True}]),

        Node(package='ammr_bringup', executable='map_publisher',
             arguments=[map_file]),

        Node(package='ammr_bringup', executable='odom_tf_broadcaster',
             parameters=[{'use_sim_time': True}], output='screen'),

        # same LiDAR mask as the navigation stack; the arm risers sit inside
        # these already-blocked sectors on purpose (see arm_mount.xacro)
        Node(package='ammr_bringup', executable='scan_relay',
             parameters=[{'use_sim_time': True,
                          'blocked_centers_deg': '45,135,225,315',
                          'blocked_halfwidth_deg': 15.0}],
             output='screen'),

        TimerAction(period=6.0, actions=[spawn]),

        RegisterEventHandler(OnProcessExit(target_action=spawn,
                                           on_exit=[load_jsb])),
        RegisterEventHandler(OnProcessExit(target_action=load_jsb,
                                           on_exit=[load_traj])),

        TimerAction(period=10.0, actions=[Node(
            package='rviz2', executable='rviz2', name='rviz2',
            parameters=[{'use_sim_time': True}],
            output='screen', condition=IfCondition(rviz))]),
    ])
