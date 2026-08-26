"""Phase 2 bring-up: omni_bot WITH the UFACTORY Lite 6 arm, in gz Harmonic.

Spawns the combined robot (base + mast + 6-DOF arm), starts the ros2_control
controllers for the arm, and the Foxglove bridge. The base is still driven by gz
VelocityControl on /cmd_vel exactly as in the validated navigation stack -- only
the arm goes through ros2_control -- so nothing about the 98% navigation result
changes.

    ros2 launch my_omnibot_description omni_bot_arm.launch.py

Arguments
    gui:=false        headless (gz sim -s)
    foxglove:=false   do not start the Foxglove bridge (ws://localhost:8765)
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
    foxglove = LaunchConfiguration('foxglove')
    world = LaunchConfiguration('world')
    use_arm = LaunchConfiguration('use_arm')
    add_gripper = LaunchConfiguration('add_gripper')

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_arm:=', use_arm,
                 ' add_gripper:=', add_gripper, ' use_camera:=true']),
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

    # gz loads <plugin filename="gz_ros2_control-system"> from this path only.
    # Without it the plugin is silently skipped, no controller_manager ever
    # starts, the spawners hang forever, and with no controller holding the
    # joints the arm simply collapses under gravity.
    plugin_paths = os.pathsep.join(
        [p for p in ['/opt/ros/jazzy/lib',
                     os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')] if p])

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
                   '/controller_manager', '--controller-manager-timeout', '60'],
        output='screen',
    )
    load_traj = Node(
        package='controller_manager', executable='spawner',
        arguments=['lite6_traj_controller', '--controller-manager',
                   '/controller_manager', '--controller-manager-timeout', '60'],
        output='screen',
    )

    # Conditional, not listed unconditionally: configuring a controller whose
    # joint the hardware does not expose fails the whole controller_manager, so
    # without the gripper in the model this must not be spawned at all.
    load_gripper = Node(
        package='controller_manager', executable='spawner',
        arguments=['lite6_gripper_controller', '--controller-manager',
                   '/controller_manager', '--controller-manager-timeout', '60'],
        condition=IfCondition(add_gripper),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('add_gripper', default_value='false'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_paths),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', plugin_paths),
        # Hybrid-graphics (PRIME on-demand) machine: without these, gz/OGRE's EGL
        # is routed to Mesa, which cannot drive the NVIDIA card, giving
        # "failed to create dri2 screen", a software-rendered viewport and a
        # gpu_lidar that produces no /scan_raw. Same three variables the
        # navigation launches use.
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('world', default_value=default_world),
        # use_arm:=false spawns the plain navigation robot through this same
        # launch, which is what makes an arm / no-arm A-B comparison fair.
        DeclareLaunchArgument('use_arm', default_value='true'),

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
        RegisterEventHandler(OnProcessExit(target_action=load_traj,
                                           on_exit=[load_gripper])),

        # Foxglove (ws://localhost:8765) is the visualiser used in this project,
        # not RViz.
        TimerAction(period=8.0, actions=[Node(
            package='foxglove_bridge', executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{'use_sim_time': True, 'port': 8765}],
            output='screen', condition=IfCondition(foxglove))]),
    ])
