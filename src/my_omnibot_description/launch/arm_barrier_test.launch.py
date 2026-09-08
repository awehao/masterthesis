"""Dedicated launch for the link safety layer's in-loop test.

Separate from omni_bot_arm.launch.py, which stays as 5B validated it. Three
things differ and each exists because it caused a failure the last time round:

  world       arm_barrier_test.sdf, one box whose pose lives only in that file.
              The distance node's obstacle list is generated from the same
              file, so simulator and safety layer agree by construction. Moving
              a box at runtime with set_pose left the node's static parameter
              behind and the two ends were looking at different scenes.

  controller  the velocity set, so the filter's output reaches the joints
              without an integrator in between. Exactly one arm command
              controller is activated.

  gui         defaults to TRUE. In this environment the headless configuration
              updated the controller reference and advanced the simulation
              clock while the joints did not move; with the GUI the same
              trajectory executes. The underlying cause is not identified, so
              the working configuration is the default and the observation is
              recorded rather than explained.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc = get_package_share_directory('my_omnibot_description')
    bringup = get_package_share_directory('ammr_bringup')
    urdf = os.path.join(desc, 'urdf', 'omni_bot.urdf.xacro')
    world = os.path.join(bringup, 'worlds', 'arm_barrier_test.sdf')

    gui = LaunchConfiguration('gui')
    robot_desc = Command([
        'xacro ', urdf, ' use_arm:=true add_gripper:=true',
        ' controllers_yaml:=lite6_velocity_controllers.yaml'])

    resource = ':'.join(
        p for p in (os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
                    os.path.dirname(desc), os.path.dirname(bringup)) if p)

    # gz loads <plugin filename="gz_ros2_control-system"> from this path only.
    # It was missing here and the launch happened to work anyway, because the
    # shell that first ran it already had the variable exported. Started from a
    # clean environment the plugin is silently skipped, no controller_manager
    # ever appears, every spawner retries until it gives up, and with nothing
    # holding the joints the arm collapses under gravity.
    plugin_paths = os.pathsep.join(
        [p for p in ['/opt/ros/jazzy/lib',
                     os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')] if p])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', plugin_paths),
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),

        ExecuteProcess(cmd=['gz', 'sim', '-r', world], output='screen',
                       condition=IfCondition(gui)),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', world], output='screen',
                       condition=UnlessCondition(gui)),

        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
             output='screen'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             # value_type=str, or launch tries to parse the URDF as YAML and
             # dies on the first colon in the XML.
             parameters=[{'robot_description': ParameterValue(robot_desc,
                                                              value_type=str),
                          'use_sim_time': True}],
             output='screen'),

        # The model has to exist before any spawner runs: a controller_manager
        # that is not there yet makes the spawner wait its whole timeout and
        # then fail.
        TimerAction(period=6.0, actions=[
            Node(package='ros_gz_sim', executable='create',
                 arguments=['-name', 'omni_bot', '-topic', 'robot_description',
                            '-x', '0.0', '-y', '0.0', '-z', '0.0'],
                 output='screen')]),

        TimerAction(period=10.0, actions=[
            Node(package='controller_manager', executable='spawner',
                 arguments=['joint_state_broadcaster', '--controller-manager',
                            '/controller_manager',
                            '--controller-manager-timeout', '60'],
                 output='screen')]),
        TimerAction(period=13.0, actions=[
            Node(package='controller_manager', executable='spawner',
                 arguments=['lite6_vel_controller', '--controller-manager',
                            '/controller_manager',
                            '--controller-manager-timeout', '60'],
                 output='screen')]),
        TimerAction(period=16.0, actions=[
            Node(package='controller_manager', executable='spawner',
                 arguments=['lite6_gripper_controller', '--controller-manager',
                            '/controller_manager',
                            '--controller-manager-timeout', '60'],
                 output='screen')]),
    ])
