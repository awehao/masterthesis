"""View and drive the whole-body kinematics model, without gz.

omni_bot_wholebody.urdf.xacro has no simulation behind it: no gz, no
controllers, no hardware. Nothing publishes /joint_states for it, so on its own
a visualiser sees an empty scene -- the description exists but no TF tree is
ever computed from it. This launch supplies the two missing pieces:
robot_state_publisher to turn joint angles into TF, and joint_state_publisher
to invent the angles.

    ros2 launch my_omnibot_description omni_bot_wholebody.launch.py

Then open Foxglove on ws://localhost:8765 (this project's visualiser, not
RViz), add a 3D panel and set its frame to `world`.

Driving the 9 joints from the command line, base included:

    ros2 topic pub -1 /joint_states sensor_msgs/msg/JointState \\
      "{name: [base_x, base_y, base_theta, joint1, joint2, joint3, joint4,
               joint5, joint6],
        position: [1.0, 0.5, 0.3, 0.0, -0.082, 0.089, 0.0, 1.679, 0.0]}"

base_x / base_y / base_theta being ordinary joints is the whole point of the
model: the chassis moves by setting them, exactly as the arm does.

Mesh paths: the URDF carries absolute `file://` URIs because gz Harmonic cannot
resolve `package://`. Foxglove Desktop reads those; Foxglove Web will not,
because the browser has no access to the local filesystem. If the robot appears
as bare axes with no geometry, that is which viewer you are on, not a broken
model.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    desc_pkg = get_package_share_directory('my_omnibot_description')
    urdf = os.path.join(desc_pkg, 'urdf', 'omni_bot_wholebody.urdf.xacro')

    robot_description = ParameterValue(Command(['xacro ', urdf]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('foxglove', default_value='true'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen',
             parameters=[{'robot_description': robot_description,
                          'use_sim_time': False}]),

        # Publishes every movable joint at its default, which keeps the TF tree
        # complete so the model renders. Override any joint by publishing
        # /joint_states yourself; source_list makes this node yield to it
        # rather than fight it at 10 Hz.
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             output='screen',
             parameters=[{'rate': 10,
                          'source_list': ['/joint_states_cmd'],
                          'zeros': {'joint2': -0.082,
                                    'joint3': 0.089,
                                    'joint5': 1.679}}]),

        Node(package='foxglove_bridge', executable='foxglove_bridge',
             name='foxglove_bridge', output='screen',
             condition=IfCondition(LaunchConfiguration('foxglove')),
             parameters=[{'port': 8765}]),
    ])
