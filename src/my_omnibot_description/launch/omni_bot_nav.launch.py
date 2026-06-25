"""One-shot navigation demo on the ported omni_bot.

Brings up, in order:
  1. gz Sim + omni_bot + bridge + odom + scan_relay   (our gazebo launch,
     map_publisher OFF because Nav2 map_server owns /map; camera OFF, not
     needed for nav)
  2. Stripped Nav2 + SE(2) GMPC controller            (ammr_wholebody_mpc
     gmpc_nav2.launch.py: map_server + amcl + planner + goal_to_plan_relay +
     gmpc_node)
  3. foxglove_bridge                                  (ws://localhost:8765)

Connect Foxglove Studio to ws://localhost:8765, then send a goal with a
Foxglove "Publish" panel on /goal_pose, or from a shell:
    ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 5.0}, \
        orientation: {w: 1.0}}}" --once
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    desc_pkg    = get_package_share_directory('my_omnibot_description')
    wbmpc_pkg   = get_package_share_directory('ammr_wholebody_mpc')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_pkg, 'launch', 'omni_bot_gazebo.launch.py')
        ),
        launch_arguments={'with_map': 'false', 'use_camera': 'false'}.items(),
    )

    # Nav2 + GMPC: wait for gz + robot + TF to be up
    nav = TimerAction(period=10.0, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(wbmpc_pkg, 'launch', 'gmpc_nav2.launch.py')
        )
    )])

    # Foxglove bridge: connect Foxglove Studio to ws://localhost:8765
    foxglove = TimerAction(period=8.0, actions=[Node(
        package='foxglove_bridge', executable='foxglove_bridge',
        name='foxglove_bridge', output='screen',
        parameters=[{'use_sim_time': True, 'port': 8765}],
    )])

    return LaunchDescription([gazebo, nav, foxglove])
