"""Run GMPC alongside a stripped-down Nav2 stack.

Stack composition
-----------------
  - map_server         (random_room.yaml)
  - amcl               (Omni motion model — same config as Baseline B)
  - planner_server     (SmacPlanner2D — same config as Baseline B)
  - lifecycle_manager_navigation
  - goal_to_plan_relay (our bridge: /goal_pose → ComputePathToPose action)
  - gmpc_controller    (our SE(2) GMPC node)

What is intentionally NOT launched (replaced by GMPC):
  - controller_server     (GMPC takes its place)
  - velocity_smoother     (GMPC has built-in acceleration limits)
  - behavior_server       (no spin/backup recovery yet)
  - bt_navigator          (we bypass the BT — goal_to_plan_relay handles the
                          /goal_pose → /plan flow directly)

Assumes Gazebo + the omni chassis are already running:
    ros2 launch ammr_bringup gazebo.launch.py

Then in a second shell:
    ros2 launch ammr_wholebody_mpc gmpc_nav2.launch.py

Send goals via RViz "2D Goal Pose" or:
    ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
        "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 5.0}, \
          orientation: {w: 1.0}}}" --once
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch                       import LaunchDescription
from launch_ros.actions           import Node
from nav2_common.launch           import RewrittenYaml


def generate_launch_description():
    bringup_pkg  = get_package_share_directory('ammr_bringup')
    nav_pkg      = get_package_share_directory('ammr_navigation')
    wbmpc_pkg    = get_package_share_directory('ammr_wholebody_mpc')

    map_file     = os.path.join(bringup_pkg, 'maps',   'random_room.yaml')
    nav_params   = os.path.join(nav_pkg,     'config', 'nav2_params_mppi.yaml')
    gmpc_params  = os.path.join(wbmpc_pkg,   'config', 'gmpc_params.yaml')

    # Inject map path into map_server.yaml_filename
    configured_nav_params = RewrittenYaml(
        source_file=nav_params,
        param_rewrites={'yaml_filename': map_file},
        convert_types=True,
    )

    return LaunchDescription([

        # ---- Nav2 servers we keep -------------------------------------
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[configured_nav_params],
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[configured_nav_params],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[configured_nav_params],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart':    True,
                'node_names':   ['map_server', 'amcl', 'planner_server'],
            }],
        ),

        # ---- Our nodes -------------------------------------------------
        Node(
            package='ammr_wholebody_mpc',
            executable='goal_to_plan_relay',
            name='goal_to_plan_relay',
            output='screen',
            parameters=[{
                'use_sim_time':     True,
                'global_frame':     'map',
                'robot_base_frame': 'base_footprint',
                'planner_id':       'GridBased',
            }],
        ),
        Node(
            package='ammr_wholebody_mpc',
            executable='gmpc_node',
            name='gmpc_controller',
            output='screen',
            parameters=[gmpc_params],
        ),
    ])
