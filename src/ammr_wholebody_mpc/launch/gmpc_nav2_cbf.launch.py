"""GMPC + CBF stack for dynamic-obstacle benchmark.

Same as gmpc_nav2.launch.py but with:
  - CBF enabled in the GMPC controller (parameter override)
  - obstacle_aggregator node bridging Gazebo ground-truth pose -> /gmpc/obstacles

Assumes Gazebo *dynamic* world is running:
    ros2 launch ammr_bringup gazebo_dynamic.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch                       import LaunchDescription
from launch_ros.actions           import Node
from nav2_common.launch           import RewrittenYaml


def generate_launch_description():
    bringup_pkg = get_package_share_directory('ammr_bringup')
    nav_pkg     = get_package_share_directory('ammr_navigation')
    wbmpc_pkg   = get_package_share_directory('ammr_wholebody_mpc')

    map_file    = os.path.join(bringup_pkg, 'maps',   'random_room.yaml')
    nav_params  = os.path.join(nav_pkg,     'config', 'nav2_params_mppi.yaml')
    gmpc_params = os.path.join(wbmpc_pkg,   'config', 'gmpc_params.yaml')
    traj_file   = os.path.join(bringup_pkg, 'config', 'dynamic_trajectories.yaml')

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

        # ---- Goal-to-plan relay --------------------------------------------
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

        # ---- Obstacle aggregator (Gazebo poses -> /gmpc/obstacles) ---------
        Node(
            package='ammr_wholebody_mpc',
            executable='obstacle_aggregator',
            name='obstacle_aggregator',
            output='screen',
            parameters=[{
                'use_sim_time':       True,
                'trajectories_file':  traj_file,
                'publish_rate':       20.0,
                'output_topic':       '/gmpc/obstacles',
            }],
        ),

        # ---- GMPC controller with CBF ENABLED ----------------------
        Node(
            package='ammr_wholebody_mpc',
            executable='gmpc_node',
            name='gmpc_controller',
            output='screen',
            parameters=[
                gmpc_params,                       # base config from yaml
                {                                  # CBF overrides
                    'cbf_enable':       True,
                    'cbf_alpha':        2.0,
                    'cbf_safe_margin':  0.30,
                    'cbf_slack_weight': 1.0e4,
                    'obstacles_topic':  '/gmpc/obstacles',
                },
            ],
        ),
    ])
