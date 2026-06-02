"""GMPC (no CBF) with an obstacle-BLIND global planner.

Companion to gmpc_nav2_cbf_blind.launch.py for an A/B test:
  • this launch  : planner blind, CBF off  → robot SHOULD crash into cylinders
  • _cbf_blind   : planner blind, CBF on   → robot SHOULD avoid (CBF carrying)

Same Gazebo world (gazebo_dynamic.launch.py dynamic:=false).
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
    nav_params  = os.path.join(nav_pkg,     'config', 'nav2_params_mppi_blind.yaml')
    gmpc_params = os.path.join(wbmpc_pkg,   'config', 'gmpc_params.yaml')

    configured_nav_params = RewrittenYaml(
        source_file=nav_params,
        param_rewrites={'yaml_filename': map_file},
        convert_types=True,
    )

    return LaunchDescription([
        Node(package='nav2_map_server',
             executable='map_server', name='map_server',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_amcl',
             executable='amcl', name='amcl',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_planner',
             executable='planner_server', name='planner_server',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_lifecycle_manager',
             executable='lifecycle_manager',
             name='lifecycle_manager_navigation',
             output='screen',
             parameters=[{
                'use_sim_time': True,
                'autostart':    True,
                'node_names':   ['map_server', 'amcl', 'planner_server'],
             }]),
        Node(package='ammr_wholebody_mpc',
             executable='goal_to_plan_relay',
             name='goal_to_plan_relay', output='screen',
             parameters=[{'use_sim_time': True,
                          'global_frame': 'map',
                          'robot_base_frame': 'base_footprint',
                          'planner_id': 'GridBased'}]),
        # No obstacle_aggregator and CBF off
        Node(package='ammr_wholebody_mpc',
             executable='gmpc_node',
             name='gmpc_controller', output='screen',
             parameters=[
                 gmpc_params,
                 {'cbf_enable': False},
             ]),
    ])
