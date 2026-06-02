"""GMPC + CBF stack with an obstacle-BLIND global planner.

Same as gmpc_nav2_cbf.launch.py except the planner uses
`nav2_params_mppi_blind.yaml`, which removes the `obstacle_layer` from
`global_costmap`. The planner therefore only sees the static map and the
SmacPlanner-generated /plan will pass straight through any unmapped cylinder.

This isolates the contribution of the CBF safety filter:
  • path → DOES intersect obstacles
  • only the CBF can avoid them

Pair with:
    ros2 launch ammr_bringup gazebo_dynamic.launch.py dynamic:=false
to ensure the cylinders sit on the path.
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
    traj_file   = os.path.join(bringup_pkg, 'config', 'dynamic_trajectories.yaml')

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
        Node(package='ammr_wholebody_mpc',
             executable='obstacle_aggregator',
             name='obstacle_aggregator', output='screen',
             parameters=[{'use_sim_time': True,
                          'trajectories_file': traj_file,
                          'publish_rate': 20.0,
                          'output_topic': '/gmpc/obstacles'}]),
        Node(package='ammr_wholebody_mpc',
             executable='gmpc_node',
             name='gmpc_controller', output='screen',
             parameters=[
                 gmpc_params,
                 # CBF with gain scheduling (FIX-2):
                 # Q drops to 20% and slack 30x when robot enters danger zone.
                 {'cbf_enable':           True,
                  'cbf_alpha':            5.0,
                  'cbf_safe_margin':      0.35,
                  'cbf_slack_weight':     1.0e4,
                  'cbf_danger_thresh':    0.5,
                  'cbf_Q_min_scale':      0.2,
                  'cbf_slack_max_scale':  30.0,
                  'obstacles_topic':      '/gmpc/obstacles'},
             ]),
    ])
