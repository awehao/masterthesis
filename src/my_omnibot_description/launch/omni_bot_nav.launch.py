"""One-shot navigation demo on the ported omni_bot (static world).

Brings up:
  1. gz Sim + omni_bot + bridge + odom + scan_relay   (our gazebo launch;
     map_publisher OFF — Nav2 map_server owns /map)
  2. Stripped Nav2 + SE(2) GMPC controller            (map_server + amcl +
     planner + goal_to_plan_relay + gmpc_node)
  3. foxglove_bridge                                  (ws://localhost:8765)

The shared baseline config (ammr_navigation/nav2_params_mppi.yaml) is NOT
edited; costmap robot_radius / inflation_radius are overridden at launch time
to fit the larger ported chassis (0.45 m square -> ~0.32 m half-diagonal).

Args:
  use_camera   (default false)  spawn the RealSense D435i too
  robot_radius (default 0.33)   costmap radius for the bigger chassis
  inflation    (default 0.45)   inflation radius (keep clearance from walls)

Send a goal (Foxglove Publish panel on /goal_pose, or):
    ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 5.0}, \
        orientation: {w: 1.0}}}" --once
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    desc_pkg    = get_package_share_directory('my_omnibot_description')
    bringup_pkg = get_package_share_directory('ammr_bringup')
    nav_pkg     = get_package_share_directory('ammr_navigation')
    wbmpc_pkg   = get_package_share_directory('ammr_wholebody_mpc')

    map_file    = os.path.join(bringup_pkg, 'maps',   'random_room.yaml')
    nav_params  = os.path.join(nav_pkg,     'config', 'nav2_params_mppi.yaml')
    gmpc_params = os.path.join(wbmpc_pkg,   'config', 'gmpc_params.yaml')
    traj_file   = os.path.join(bringup_pkg, 'config', 'dynamic_trajectories.yaml')

    use_camera   = LaunchConfiguration('use_camera')
    robot_radius = LaunchConfiguration('robot_radius')
    inflation    = LaunchConfiguration('inflation')
    gui          = LaunchConfiguration('gui')
    cbf          = LaunchConfiguration('cbf')

    # CBF param overrides (same values as gmpc_nav2_cbf.launch.py). On a static
    # world obstacle_aggregator finds no dynamic obstacles, so /gmpc/obstacles is
    # empty and the CBF rows are inactive -> behaves like plain GMPC, but the full
    # CBF stack is exercised.
    cbf_overrides = {
        'cbf_enable': True, 'cbf_alpha': 3.0, 'cbf_safe_margin': 0.30,
        'cbf_slack_weight': 5.0e2, 'cbf_eps0_scale': 30.0,
        'cbf_danger_thresh': 0.4, 'cbf_Q_min_scale': 0.20,
        'cbf_slack_max_scale': 20.0, 'obstacles_topic': '/gmpc/obstacles',
    }

    # Override map path + costmap sizing for the bigger chassis, without
    # touching the shared baseline file.
    configured_nav_params = RewrittenYaml(
        source_file=nav_params,
        param_rewrites={
            'yaml_filename':    map_file,
            'robot_radius':     robot_radius,
            'inflation_radius': inflation,
        },
        convert_types=True,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_pkg, 'launch', 'omni_bot_gazebo.launch.py')
        ),
        launch_arguments={'with_map': 'false', 'use_camera': use_camera,
                          'gui': gui}.items(),
    )

    nav_nodes = TimerAction(period=10.0, actions=[
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_amcl', executable='amcl', name='amcl',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['map_server', 'amcl', 'planner_server']}]),
        Node(package='ammr_wholebody_mpc', executable='goal_to_plan_relay',
             name='goal_to_plan_relay', output='screen',
             parameters=[{'use_sim_time': True, 'global_frame': 'map',
                          'robot_base_frame': 'base_footprint',
                          'planner_id': 'GridBased'}]),
        # plain GMPC (cbf:=false)
        Node(package='ammr_wholebody_mpc', executable='gmpc_node',
             name='gmpc_controller', output='screen', parameters=[gmpc_params],
             condition=UnlessCondition(cbf)),
        # GMPC + CBF (cbf:=true): CBF-enabled controller + obstacle aggregator
        Node(package='ammr_wholebody_mpc', executable='gmpc_node',
             name='gmpc_controller', output='screen',
             parameters=[gmpc_params, cbf_overrides],
             condition=IfCondition(cbf)),
        Node(package='ammr_wholebody_mpc', executable='obstacle_aggregator',
             name='obstacle_aggregator', output='screen',
             condition=IfCondition(cbf),
             parameters=[{'use_sim_time': True, 'trajectories_file': traj_file,
                          'publish_rate': 20.0, 'output_topic': '/gmpc/obstacles',
                          'kf_sigma_meas': 0.05, 'kf_sigma_vel': 0.4,
                          'kf_sigma_pos': 0.01, 'kf_init_vel_var': 1.0}]),
    ])

    # Foxglove only when interactive (skip for headless batch trials)
    foxglove = TimerAction(period=8.0, actions=[Node(
        package='foxglove_bridge', executable='foxglove_bridge',
        name='foxglove_bridge', output='screen',
        parameters=[{'use_sim_time': True, 'port': 8765}],
        condition=IfCondition(gui),
    )])

    return LaunchDescription([
        DeclareLaunchArgument('use_camera', default_value='false',
                              description='Also spawn the RealSense D435i camera'),
        DeclareLaunchArgument('robot_radius', default_value='0.33',
                              description='Costmap robot radius for the ported chassis'),
        DeclareLaunchArgument('inflation', default_value='0.45',
                              description='Costmap inflation radius'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='GUI + Foxglove. Set false for headless batch.'),
        DeclareLaunchArgument('cbf', default_value='false',
                              description='Enable GMPC+CBF (CBF controller + obstacle_aggregator).'),
        gazebo,
        foxglove,
        nav_nodes,
    ])
