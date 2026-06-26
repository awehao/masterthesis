"""omni_bot baseline benchmark: Nav2 SOTA controllers (MPPI / RPP).

Same robot, same dynamic+unknown-static world, SAME localization (AMCL beam-skip
+ robot_localization EKF) as the GMPC stack -> fair comparison. The difference is
the local controller:
  method:=mppi  -> nav2 controller_server with MPPIController
  method:=rpp   -> nav2 controller_server with RegulatedPurePursuit

These baselines have NO CBF, so their costmap stays on /scan (they must see the
dynamic obstacles to react). Goal is sent on /goal_pose (default Nav2 BT).

Run (GUI):   ros2 launch my_omnibot_description omni_bot_baseline.launch.py method:=mppi
Headless:    ... gui:=false
"""
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def _dyn_names(traj_file):
    try:
        with open(traj_file) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return []
    return [d['name'] for d in cfg.get('dynamic_obstacles', [])]


def generate_launch_description():
    desc_pkg = get_package_share_directory('my_omnibot_description')
    bringup  = get_package_share_directory('ammr_bringup')

    urdf_file  = os.path.join(desc_pkg, 'urdf',   'omni_bot.urdf.xacro')
    world_file = os.path.join(bringup,  'worlds', 'random_room_dynamic.sdf')
    map_file   = os.path.join(bringup,  'maps',   'random_room.yaml')
    traj_file  = os.path.join(bringup,  'config', 'dynamic_trajectories.yaml')
    ekf_config = os.path.join(desc_pkg, 'config', 'ekf_fusion.yaml')
    mppi_cfg   = os.path.join(desc_pkg, 'config', 'nav2_baseline_mppi.yaml')
    rpp_cfg    = os.path.join(desc_pkg, 'config', 'nav2_baseline_rpp.yaml')

    gui    = LaunchConfiguration('gui')
    method = LaunchConfiguration('method')          # mppi | rpp
    speed_scale = LaunchConfiguration('obstacle_speed_scale')   # L1 difficulty knob
    is_mppi = PythonExpression(["'", method, "' == 'mppi'"])
    is_rpp  = PythonExpression(["'", method, "' == 'rpp'"])

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_camera:=false']), value_type=str)

    def params(cfg):
        return RewrittenYaml(source_file=cfg, convert_types=True,
                             param_rewrites={'yaml_filename': map_file,
                                             'tf_broadcast': 'false'})  # EKF owns map->odom
    mppi_params = params(mppi_cfg)
    rpp_params  = params(rpp_cfg)

    rp = os.path.dirname(desc_pkg)
    prev = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if prev:
        rp = rp + os.pathsep + prev

    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/odom_raw@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
    ]
    for name in _dyn_names(traj_file):
        bridge_args.append(f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist')
        bridge_args.append(f'/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose')

    nav_names = ['controller_server', 'smoother_server', 'planner_server',
                 'behavior_server', 'bt_navigator', 'waypoint_follower',
                 'velocity_smoother']

    def nav2_stack(cfg_params, cond):
        """Full Nav2 controller stack for one baseline (MPPI or RPP)."""
        common = dict(output='screen', parameters=[cfg_params], condition=cond)
        return [
            Node(package='nav2_controller', executable='controller_server',
                 name='controller_server', remappings=[('cmd_vel', 'cmd_vel_nav')], **common),
            Node(package='nav2_smoother', executable='smoother_server',
                 name='smoother_server', **common),
            Node(package='nav2_planner', executable='planner_server',
                 name='planner_server', **common),
            Node(package='nav2_behaviors', executable='behavior_server',
                 name='behavior_server', **common),
            Node(package='nav2_bt_navigator', executable='bt_navigator',
                 name='bt_navigator', **common),
            Node(package='nav2_waypoint_follower', executable='waypoint_follower',
                 name='waypoint_follower', **common),
            Node(package='nav2_velocity_smoother', executable='velocity_smoother',
                 name='velocity_smoother',
                 remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')],
                 **common),
        ]

    # localization + map (method-specific config so amcl/costmap params match)
    def loc_stack(cfg_params, cond):
        return [
            Node(package='nav2_map_server', executable='map_server', name='map_server',
                 output='screen', parameters=[cfg_params], condition=cond),
            Node(package='nav2_amcl', executable='amcl', name='amcl',
                 output='screen', parameters=[cfg_params], condition=cond),
        ]

    nav_nodes = (
        loc_stack(mppi_params, IfCondition(is_mppi)) +
        loc_stack(rpp_params,  IfCondition(is_rpp)) +
        [Node(package='robot_localization', executable='ekf_node', name='ekf_global',
              output='screen', parameters=[ekf_config])] +
        nav2_stack(mppi_params, IfCondition(is_mppi)) +
        nav2_stack(rpp_params,  IfCondition(is_rpp)) +
        [Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
              name='lifecycle_manager_localization', output='screen',
              parameters=[{'use_sim_time': True, 'autostart': True,
                           'node_names': ['map_server', 'amcl']}]),
         Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
              name='lifecycle_manager_navigation', output='screen',
              parameters=[{'use_sim_time': True, 'autostart': True,
                           'node_names': nav_names}])]
    )

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', rp),
        # Force gz/OGRE onto the NVIDIA dGPU (hybrid PRIME on-demand machine);
        # otherwise EGL routes to Mesa -> "driver (null)" + flickering viewport.
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('method', default_value='mppi',
                              description='mppi | rpp'),
        DeclareLaunchArgument('obstacle_speed_scale', default_value='1.0',
                              description='L1: scale dyn-obstacle speed (1.0=0.3m/s, '
                                          '2.0=0.6, 3.33=1.0)'),

        ExecuteProcess(cmd=['gz', 'sim', '-r', world_file], output='screen',
                       condition=IfCondition(gui)),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', world_file], output='screen',
                       condition=UnlessCondition(gui)),

        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=bridge_args, output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),
        Node(package='ammr_bringup', executable='odom_tf_broadcaster',
             parameters=[{'use_sim_time': True}], output='screen'),
        Node(package='ammr_bringup', executable='scan_relay',
             parameters=[{'use_sim_time': True,
                          'blocked_centers_deg': '45,135,225,315',
                          'blocked_halfwidth_deg': 15.0}], output='screen'),

        TimerAction(period=6.0, actions=[Node(
            package='ros_gz_sim', executable='create',
            arguments=['-name', 'omni_bot', '-topic', 'robot_description',
                       '-x', '0.0', '-y', '0.0', '-z', '0.05'], output='screen')]),
        TimerAction(period=8.0, actions=[Node(
            package='ammr_bringup', executable='dynamic_obstacle_driver',
            parameters=[{'use_sim_time': True}, {'trajectories_file': traj_file},
                        {'speed_scale': ParameterValue(speed_scale, value_type=float)}],
            output='screen')]),
        TimerAction(period=10.0, actions=nav_nodes),

        TimerAction(period=8.0, actions=[Node(
            package='foxglove_bridge', executable='foxglove_bridge',
            name='foxglove_bridge', output='screen',
            parameters=[{'use_sim_time': True, 'port': 8765}],
            condition=IfCondition(gui))]),
    ])
