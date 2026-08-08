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
    # `or []`: see omni_bot_dynamic.launch.py -- an empty key parses to None.
    return [d['name'] for d in (cfg.get('dynamic_obstacles') or [])]


def generate_launch_description():
    desc_pkg = get_package_share_directory('my_omnibot_description')
    bringup  = get_package_share_directory('ammr_bringup')

    urdf_file  = os.path.join(desc_pkg, 'urdf',   'omni_bot.urdf.xacro')
    # Same world/scenario selection as omni_bot_dynamic.launch.py. Without this
    # the baselines could only run on random_room, so the GMPC numbers on
    # bigarena had nothing to be compared against -- the comparison table would
    # have mixed two different scenarios.
    if os.environ.get('BIGARENA', '0') == '1':
        world_file = os.path.join(bringup, 'worlds', 'bigarena.sdf')
        map_file   = os.path.join(bringup, 'maps',   'bigarena.yaml')
    elif os.environ.get('ARENA', '0') == '1':
        world_file = os.path.join(bringup, 'worlds', 'arena.sdf')
        map_file   = os.path.join(bringup, 'maps',   'arena.yaml')
    else:
        world_file = os.path.join(bringup,  'worlds', 'random_room_dynamic.sdf')
        map_file   = os.path.join(bringup,  'maps',   'random_room.yaml')
    _traj = os.environ.get('TRAJ', '')
    traj_file  = os.path.join(
        bringup, 'config',
        f'dynamic_trajectories_{_traj}.yaml' if _traj else 'dynamic_trajectories.yaml')
    ekf_config = os.path.join(desc_pkg, 'config', 'ekf_fusion.yaml')
    mppi_cfg   = os.path.join(desc_pkg, 'config', 'nav2_baseline_mppi.yaml')
    rpp_cfg    = os.path.join(desc_pkg, 'config', 'nav2_baseline_rpp.yaml')

    gui    = LaunchConfiguration('gui')
    method = LaunchConfiguration('method')          # mppi | rpp
    is_mppi = PythonExpression(["'", method, "' == 'mppi'"])
    is_rpp  = PythonExpression(["'", method, "' == 'rpp'"])

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_camera:=false']), value_type=str)

    def params(cfg):
        rw = {'yaml_filename': map_file,
              'tf_broadcast': 'false'}          # EKF owns map->odom
        # BASE_ACCEL=1 clamps the baseline's acceleration to the same box the
        # GMPC runs in (0.8 / 0.6 / 1.2). MPPI ships with 1.5 / 1.0 / 2.0 --
        # nearly double the authority on the same chassis. Acceleration is a
        # property of the robot, so the comparison should hold it fixed; the
        # horizon is not (MPPI sees 2.8 s to the GMPC's 1.0 s because sampling
        # tolerates prediction error that hard CBF constraints do not), so that
        # difference is left alone as part of each method's design.
        # BASE_ACCEL is obsolete: the baseline yamls now carry the same
        # hardware-derived limits as the GMPC (0.2775 m/s per axis, 6.25 m/s^2,
        # 1.1327 rad/s), so there is nothing left to clamp. The flag existed
        # when the GMPC was self-limited to 0.8/0.6/1.2 and the baselines were
        # not; both sides were wrong, and both are now the chassis's real box.
        if os.environ.get('BASE_ACCEL') == '1':
            rw.update({'ax_max': '6.25', 'ay_max': '6.25', 'az_max': '25.51'})
        return RewrittenYaml(source_file=cfg, convert_types=True,
                             param_rewrites=rw)
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
        # Hybrid-graphics (PRIME on-demand) machine: without these, gz/OGRE's
        # EGL is routed to Mesa, which cannot drive the NVIDIA card -- the sim
        # runs but the gpu_lidar returns nothing usable. The dynamic launch
        # carries the same three.
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('method', default_value='mppi',
                              description='mppi | rpp'),

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
                       # Random-pose batches drive the spawn per trial, exactly
                       # as omni_bot_dynamic.launch.py does; the batch harness
                       # exports these from its POSES_CSV row.
                       '-x', os.environ.get('SPAWN_X', '0.0'),
                       '-y', os.environ.get('SPAWN_Y', '0.0'),
                       '-z', '0.05'], output='screen')]),
        TimerAction(period=8.0, actions=[Node(
            package='ammr_bringup', executable='dynamic_obstacle_driver',
            parameters=[{'use_sim_time': True}, {'trajectories_file': traj_file}],
            output='screen')]),
        TimerAction(period=10.0, actions=nav_nodes),

        TimerAction(period=8.0, actions=[Node(
            package='foxglove_bridge', executable='foxglove_bridge',
            name='foxglove_bridge', output='screen',
            parameters=[{'use_sim_time': True, 'port': 8765}],
            condition=IfCondition(gui))]),
    ])
