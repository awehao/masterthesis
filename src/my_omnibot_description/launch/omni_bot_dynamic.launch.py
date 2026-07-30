"""omni_bot + dynamic obstacles + GMPC-CBF, obstacles detected from /scan.

This is the Sprint-B integration: instead of obstacle_aggregator (which reads
Gazebo ground-truth poses), it runs scan_obstacle_tracker, which detects and
tracks UNKNOWN dynamic obstacles purely from /scan (static-map subtraction +
clustering + per-track Kalman filter) and publishes /gmpc/obstacles. The
GMPC + horizon-CBF controller consumes /gmpc/obstacles unchanged.

  obstacle_source:=scan   (default) -> scan_obstacle_tracker  (real perception)
  obstacle_source:=truth            -> obstacle_aggregator     (ground-truth baseline)

Run (GUI):
  ros2 launch my_omnibot_description omni_bot_dynamic.launch.py
Headless batch:
  ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false
Send a goal:
  ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'map'}, pose: {position: {x: 17.0, y: 17.0}, orientation: {w: 1.0}}}" --once
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
    desc_pkg  = get_package_share_directory('my_omnibot_description')
    bringup   = get_package_share_directory('ammr_bringup')
    nav_pkg   = get_package_share_directory('ammr_navigation')
    wbmpc_pkg = get_package_share_directory('ammr_wholebody_mpc')

    urdf_file  = os.path.join(desc_pkg, 'urdf',   'omni_bot.urdf.xacro')
    world_file = os.path.join(bringup,  'worlds', 'random_room_dynamic.sdf')
    map_file   = os.path.join(bringup,  'maps',   'random_room.yaml')
    nav_params = os.path.join(desc_pkg, 'config', 'nav2_localization.yaml')
    gmpc_params= os.path.join(wbmpc_pkg,'config', 'gmpc_params.yaml')
    traj_file  = os.path.join(bringup,  'config', 'dynamic_trajectories.yaml')
    ekf_config = os.path.join(desc_pkg, 'config', 'ekf_fusion.yaml')

    gui          = LaunchConfiguration('gui')
    cbf          = LaunchConfiguration('cbf')
    robot_radius = LaunchConfiguration('robot_radius')
    inflation    = LaunchConfiguration('inflation')
    use_scan     = PythonExpression(["'", LaunchConfiguration('obstacle_source'), "' == 'scan'"])
    use_truth    = PythonExpression(["'", LaunchConfiguration('obstacle_source'), "' == 'truth'"])
    use_smoother = LaunchConfiguration('use_smoother')
    # When the velocity_smoother is ON, GMPC publishes to cmd_vel_nav and the
    # smoother rate-limits it onto /cmd_vel. When OFF, GMPC drives /cmd_vel
    # directly (one fewer lifecycle dependency -> fewer launch flakes; jerk is
    # already handled by gmpc.py's accept-suboptimal-OSQP-solution path).
    gmpc_cmd_topic = PythonExpression(
        ["'cmd_vel_nav' if '", use_smoother, "' == 'true' else 'cmd_vel'"])

    use_arm = LaunchConfiguration('use_arm')
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_camera:=false use_arm:=', use_arm]),
        value_type=str)

    configured_nav_params = RewrittenYaml(
        source_file=nav_params,
        param_rewrites={'yaml_filename': map_file,
                        'robot_radius': robot_radius,
                        'inflation_radius': inflation,
                        # EKF owns map->odom; AMCL (beam-skip) only emits /amcl_pose
                        'tf_broadcast': 'false'},
        convert_types=True)

    # gz mesh resolution for the chassis/roller STLs (see omni_bot_gazebo.launch)
    rp = os.pathsep.join([os.path.dirname(desc_pkg),
                          os.path.dirname(get_package_share_directory('xarm_description'))])
    prev = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if prev:
        rp = rp + os.pathsep + prev
    # gz_ros2_control-system lives here; without this path gz silently skips the
    # plugin, no controller_manager starts and the arm joints go limp.
    plugin_paths = os.pathsep.join(
        [q for q in ['/opt/ros/jazzy/lib',
                     os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')] if q])

    # one bridge: robot topics + per-obstacle cmd_vel (driver -> gz)
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/odom_raw@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
    ]
    for name in _dyn_names(traj_file):
        bridge_args.append(f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist')
        # pose feedback is for the DRIVER's closed-loop ping-pong (sim
        # infrastructure), NOT for perception — scan_obstacle_tracker only uses
        # /scan. Without it the driver never detects "reached" and the obstacle
        # drives straight through the robot.
        bridge_args.append(f'/model/{name}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose')

    cbf_overrides = {
        # safe_margin 0.38 = robot radius 0.30 + 0.08 m buffer. At exactly the
        # robot radius (old 0.30) the CBF target was "edge just grazes surface",
        # so any QP slack / scan-tracking latency penetrated the obstacle
        # (measured up to 7 cm). 0.08 m buffer keeps clearance >= 0 while still
        # fitting the map's ~0.9 m narrow passages (static keep-out 0.05+0.38).
        'cbf_enable': True, 'cbf_safe_margin': 0.38,
        # CBF class-K gain: h_dot >= -alpha*h. Smaller = the barrier engages
        # EARLIER and more gently, instead of waiting until it must spend the
        # whole acceleration budget. Measured at 3.0: min_h is under the danger
        # threshold ~48% of a run and the acceleration box saturates 21% of the
        # time, which is the "constant jitter" -- not any one event.
        'cbf_alpha': float(os.environ.get('CBF_ALPHA', '3.0')),
        'cbf_slack_weight': 5.0e2, 'cbf_eps0_scale': 30.0,
        'cbf_danger_thresh': 0.4, 'cbf_Q_min_scale': 0.20,
        'cbf_slack_max_scale': 20.0, 'obstacles_topic': '/gmpc/obstacles',
        # Wall-vs-dynamic slack pricing; overridable from the harness for A/B.
        'cbf_static_slack_scale': float(
            os.environ.get('CBF_STATIC_SLACK_SCALE', '1.0')),
        # Acceleration ceiling, overridable for the a_max study. jerk p95 comes
        # out EXACTLY at these numbers on every axis and every configuration
        # tested, which means the acceleration box is the binding constraint
        # more than 5% of the time; the sweep asks whether the reported jerk is
        # a controller defect or simply the limit the controller was given.
        # MPPI for comparison runs 1.5 / 1.0 / 2.0 and uses about 13% of it.
        'ax_max': float(os.environ.get('AX_MAX', '0.8')),
        'ay_max': float(os.environ.get('AY_MAX', '0.6')),
        'az_max': float(os.environ.get('AZ_MAX', '1.2')),
        # Spatio-temporal cost field (proactive detour). ST_WEIGHT=0 is the
        # validated configuration. CBF_ENABLE is separate so all four
        # combinations of {CBF, ST} can be benchmarked.
        'st_weight': float(os.environ.get('ST_WEIGHT', '0.0')),
        'st_sigma0': float(os.environ.get('ST_SIGMA0', '0.6')),
        'st_growth': float(os.environ.get('ST_GROWTH', '0.02')),
        'cbf_margin_growth': float(os.environ.get('CBF_MARGIN_GROWTH', '0.0')),
        'prog_weight': float(os.environ.get('PROG_WEIGHT', '0.0')),
        # Committed detour (detour.py): DETOUR=1 to enable.
        'detour_enable': os.environ.get('DETOUR', '0') == '1',
        # 0.35, the value the five validation rounds actually used. The
        # dataclass default of 0.60 was never validated end to end.
        'detour_max_offset': float(os.environ.get('DETOUR_OFFSET', '0.35')),
        'detour_trigger_range': float(os.environ.get('DETOUR_RANGE', '2.0')),
        'detour_vx_floor': float(os.environ.get('DETOUR_VX_FLOOR', '0.0')),
        'detour_clear_ref': os.environ.get('DETOUR_CLEAR_REF', '1') == '1',
        'detour_clear_pad': float(os.environ.get('DETOUR_CLEAR_PAD', '0.05')),
        'detour_side_proj': os.environ.get('DETOUR_SIDE_PROJ', '1') == '1',
        # Reference heading from a look-ahead chord; 0 = validated tangent.
        'ref_yaw_lookahead': float(os.environ.get('YAW_LOOKAHEAD', '0.0')),
        # Cross-fade seconds when a new /plan arrives; 0 = adopt instantly.
        'plan_blend_s': float(os.environ.get('PLAN_BLEND', '0.0')),
    }
    cbf_overrides['cbf_enable'] = os.environ.get('CBF_ENABLE', '1') == '1'

    nav_nodes = [
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen', parameters=[configured_nav_params]),
        Node(package='nav2_amcl', executable='amcl', name='amcl',
             output='screen', parameters=[configured_nav_params]),
        # EKF: fuse /odom (smooth) + /amcl_pose (beam-skip, jumps rejected)
        # -> stable map->odom. beam-skip cleans AMCL's input, EKF smooths output.
        Node(package='robot_localization', executable='ekf_node', name='ekf_global',
             output='screen', parameters=[ekf_config]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[configured_nav_params]),
        # lifecycle manager: two variants so the smoother isn't a managed node
        # when it's disabled (otherwise the manager blocks waiting for a node
        # that never appears -> whole nav stack fails to activate).
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['map_server', 'amcl', 'planner_server',
                                         'velocity_smoother']}],
             condition=IfCondition(use_smoother)),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['map_server', 'amcl', 'planner_server']}],
             condition=UnlessCondition(use_smoother)),
        # velocity_smoother (only when use_smoother:=true): rate-limit GMPC
        # output. gmpc -> cmd_vel_nav -> [smoother] -> /cmd_vel -> bridge -> gz.
        Node(package='nav2_velocity_smoother', executable='velocity_smoother',
             name='velocity_smoother', output='screen',
             parameters=[configured_nav_params],
             remappings=[('cmd_vel', 'cmd_vel_nav'),
                         ('cmd_vel_smoothed', 'cmd_vel')],
             condition=IfCondition(use_smoother)),
        Node(package='ammr_wholebody_mpc', executable='goal_to_plan_relay',
             name='goal_to_plan_relay', output='screen',
             parameters=[{'use_sim_time': True, 'global_frame': 'map',
                          'robot_base_frame': 'base_footprint',
                          'planner_id': 'GridBased', 'replan_period': 3.0}]),
        # GMPC + CBF (cbf:=true) or plain GMPC (cbf:=false). Publishes to
        # cmd_vel_nav so velocity_smoother (above) can rate-limit it.
        Node(package='ammr_wholebody_mpc', executable='gmpc_node',
             name='gmpc_controller', output='screen',
             parameters=[gmpc_params, cbf_overrides, {'cmd_vel_topic': gmpc_cmd_topic}],
             condition=IfCondition(cbf)),
        Node(package='ammr_wholebody_mpc', executable='gmpc_node',
             name='gmpc_controller', output='screen',
             parameters=[gmpc_params, {'cmd_vel_topic': gmpc_cmd_topic}],
             condition=UnlessCondition(cbf)),
        # Perception (scan mode): real /scan -> dynamic /gmpc/obstacles + static
        # wall points /gmpc/static_obstacles.
        Node(package='ammr_wholebody_mpc', executable='scan_obstacle_tracker',
             name='scan_obstacle_tracker', output='screen',
             condition=IfCondition(use_scan),
             parameters=[{'use_sim_time': True,
                          # A/B knobs; defaults reproduce the single-threshold
                          # gate the N=40 benchmark ran with.
                          'release_track_speed': float(
                              os.environ.get('TRACK_RELEASE_SPEED', '0.10')),
                          'track_release_frames': int(
                              os.environ.get('TRACK_RELEASE_FRAMES', '0'))}]),
        # Truth mode: obstacle_aggregator gives the DYNAMIC obstacles from ground
        # truth; we ALSO run scan_obstacle_tracker purely for the STATIC wall
        # points (its dynamic output is dumped to an unused topic) so gmpc_truth
        # gets the SAME static-CBF as gmpc_scan -> fair perception ablation.
        Node(package='ammr_wholebody_mpc', executable='scan_obstacle_tracker',
             name='scan_obstacle_tracker_static', output='screen',
             condition=IfCondition(use_truth),
             parameters=[{'use_sim_time': True,
                          'output_topic': '/gmpc/_scan_dyn_unused'}]),
        Node(package='ammr_wholebody_mpc', executable='obstacle_aggregator',
             name='obstacle_aggregator', output='screen',
             condition=IfCondition(use_truth),
             parameters=[{'use_sim_time': True, 'trajectories_file': traj_file,
                          'publish_rate': 20.0, 'output_topic': '/gmpc/obstacles',
                          'kf_sigma_meas': 0.05, 'kf_sigma_vel': 0.4,
                          'kf_sigma_pos': 0.01, 'kf_init_vel_var': 1.0}]),
    ]

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', rp),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', plugin_paths),
        DeclareLaunchArgument('gui', default_value='true'),
        # Phase 2: mount the Lite 6. Default false so the benchmark robot and
        # the recorded N=40 results stay exactly as validated.
        DeclareLaunchArgument('use_arm', default_value='false'),
        DeclareLaunchArgument('cbf', default_value='true'),
        DeclareLaunchArgument('obstacle_source', default_value='scan',
                              description='scan = real /scan perception; truth = ground-truth'),
        DeclareLaunchArgument('robot_radius', default_value='0.33'),
        DeclareLaunchArgument('inflation', default_value='0.45'),
        DeclareLaunchArgument('use_smoother', default_value='true',
                              description='true = GMPC->cmd_vel_nav->velocity_smoother->/cmd_vel; '
                                          'false = GMPC drives /cmd_vel directly'),

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

        # move the obstacles (ping-pong)
        TimerAction(period=8.0, actions=[Node(
            package='ammr_bringup', executable='dynamic_obstacle_driver',
            parameters=[{'use_sim_time': True}, {'trajectories_file': traj_file}],
            output='screen')]),

        # nav + control + perception (wait for gz/robot/TF)
        TimerAction(period=10.0, actions=nav_nodes),

        TimerAction(period=8.0, actions=[Node(
            package='foxglove_bridge', executable='foxglove_bridge',
            name='foxglove_bridge', output='screen',
            parameters=[{'use_sim_time': True, 'port': 8765}],
            condition=IfCondition(gui))]),
    ])
