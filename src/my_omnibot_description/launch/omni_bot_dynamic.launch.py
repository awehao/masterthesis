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
    # `or []` and not just the get() default: a YAML with the key present
    # but empty ('dynamic_obstacles:' and nothing under it) parses to None,
    # and the default only applies when the key is MISSING. That made the
    # no-movers control scenario fail bring-up on all ten trials.
    return [d['name'] for d in (cfg.get('dynamic_obstacles') or [])]


def generate_launch_description():
    desc_pkg  = get_package_share_directory('my_omnibot_description')
    bringup   = get_package_share_directory('ammr_bringup')
    nav_pkg   = get_package_share_directory('ammr_navigation')
    wbmpc_pkg = get_package_share_directory('ammr_wholebody_mpc')

    urdf_file  = os.path.join(desc_pkg, 'urdf',   'omni_bot.urdf.xacro')
    # ARENA=1 swaps the 20 m random room for the 9 m purpose-built arena: known
    # walls, unknown static cylinders, unknown movers, and a divider with two
    # gaps so route choice is a real decision. A trial there costs ~45 s instead
    # of ~170 s, which is the cheapest way to buy the statistical power that
    # collision rate needs -- at n=10 its 95% interval spans 20-30 points.
    # BIGARENA=1 is the 20 x 20 m floor: eight bays joined by staggered 2.0-2.5 m
    # doorways, known and unknown clutter placed by rejection sampling, and
    # traffic making long traversals. The 9 m arena has essentially one route,
    # so anything put on it is a roadblock; this one has alternatives, which is
    # what makes route choice a decision rather than a formality.
    if os.environ.get('BIGARENA', '0') == '1':
        # SCENE names a member of the bigarena family: same generator, same
        # rules, different seed. Defaults to the development scene, so every
        # earlier result is reproduced unchanged.
        _scene = os.environ.get('SCENE', 'bigarena')
        world_file = os.path.join(bringup, 'worlds', f'{_scene}.sdf')
        map_file   = os.path.join(bringup, 'maps',   f'{_scene}.yaml')
    elif os.environ.get('ARENA', '0') == '1':
        world_file = os.path.join(bringup, 'worlds', 'arena.sdf')
        map_file   = os.path.join(bringup, 'maps',   'arena.yaml')
    else:
        world_file = os.path.join(bringup,  'worlds', 'random_room_dynamic.sdf')
        map_file   = os.path.join(bringup,  'maps',   'random_room.yaml')
    nav_params = os.path.join(desc_pkg, 'config', 'nav2_localization.yaml')
    gmpc_params= os.path.join(wbmpc_pkg,'config', 'gmpc_params.yaml')
    # Scenario selection. Every scenario is just a trajectories YAML -- the world
    # provides four dynamic model slots and four unknown static cylinders, so a
    # new encounter geometry costs a file, not a world edit.
    # TRAJ=crossing -> config/dynamic_trajectories_crossing.yaml
    _traj = os.environ.get('TRAJ', '')
    if os.environ.get('BIGARENA', '0') == '1' and not _traj:
        _traj = f'{_scene}_traffic'
    traj_file  = os.path.join(
        bringup, 'config',
        f'dynamic_trajectories_{_traj}.yaml' if _traj else 'dynamic_trajectories.yaml')
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

    # Read once: the smoother's output topic depends on it, and a launch-time
    # substitution cannot be used to pick a remapping.
    _shield = os.environ.get('SHIELD', '0') == '1'
    truth_pose = PythonExpression(
        ["'", str(os.environ.get('TRUTH_POSE', '0')), "' == '1'"])
    use_arm = LaunchConfiguration('use_arm')
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file, ' use_camera:=false use_arm:=', use_arm]),
        value_type=str)

    configured_nav_params = RewrittenYaml(
        source_file=nav_params,
        param_rewrites={'yaml_filename': map_file,
                        'robot_radius': robot_radius,
                        'inflation_radius': inflation,
                        # Which scan the PLANNER's costmap marks from.
                        # /scan          -> it sees movers at their CURRENT position,
                        #                   so the plan is re-routed around a target
                        #                   that has moved by the time we get there:
                        #                   measured 24% of replans shifting the path
                        #                   ahead of the robot by >0.25 m (max 1.82 m,
                        #                   heading up to 63 deg), and in the 0.3 s
                        #                   after such a jump the controller sits at
                        #                   0.96 of a_max, saturating 88.5% of the
                        #                   time against 13.5% otherwise.
                        # /scan_filtered -> movers masked out (the tracker already
                        #                   publishes this); the planner sees only
                        #                   static geometry, so its cost field stops
                        #                   changing and the plan stops flipping.
                        #                   Dynamic avoidance is then entirely the
                        #                   CBF's job. NOTE this also means the plan
                        #                   will route straight down a corridor a
                        #                   mover sweeps -- see the envelope layer
                        #                   discussion; the two are complementary.
                        # Only the GLOBAL costmap is affected here: this launch runs
                        # no controller_server, so the local costmap block is inert.
                        'topic': os.environ.get('PLANNER_SCAN', '/scan'),
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
        # The surface points carry radius 0 and the CBF treats the robot as a
        # point, so the robot's own 0.30 m lives INSIDE this number: 0.38 buys
        # 0.08 m of real buffer. One control period at a 0.5 m/s closing speed
        # is already 0.025 m, so 0.08 leaves almost nothing for perception
        # latency -- the measured graze stopped at exactly 0.300 m, i.e. exactly
        # HORIZON: how far ahead the barrier can see, in control steps of dt.
        #
        # At 20 steps x 0.05 s the CBF sees 1.0 s. Against a mover closing at
        # 0.30 + 0.22 = 0.52 m/s that is 0.52 m of approach, so a 0.45 m
        # keep-out only becomes active when the obstacle is ~0.97 m away --
        # 1.3 s before contact. In 1.3 s the only thing the robot can do is
        # stop, and stopping does not work against something that is still
        # moving: of 24 contacts measured over 146 trials, 23 were with the
        # fastest mover and the robot was doing 0.03-0.09 m/s at the moment of
        # contact while the obstacle did 0.30. It was not cutting a corner --
        # it had braked, and been driven into. Seeing the whole encounter needs
        # 2-3 s. The QP grows with N, but there is room: solve time is 0.18 ms.
        'horizon': int(os.environ.get('HORIZON', '20')),
        'cbf_enable': True,
        'cbf_safe_margin': float(os.environ.get('CBF_SAFE_MARGIN', '0.38')),
        # CBF class-K gain: h_dot >= -alpha*h. Smaller = the barrier engages
        # EARLIER and more gently, instead of waiting until it must spend the
        # whole acceleration budget. Measured at 3.0: min_h is under the danger
        # threshold ~48% of a run and the acceleration box saturates 21% of the
        # time, which is the "constant jitter" -- not any one event.
        'cbf_alpha': float(os.environ.get('CBF_ALPHA', '3.0')),
        # How hard the QP is punished for relaxing a CBF constraint, relative to
        # tracking. At 5e2 the measured wall clearance settles at +0.044 m against
        # a +0.08 m target -- the QP is choosing to give up clearance to stay on
        # the path. Raising this shifts that trade towards clearance.
        'cbf_slack_weight': float(os.environ.get('CBF_SLACK_W', '5.0e2')),
        'cbf_eps0_scale': 30.0,
        'cbf_danger_thresh': 0.4, 'cbf_Q_min_scale': 0.20,
        'cbf_slack_max_scale': 20.0, 'obstacles_topic': '/gmpc/obstacles',
        # Wall-vs-dynamic slack pricing; overridable from the harness for A/B.
        'static_cbf_safe_margin': float(
            os.environ.get('STATIC_MARGIN', '0.38')),
        'cbf_static_slack_scale': float(
            os.environ.get('CBF_STATIC_SLACK_SCALE', '1.0')),
        # Acceleration ceiling, overridable for the a_max study. jerk p95 comes
        # out EXACTLY at these numbers on every axis and every configuration
        # tested, which means the acceleration box is the binding constraint
        # more than 5% of the time; the sweep asks whether the reported jerk is
        # a controller defect or simply the limit the controller was given.
        # MPPI for comparison runs 1.5 / 1.0 / 2.0 and uses about 13% of it.
        # Defaults now match gmpc_params.yaml, which carries the hardware-derived
        # limits (wheel alpha_max 125 rad/s^2 -> 6.25 m/s^2 per axis). The old
        # 0.8/0.6/1.2 silently overrode the yaml -- they are applied AFTER it --
        # so every result before this ran at an eighth of the chassis's actual
        # acceleration, and the "stage 4: full acceleration" comparison measured
        # nothing.
        'ax_max': float(os.environ.get('AX_MAX', '6.25')),
        'ay_max': float(os.environ.get('AY_MAX', '6.25')),
        'az_max': float(os.environ.get('AZ_MAX', '25.51')),
        # Spatio-temporal cost field (proactive detour). ST_WEIGHT=0 is the
        # validated configuration. CBF_ENABLE is separate so all four
        # combinations of {CBF, ST} can be benchmarked.
        'st_weight': float(os.environ.get('ST_WEIGHT', '0.0')),
        'st_sigma0': float(os.environ.get('ST_SIGMA0', '0.6')),
        'st_growth': float(os.environ.get('ST_GROWTH', '0.02')),
        'cbf_margin_growth': float(os.environ.get('CBF_MARGIN_GROWTH', '0.0')),
        # Extra keep-out proportional to the closing rate [s]. A fixed margin
        # must be sized for the worst approach and then costs that width in
        # every corridor: at 0.75 m arrival fell and contacts rose, because a
        # 2.0 m doorway leaves only 0.5 m of feasible corridor. 0 = fixed.
        'cbf_vel_margin_gain': float(os.environ.get('CBF_VEL_MARGIN', '0.0')),
        # Constraint pruning. Rows are n_obs x N and n_obs counts SURFACE
        # POINTS: measured 8 at the median (160 rows) and 40 at the peak (800),
        # with OSQP's p99 solve time at 425 ms against a 50 ms period. A point
        # 3 m away cannot be violated inside a 1 s horizon, and the far half of
        # the horizon is constant-velocity extrapolation anyway. 0 = keep all.
        # MARGIN_MODE=derived sizes each keep-out from what it has to absorb
        # (localisation error + braking distance + one control step, plus the
        # obstacle's own travel during the perception lag) instead of the tuned
        # 0.38 / 0.60. It collapses as the robot slows, which is what lets it
        # settle on a goal near a wall -- the fixed 0.68 m keep-out put such a
        # goal inside the barrier and the robot ground along the wall instead.
        'margin_mode': os.environ.get('MARGIN_MODE', 'fixed'),
        # HARD_K0=1 pins the k=0 slack to zero: the barrier becomes a hard
        # constraint at the current state. When it cannot be met the QP is
        # primal infeasible and gmpc.py falls through to the emergency-brake
        # path (u = 0) -- so this trades a shallow violation for stopping, which
        # against an approaching mover is not obviously better. Measured, not
        # assumed.
        # POSE_SOURCE=odom takes map->base_footprint from the EKF topic
        # instead of composing it through TF. See gmpc_node's pose_source.
        'pose_source': os.environ.get('POSE_SOURCE', 'tf'),
        'cbf_hard_k0': os.environ.get('HARD_K0', '0') == '1',
        # Walls only, and only at k = 0. Separate knob from HARD_K0
        # because that one hardens the dynamic rows too (shared slack).
        'cbf_hard_k0_static': os.environ.get('HARD_K0_STATIC', '0') == '1',
        'cbf_prune_range': float(os.environ.get('CBF_PRUNE_RANGE', '0.0')),
        'cbf_near_steps': int(os.environ.get('CBF_NEAR_STEPS', '6')),
        'cbf_far_stride': int(os.environ.get('CBF_FAR_STRIDE', '1')),
        'prog_weight': float(os.environ.get('PROG_WEIGHT', '0.0')),
        # Committed detour (detour.py): DETOUR=1 to enable.
        'detour_enable': os.environ.get('DETOUR', '0') == '1',
        # The lateral lane the detour aims for must clear the keep-out, which
        # is the mover's radius plus cbf_safe_margin = 0.25 + 0.60 = 0.85 m. The
        # old 0.35 was chosen when the margin was 0.38 (keep-out 0.63) and never
        # revisited, so the offset reference stayed inside the barrier and the
        # detour could not resolve anything.
        'detour_max_offset': float(os.environ.get('DETOUR_OFFSET', '0.90')),
        # Far enough that the ramp finishes before the encounter: 0.9 m at
        # 0.01 m/step is 90 steps = 4.5 s, and 2.5 m of approach at
        # 0.28 m/s gives about 9 s.
        'detour_trigger_range': float(os.environ.get('DETOUR_RANGE', '2.5')),
        'detour_vx_floor': float(os.environ.get('DETOUR_VX_FLOOR', '0.0')),
        'detour_clear_ref': os.environ.get('DETOUR_CLEAR_REF', '1') == '1',
        'detour_clear_pad': float(os.environ.get('DETOUR_CLEAR_PAD', '0.05')),
        'detour_side_proj': os.environ.get('DETOUR_SIDE_PROJ', '1') == '1',
        # Reference heading from a look-ahead chord; 0 = validated tangent.
        # Cross-fade seconds when a new /plan arrives; 0 = adopt instantly.
        'plan_blend_s': float(os.environ.get('PLAN_BLEND', '0.0')),
        # Hold still when blocked instead of pushing a reference that insists
        # the way is clear. 0 = off, which is what every result so far used.
        'stuck_window_s': float(os.environ.get('STUCK_WINDOW', '0.0')),
        'stuck_progress_m': float(os.environ.get('STUCK_PROGRESS', '0.15')),
        'stuck_release_h': float(os.environ.get('STUCK_RELEASE_H', '0.6')),
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
             output='screen',
             # EKF_REJECT overrides pose0_rejection_threshold (Mahalanobis gate
             # on /amcl_pose). At 2.5 the filter locks itself out: once its own
             # estimate drifts, correct AMCL poses look like outliers and are
             # discarded, so it drifts further. Measured on one trial, the EKF
             # reached 2.49 m of error while AMCL stayed within 0.07 m and kept
             # publishing 13-20 times per 10 s -- the correction was available
             # and refused. Across 89 trials, 13 had EKF peaks over 0.5 m and
             # three exceeded 8 m. Empty keeps the file's value.
             parameters=[ekf_config] + (
                 [{'pose0_rejection_threshold':
                   float(os.environ['EKF_REJECT'])}]
                 if os.environ.get('EKF_REJECT') else []),
             condition=UnlessCondition(truth_pose)),
        # TRUTH_POSE=1: replace the whole localisation stack with a perfect one.
        # /odom already carries gz's true pose IN MAP COORDINATES (verified:
        # seed1 starts at 17.10,14.80 against a spawn of 17.1,14.8), and
        # odom_tf_broadcaster publishes odom->base_footprint from it, so an
        # identity map->odom makes TF map->base_footprint exactly the truth.
        #
        # This isolates how much of the contact rate is localisation error. It
        # matters because 43 of 48 contacts are shallower than AMCL's own p90
        # (0.076 m) -- the barrier may be satisfied in the frame it can see
        # while the true body grazes. Without this ablation that is a hypothesis;
        # with it, it is a number.
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='truth_map_odom', output='screen',
             arguments=['--x', '0', '--y', '0', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'map', '--child-frame-id', 'odom'],
             condition=IfCondition(truth_pose)),
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
        # SHIELD=1 inserts the raw-scan safety layer between the smoother and
        # the chassis, so it constrains the command that is actually executed --
        # anything upstream of the smoother could still be reshaped afterwards.
        # Every command path goes through it; there is no bypass topic.
        Node(package='nav2_velocity_smoother', executable='velocity_smoother',
             name='velocity_smoother', output='screen',
             parameters=[configured_nav_params],
             remappings=[('cmd_vel', 'cmd_vel_nav'),
                         ('cmd_vel_smoothed',
                          'cmd_vel_pre_shield' if _shield else 'cmd_vel')],
             condition=IfCondition(use_smoother)),
        Node(package='ammr_wholebody_mpc', executable='scan_safety_shield',
             name='scan_safety_shield', output='screen',
             parameters=[{'use_sim_time': True,
                          'enable': True,
                          # RAW /scan, not /scan_filtered: the filtered stream
                          # has the confirmed movers masked out, which is
                          # exactly the set this layer must not depend on.
                          'scan_topic': '/scan',
                          'cmd_in_topic': '/cmd_vel_pre_shield',
                          'cmd_out_topic': '/cmd_vel',
                          'robot_radius': 0.30,
                          'alpha': float(os.environ.get('SHIELD_ALPHA', '2.0')),
                          'a_brake': 6.25,
                          'tau': float(os.environ.get('SHIELD_TAU', '0.15')),
                          'vx_max': 0.2775, 'vy_max': 0.2775,
                          'wz_max': 1.1327}],
             condition=IfCondition(str(int(_shield)))),
        Node(package='ammr_wholebody_mpc', executable='goal_to_plan_relay',
             name='goal_to_plan_relay', output='screen',
             parameters=[{'use_sim_time': True, 'global_frame': 'map',
                          'robot_base_frame': 'base_footprint',
                          'planner_id': 'GridBased',
                          # 3 s dates from when the planner marked its costmap
                          # from the raw /scan: it re-routed around movers at
                          # their CURRENT position, so the path flipped every
                          # replan and slowing the rate was the fix. The planner
                          # now reads /scan_filtered with movers masked out, so
                          # its cost field no longer changes as they pass and
                          # the reason for the delay is gone. nav2's own BT
                          # replans at 1 Hz, which is what MPPI and RPP get --
                          # three times our rate, on the same scenario.
                          #
                          # Measured at 1 Hz, paired over the same 30 routes:
                          # contacts 3 -> 1, worst penetration -0.313 -> -0.053,
                          # arrival 97 s -> 85 s. Nothing traded away for it.
                          'replan_period': float(
                              os.environ.get('REPLAN', '1.0'))}]),
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
                              os.environ.get('TRACK_RELEASE_FRAMES', '0')),
                          # Static/dynamic routing. The net-displacement gate
                          # asks for min_net_speed averaged over static_window_s,
                          # i.e. 0.10 m over 2 s at the defaults. A 0.10 m/s
                          # mover covers 0.20 m -- only 2x the threshold, so
                          # occlusion tips it either way, and measured on batch N
                          # the two slowest movers were called static at 2 of
                          # their closest approaches each.
                          'static_window_s': float(
                              os.environ.get('STATIC_WINDOW', '2.0')),
                          'min_net_speed': float(
                              os.environ.get('MIN_NET_SPEED', '0.05')),
                          'static_keep_velocity':
                              os.environ.get('STATIC_KEEP_VEL', '0') == '1',
                          # Smallest cluster that becomes a track. At 2 returns
                          # a 0.15 m body is invisible past ~2 m: the lidar is
                          # 360 beams over 360 deg, so beams are 5.2 cm apart at
                          # 3 m and such a body subtends about 5.7 deg head-on
                          # and less obliquely. Measured over 3 s before closest
                          # approach, dyn_obs_3 (r = 0.15) sits in the CBF's
                          # constraint set only 83% of the time (q1 75%), while
                          # every body of r >= 0.25 is at 100%. It is also 4 of
                          # the 8 contacts in 172 trials. Allowing single-return
                          # clusters trades that against noise, which
                          # min_track_age = 3 and the association gate still have
                          # to pass.
                          'min_cluster_pts': int(
                              os.environ.get('MIN_CLUSTER_PTS', '2')),
                          # Close-range fragmentation recovery. Each part is a
                          # separate switch so seed27 can be replayed with one
                          # change at a time; all default to the old behaviour.
                          # min_track_speed 0.10 sits exactly on dyn_obs_5 and
                          # dyn_obs_2, whose configured speed IS 0.10 m/s, so
                          # measurement noise decides each cycle whether they
                          # are movers at all. The net-displacement gate does
                          # the same job more reliably (0.10 m/s over a 2 s
                          # window is 0.20 m of travel, far above its 0.05
                          # threshold), which is why it was added.
                          'min_track_speed': float(
                              os.environ.get('MIN_TRACK_SPEED', '0.10')),
                          'release_track_speed': float(
                              os.environ.get('RELEASE_TRACK_SPEED', '0.10')),
                          'assoc_predict':
                              os.environ.get('ASSOC_PREDICT', '0') == '1',
                          'assoc_maha':
                              os.environ.get('ASSOC_MAHA', '0') == '1',
                          'fragment_merge':
                              os.environ.get('FRAG_MERGE', '0') == '1',
                          'coast_publish_s': float(
                              os.environ.get('COAST_S', '0.0')),
                          'coast_radius_growth': float(
                              os.environ.get('COAST_GROWTH', '0.0'))}]),
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
        # Hybrid-graphics (PRIME on-demand) machine: without these, gz/OGRE's
        # EGL is routed to Mesa, which cannot drive the NVIDIA card. The log
        # then carries "failed to create dri2 screen" and the viewport renders
        # nothing -- the robot is simulated and driving, but not drawn.
        # omni_bot_arm.launch.py sets the same three and claims the navigation
        # launches do too; this one did not, which is why GUI runs here showed
        # an empty world.
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
        DeclareLaunchArgument('gui', default_value='true'),
        # Phase 2: mount the Lite 6. Default false so the benchmark robot and
        # the recorded N=40 results stay exactly as validated.
        DeclareLaunchArgument('use_arm', default_value='false'),
        DeclareLaunchArgument('cbf', default_value='true'),
        DeclareLaunchArgument('obstacle_source', default_value='scan',
                              description='scan = real /scan perception; truth = ground-truth'),
        DeclareLaunchArgument('robot_radius', default_value='0.33'),
        # The planner's clearance must be strictly LARGER than the CBF's
        # keep-out, or the two layers fight: measured in the 20 m floor, the
        # plan passed 0.43 m from a known box while the CBF's keep-out was
        # 0.45 m, so the reference pulled the robot into a region its own
        # barrier forbade. The result was three seconds of full-rate yaw
        # reversal and a -0.02 m touch on a KNOWN obstacle -- the "drunk"
        # weaving. 0.45 was tuned for the 20 m random room, whose obstacles
        # sit well off the route; it was never enough once clutter sits beside
        # the path. Env-overridable so the trade against doorway width can be
        # tested: at 0.70 a 2.0 m doorway still leaves 0.6 m of zero-cost centre.
        DeclareLaunchArgument(
            'inflation', default_value=os.environ.get('INFLATION', '0.70')),
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
                          # Centres are exact: 21 trials at different positions
                          # all show the same four body-fixed sectors at +-45
                          # and +-135, 8-9 deg wide, 34 beams total. The
                          # half-width was not -- 15 deg masked 120 of 360
                          # beams, discarding a quarter of the lidar.
                          #
                          # Over-masking did more than cost beams. The clear
                          # gaps between sectors were only 60 deg, and a mover
                          # of radius 0.82 m at 0.5 m subtends 60.8 deg, so its
                          # visible arc straddled a masked sector and split into
                          # two clusters; the centroid then jumped past the
                          # 0.80 m association gate and the track was rebuilt,
                          # invisible for min_track_age frames. Measured: 47 of
                          # 60 close encounters with the large movers had their
                          # arc cut by the mask, and their CBF coverage at
                          # 0.5-1.5 m was 31-53% against 82-100% for the small
                          # ones. At 10 deg the gaps widen to 70 deg and neither
                          # large mover straddles them inside 1 m.
                          'blocked_centers_deg': '45,135,225,315',
                          # MASK_HW=15 restores the old width for A/B testing.
                          'blocked_halfwidth_deg': float(
                              os.environ.get('MASK_HW', '10.0'))}],
             output='screen'),

        TimerAction(period=6.0, actions=[Node(
            package='ros_gz_sim', executable='create',
            arguments=['-name', 'omni_bot', '-topic', 'robot_description',
                       # Spawn pose, so a batch can randomise where a traverse
                       # begins instead of repeating one route. Defaults to the
                       # origin, which is what every recorded result used.
                       '-x', os.environ.get('SPAWN_X', '0.0'),
                       '-y', os.environ.get('SPAWN_Y', '0.0'),
                       '-z', '0.05'], output='screen')]),

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
