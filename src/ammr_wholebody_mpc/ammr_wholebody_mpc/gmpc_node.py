"""ROS2 standalone GMPC controller node.

Subscribes  : /plan  (nav_msgs/Path)            in `global_frame`
Publishes   : /cmd_vel (geometry_msgs/Twist)    body-frame command for the omni base
TF needed   : `global_frame` → `robot_base_frame`   (e.g. map → base_footprint)

Control loop runs at `control_frequency` Hz:
  1. Look up the current robot pose in `global_frame` via tf2.
  2. Build a horizon reference (X_ref_win, xi_ref_win) by arclength sampling
     the latest /plan starting from the projection of the robot.
  3. Solve the SE(2) GMPC QP (see ammr_wholebody_mpc.gmpc).
  4. Publish the optimal body twist as /cmd_vel.

If no plan has been received, the node publishes zero twist.
If the robot is within `goal_tolerance_xy` of the path end, the node holds zero.
If TF lookup fails, the node publishes zero and warns (throttled).
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Twist, TransformStamped, Point
from nav_msgs.msg      import Path
from std_msgs.msg      import Float32, Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker

import tf2_ros

from collections import deque

from .gmpc           import GMPC, GMPCConfig
from .se2            import from_xytheta, to_xytheta
from .detour import (DetourConfig, DetourState, apply_offset,
                     clear_reference, FREE)
from .path_processor import (path_msg_to_xyth,
                             build_reference_window,
                             blend_reference,
                             quaternion_to_yaw)


class GMPCNode(Node):

    def __init__(self):
        super().__init__('gmpc_controller')

        # ---- Parameters ---------------------------------------------------
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('horizon',           20)
        self.declare_parameter('v_nominal',         0.30)

        self.declare_parameter('vx_min', -0.20)
        self.declare_parameter('vx_max',  0.35)
        self.declare_parameter('vy_min', -0.25)
        self.declare_parameter('vy_max',  0.25)
        self.declare_parameter('wz_min', -0.80)
        self.declare_parameter('wz_max',  0.80)
        self.declare_parameter('ax_max',  1.5)
        self.declare_parameter('ay_max',  1.0)
        self.declare_parameter('az_max',  2.0)

        self.declare_parameter('Q_xy',    10.0)   # forward (body-x) tracking weight
        # Lateral (body-y) tracking weight, separate from forward. Relaxing it
        # lets the robot, after the CBF pushes it off-path to dodge, merge back
        # to the global plan along a GENTLE arc instead of snapping the heading
        # hard (which shows up as xy-path zig-zag). <=0 -> use Q_xy (isotropic).
        self.declare_parameter('Q_y',      0.0)
        self.declare_parameter('R_vx',     0.5)
        self.declare_parameter('R_vy',     0.5)
        self.declare_parameter('R_w',      0.2)
        self.declare_parameter('Qf_mult',  5.0)
        # Input-increment (Δu) smoothness weights S (penalise control jerk).
        # 0 = off. Larger -> smoother cruising, CBF still allowed to burst.
        self.declare_parameter('S_vx',     0.0)
        self.declare_parameter('S_vy',     0.0)
        self.declare_parameter('S_w',      0.0)

        self.declare_parameter('global_frame',     'map')
        self.declare_parameter('robot_base_frame', 'base_footprint')
        # Pose low-pass filter: EMA on the (map-frame) robot pose before it feeds
        # the controller/CBF. AMCL injects ~3-5 cm/cycle jitter into the pose;
        # the near-hard CBF chases that jitter -> the xy path zig-zags near walls.
        # Smoothing the INPUT (not the output) is the right fix: filtering the
        # control output instead makes the reactive loop sluggish and oscillate
        # MORE. alpha in [0,1): 0 = off, higher = smoother (more lag). ~0.7 ->
        # ~0.12 s time constant (a few cm lag, absorbed by the CBF margin).
        self.declare_parameter('pose_lpf_alpha', 0.0)
        self.declare_parameter('plan_topic',       '/plan')
        self.declare_parameter('cmd_vel_topic',    '/cmd_vel')
        # Cross-fade the tracked reference from the previous plan to the new one
        # over this many seconds. 0 = adopt the new plan instantly (validated
        # behaviour). The global planner republishes every 3 s and a quarter of
        # those updates move the path in front of the robot by more than 0.25 m
        # (measured: median 0.05 m, p90 0.44 m, max 1.82 m; reference heading
        # p90 26 deg, max 63 deg). In the 0.3 s after such a jump the controller
        # sits at 0.96 of a_max and saturates 88.5% of the time, against 13.5%
        # otherwise -- it is chasing a step, not misbehaving.
        self.declare_parameter('plan_blend_s', 0.0)
        # Hold still when there is demonstrably no way forward, instead of
        # tracking a reference that insists there is. With the planner decoupled
        # from the movers it cannot know an opening is blocked, so it keeps
        # routing through one; the CBF stops the robot, the plan flips to the
        # other opening three seconds later, and the robot turns around. In the
        # both-gaps-patrolled scenario that produced 8 left/right plan flips in
        # 29 replans, 10.9 reversals per run, 12.7% of commands in reverse, and
        # the robot never once reached the divider in 88 s -- 2 of 10 trials
        # arrived and 9 collided. Waiting is the correct answer there and
        # nothing in the stack could express it.
        # 0 disables (the behaviour every result so far was measured with).
        self.declare_parameter('stuck_window_s', 0.0)
        self.declare_parameter('stuck_progress_m', 0.15)
        self.declare_parameter('stuck_release_h', 0.6)
        self.declare_parameter('stuck_max_hold_s', 25.0)
        self.declare_parameter('goal_tolerance_xy', 0.20)
        self.declare_parameter('tf_timeout',        0.10)

        # ---- CBF safety filter ------------------------------------------
        self.declare_parameter('cbf_enable',         False)
        self.declare_parameter('cbf_alpha',          2.0)
        self.declare_parameter('cbf_safe_margin',    0.30)
        self.declare_parameter('cbf_slack_weight',   1.0e4)
        self.declare_parameter('cbf_eps0_scale',     100.0)
        # Gain scheduling — when robot enters the danger zone (h < threshold)
        # the QP downweights tracking and upweights slack penalty.
        self.declare_parameter('cbf_danger_thresh',   0.5)
        self.declare_parameter('cbf_Q_min_scale',     0.2)
        self.declare_parameter('cbf_slack_max_scale', 100.0)
        # Price wall-constraint relaxation this many times higher than dynamic
        # relaxation, so a robot pinched between a mover and a wall gives up
        # dynamic clearance (0.08 m of buffer) instead of wall clearance
        # (0.03 m). 1.0 = the original single shared slack.
        self.declare_parameter('cbf_static_slack_scale', 1.0)
        # Spatio-temporal cost field: soft Gaussian barrier on the PREDICTED
        # obstacle positions across the horizon. 0 = off (identical to the
        # validated configuration); independent of cbf_enable so the two can be
        # switched separately for A/B.
        self.declare_parameter('st_weight', 0.0)
        self.declare_parameter('st_sigma0', 0.6)
        self.declare_parameter('st_growth', 0.02)
        # Grow the CBF keep-out with the horizon step to discount far-future
        # constant-velocity predictions. 0 = the validated constant margin.
        # See gmpc.py: keep-out that grows with the closing rate.
        self.declare_parameter('cbf_vel_margin_gain', 0.0)
        # See gmpc.py: drop CBF rows that cannot bind (far points, far steps).
        self.declare_parameter('cbf_prune_range', 0.0)
        self.declare_parameter('cbf_near_steps', 6)
        self.declare_parameter('cbf_far_stride', 1)
        self.declare_parameter('cbf_hard_k0', False)
        self.declare_parameter('cbf_margin_growth', 0.0)
        # Prefer giving way FORWARD rather than backward when the CBF pushes the
        # robot off the reference. 0 = no preference (validated behaviour).
        self.declare_parameter('prog_weight', 0.0)
        # Committed detour (see detour.py): lock a side, bend the reference
        # around the obstacle, keep a forward-speed floor. Off by default.
        self.declare_parameter('detour_enable', False)
        self.declare_parameter('detour_trigger_range', 2.0)
        self.declare_parameter('detour_cone_deg', 30.0)
        self.declare_parameter('detour_max_offset', 0.60)
        self.declare_parameter('detour_offset_rate', 0.08)
        self.declare_parameter('detour_vx_floor', 0.0)
        # Keep the tracked reference out of the CBF's keep-out discs. Without
        # it Q pulls towards points the CBF forbids (see clear_reference).
        self.declare_parameter('detour_clear_ref', True)
        self.declare_parameter('detour_clear_pad', 0.05)
        # Push a blocked reference point out ALONG the committed side rather
        # than radially. Radial is "away from the obstacle centre", which with
        # the obstacle dead ahead means BACKWARDS -- measured 0.235 m of
        # backward reference shift, and the robot answered by retreating 1.8 m
        # down the spawn corridor. False restores the radial behaviour so the
        # two can be compared.
        self.declare_parameter('detour_side_proj', True)
        self.declare_parameter('obstacles_topic',    '/gmpc/obstacles')
        # static-CBF (Solution 1): nearest wall points (v=0) so the CBF also
        # repels from known static geometry and won't dodge into walls.
        self.declare_parameter('static_obstacles_topic', '/gmpc/static_obstacles')
        # drop static-CBF points within this distance of the goal: the goal can
        # hug a wall, and the planner's path already handles the (static-safe)
        # final approach -> don't let the wall-CBF block reaching the goal.
        self.declare_parameter('static_goal_clear', 0.6)
        # Static obstacles get their OWN (smaller) CBF margin: the moving
        # obstacles in open space can afford the full cbf_safe_margin, but the
        # same keep-out applied to wall points boxes the robot in narrow
        # passages. Static walls are already routed around by the planner, so
        # the static-CBF is only a backstop -> a tighter margin keeps narrow
        # passages drivable while still preventing dodge-into-wall.
        # NOTE the 0.30 m robot radius is INSIDE this number -- the CBF treats
        # the robot as a point -- so 0.33 buys only 3 cm of true buffer, against
        # the 8 cm the dynamic margin (0.38) buys. That difference is invisible
        # for wall points, whose own radius is 0.05, but not for an unknown
        # cylinder of radius 0.30: keep-out 0.63 against a 0.60 m contact
        # distance. Measured across 187 arena trials, EVERY collision was a
        # graze of an unknown static pillar, at -0.006 to +0.011 m -- exactly
        # that 3 cm target minus slack -- while walls and movers were never
        # touched. Measured at 0.38 against 0.33, ten trials each: shapes 8 -> 3
        # collisions, stopgo 2 -> 0 and 14 s faster, crossing's worst-case
        # clearance +0.070 -> +0.327, with arrival time and path length
        # unchanged on the scenarios that already passed.
        self.declare_parameter('static_cbf_safe_margin', 0.38)
        # Static-CBF is a BACKSTOP for "don't dodge a dynamic obstacle into a
        # wall" — only engage it when a dynamic obstacle is within this range of
        # the robot. With no dynamic threat nearby the robot just tracks the
        # (smooth) global plan through gaps, instead of the two-sided wall CBF
        # constraints squeezing it into a left-right zig-zag in a passage.
        # 0 = always on (old behaviour).
        self.declare_parameter('static_activate_range', 0.0)

        # DERIVED margins. 'fixed' keeps cbf_safe_margin / static_cbf_safe_margin
        # exactly as they are; 'derived' computes each obstacle's keep-out from
        # what it physically has to absorb:
        #
        #   static   loc_err + v^2/(2 a) + v dt
        #   dynamic  the same, plus v_obs * percep_lag + obs_pos_err
        #
        # The fixed values have no derivation behind them -- 0.38 and 0.60 were
        # tuned, and the measurements say they are far larger than needed. At the
        # wall graze of C_seed22 the robot was doing 0.254 m/s, for which the
        # terms above sum to 0.094 m against a margin of 0.380. Oversizing is not
        # free: the goal there sat 0.55 m from a wall, inside a 0.68 m keep-out,
        # so the barrier pushed the robot off its own goal and it ground along
        # the wall for two seconds before converging.
        #
        # A speed-scaled margin also removes the need to tune static_goal_clear:
        # the keep-out shrinks as the robot slows into the goal.
        self.declare_parameter('margin_mode', 'fixed')      # fixed | derived
        self.declare_parameter('margin_loc_err', 0.06)      # localisation p90 [m]
        self.declare_parameter('margin_percep_lag', 0.30)   # min_track_age/rate [s]
        self.declare_parameter('margin_obs_pos_err', 0.05)  # surface-point error [m]
        self.declare_parameter('margin_floor', 0.08)        # never below this [m]

        # ---- Diagnostic topics ------------------------------------------
        self.declare_parameter('solve_time_topic',   '/gmpc/solve_time_ms')
        self.declare_parameter('min_h_topic',        '/gmpc/min_h')
        self.declare_parameter('cbf_zone_topic',     '/gmpc/cbf_zones')

        # Read into instance state
        f       = float(self.get_parameter('control_frequency').value)
        N       = int(  self.get_parameter('horizon').value)
        self.dt = 1.0 / f
        self.v_nom = float(self.get_parameter('v_nominal').value)

        self.global_frame     = str(self.get_parameter('global_frame').value)
        self.base_frame       = str(self.get_parameter('robot_base_frame').value)
        self.pose_lpf_alpha   = float(self.get_parameter('pose_lpf_alpha').value)
        self._pose_filt       = None        # EMA state for the robot pose
        self.goal_tol_xy      = float(self.get_parameter('goal_tolerance_xy').value)
        self.plan_blend_s = float(self.get_parameter('plan_blend_s').value)
        self.stuck_window_s = float(self.get_parameter('stuck_window_s').value)
        self.stuck_progress_m = float(self.get_parameter('stuck_progress_m').value)
        self.stuck_release_h = float(self.get_parameter('stuck_release_h').value)
        self.stuck_max_hold_s = float(self.get_parameter('stuck_max_hold_s').value)
        self._goal_hist = deque(maxlen=2000)   # (t, distance to goal)
        self._holding_since = None
        self._last_min_h = None
        # "the CBF is doing something" -- the same threshold gain scheduling
        # uses, so being blocked and being in the danger zone mean the same thing
        self.cbf_danger_hold = float(
            self.get_parameter('cbf_danger_thresh').value)
        self._prev_path_xyth = None      # path being faded OUT
        self._blend_t0 = None            # when the current fade started
        self.tf_timeout_s     = float(self.get_parameter('tf_timeout').value)

        Qxy  = float(self.get_parameter('Q_xy').value)
        # No heading is ever referenced (see path_processor), so heading error
        # is not a tracking objective and carries no weight. An explicit zero
        # rather than a deletion: Q must stay the 3x3 the SE(2) cost needs.
        Qyaw = 0.0
        Qy   = float(self.get_parameter('Q_y').value)
        if Qy <= 0.0:
            Qy = Qxy                                   # default: isotropic xy
        Qf_m = float(self.get_parameter('Qf_mult').value)

        cfg = GMPCConfig(
            N=N, dt=self.dt,
            u_min=np.array([float(self.get_parameter('vx_min').value),
                            float(self.get_parameter('vy_min').value),
                            float(self.get_parameter('wz_min').value)]),
            u_max=np.array([float(self.get_parameter('vx_max').value),
                            float(self.get_parameter('vy_max').value),
                            float(self.get_parameter('wz_max').value)]),
            a_max=np.array([float(self.get_parameter('ax_max').value),
                            float(self.get_parameter('ay_max').value),
                            float(self.get_parameter('az_max').value)]),
            Q =np.diag([Qxy, Qy, Qyaw]),
            R =np.diag([float(self.get_parameter('R_vx').value),
                        float(self.get_parameter('R_vy').value),
                        float(self.get_parameter('R_w').value)]),
            S =np.diag([float(self.get_parameter('S_vx').value),
                        float(self.get_parameter('S_vy').value),
                        float(self.get_parameter('S_w').value)]),
            Qf=np.diag([Qxy * Qf_m, Qxy * Qf_m, Qyaw * Qf_m]),
            cbf_alpha         =float(self.get_parameter('cbf_alpha').value),
            cbf_safe_margin   =float(self.get_parameter('cbf_safe_margin').value),
            cbf_slack_weight  =float(self.get_parameter('cbf_slack_weight').value),
            cbf_eps0_scale    =float(self.get_parameter('cbf_eps0_scale').value),
            cbf_danger_thresh =float(self.get_parameter('cbf_danger_thresh').value),
            cbf_Q_min_scale   =float(self.get_parameter('cbf_Q_min_scale').value),
            cbf_slack_max_scale=float(self.get_parameter('cbf_slack_max_scale').value),
            cbf_static_slack_scale=float(
                self.get_parameter('cbf_static_slack_scale').value),
            st_weight=float(self.get_parameter('st_weight').value),
            st_sigma0=float(self.get_parameter('st_sigma0').value),
            st_growth=float(self.get_parameter('st_growth').value),
            cbf_margin_growth=float(
                self.get_parameter('cbf_margin_growth').value),
            cbf_vel_margin_gain=float(
                self.get_parameter('cbf_vel_margin_gain').value),
            cbf_prune_range=float(
                self.get_parameter('cbf_prune_range').value),
            cbf_near_steps=int(self.get_parameter('cbf_near_steps').value),
            cbf_far_stride=int(self.get_parameter('cbf_far_stride').value),
            cbf_hard_k0=bool(self.get_parameter('cbf_hard_k0').value),
            prog_weight=float(self.get_parameter('prog_weight').value),
        )
        self.mpc = GMPC(cfg)
        self.cfg = cfg        # derived margins read a_max from here
        self.detour = DetourState(DetourConfig(
            enable=bool(self.get_parameter('detour_enable').value),
            trigger_range=float(self.get_parameter('detour_trigger_range').value),
            trigger_cone_deg=float(self.get_parameter('detour_cone_deg').value),
            max_offset=float(self.get_parameter('detour_max_offset').value),
            offset_rate=float(self.get_parameter('detour_offset_rate').value),
            vx_floor=float(self.get_parameter('detour_vx_floor').value)))
        self.detour_clear_ref = bool(
            self.get_parameter('detour_clear_ref').value)
        self.detour_clear_pad = float(
            self.get_parameter('detour_clear_pad').value)
        self.detour_side_proj = bool(
            self.get_parameter('detour_side_proj').value)
        self._cfg = cfg
        self.N   = cfg.N
        self.cbf_enable = bool(self.get_parameter('cbf_enable').value)
        self.static_goal_clear = float(self.get_parameter('static_goal_clear').value)
        self.static_cbf_safe_margin = float(self.get_parameter('static_cbf_safe_margin').value)
        self.margin_mode        = str(self.get_parameter('margin_mode').value)
        self.margin_loc_err     = float(self.get_parameter('margin_loc_err').value)
        self.margin_percep_lag  = float(self.get_parameter('margin_percep_lag').value)
        self.margin_obs_pos_err = float(self.get_parameter('margin_obs_pos_err').value)
        self.margin_floor       = float(self.get_parameter('margin_floor').value)
        self.static_activate_range = float(self.get_parameter('static_activate_range').value)

        # ---- State --------------------------------------------------------
        self.latest_path  = None
        self.xi_prev      = np.zeros(3)
        self._arrived     = False
        self._obstacles   = []           # dynamic: list of dict {x, y, radius, vx, vy}
        self._static_obstacles = []      # static walls (v=0) for the CBF

        # ---- TF + I/O -----------------------------------------------------
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        plan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history    =QoSHistoryPolicy.KEEP_LAST,
            depth      =1,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter('plan_topic').value),
            self._plan_cb, plan_qos,
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('cmd_vel_topic').value),
            10,
        )

        # Diagnostic publishers (for analyze.py + Foxglove visualisation)
        self.solve_time_pub = self.create_publisher(
            Float32, str(self.get_parameter('solve_time_topic').value), 10)
        self.min_h_pub      = self.create_publisher(
            Float32, str(self.get_parameter('min_h_topic').value), 10)
        self.cbf_zone_pub   = self.create_publisher(
            MarkerArray, str(self.get_parameter('cbf_zone_topic').value), 10)

        # Obstacles: subscribe to a Float32MultiArray that the aggregator publishes.
        # Layout: [x1, y1, r1, x2, y2, r2, ...] flat array, length = 3*N_obs.
        if self.cbf_enable:
            self.create_subscription(
                Float32MultiArray,
                str(self.get_parameter('obstacles_topic').value),
                self._obstacles_cb, 10,
            )
            self.create_subscription(
                Float32MultiArray,
                str(self.get_parameter('static_obstacles_topic').value),
                self._static_obstacles_cb, 10,
            )

        self.create_timer(self.dt, self._control_step)

        self.get_logger().info(
            f'GMPC controller up: N={cfg.N}, dt={self.dt:.3f}s, v_nom={self.v_nom:.2f} m/s, '
            f'frame {self.global_frame}->{self.base_frame}, '
            f'CBF={"ON" if self.cbf_enable else "OFF"}'
            + (f' (α={cfg.cbf_alpha:.1f}, margin={cfg.cbf_safe_margin:.2f}m)'
               if self.cbf_enable else '')
        )

    def _blend_alpha(self):
        """Fade weight in [0,1) for the new plan, or None when not fading.

        1.0 means the fade is over, so it is reported as None and the previous
        path is dropped -- that keeps the common case free of the second
        reference build.
        """
        if (self.plan_blend_s <= 0.0 or self._prev_path_xyth is None
                or self._blend_t0 is None):
            return None
        dt = (self.get_clock().now() - self._blend_t0).nanoseconds * 1e-9
        if dt < 0.0 or dt >= self.plan_blend_s:
            self._prev_path_xyth = None
            self._blend_t0 = None
            return None
        return dt / self.plan_blend_s

    def _hold_if_stuck(self, dist_to_goal) -> bool:
        """Stop and wait when the route is blocked, rather than keep pushing.

        Detection is deliberately about OUTCOME, not cause: if the distance to
        the goal has not fallen over a whole window, the robot is not getting
        anywhere, whatever the reason. It only counts as blocked when the CBF is
        also active -- otherwise a robot that is simply slow, or circling a wide
        detour, would be mistaken for a stuck one.

        Holding releases as soon as min_h recovers, which for a patrolling
        obstacle happens on its own, and is capped so a permanent blockage does
        not become a permanent stop: after the cap the robot resumes and the
        detector starts again from scratch.

        Returns True if the caller should publish zero and skip this step.
        """
        if self.stuck_window_s <= 0.0:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        self._goal_hist.append((now, dist_to_goal))
        while self._goal_hist and now - self._goal_hist[0][0] > self.stuck_window_s:
            self._goal_hist.popleft()

        h = self._last_min_h if self._last_min_h is not None else 9.9
        if self._holding_since is not None:
            held = now - self._holding_since
            if h > self.stuck_release_h or held > self.stuck_max_hold_s:
                self.get_logger().info(
                    f'\033[1;32mresuming\033[0m after holding {held:.1f} s '
                    f'(min_h={h:.2f})')
                self._holding_since = None
                self._goal_hist.clear()
                return False
            self._publish_zero()
            return True

        if (len(self._goal_hist) > 10
                and now - self._goal_hist[0][0] >= self.stuck_window_s * 0.95):
            progress = self._goal_hist[0][1] - dist_to_goal
            if progress < self.stuck_progress_m and h < self.cbf_danger_hold:
                self.get_logger().warn(
                    f'\033[1;33mblocked\033[0m: {progress:+.2f} m of progress in '
                    f'{self.stuck_window_s:.0f} s with min_h={h:.2f} -- holding')
                self._holding_since = now
                self._publish_zero()
                return True
        return False

    def _plan_cb(self, msg: Path):
        if len(msg.poses) == 0:
            self.get_logger().warn('Received empty plan')
        if (self.plan_blend_s > 0.0 and self.latest_path is not None
                and len(self.latest_path.poses) >= 2 and len(msg.poses) >= 2):
            # Keep the outgoing path so the reference can cross-fade instead of
            # teleporting. Restarting the fade on every plan is deliberate: a
            # second jump during a fade should also be smoothed, not snapped to.
            self._prev_path_xyth = path_msg_to_xyth(self.latest_path)
            self._blend_t0 = self.get_clock().now()
        self.latest_path = msg
        # New plan means a new goal (or continuous replan) — re-arm the arrival
        # detector so the next arrival also gets logged.
        self._arrived = False

    def _obstacles_cb(self, msg: Float32MultiArray):
        """Flat [x, y, r, vx, vy, ...] (5 floats per obstacle) in global frame."""
        data = list(msg.data)
        stride = 5
        n = len(data) // stride
        self._obstacles = [
            {'x':      float(data[stride*i + 0]),
             'y':      float(data[stride*i + 1]),
             'radius': float(data[stride*i + 2]),
             'vx':     float(data[stride*i + 3]),
             'vy':     float(data[stride*i + 4])}
            for i in range(n)
        ]

    def _static_obstacles_cb(self, msg: Float32MultiArray):
        """Nearest WALL points (v=0) from the tracker. Same wire format as the
        dynamic obstacles; merged into the CBF set so the controller doesn't
        dodge a moving obstacle straight into static geometry."""
        data = list(msg.data)
        stride = 5
        n = len(data) // stride
        self._static_obstacles = [
            {'x':      float(data[stride*i + 0]),
             'y':      float(data[stride*i + 1]),
             'radius': float(data[stride*i + 2]),
             'vx':     0.0,
             'vy':     0.0}
            for i in range(n)
        ]

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())
        self.xi_prev = np.zeros(3)

    @staticmethod
    def _tf_to_xyth(tf: TransformStamped) -> np.ndarray:
        t = tf.transform.translation
        r = tf.transform.rotation
        return np.array([t.x, t.y, quaternion_to_yaw(r.x, r.y, r.z, r.w)])

    def _lookup_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.tf_timeout_s),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {self.global_frame}->{self.base_frame} failed: {e}',
                throttle_duration_sec=2.0)
            return None
        pose = self._tf_to_xyth(tf)
        a = self.pose_lpf_alpha
        if a > 0.0:
            if self._pose_filt is None:
                self._pose_filt = pose.copy()
            else:
                f = self._pose_filt
                f[0] = a * f[0] + (1.0 - a) * pose[0]
                f[1] = a * f[1] + (1.0 - a) * pose[1]
                # yaw via shortest-angle increment (avoid ±pi wrap artefacts)
                dyaw = np.arctan2(np.sin(pose[2] - f[2]), np.cos(pose[2] - f[2]))
                f[2] = f[2] + (1.0 - a) * dyaw
            pose = self._pose_filt.copy()
        return pose

    # ----------------------------------------------------------------------
    def _control_step(self):
        # 0. Need a plan
        if self.latest_path is None or len(self.latest_path.poses) < 2:
            self._publish_zero()
            return

        # 1. Robot pose in global_frame
        robot_xyth = self._lookup_robot_pose()
        if robot_xyth is None:
            self._publish_zero()
            return

        # 2. Distance-to-goal check (just hold zero when there)
        goal_xy = np.array([self.latest_path.poses[-1].pose.position.x,
                            self.latest_path.poses[-1].pose.position.y])
        dist_to_goal = float(np.linalg.norm(goal_xy - robot_xyth[:2]))
        if self._hold_if_stuck(dist_to_goal):
            return
        if dist_to_goal < self.goal_tol_xy:
            if not self._arrived:
                self.get_logger().info(
                    f'\033[1;32mGoal reached\033[0m '
                    f'(within {dist_to_goal:.3f} m of '
                    f'({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), tol={self.goal_tol_xy:.2f} m) '
                    f'-- holding zero twist'
                )
                self._arrived = True
            self._publish_zero()
            return

        # 3. Build horizon reference from latest /plan
        path_xyth = path_msg_to_xyth(self.latest_path)
        # (reference yaw is rate-limited after blending; see step 3b below)
        X_ref_win, xi_ref_win = build_reference_window(
            path_xyth, robot_xyth,
            N=self.N, dt=self.dt, v_nom=self.v_nom,
        )
        a = self._blend_alpha()
        if a is not None:
            X_old, xi_old = build_reference_window(
                self._prev_path_xyth, robot_xyth,
                N=self.N, dt=self.dt, v_nom=self.v_nom,
                )
            X_ref_win, xi_ref_win = blend_reference(
                X_old, xi_old, X_ref_win, xi_ref_win, a)

        # 4. Solve
        X_now  = from_xytheta(*robot_xyth)
        # Derived keep-outs, sized from what each obstacle actually has to
        # absorb rather than from a tuned constant. v is the last commanded
        # body speed, so the margin collapses as the robot slows -- which is
        # what lets it settle on a goal that sits close to a wall.
        v_rob = float(np.hypot(self.xi_prev[0], self.xi_prev[1]))
        if self.margin_mode == 'derived':
            a_brake = max(1e-3, min(self.cfg.a_max[0], self.cfg.a_max[1]))
            m_base = (self.margin_loc_err
                      + v_rob * v_rob / (2.0 * a_brake)
                      + v_rob * self.dt)
            m_static = max(self.margin_floor, m_base)
        else:
            m_base = m_static = None
        if self.cbf_enable:
            obstacles = list(self._obstacles)                     # dynamic
            if m_base is not None:
                for o in obstacles:
                    v_obs = float(np.hypot(o.get('vx', 0.0), o.get('vy', 0.0)))
                    o['margin'] = max(
                        self.margin_floor,
                        m_base + v_obs * self.margin_percep_lag
                        + self.margin_obs_pos_err)
            # Engage the static-CBF (wall backstop) only while a dynamic obstacle
            # is actually nearby; otherwise let the smooth plan carry the robot
            # through gaps without two-sided wall constraints zig-zagging it.
            rx, ry = robot_xyth[0], robot_xyth[1]
            dyn_near = (self.static_activate_range <= 0.0) or any(
                np.hypot(o['x'] - rx, o['y'] - ry) < self.static_activate_range
                for o in self._obstacles)
            if dyn_near:
                for s in self._static_obstacles:                  # walls (v=0),
                    if np.hypot(s['x'] - goal_xy[0],              # but not the
                                s['y'] - goal_xy[1]) > self.static_goal_clear:
                        # smaller keep-out for walls (see static_cbf_safe_margin).
                        # 'static' tags these as geometry to route AROUND rather
                        # than movers to commit a detour against -- detour.py
                        # must never lock a side against a wall point.
                        obstacles.append({
                            **s, 'static': True,
                            'margin': (m_static if m_static is not None
                                       else self.static_cbf_safe_margin)})
        else:
            obstacles = None
        # Committed detour: pick a side once, then bend the horizon reference
        # around the obstacle so the tracking cost pulls the robot AROUND it
        # instead of resisting the CBF's sideways push. See detour.py.
        side, off = self.detour.update(robot_xyth, obstacles or [])
        if abs(off) > 1e-6:
            X_ref_win = apply_offset(X_ref_win, off, self.detour.cfg.max_offset)
            if self.detour_clear_ref:
                X_ref_win = clear_reference(
                    X_ref_win, obstacles or [], self.dt,
                    default_margin=self._cfg.cbf_safe_margin,
                    pad=self.detour_clear_pad, side=side,
                    sideways=self.detour_side_proj)
        # While detouring, floor the forward speed so "stop and wait" leaves the
        # solution space; restore the nominal limit as soon as we are free.
        floor = self.detour.cfg.vx_floor if (
            side != FREE and self.detour.cfg.enable) else None
        u_min_saved = self._cfg.u_min
        if floor is not None and floor > self._cfg.u_min[0]:
            self._cfg.u_min = np.array([floor, u_min_saved[1], u_min_saved[2]])
        try:
            result = self.mpc.solve(X_now, X_ref_win, xi_ref_win, self.xi_prev,
                                    obstacles=obstacles)
        finally:
            self._cfg.u_min = u_min_saved

        # 5. Publish cmd_vel (saturate one more time as a belt-and-braces safety)
        u = result.u_opt
        twist = Twist()
        twist.linear.x  = float(u[0])
        twist.linear.y  = float(u[1])
        twist.angular.z = float(u[2])
        self.cmd_pub.publish(twist)
        self.xi_prev = u

        # 6. Diagnostic topics (for analyze.py + Foxglove)
        m = Float32(); m.data = float(result.solve_time_s * 1e3)
        self.solve_time_pub.publish(m)
        if self.cbf_enable and result.cbf_active > 0:
            mh = Float32(); mh.data = float(result.min_h)
            self.min_h_pub.publish(mh)
            # kept for the stuck detector: how close the barrier came this step
            self._last_min_h = float(result.min_h)
            self._publish_cbf_zones()

        if result.status not in ('solved', 'solved inaccurate'):
            self.get_logger().warn(
                f'OSQP status={result.status}, emergency-braking',
                throttle_duration_sec=1.0)

    def _publish_cbf_zones(self):
        """Visualise CBF safety zones as MarkerArray for Foxglove."""
        ma = MarkerArray()
        for i, obs in enumerate(self._obstacles):
            mk = Marker()
            mk.header.frame_id = self.global_frame
            mk.header.stamp    = self.get_clock().now().to_msg()
            mk.ns       = 'cbf_zone'
            mk.id       = i
            mk.type     = Marker.CYLINDER
            mk.action   = Marker.ADD
            r = obs['radius'] + self.mpc.cfg.cbf_safe_margin
            mk.scale.x = 2.0 * r
            mk.scale.y = 2.0 * r
            mk.scale.z = 0.02
            mk.pose.position.x = obs['x']
            mk.pose.position.y = obs['y']
            mk.pose.position.z = 0.01
            mk.pose.orientation.w = 1.0
            mk.color.r = 1.0
            mk.color.g = 0.5
            mk.color.b = 0.0
            mk.color.a = 0.25            # translucent orange ring
            ma.markers.append(mk)
        self.cbf_zone_pub.publish(ma)


def main():
    rclpy.init()
    node = GMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
