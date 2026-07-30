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

from .gmpc           import GMPC, GMPCConfig
from .se2            import from_xytheta
from .path_processor import (path_msg_to_xyth,
                             build_reference_window,
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
        self.declare_parameter('Q_yaw',    5.0)
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
        self.declare_parameter('cbf_margin_growth', 0.0)
        # Prefer giving way FORWARD rather than backward when the CBF pushes the
        # robot off the reference. 0 = no preference (validated behaviour).
        self.declare_parameter('prog_weight', 0.0)
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
        self.declare_parameter('static_cbf_safe_margin', 0.33)
        # Static-CBF is a BACKSTOP for "don't dodge a dynamic obstacle into a
        # wall" — only engage it when a dynamic obstacle is within this range of
        # the robot. With no dynamic threat nearby the robot just tracks the
        # (smooth) global plan through gaps, instead of the two-sided wall CBF
        # constraints squeezing it into a left-right zig-zag in a passage.
        # 0 = always on (old behaviour).
        self.declare_parameter('static_activate_range', 0.0)

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
        self.tf_timeout_s     = float(self.get_parameter('tf_timeout').value)

        Qxy  = float(self.get_parameter('Q_xy').value)
        Qyaw = float(self.get_parameter('Q_yaw').value)
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
            prog_weight=float(self.get_parameter('prog_weight').value),
        )
        self.mpc = GMPC(cfg)
        self.N   = cfg.N
        self.cbf_enable = bool(self.get_parameter('cbf_enable').value)
        self.static_goal_clear = float(self.get_parameter('static_goal_clear').value)
        self.static_cbf_safe_margin = float(self.get_parameter('static_cbf_safe_margin').value)
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

    # ----------------------------------------------------------------------
    def _plan_cb(self, msg: Path):
        if len(msg.poses) == 0:
            self.get_logger().warn('Received empty plan')
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
        X_ref_win, xi_ref_win = build_reference_window(
            path_xyth, robot_xyth,
            N=self.N, dt=self.dt, v_nom=self.v_nom,
        )

        # 4. Solve
        X_now  = from_xytheta(*robot_xyth)
        if self.cbf_enable:
            obstacles = list(self._obstacles)                     # dynamic
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
                        # smaller keep-out for walls (see static_cbf_safe_margin)
                        obstacles.append({**s, 'margin': self.static_cbf_safe_margin})
        else:
            obstacles = None
        result = self.mpc.solve(X_now, X_ref_win, xi_ref_win, self.xi_prev,
                                obstacles=obstacles)

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
