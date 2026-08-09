"""Last-resort safety layer driven by raw /scan, independent of tracking.

Why this exists
---------------
Perception being able to SEE an obstacle does not mean the barrier is given a
constraint for it. Between the scan and the CBF sit a cluster step, a track, an
age test, an instantaneous-speed test and a net-displacement test, and every one
of them is a hard gate: fail it and the object does not enter the QP at all --
not weakly, not with a small weight, but absent. Measured on seed27, a
1.6 x 0.4 m box drove into a stopped robot while a healthy track (age median
218, KF speed 0.097 m/s against a true 0.096) sat right on it, published in only
29% of cycles because the box's configured speed IS 0.10 m/s and the publish
threshold was 0.10.

Fixing those thresholds one at a time closes one case each. This closes the
class, by asserting a single invariant that needs no classification:

    while a valid return exists, the robot may not keep closing on it

The high-level GMPC + horizon CBF still owns efficient avoidance, prediction and
routing. This owns only the guarantee that a classification failure cannot
become an immediate contact.

Formulation
-----------
For each valid scan point p_i in the base frame, with n_i the unit vector from
the robot toward it and r_i the point on the body surface facing it, the
approach rate of that surface point is

    v_n,i = n_i^T (v + w * J r_i),      J = [[0,-1],[1,0]]

and the shield enforces a first-order barrier

    v_n,i >= -alpha * (d_i - d_stop,i)

so the permitted approach speed shrinks with distance and reverses once inside
d_stop. Tangential motion and retreat are never restricted -- an all-stop is far
more likely to strand the robot in a doorway than to save it.

The rotation term is carried explicitly even though it vanishes for the current
circular footprint (n^T J n = 0: spinning a disc moves no surface point toward
an external obstacle). It is not dropped, because a non-circular footprint makes
it the dominant term, and a shield that only limited v_x could be swept into an
obstacle by yaw alone.

d_stop is built from the chassis's real numbers rather than a tuned constant:

    d_stop = d0 + v*tau + v^2 / (2 a_brake) + eps_scan + eps_footprint

A fixed threshold is simultaneously too timid when crawling and too short at
speed.

Deliberately NOT done here: clustering, tracking, association, ageing,
prediction. Rebuilding any of that would reintroduce the failure this layer
exists to cover. The only filtering is range validity, self-return rejection,
and an optional neighbour-consistency test that is bypassed inside the
emergency distance -- a single stray return close enough to matter is treated as
real.
"""

from __future__ import annotations

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import tf2_ros
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class ScanSafetyShield(Node):

    def __init__(self):
        super().__init__('scan_safety_shield')
        p = self.declare_parameter
        p('scan_topic', '/scan')
        p('cmd_in_topic', '/cmd_vel_pre_shield')
        p('cmd_out_topic', '/cmd_vel')
        p('base_frame', 'base_footprint')
        p('enable', True)

        p('robot_radius', 0.30)         # m, circular collision footprint
        p('alpha', 2.0)                 # 1/s, barrier relaxation rate
        p('d0', 0.05)                   # m, standoff at zero speed
        p('tau', 0.15)                  # s, sense + control + actuation latency
        p('a_brake', 6.25)              # m/s^2, from the wheel Jacobian
        p('eps_scan', 0.03)             # m, range noise allowance
        p('eps_footprint', 0.02)        # m, footprint fit allowance
        p('range_of_interest', 2.0)     # m, ignore returns further than this
        p('self_return_m', 0.02)        # m, drop hits inside the footprint+this
        p('neighbour_consistency', True)
        p('emergency_d', 0.15)          # m, below this a lone return still counts
        p('scan_timeout', 0.5)          # s, stale scan -> clamp
        p('stale_speed', 0.05)          # m/s, cap while the scan is stale
        p('max_points', 360)            # cost bound per cycle
        p('proj_iters', 6)              # fixed count -> deterministic runtime
        p('vx_max', 0.2775)
        p('vy_max', 0.2775)
        p('wz_max', 1.1327)

        g = lambda k: self.get_parameter(k).value
        self.enable = bool(g('enable'))
        self.R = float(g('robot_radius'))
        self.alpha = float(g('alpha'))
        self.d0 = float(g('d0'))
        self.tau = float(g('tau'))
        self.a_brake = max(1e-3, float(g('a_brake')))
        self.eps = float(g('eps_scan')) + float(g('eps_footprint'))
        self.roi = float(g('range_of_interest'))
        self.self_m = float(g('self_return_m'))
        self.neigh = bool(g('neighbour_consistency'))
        self.emerg = float(g('emergency_d'))
        self.scan_timeout = float(g('scan_timeout'))
        self.stale_speed = float(g('stale_speed'))
        self.max_pts = int(g('max_points'))
        self.iters = int(g('proj_iters'))
        self.u_max = np.array([float(g('vx_max')), float(g('vy_max')),
                               float(g('wz_max'))])
        self.base_frame = str(g('base_frame'))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._pts = None            # (N,2) in base frame
        self._scan_t = 0.0

        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, str(g('scan_topic')),
                                 self._scan_cb, scan_qos)
        self.create_subscription(Twist, str(g('cmd_in_topic')),
                                 self._cmd_cb, 10)
        self.pub = self.create_publisher(Twist, str(g('cmd_out_topic')), 10)
        # cycle, active, n_pts, d_min, dv, vx_in, vy_in, wz_in,
        # vx_out, vy_out, wz_out, scan_age, stale
        self.diag = self.create_publisher(Float32MultiArray, '/shield/diag', 10)
        self._cycle = 0
        self.get_logger().info(
            f'scan shield {"ON" if self.enable else "OFF (pass-through)"}: '
            f'{g("cmd_in_topic")} -> {g("cmd_out_topic")}, alpha={self.alpha}, '
            f'a_brake={self.a_brake}')

    # ------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        """Raw returns to base-frame points. No clustering, by design."""
        n = len(msg.ranges)
        if n == 0:
            return
        r = np.asarray(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(n) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        ok &= r <= self.roi + self.R
        if self.neigh:
            # A lone return with no neighbour within 0.15 m is treated as noise
            # -- EXCEPT inside the emergency distance, where a single hit is
            # already too close to argue with.
            rr = np.where(ok, r, np.inf)
            near = np.full(n, False)
            for shift in (-1, 1):
                s = np.roll(rr, shift)
                near |= np.abs(rr - s) < 0.15
            ok &= (near | (r <= self.R + self.emerg))
        if not np.any(ok):
            self._pts = np.empty((0, 2))
            self._scan_t = self.get_clock().now().nanoseconds * 1e-9
            return
        idx = np.where(ok)[0]
        if len(idx) > self.max_pts:
            idx = idx[np.argsort(r[idx])[:self.max_pts]]
        xs = r[idx] * np.cos(ang[idx])
        ys = r[idx] * np.sin(ang[idx])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.05))
            q = tf.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            c, s = math.cos(yaw), math.sin(yaw)
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            bx = tx + c * xs - s * ys
            by = ty + s * xs + c * ys
        except Exception:
            # Sensor-to-base is fixed geometry; if TF is momentarily missing,
            # using the raw sensor frame is a small error, whereas discarding
            # the scan blinds the last safety layer.
            bx, by = xs, ys

        d = np.hypot(bx, by)
        keep = d > self.R + self.self_m
        self._pts = np.stack([bx[keep], by[keep]], axis=1)
        self._scan_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------
    def _limit(self, u: np.ndarray, pts: np.ndarray):
        """Project u onto {a_i . u <= b_i} by successive projection.

        A fixed iteration count keeps the runtime bounded and the output
        deterministic; a solver call in the last safety layer would put an
        unbounded latency in the one place that must never miss its deadline.
        Unit normals make each projection a single axpy.
        """
        if not len(pts):
            return u, 0.0, float('inf')
        d = np.hypot(pts[:, 0], pts[:, 1]) - self.R      # surface to surface
        nrm = pts / np.maximum(np.hypot(pts[:, 0], pts[:, 1])[:, None], 1e-9)
        v = float(np.hypot(u[0], u[1]))
        d_stop = (self.d0 + v * self.tau + v * v / (2.0 * self.a_brake)
                  + self.eps)
        b = self.alpha * (d - d_stop)                    # allowed approach rate

        # n_i points robot -> obstacle, so the separation rate is d_dot =
        # -n_i^T v_surface and the barrier d_dot + alpha*(d - d_stop) >= 0
        # becomes  n_i^T (v + w J r_i) <= alpha (d_i - d_stop), i.e.
        #   a_i = [n_x, n_y, n^T J r_i],  b_i = alpha (d_i - d_stop).
        # The yaw column is n^T J n * R = 0 for a disc; it is kept explicit so a
        # non-circular footprint only has to supply a different r_i.
        Jr = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1) * self.R
        a = np.concatenate([nrm, (nrm * Jr).sum(axis=1, keepdims=True)],
                           axis=1)
        a2 = np.maximum((a * a).sum(axis=1), 1e-9)

        out = u.copy()
        for _ in range(self.iters):
            viol = a @ out - b
            if np.all(viol <= 1e-6):
                break
            i = int(np.argmax(viol))
            out = out - (viol[i] / a2[i]) * a[i]
            out = np.clip(out, -self.u_max, self.u_max)
        return out, float(np.linalg.norm(out - u)), float(np.min(d))

    def _cmd_cb(self, msg: Twist):
        u = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        now = self.get_clock().now().nanoseconds * 1e-9
        age = now - self._scan_t if self._scan_t else float('inf')
        stale = age > self.scan_timeout
        dv, dmin, npts = 0.0, float('inf'), 0

        if not self.enable:
            out = u
        elif stale:
            # No fresh view: do not travel fast into what cannot be seen. Not a
            # full stop -- the robot may still creep and turn, which is what
            # lets it recover rather than blocking a corridor.
            sp = float(np.hypot(u[0], u[1]))
            out = u.copy()
            if sp > self.stale_speed:
                out[:2] = u[:2] * (self.stale_speed / sp)
            dv = float(np.linalg.norm(out - u))
        else:
            pts = self._pts if self._pts is not None else np.empty((0, 2))
            npts = len(pts)
            out, dv, dmin = self._limit(u, pts)

        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = (float(out[0]), float(out[1]),
                                               float(out[2]))
        self.pub.publish(t)

        self._cycle += 1
        d = Float32MultiArray()
        d.data = [float(self._cycle), 1.0 if dv > 1e-4 else 0.0, float(npts),
                  dmin if math.isfinite(dmin) else -1.0, dv,
                  float(u[0]), float(u[1]), float(u[2]),
                  float(out[0]), float(out[1]), float(out[2]),
                  age if math.isfinite(age) else -1.0, 1.0 if stale else 0.0]
        self.diag.publish(d)


def main():
    rclpy.init()
    node = ScanSafetyShield()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
