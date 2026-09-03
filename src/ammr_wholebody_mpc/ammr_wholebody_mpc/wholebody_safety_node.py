"""ROS wrapper around the whole-body velocity safety filter.

    /wholebody_safety/cmd_in    Float64MultiArray, 9   desired [vx vy wz q1..q6]
    /arm_link_distance/points   PointCloud2            p_i, n_i, d_i, status, occluded
    /joint_states               JointState             the six arm joints
    TF report_frame -> base_link                       the base part of q
        |
    /wholebody_safety/cmd_out   Float64MultiArray, 9   filtered
    /wholebody_safety/diag      Float32MultiArray      what happened this cycle

The filter itself is verified offline. What only running it can test is the
part around it: message age, TF timing, and what happens when an input stops
arriving. So the node's job is mostly to decide what to do with data it does
not have, and every one of those decisions fails toward stopping:

    no cmd yet            output zero
    cmd older than max_cmd_age        output zero -- a stale command is not a
                                      request to keep moving
    points older than max_points_age  every row becomes NODATA, which the filter
                                      already degrades to a speed cap
    joints older than max_joint_age   output zero: without q there is no
                                      Jacobian, so nothing can be constrained
    TF missing                        base DOF are held at zero and the arm is
                                      still filtered, rather than dropping the
                                      whole command

Publishing zero on missing data is the only defensible default here. Passing
the command through unfiltered would mean the safety layer silently disappears
at exactly the moment its inputs broke.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from tf2_ros import Buffer, TransformListener

from .arm_detection_points import _quat_to_rot
from .wholebody_kinematics import DOF_NAMES, WholeBodyKinematics
from .wholebody_safety_filter import (STATUS_NODATA, DetectionPoint,
                                      SafetyConfig, filter_velocity)

ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
PC_FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status', 'age', 'occluded']
FRAMES = ['detect0_1', 'detect0_2', 'detect1', 'detect2_1', 'detect2_2',
          'detect2_3', 'detect3_1', 'detect3_2', 'detect4_1', 'detect4_2',
          'detect5', 'detect6']


class WholeBodySafetyNode(Node):

    def __init__(self) -> None:
        super().__init__('wholebody_safety')
        p = self.declare_parameter
        # The SOLVER's model, not the simulator's. /robot_description carries
        # the gz robot, whose base is a floating body with no base_x/base_y/
        # base_theta joints, so a 9-DOF Jacobian cannot be built from it. The
        # whole-body model declares those three as real joints precisely so one
        # Jacobian covers base and arm; this node must load THAT file. Accepts
        # a .xacro (expanded here) or an already-expanded .urdf.
        p('wholebody_urdf', '')
        p('report_frame', 'odom')
        p('base_frame', 'base_link')
        p('control_rate', 20.0)
        p('max_cmd_age', 0.25)
        p('max_points_age', 0.30)
        p('max_joint_age', 0.30)
        p('use_base_dof', True)
        p('alpha', 2.0)
        p('d0', 0.05)
        p('tau', 0.15)
        p('a_brake', 1.0)
        p('eps', 0.03)

        g = lambda k: self.get_parameter(k).value
        self.report_frame = str(g('report_frame'))
        self.base_frame = str(g('base_frame'))
        self.max_cmd_age = float(g('max_cmd_age'))
        self.max_points_age = float(g('max_points_age'))
        self.max_joint_age = float(g('max_joint_age'))
        self.use_base = bool(g('use_base_dof'))

        self.cfg = SafetyConfig(alpha=float(g('alpha')), d0=float(g('d0')),
                                tau=float(g('tau')), a_brake=float(g('a_brake')),
                                eps=float(g('eps')),
                                dt=1.0 / max(1.0, float(g('control_rate'))))

        self.K: WholeBodyKinematics | None = None
        path = str(g('wholebody_urdf'))
        if path:
            try:
                import subprocess
                xml = (subprocess.check_output(['xacro', path], text=True)
                       if path.endswith('.xacro') else open(path).read())
                self.K = WholeBodyKinematics.from_urdf_string(xml)
                self.get_logger().info(f'kinematics loaded from {path}')
            except Exception as exc:                            # noqa: BLE001
                self.get_logger().error(f'cannot load {path}: {exc}')
        self.q_arm = np.zeros(6)
        self.q_arm_t = 0.0
        self.cmd = np.zeros(9)
        self.cmd_t = 0.0
        self.pts_raw = None
        self.pts_t = 0.0
        self._cycle = 0

        from std_msgs.msg import String
        from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                               QoSProfile, QoSReliabilityPolicy)
        latched = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self._on_urdf,
                                 latched)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 self._on_pts, 10)
        self.create_subscription(Float64MultiArray, '~/cmd_in', self._on_cmd, 10)
        self.pub = self.create_publisher(Float64MultiArray, '~/cmd_out', 10)
        self.diag = self.create_publisher(Float32MultiArray, '~/diag', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(self.cfg.dt, self._tick)
        self.get_logger().info(
            f'wholebody_safety: frame {self.report_frame}, '
            f'{1.0/self.cfg.dt:.0f} Hz, base DOF {"on" if self.use_base else "off"}')

    # ------------------------------------------------------------ inputs
    def _on_urdf(self, msg) -> None:
        # Fallback only. /robot_description is normally the simulator's robot,
        # which has no virtual base joints; loading it succeeds only if someone
        # published the whole-body model there instead.
        if self.K is not None:
            return
        try:
            self.K = WholeBodyKinematics.from_urdf_string(msg.data)
            self.get_logger().info('kinematics loaded from /robot_description')
        except Exception as exc:                                # noqa: BLE001
            self.get_logger().error(
                f'/robot_description is not a whole-body model ({exc}); '
                f'set the wholebody_urdf parameter')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_js(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in ARM_JOINTS):
            return
        self.q_arm = np.array([msg.position[idx[j]] for j in ARM_JOINTS])
        self.q_arm_t = self._now()

    def _on_pts(self, msg: PointCloud2) -> None:
        self.pts_raw = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.width, len(PC_FIELDS)).astype(float)
        self.pts_t = self._now()

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        d = list(msg.data)
        if len(d) != 9:
            self.get_logger().warn(f'cmd_in must be 9 long, got {len(d)}',
                                   throttle_duration_sec=2.0)
            return
        self.cmd = np.array(d, dtype=float)
        self.cmd_t = self._now()

    # -------------------------------------------------------------- loop
    def _base_q(self):
        try:
            t = self.tf_buffer.lookup_transform(self.report_frame,
                                                self.base_frame,
                                                rclpy.time.Time())
        except Exception:
            return None
        q, tr = t.transform.rotation, t.transform.translation
        R = _quat_to_rot(q.x, q.y, q.z, q.w)
        return np.array([tr.x, tr.y, float(np.arctan2(R[1, 0], R[0, 0]))])

    def _tick(self) -> None:
        self._cycle += 1
        now = self._now()
        out = np.zeros(9)
        reason = 0.0        # 0 ok, 1 no cmd, 2 stale cmd, 3 stale joints,
                            # 4 no kinematics, 5 no points
        res = SafetyLike()

        if self.K is None:
            reason = 4.0
        elif self.cmd_t == 0.0:
            reason = 1.0
        elif now - self.cmd_t > self.max_cmd_age:
            reason = 2.0
        elif self.q_arm_t == 0.0 or now - self.q_arm_t > self.max_joint_age:
            # Without q there is no Jacobian and nothing can be constrained.
            reason = 3.0
        else:
            q = np.zeros(9)
            q[3:] = self.q_arm
            base = self._base_q() if self.use_base else None
            if base is not None:
                q[:3] = base
            v_in = self.cmd.copy()
            if base is None:
                v_in[:3] = 0.0          # cannot reason about a base we cannot locate

            pts = self._points(now)
            if pts is None:
                reason = 5.0
            else:
                r = filter_velocity(self.K, q, v_in, pts, self.cfg)
                out = r.v
                res = r

        m = Float64MultiArray()
        m.data = [float(v) for v in out]
        self.pub.publish(m)

        d = Float32MultiArray()
        #  0 cycle 1 reason 2 n_rows 3 n_active 4 resid_before 5 resid_after
        #  6 iters 7 fallback 8 unresolved 9 runtime_ms 10 speed_cap
        # 11 min_d 12 n_stale 13 n_nodata 14 n_occluded
        d.data = [float(self._cycle), reason, float(res.n_rows),
                  float(res.n_active), float(res.max_resid_before),
                  float(res.max_resid_after), float(res.iters),
                  1.0 if res.fallback else 0.0,
                  1.0 if res.unresolved else 0.0,
                  float(res.runtime_s * 1e3),
                  float(res.speed_cap if np.isfinite(res.speed_cap) else -1.0),
                  float(self._min_d), float(self._n_stale),
                  float(self._n_nodata), float(self._n_occl)]
        self.diag.publish(d)

    _min_d = -1.0
    _n_stale = 0
    _n_nodata = 0
    _n_occl = 0

    def _points(self, now: float):
        if self.pts_raw is None:
            return None
        stale_feed = (now - self.pts_t) > self.max_points_age
        pts, mind = [], np.inf
        self._n_stale = self._n_nodata = self._n_occl = 0
        for i, row in enumerate(self.pts_raw[:len(FRAMES)]):
            x, y, z, nx, ny, nz, dd, st, age, occ = row
            st = int(st)
            if stale_feed:
                # The whole feed is late: no row may be trusted as current.
                st = STATUS_NODATA
            if st == STATUS_NODATA:
                self._n_nodata += 1
            elif st != 0:
                self._n_stale += 1
            if occ >= 0.5:
                self._n_occl += 1
            nvec = np.array([nx, ny, nz])
            nn = float(np.linalg.norm(nvec))
            if nn < 1e-6:
                nvec = np.array([1.0, 0.0, 0.0])
            else:
                nvec = nvec / nn
            if st == 0:
                mind = min(mind, dd)
            pts.append(DetectionPoint(FRAMES[i], np.array([x, y, z]), nvec,
                                      float(dd), st, float(max(age, 0.0)),
                                      occ >= 0.5))
        self._min_d = mind if np.isfinite(mind) else -1.0
        return pts


class SafetyLike:
    """Zero-valued stand-in so diagnostics publish every cycle, including the
    cycles where nothing was computed. A missing diag message and a diag
    message saying 'refused' are not the same thing."""
    n_rows = 0
    n_active = 0
    max_resid_before = 0.0
    max_resid_after = 0.0
    iters = 0
    fallback = False
    unresolved = False
    speed_cap = float('inf')
    runtime_s = 0.0


def main() -> None:
    rclpy.init()
    node = WholeBodySafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
