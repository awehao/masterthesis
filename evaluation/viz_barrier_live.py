"""Live Foxglove view of the link barrier while it is acting.

What this draws, and why each piece is separate
-----------------------------------------------
The claim under test is "the barrier kept the whole arm clear of the box". Three
different quantities get confused with each other when only one number is shown,
so all three are published separately and never merged:

  min_d_model    min over the reported rows of (d - rho). This is the quantity
                 the barrier actually enforces. It is a LOWER bound on the true
                 clearance by the Lipschitz argument in section 10.11, which is
                 exactly why it is not the true clearance and must not be
                 labelled as one.

  clearance_ub   min distance from an INDEPENDENT dense random surface cloud
                 (seed 4242, a seed the barrier's sampler never saw) to the
                 obstacle. A minimum taken over a subset of the surface can only
                 be too large, so this is an UPPER bound on the true clearance.

  Together they bracket it:  min_d_model <= d_true <= clearance_ub. The gap
  between them is the price of discretisation, published as `conservatism`. If
  min_d_model ever rose ABOVE clearance_ub the safety model would be proven
  optimistic -- that is the falsification this trace exists to catch, and it is
  published as `bound_violation` so it cannot be missed by eye.

  Neither is the exact mesh distance. The exact triangle-to-triangle check is
  evaluation/mesh_distance.py, which is far too slow for a 10 Hz loop; it is run
  offline on the snapshots this node's numbers point at.

Markers, on /barrier_viz/markers in the report frame:

  grey box       the obstacle, drawn from the same world SDF the distance node
                 parses, so the picture cannot disagree with the constraint.
  yellow lines   every reported sample to its nearest obstacle point.
  red lines      the samples whose row the input command VIOLATED this cycle
                 (residual in > 0). These are the constraints doing the work;
                 everything else is inactive and drawn thin.
  blue arrow     the raw commanded TCP linear velocity, before the filter.
  green arrow    the corrected TCP linear velocity, after it.

The arrows are drawn at the TCP pose from TF but computed with the whole-body
Jacobian. A constant translation between the two models does not change a
Jacobian, so this is exact and not an approximation worth flagging.

    python3 evaluation/viz_barrier_live.py --ros-args \
        -p wholebody_urdf:=... -p world_sdf:=...
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import ColorRGBA, Float32, Float64MultiArray, Float32MultiArray
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))
sys.path.insert(0, _HERE)

from ammr_wholebody_mpc.arm_link_geometry import (  # noqa: E402
    arm_link_names, link_collision_tris, obstacle_distances, _surface_points)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import verify_pregrasp as VP  # noqa: E402

STATUS_OK = 0.0
ARM = [f'joint{i}' for i in range(1, 7)]


def _expand(path: str) -> str:
    import subprocess
    return (subprocess.check_output(['xacro', path], text=True)
            if path.endswith('.xacro') else open(path).read())


def _quat_to_rot(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pt(v):
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


class BarrierViz(Node):

    def __init__(self) -> None:
        super().__init__('barrier_viz')
        p = self.declare_parameter
        p('report_frame', 'world')
        p('wholebody_urdf', '')
        p('world_sdf', 'src/ammr_bringup/worlds/arm_barrier_test.sdf')
        p('tcp_link', 'uflite_gripper_link')
        # Per link. Measured on this machine: 15000 x 10 links = 240k
        # point-to-box distances costs 58 ms per pass, which at 4 Hz is a
        # quarter of a core running beside the simulator. 10000 at 3 Hz is
        # about 12%. A coarser cloud only widens the bracket -- a minimum over
        # fewer surface points can only come out larger -- so it stays a valid
        # upper bound and costs readability, not soundness.
        p('indep_n', 10000)
        # A seed that took no part in building the barrier's own samples. If
        # this ever matches the sampler's, the trace stops being independent
        # and starts measuring two copies of the same procedure against
        # each other.
        p('indep_seed', 4242)
        p('indep_rate', 3.0)
        p('marker_rate', 10.0)
        p('arrow_scale', 0.5)        # metres of arrow per m/s
        g = lambda k: self.get_parameter(k).value

        self.frame = str(g('report_frame'))
        self.tcp = str(g('tcp_link'))
        self.arrow_scale = float(g('arrow_scale'))

        src = str(g('wholebody_urdf'))
        if not src:
            raise RuntimeError('viz needs wholebody_urdf: the link names and the '
                               'Jacobian both come from the 9-DOF model, and '
                               'guessing either makes the picture a lie')
        xml = _expand(src)
        self.K = WholeBodyKinematics.from_urdf_string(xml)
        self.n_dof = len(self.K.dof_names)
        self.arm_idx = [self.K.dof_names.index(j) for j in ARM]
        self.link_names = arm_link_names(xml)

        world = str(g('world_sdf'))
        self.obs = VP.obstacles_from_world(world)
        if not self.obs:
            raise RuntimeError(f'no obs_* model found in {world}')
        self.get_logger().info(
            'obstacles: ' + ', '.join(f'{o.name} {o.kind}' for o in self.obs))

        rng = np.random.default_rng(int(g('indep_seed')))
        tris = link_collision_tris(xml)
        self.ref = {k: _surface_points(v, int(g('indep_n')), rng)
                    for k, v in tris.items() if k in self.link_names}
        self.get_logger().info(
            'independent reference cloud: '
            + ', '.join(f'{k} {len(v)}' for k, v in self.ref.items())
            + f'  (seed {int(g("indep_seed"))}, not the sampler\'s)')

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.rows = None            # (m, nf) float32, latest cloud
        self.binding = set()        # detection point indices with r_in > 0
        self.r_in = self.r_out = 0.0
        self.cmd_in = self.cmd_out = None
        self.q_arm = None
        self.diag = None

        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 self._on_pts, 10)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/barrier',
                                 self._on_barrier, 10)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/diag',
                                 self._on_diag, 10)
        self.create_subscription(Float64MultiArray, '/wholebody_safety/cmd_in',
                                 self._on_in, 10)
        self.create_subscription(Float64MultiArray, '/wholebody_safety/cmd_out',
                                 self._on_out, 10)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        self.mk = self.create_publisher(MarkerArray, '/barrier_viz/markers', 10)
        self.scal = {k: self.create_publisher(Float32, f'/barrier_viz/{k}', 10)
                     for k in ('min_d_model', 'clearance_ub', 'conservatism',
                               'bound_violation', 'resid_in', 'resid_out',
                               'n_binding', 'tcp_speed_in', 'tcp_speed_out')}

        self.create_timer(1.0 / float(g('marker_rate')), self._draw)
        self.create_timer(1.0 / float(g('indep_rate')), self._independent)
        self._ub = float('nan')

    # ---------------------------------------------------------------- inputs
    def _on_pts(self, msg: PointCloud2) -> None:
        nf = msg.point_step // 4
        if msg.width == 0:
            self.rows = None
            return
        self.rows = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.width, nf)

    def _on_barrier(self, msg: Float32MultiArray) -> None:
        a = np.asarray(msg.data, dtype=np.float64).reshape(-1, 3)
        self.binding = {int(i) for i, ri, _ in a if ri > 1e-9}
        self.r_in = float(a[:, 1].max()) if len(a) else 0.0
        self.r_out = float(a[:, 2].max()) if len(a) else 0.0

    def _on_diag(self, msg: Float32MultiArray) -> None:
        self.diag = list(msg.data)

    def _on_in(self, msg: Float64MultiArray) -> None:
        self.cmd_in = np.asarray(msg.data, dtype=float)

    def _on_out(self, msg: Float64MultiArray) -> None:
        self.cmd_out = np.asarray(msg.data, dtype=float)

    def _on_js(self, msg: JointState) -> None:
        ix = {n: i for i, n in enumerate(msg.name)}
        if all(a in ix for a in ARM) and len(msg.position) > max(ix[a] for a in ARM):
            self.q_arm = np.array([msg.position[ix[a]] for a in ARM])

    # ------------------------------------------------------------------ util
    def _tf(self, child):
        try:
            t = self.tf_buffer.lookup_transform(self.frame, child,
                                                rclpy.time.Time())
        except Exception:
            return None
        r, q = t.transform.translation, t.transform.rotation
        T = np.eye(4)
        T[:3, :3] = _quat_to_rot(q.x, q.y, q.z, q.w)
        T[:3, 3] = [r.x, r.y, r.z]
        return T

    def _q9(self):
        """Generalised coordinate for the Jacobian.

        The base is fixed for this test, so its three coordinates are zero and
        a constant offset between the simulated base pose and the model origin
        does not enter: a rigid translation leaves every Jacobian unchanged.
        """
        if self.q_arm is None:
            return None
        q = np.zeros(self.n_dof)
        q[self.arm_idx] = self.q_arm
        return q

    def _tcp_vel(self, v):
        q = self._q9()
        if q is None or v is None or len(v) != self.n_dof:
            return None
        J = self.K.jacobian(q, self.tcp)
        return J[:3] @ v

    # --------------------------------------------------- independent bracket
    def _independent(self) -> None:
        """Upper bound on true clearance, from a cloud the barrier never saw."""
        live = [o for o in self.obs if o.T_world_link is not None]
        best = float('inf')
        for name in self.link_names:
            pts = self.ref.get(name)
            T = self._tf(name)
            if pts is None or T is None:
                continue
            W = (pts @ T[:3, :3].T) + T[:3, 3]
            d, _, _ = obstacle_distances(W, live)
            best = min(best, float(d.min()))
        self._ub = best if np.isfinite(best) else float('nan')

    # ------------------------------------------------------------------ draw
    def _draw(self) -> None:
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()

        def base(mid, typ, scale, colour):
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = now
            m.ns = 'barrier'
            m.id = mid
            m.type = typ
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = colour
            return m

        # obstacle, from the same file the constraint reads
        for i, o in enumerate(self.obs):
            if o.kind != 'box':
                continue
            m = base(100 + i, Marker.CUBE, 1.0, _rgba(0.55, 0.55, 0.58, 0.35))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                (float(x) for x in o.T_world_link[:3, 3])
            m.scale.x, m.scale.y, m.scale.z = (float(s) for s in o.size)
            arr.markers.append(m)

        min_model = float('nan')
        n_bind = 0
        if self.rows is not None and len(self.rows):
            R = self.rows
            nf = R.shape[1]
            ok = R[:, 7] == STATUS_OK
            rho = R[:, 14] if nf >= 15 else np.zeros(len(R))
            if ok.any():
                min_model = float((R[ok, 6] - rho[ok]).min())

            slack = base(1, Marker.LINE_LIST, 0.0015, _rgba(0.95, 0.85, 0.15, 0.75))
            bind = base(2, Marker.LINE_LIST, 0.0045, _rgba(0.95, 0.15, 0.15, 1.0))
            pts_b = base(3, Marker.SPHERE_LIST, 0.014, _rgba(0.95, 0.15, 0.15, 1.0))
            for i in range(len(R)):
                if R[i, 7] != STATUS_OK:
                    continue
                p = R[i, 0:3].astype(float)
                s = p + R[i, 3:6].astype(float) * abs(float(R[i, 6]))
                a, b = _pt(p), _pt(s)
                if i in self.binding:
                    n_bind += 1
                    bind.points += [a, b]
                    pts_b.points.append(a)
                else:
                    slack.points += [a, b]
            arr.markers += [slack, bind, pts_b]

        # velocity arrows at the TCP
        T = self._tf(self.tcp)
        vin = self._tcp_vel(self.cmd_in)
        vout = self._tcp_vel(self.cmd_out)
        s_in = float(np.linalg.norm(vin)) if vin is not None else 0.0
        s_out = float(np.linalg.norm(vout)) if vout is not None else 0.0
        if T is not None:
            o = T[:3, 3]
            for mid, v, col in ((4, vin, _rgba(0.20, 0.45, 0.95)),
                                (5, vout, _rgba(0.15, 0.85, 0.35))):
                if v is None or np.linalg.norm(v) < 1e-6:
                    continue
                m = base(mid, Marker.ARROW, 0.0, col)
                m.scale.x, m.scale.y, m.scale.z = 0.008, 0.018, 0.03
                tip = o + v * self.arrow_scale
                m.points = [_pt(o), _pt(tip)]
                arr.markers.append(m)

            txt = base(6, Marker.TEXT_VIEW_FACING, 0.035, _rgba(1, 1, 1, 0.95))
            txt.pose.position.x, txt.pose.position.y = float(o[0]), float(o[1])
            txt.pose.position.z = float(o[2]) + 0.16
            ub = self._ub
            txt.text = (f'model min(d-rho) {min_model*1000:7.1f} mm\n'
                        f'indep upper bd   {ub*1000:7.1f} mm\n'
                        f'binding rows     {n_bind}\n'
                        f'resid in/out  {self.r_in*1000:+.2f} / '
                        f'{self.r_out*1000:+.2f} mm/s\n'
                        f'TCP |v| {s_in*1000:.1f} -> {s_out*1000:.1f} mm/s')
            arr.markers.append(txt)

        self.mk.publish(arr)

        ub = self._ub
        cons = ub - min_model if np.isfinite(ub) and np.isfinite(min_model) else float('nan')
        out = {'min_d_model': min_model, 'clearance_ub': ub,
               'conservatism': cons,
               # Positive means the enforced lower bound exceeded an upper
               # bound on the truth, which cannot happen if the model is sound.
               'bound_violation': (min_model - ub) if np.isfinite(cons) else float('nan'),
               'resid_in': self.r_in, 'resid_out': self.r_out,
               'n_binding': float(n_bind),
               'tcp_speed_in': s_in, 'tcp_speed_out': s_out}
        for k, v in out.items():
            msg = Float32()
            msg.data = float(v)
            self.scal[k].publish(msg)


def main() -> None:
    rclpy.init()
    n = BarrierViz()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
