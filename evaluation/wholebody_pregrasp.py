"""Whole-body pre-grasp: one solver decides the base and the arm together.

The target is a gripper pose fixed in WORLD coordinates. Where the base ends up
is an output, not an input -- nothing here says the chassis must stop anywhere
in particular, and no base trajectory is planned.

What the solver is, precisely
-----------------------------
Whole-body resolved-motion rate control: at every cycle it builds the 9-column
Jacobian of the tool frame with respect to q = (base_x, base_y, base_theta,
joint1..joint6) and solves ONE damped least-squares problem for the 6-D task
error. It is not the geometric MPC and is not called that here. What matters
for the request is the property it does have: base and arm columns sit in the
same matrix and are chosen by the same solve, so the chassis moving forward and
the arm unfolding are two parts of one answer rather than two trajectories
played side by side.

A weight makes the base cheaper than the arm in that solve, which is what
produces the behaviour worth watching -- the base carries the gross translation
while the arm handles orientation and the last of the reach -- but the weight
does not decouple them: a single pseudo-inverse still decides both.

Nine DOF against a six-dimensional task leaves three redundant, and the posture
is what resolves that redundancy. It is a WEIGHTED TERM IN THE SAME LEAST
SQUARES, not a null-space projection:

    min_v  ||J v - e||^2  +  mu^2 ||v_arm - kp (q_pref - q_arm)||^2
                          +  lam^2 ||v / w||^2

Null-space projection was tried first and does not work here, for a reason
worth recording. The projected term carries no base component, so it can bias
the arm but can never park the CHASSIS. The task was then free to satisfy
itself by driving the base, which it did: the base ran to x = +0.39 while the
target sat at x = +0.30, i.e. past it, leaving the arm to reach backwards and
fold into itself. Position error settled at 2-4 mm while orientation never
converged at all, and the arm ended in self-collision.

As a weighted term the posture competes with the task over ALL nine columns.
Holding the arm near a chosen configuration fixes the tool's offset from the
base, so satisfying the task then requires the base to stop at the one place
where that offset lands the tool on the target -- the parking distance is an
output of the same solve rather than a number anyone typed in.

q_pref is the pre-grasp arm configuration whose self-clearance and path were
checked offline in section 10.14. Joint limits are handled by a potential in
the same term: joint3's safe range is [-0.061, +2.936], one-sided, and the
checked start pose sits only 0.36 rad above the bottom of it.

The command goes to /wholebody_safety/cmd_in with fix_base OFF, so all nine
components pass through the barrier and the velocity/acceleration boxes, and
leave as one message that the gate forwards or zeroes as a unit.

    python3 evaluation/wholebody_pregrasp.py --target 0.30 0.0 0.55
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray
from tf2_ros import Buffer, TransformListener

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, _HERE)

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS, rot_error   # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (            # noqa: E402
    STATUS_OK as F_OK, DetectionPoint, SafetyConfig, _box_rows,
    _joint_limit_rows, _rows_from_points)
from arm_poses import pose as named_pose                            # noqa: E402

# World z of the model root (base_footprint). base_footprint is DEFINED as the
# ground contact frame, so this is zero -- and it stayed zero only after the
# support spheres were corrected. They used to reach one wheel radius below
# base_footprint, so gz settled the model with base_footprint 0.05 m up and the
# wheels floating; the spawn height, this constant and the static TF all carried
# a matching 0.05 to compensate, and the compensation was invisible because all
# three agreed. Changing the model without changing all of them would have left
# the controller working in the old frame.
MODEL_Z = 0.0
STATUS_OK = 0.0


def _expand(path):
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


def wait_fresh(node, getters, timeout=20.0, fresh=0.08):
    """Spin until every feed is not just present but CURRENT.

    Waiting for "has arrived at least once" let a run begin with a joint state
    that was 335 ms old, because the wait loop exited the moment the LAST feed
    appeared while the first one had gone quiet during startup contention. The
    guard then fired on its first check and the run aborted before commanding
    anything. Freshness is the property the guard tests, so it is the property
    to wait for.
    """
    import time as _t
    t0 = _t.monotonic()
    while _t.monotonic() - t0 < timeout:
        node.exec.spin_once(timeout_sec=0.02)
        ages = [g() for g in getters]
        if all(a is not None and a < fresh for a in ages):
            return True
    return False


class WholeBody(Node):

    def __init__(self, a):
        super().__init__('wholebody_pregrasp')
        self.a = a
        xml = _expand(a.urdf)
        self.K = WholeBodyKinematics.from_urdf_string(xml)
        self.n = len(self.K.dof_names)
        self.idx = [self.K.dof_names.index(j) for j in ARM_JOINTS]
        from ammr_wholebody_mpc.arm_link_geometry import arm_link_names
        self.link_names = arm_link_names(xml)
        if self.n < 9:
            raise RuntimeError('需要 9 自由度全身模型')

        self.q_arm = None
        self.q_arm_t = 0.0
        self.base = None                # (x, y, yaw) in world
        self.base_t = 0.0
        self.base_stamp = None
        self.rows = None
        self.min_d = float('nan')
        self.ub = float('nan')
        self.r_in = self.r_out = 0.0
        self.n_bind = 0
        self.cmd_out = None

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self, spin_thread=True)
        # Callbacks run in a ReentrantCallbackGroup under a MultiThreadedExecutor.
        # With the default single-threaded executor and mutually-exclusive groups, one
        # slow callback blocks every other feed, and under load that showed up as
        # guards firing on data that was not actually late: /joint_states measured
        # 140 Hz with a 9 ms worst gap on the wire while the node saw 335 ms, and
        # /odom measured 27 Hz with a 38 ms worst gap while the run aborted on a
        # "base pose timeout". Raising the thresholds would have hidden a scheduling
        # problem behind a weakened safety check.
        self.cbg = ReentrantCallbackGroup()
        self.exec = MultiThreadedExecutor(num_threads=4)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10, callback_group=self.cbg)
        self.create_subscription(Odometry, '/odom', self._on_odom,
                                 qos_profile_sensor_data, callback_group=self.cbg)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 self._on_pts, 10, callback_group=self.cbg)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/barrier',
                                 self._on_barrier, 10, callback_group=self.cbg)
        self.create_subscription(Float64MultiArray, '/wholebody_safety/cmd_out',
                                 self._on_out, 10, callback_group=self.cbg)
        self.create_subscription(Float32, '/barrier_viz/clearance_ub',
                                 self._on_ub, 10, callback_group=self.cbg)
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)
        self.exec.add_node(self)
        self.cfg = SafetyConfig(dt=1.0 / a.rate)
        self.v_prev = np.zeros(self.n)
        # NOT reset here: link_names is set above from the description, and
        # clearing it made every cloud row fail the link-index check, so the
        # constraint set came out empty, the QP branch was skipped, and the
        # run reproduced the two-stage result exactly while reporting success.
        self.n_bar_rows = 0
        self.n_qp_fail = 0
        self.log = []

    # ---------------------------------------------------------------- inputs
    def _on_js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if all(j in ix for j in ARM_JOINTS) and \
                len(m.position) > max(ix[j] for j in ARM_JOINTS):
            self.q_arm = np.array([m.position[ix[j]] for j in ARM_JOINTS])
            self.q_arm_t = time.monotonic()

    def _on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y),
                         1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        self.base = np.array([p.x, p.y, yaw])
        self.base_t = time.monotonic()
        # The message's OWN time, kept separately. Base speed differenced
        # against the control loop's clock instead reported 0.3239 m/s against
        # a 0.2775 limit on two cycles while the command never exceeded
        # 0.2617 -- the loop runs at 53 ms and /odom at 34 ms, so dividing an
        # odom displacement by a loop interval is not a speed. Same class of
        # error as differencing TF against wall time in the barrier probe.
        self.base_stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def _on_pts(self, m):
        nf = m.point_step // 4
        self.rows = (np.frombuffer(m.data, dtype=np.float32).reshape(m.width, nf)
                     if m.width else None)
        if self.rows is not None:
            ok = self.rows[:, 7] == STATUS_OK
            rho = self.rows[:, 14] if nf >= 15 else np.zeros(len(self.rows))
            self.min_d = float((self.rows[ok, 6] - rho[ok]).min()) if ok.any() \
                else float('nan')

    def _on_barrier(self, m):
        b = np.asarray(m.data, dtype=float).reshape(-1, 3)
        self.r_in = float(b[:, 1].max()) if len(b) else 0.0
        self.r_out = float(b[:, 2].max()) if len(b) else 0.0
        self.n_bind = int((b[:, 1] > 1e-9).sum()) if len(b) else 0

    def _on_out(self, m):
        self.cmd_out = np.asarray(m.data, dtype=float)

    def _on_ub(self, m):
        self.ub = float(m.data)

    # ------------------------------------------------------------------ util
    def q9(self):
        q = np.zeros(self.n)
        q[:3] = self.base
        q[self.idx] = self.q_arm
        return q

    def tool(self, q):
        """Tool pose in WORLD coordinates.

        The kinematic model's root and the simulator's world differ by the
        spawn height and nothing else, so one translation converts between them.
        """
        T = self.K.fk(q, self.a.tcp).copy()
        T[2, 3] += MODEL_Z
        return T

    def guard(self):
        ja = (time.monotonic() - self.q_arm_t) if self.q_arm is not None else 1e9
        ba = (time.monotonic() - self.base_t) if self.base is not None else 1e9
        if ja > self.a.max_data_age:
            return f'JointState 已 {ja*1e3:.0f} ms 未更新'
        if ba > self.a.max_data_age:
            return f'底盤位姿已 {ba*1e3:.0f} ms 未更新'
        if self.rows is None:
            return '尚未收到距離資料'
        lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
        if np.any(self.q_arm < lo + 0.05) or np.any(self.q_arm > hi - 0.05):
            return f'關節接近安全限位 {np.round(self.q_arm,3).tolist()}'
        if np.isfinite(self.min_d) and self.min_d < self.a.abort_d:
            return f'模型間距 {self.min_d*1000:.1f} mm 低於中止門檻'
        if np.linalg.norm(self.base[:2] - self.base0[:2]) > self.a.max_base_travel:
            return f'底盤位移超過 {self.a.max_base_travel:.2f} m 上限'
        return None

    def solve(self, T_des):
        """One damped least-squares over all nine DOF.

        Weighted, not partitioned: W scales the columns inside a single
        pseudo-inverse, so the base is preferred where it helps and the arm
        still contributes to the same task error. Solving the base first and
        the arm afterwards would be two problems, and their sum would not be
        the minimiser of either.
        """
        q = self.q9()
        T = self.tool(q)
        e_p = T_des[:3, 3] - T[:3, 3]
        # SIGN: rot_error(R_cur, R_des) is already the error to be driven to
        # zero -- the same convention solve_ik uses. Negating it drives the tool
        # AWAY from the target orientation, and the run does not simply fail:
        # it converges to the antipodal point 180 degrees away, where the
        # gradient vanishes and the solver sits. Position still reached 2.8 mm,
        # so every position-based check looked healthy while the tool pointed
        # backwards, the base overran the target by 0.45 m and the arm folded
        # into itself reaching back for it.
        e_r = rot_error(T[:3, :3], T_des[:3, :3])
        # Cap the task-space step so a large initial error does not ask for a
        # velocity the boxes must then clip in a direction nobody chose.
        if np.linalg.norm(e_p) > self.a.v_task:
            e_p = e_p * (self.a.v_task / np.linalg.norm(e_p))
        if np.linalg.norm(e_r) > self.a.w_task:
            e_r = e_r * (self.a.w_task / np.linalg.norm(e_r))
        e = np.concatenate([self.a.kp_p * e_p, self.a.kp_r * e_r])

        J = self.K.jacobian(q, self.a.tcp)                  # 6 x 9
        w = np.ones(self.n)
        w[:3] = self.a.base_weight                          # >1 = base cheaper

        # Desired arm posture rate: pull toward q_pref, plus a limit potential
        # that pushes away from a bound BEFORE it is reached rather than at it.
        lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
        m = self.a.limit_margin
        push = (np.maximum(0.0, m - (self.q_arm - lo))
                - np.maximum(0.0, m - (hi - self.q_arm))) / m
        v_post = np.zeros(self.n)
        v_post[self.idx] = (self.a.kp_post * (self.q_pref - self.q_arm)
                            + self.a.k_limit * push)
        S = np.zeros((self.n, self.n))
        S[self.idx, self.idx] = 1.0                         # arm rows only

        mu, lam = self.a.mu_post, self.a.damping
        H = (J.T @ J + mu * mu * S + lam * lam * np.diag(1.0 / (w * w)))
        g = J.T @ e + mu * mu * (S @ v_post)
        v = np.linalg.solve(H, g)
        self.n_qp_fail = getattr(self, 'n_qp_fail', 0)
        if self.a.solver == 'qp':
            # Same objective, same weights; the constraints now sit INSIDE the
            # problem instead of being applied to its answer afterwards. The
            # unconstrained solution above is kept as the linearisation point
            # for d_stop, so both architectures see the same constraint set.
            A, b, nb = self._constraints(q, v)
            if len(A) == 0:
                raise RuntimeError('QP: 約束集合為空')
            vq = self._solve_qp(H, -g, A, b)
            if vq is None:
                # A solve that did not converge is also not a licence to send
                # the unconstrained answer.
                self.n_qp_fail += 1
                raise RuntimeError('QP: 求解未收斂')
            self.n_bar_rows = nb
            v = vq
        return v, T, float(np.linalg.norm(T_des[:3, 3] - T[:3, 3])), \
            float(np.linalg.norm(rot_error(T[:3, :3], T_des[:3, :3])))

    def _constraints(self, q, v_lin):
        """The constraint set the downstream filter would build, built here.

        Same rows, same linearisation point: d_stop depends on the approach
        speed, so it is evaluated at the SAME v_in the two-stage pipeline would
        have handed the filter. That keeps the two architectures comparable --
        the difference under test is where the constraints enter the solve, not
        which constraints they are.
        """
        # fail closed. An empty constraint set is not "no constraints", it is a
        # solver running blind, and the first version of this returned None
        # here, the QP branch fell through to the unconstrained answer, and the
        # run reported success while reproducing the two-stage result exactly.
        # Anything that leaves this function without a real constraint set now
        # stops the run.
        if self.K is None:
            raise RuntimeError('QP: 沒有運動學模型')
        if self.rows is None:
            raise RuntimeError('QP: 沒有距離資料')
        if not self.link_names:
            raise RuntimeError('QP: 連桿名單為空，列的連桿索引無法解讀')
        pts = []
        R = self.rows
        for r in R:
            if r[7] != F_OK:
                continue
            li = int(r[10])
            if li >= len(self.link_names):
                continue
            pts.append(DetectionPoint(
                frame=self.link_names[li], p=np.asarray(r[0:3], float),
                n=np.asarray(r[3:6], float), d=float(r[6]), status=int(r[7]),
                age=float(r[8]), occluded=bool(r[9] >= 0.5),
                offset=np.asarray(r[11:14], float), rho=float(r[14])))
        if not pts:
            raise RuntimeError(
                f'QP: {len(R)} 列距離資料中沒有一列可用（狀態或連桿索引不符），'
                '約束集合為空')
        cfg = self.cfg
        Ab, bb, cap, _ = _rows_from_points(self.K, q, pts, cfg, v_lin)
        Aj, bj = _joint_limit_rows(self.K, q, cfg)
        Ax, bx = _box_rows(cfg, self.n, cap, self.v_prev, cfg.dt)
        A = np.array(Ab + Aj + Ax)
        b = np.array(bb + bj + bx)
        return A, b, len(Ab)

    def _solve_qp(self, H, g, A, b):
        import osqp
        from scipy import sparse
        P = sparse.csc_matrix((H + H.T) / 2.0)
        m = osqp.OSQP()
        m.setup(P=P, q=g, A=sparse.csc_matrix(A),
                l=np.full(len(b), -np.inf), u=b, verbose=False,
                eps_abs=1e-7, eps_rel=1e-7, max_iter=8000, polish=False)
        r = m.solve()
        if r.info.status_val not in (1, 2):
            return None
        return np.asarray(r.x, float)

    def stop(self):
        m = Float64MultiArray()
        m.data = [0.0] * self.n
        for _ in range(6):
            self.pub.publish(m)
            self.exec.spin_once(timeout_sec=0.01)

    # ------------------------------------------------------------------- run
    def run(self, T_des):
        a = self.a
        self.base0 = self.base.copy()
        self.n_stall = 0
        self.q_pref = np.array(a.posture, dtype=float)
        q0 = self.q9()
        print(f'  起始 底盤 ({self.base[0]:+.3f}, {self.base[1]:+.3f}, '
              f'{math.degrees(self.base[2]):+.1f}°)  手臂 '
              f'{np.round(self.q_arm,3).tolist()}')
        T0 = self.tool(q0)
        print(f'  起始 TCP ({T0[0,3]:.3f}, {T0[1,3]:.3f}, {T0[2,3]:.3f})  '
              f'→ 目標 ({T_des[0,3]:.3f}, {T_des[1,3]:.3f}, {T_des[2,3]:.3f})')
        t0 = time.monotonic()
        settle = None
        while True:
            deadline = time.monotonic() + 1.0 / a.rate
            while time.monotonic() < deadline:
                self.exec.spin_once(timeout_sec=0.002)
            why = self.guard()
            if why:
                self.stop()
                print(f'  中止：{why}', flush=True)
                return False
            t = time.monotonic() - t0
            if t > a.timeout_s:
                self.stop()
                print(f'  逾時 {a.timeout_s:.0f} s', flush=True)
                return False
            try:
                v, T, ep, er = self.solve(T_des)
            except RuntimeError as exc:
                self.stop()
                print(f'  中止（fail closed）：{exc}', flush=True)
                return False
            done = ep < a.tol_p and er < a.tol_r
            # A solve that stops moving while the error is still large is a
            # different failure from running out of time, and reporting it as a
            # timeout hides which one happened. The velocities go to zero at a
            # stationary point of the weighted least squares -- typically the
            # posture term balancing the task -- and no amount of extra time
            # changes anything after that.
            if not done and float(np.abs(v).max()) < a.stall_v:
                self.n_stall += 1
                if self.n_stall > a.stall_cycles:
                    self.stop()
                    print(f'  停滯：命令已連續 {self.n_stall} 週期低於 '
                          f'{a.stall_v:.4f}，但位置誤差仍 {ep*1000:.1f} mm、'
                          f'姿態誤差 {math.degrees(er):.2f}°。'
                          f'解算收斂到非目標的駐點，不是時間不夠。', flush=True)
                    return False
            else:
                self.n_stall = 0
            if done and settle is None:
                settle = t
            if settle is not None and t - settle > a.settle_s:
                self.stop()
                break
            m = Float64MultiArray()
            m.data = [float(x) for x in v]
            self.pub.publish(m)
            self.v_prev = v.copy()
            self.log.append(dict(
                n_bar_rows=int(self.n_bar_rows), n_qp_fail=int(self.n_qp_fail),
                t=t, base=[float(x) for x in self.base],
                base_stamp=self.base_stamp,
                q=[float(x) for x in self.q_arm],
                tcp=[float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
                cmd_in=[float(x) for x in v],
                cmd_out=None if self.cmd_out is None
                        else [float(x) for x in self.cmd_out],
                pos_err=ep, rot_err=er, resid_in=self.r_in,
                resid_out=self.r_out, n_binding=self.n_bind,
                min_d_model=self.min_d, clearance_ub=self.ub))
        # let it come to rest
        t1 = time.monotonic()
        while time.monotonic() - t1 < 1.5:
            self.exec.spin_once(timeout_sec=0.02)
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='src/my_omnibot_description/urdf/'
                                      'omni_bot_wholebody.urdf.xacro')
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--target', nargs=3, type=float, default=[0.30, 0.0, 0.55],
                    help='gripper position in WORLD coordinates')
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--kp-p', type=float, default=1.0)
    ap.add_argument('--kp-r', type=float, default=1.0)
    ap.add_argument('--v-task', type=float, default=0.10,
                    help='task-space translation step cap, m/s')
    ap.add_argument('--w-task', type=float, default=0.5,
                    help='task-space rotation step cap, rad/s')
    ap.add_argument('--base-weight', type=float, default=3.0)
    ap.add_argument('--limit-margin', type=float, default=0.35,
                    help='joint-limit potential activates within this, rad')
    ap.add_argument('--k-limit', type=float, default=1.5)
    ap.add_argument('--solver', default='dls', choices=['dls', 'qp'],
                    help="'dls' = task solve then downstream projection (the "
                         "architecture as it runs). 'qp' = task, collision and "
                         "hardware limits in ONE solve, posture demoted to a "
                         "secondary objective by the same mu.")
    ap.add_argument('--mu-post', type=float, default=1.0,
                    help='weight of the posture term against the task')
    ap.add_argument('--kp-post', type=float, default=1.2,
                    help='null-space pull back toward the preferred posture')
    # The checked pre-grasp arm configuration from section 10.14, NOT the start
    # pose. Defaulting to the start pose looks harmless and is not: the posture
    # term then pulls toward "stay folded up" while the task pulls toward the
    # box, and the solve settles where the two gradients cancel. That happened
    # -- the run stopped dead at 131 mm and 22 degrees with every velocity at
    # zero, the barrier never involved, and only the timeout to show for it.
    ap.add_argument('--posture', nargs=6, type=float,
                    default=named_pose('pregrasp_reference'),
                    help='preferred arm configuration; defaults to '
                         'pregrasp_reference in config/arm_initial_pose.yaml')
    ap.add_argument('--damping', type=float, default=0.06)
    ap.add_argument('--tol-p', type=float, default=0.005)
    ap.add_argument('--tol-r', type=float, default=0.02)
    ap.add_argument('--settle-s', type=float, default=2.0)
    ap.add_argument('--timeout-s', type=float, default=60.0)
    ap.add_argument('--stall-v', type=float, default=2e-3)
    ap.add_argument('--stall-cycles', type=int, default=40)
    # Feed-staleness threshold. NOT a tuning knob, and NOT the safety stop: it
    # is the line at which a TEST gives up on its data. The safety stop is the
    # gate's 150 ms monotonic watchdog, which is a separate process, is
    # unchanged by anything here, and still zeroes base and arm together.
    #
    # Set from measurement. An INDEPENDENT monitor process -- subscribed to the
    # feeds and doing nothing else -- measured, over 110 s spanning a full run:
    #
    #   /joint_states  141.7 Hz, median 7.0 ms, p99 8.0 ms, max 12.8 ms
    #   /odom           29.3 Hz, median 34.0 ms, p99 36.7 ms, max 309.6 ms
    #
    # and in an earlier window a 332 ms stall on /joint_states with the joints
    # at rest. So this simulator stalls a feed for up to ~335 ms occasionally
    # while running two orders of magnitude tighter the rest of the time. At
    # 300 ms the guard sat inside that stall population and aborted runs on a
    # real but harmless hiccup -- twice, once on each feed. 0.5 s clears the
    # observed maximum with margin and is still only 10 control cycles.
    #
    # This number belongs to THIS simulator. On hardware it goes back down and
    # the measurement has to be redone.
    ap.add_argument('--max-data-age', type=float, default=0.50)
    ap.add_argument('--abort-d', type=float, default=0.02)
    ap.add_argument('--max-base-travel', type=float, default=1.2)
    ap.add_argument('--out', default='evaluation/results/wholebody_pregrasp.json')
    a = ap.parse_args()

    rclpy.init()
    nd = WholeBody(a)
    print('  等待 /joint_states、/odom 與距離資料…', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20 and (nd.q_arm is None or nd.base is None
                                          or nd.rows is None):
        nd.exec.spin_once(timeout_sec=0.1)
    if nd.q_arm is not None and nd.base is not None and not wait_fresh(
            nd, [lambda: time.monotonic() - nd.q_arm_t,
                 lambda: time.monotonic() - nd.base_t]):
        print('  JointState 或底盤位姿一直不新鮮，不下命令。', file=sys.stderr)
        nd.stop()
        rclpy.try_shutdown()
        return 1
    if nd.q_arm is None or nd.base is None or nd.rows is None:
        print(f'  資料不齊（arm={nd.q_arm is not None}, base={nd.base is not None},'
              f' rows={nd.rows is not None}）：不下命令。', file=sys.stderr)
        nd.stop()
        rclpy.try_shutdown()
        return 1

    # Tool +z into the box face (+x), tool +x up. Same convention as 5B.
    zc = np.array([1.0, 0.0, 0.0])
    xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = np.eye(4)
    T_des[:3, :3] = np.column_stack([xc, np.cross(zc, xc), zc])
    T_des[:3, 3] = np.array(a.target, dtype=float)

    ok = False
    try:
        ok = nd.run(T_des)
    except KeyboardInterrupt:
        print('\n  中斷')
    finally:
        nd.stop()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(args=vars(a), completed=ok,
                       target=[float(x) for x in a.target], log=nd.log),
                  open(a.out, 'w'), ensure_ascii=False)
        print(f'  {len(nd.log)} 週期寫入 {a.out}', flush=True)
        rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
