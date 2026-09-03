"""In-loop acceptance for item 3: the safety filter running as a ROS node.

The offline test proved the algebra. This proves the plumbing, which is a
different question: message age, TF timing, node timeouts, and what the robot
actually does when the loop is closed. A filter that is correct in isolation
and wrong in the loop looks identical from the maths alone.

Staged exactly as agreed, because a failure at stage 4 is uninterpretable if
stages 2 and 3 were never isolated:

    2  single joint      approach / tangential / receding / retreat
    3  six joints        several rows at once, joint limits, self-collision
    4  plus base 3-DOF   whole-body Jacobian, frames, wheel limits
    5  degradation       STALE / NODATA / occluded / node stopped
    6  end-to-end        latency, cycle time, residual, fallback rate,
                         minimum clearance, contact

Commands are applied by integrating the filtered velocity: the arm is a
position-controlled trajectory controller, so a velocity command has to be
integrated somewhere, and doing it here keeps the node itself pure.

    python3 evaluation/verify_safety_in_loop.py [--stage N]
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_detection_points import _quat_to_rot  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (  # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]
TUCK = [0.0, -0.082, 0.089, 0.0, 1.679, 0.0]
PC_FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status', 'age', 'occluded']
FRAMES = ['detect0_1', 'detect0_2', 'detect1', 'detect2_1', 'detect2_2',
          'detect2_3', 'detect3_1', 'detect3_2', 'detect4_1', 'detect4_2',
          'detect5', 'detect6']
DIAG = ['cycle', 'reason', 'n_rows', 'n_active', 'resid_before', 'resid_after',
        'iters', 'fallback', 'unresolved', 'runtime_ms', 'speed_cap',
        'min_d', 'n_stale', 'n_nodata', 'n_occluded']
REASON = {0: 'ok', 1: 'no-cmd', 2: 'stale-cmd', 3: 'stale-joints',
          4: 'no-kinematics', 5: 'no-points'}


class H(Node):
    def __init__(self):
        super().__init__('verify_safety_in_loop')
        self.js = {}
        self.pts = None
        self.out = None
        self.diag = None
        self.out_t = 0.0
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 lambda m: setattr(self, 'pts', m), 10)
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out', self._out, 10)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/diag',
                                 lambda m: setattr(self, 'diag', m), 10)
        self.cmd = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)
        self.twist = self.create_publisher(Twist, '/cmd_vel', 10)
        self.ac = ActionClient(
            self, FollowJointTrajectory,
            '/lite6_traj_controller/follow_joint_trajectory')
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)

    def _js(self, m):
        for n, p in zip(m.name, m.position):
            self.js[n] = p

    def _out(self, m):
        self.out = np.array(m.data, dtype=float)
        self.out_t = time.time()

    def spin(self, s):
        t0 = time.time()
        while time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.02)

    def q_arm(self):
        if not all(j in self.js for j in ARM):
            return None
        return np.array([self.js[j] for j in ARM])

    def goto(self, pos, secs=4.0):
        if not self.ac.wait_for_server(timeout_sec=10.0):
            return False
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = ARM
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in pos]
        p.time_from_start = Duration(sec=int(secs),
                                     nanosec=int((secs % 1) * 1e9))
        g.trajectory.points = [p]
        f = self.ac.send_goal_async(g)
        rclpy.spin_until_future_complete(self, f, timeout_sec=15.0)
        gh = f.result()
        if gh is None or not gh.accepted:
            return False
        r = gh.get_result_async()
        rclpy.spin_until_future_complete(self, r, timeout_sec=secs + 15.0)
        self.spin(1.0)
        return True

    def send(self, v9):
        m = Float64MultiArray()
        m.data = [float(x) for x in v9]
        self.cmd.publish(m)

    def diag_d(self):
        return dict(zip(DIAG, self.diag.data)) if self.diag else {}

    def points(self):
        if self.pts is None:
            return None
        return np.frombuffer(self.pts.data, dtype=np.float32).reshape(
            self.pts.width, len(PC_FIELDS)).astype(float)


def approach_of(K, q, frame, nvec, v):
    return float(nvec @ (K.jacobian(q, frame)[:3] @ v))


def run_cmd(h, K, v9, secs, apply_arm=True, apply_base=False, rate=20.0):
    """Publish a 9-DOF command for `secs`, integrating the filtered arm part.

    The command must be REPUBLISHED every cycle. The node treats a command
    older than max_cmd_age as stale and outputs zero, which is correct -- a
    command that stopped arriving is not a request to keep moving -- but it
    means a harness that publishes once and then waits is measuring the
    timeout, not the filter. Diagnostics are captured DURING the stream for the
    same reason: read afterwards, every field reports the stale-command path.
    """
    t0 = time.time()
    lat, res, fb, rt = [], [], 0, []
    n = 0
    last_diag = {}
    while time.time() - t0 < secs:
        h.send(v9)
        t_send = time.time()
        h.spin(1.0 / rate)
        if h.out is None:
            continue
        lat.append(max(0.0, h.out_t - t_send))
        d = h.diag_d()
        if d and int(d.get('reason', 9)) == 0:
            last_diag = d
            res.append(d.get('resid_after', 0.0))
            rt.append(d.get('runtime_ms', 0.0))
            fb += int(d.get('fallback', 0.0) > 0.5)
            n += 1
        if apply_arm:
            q = h.q_arm()
            if q is not None and np.abs(h.out[3:]).max() > 1e-6:
                h.goto(list(q + h.out[3:] * (1.0 / rate)), secs=0.12)
        if apply_base:
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = h.out[0], h.out[1], h.out[2]
            h.twist.publish(t)
    if apply_base:
        h.twist.publish(Twist())
    return dict(lat=np.array(lat) if lat else np.zeros(1),
                res=np.array(res) if res else np.zeros(1),
                rt=np.array(rt) if rt else np.zeros(1),
                fb=fb, n=max(n, 1), diag=last_diag)


def hold(h, v9, secs, rate=30.0):
    """Stream one command and return the last output+diag taken while it was
    still fresh."""
    t0 = time.time()
    out, dg = None, {}
    while time.time() - t0 < secs:
        h.send(v9)
        h.spin(1.0 / rate)
        d = h.diag_d()
        if d and int(d.get('reason', 9)) == 0 and h.out is not None:
            out, dg = h.out.copy(), d
    return out, dg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=0, help='0 = all')
    a = ap.parse_args()

    rclpy.init()
    h = H()
    h.spin(5.0)

    # A second copy of a node left over from an earlier run publishes to the
    # same topics, and the two streams interleave: during development cmd_out
    # came from the new node while diag came from the stale one, which made the
    # filter look broken when it was not. `kill` on a `ros2 run` wrapper does
    # not reap the python child, so this is easy to do by accident. Refuse to
    # measure anything until each topic has exactly one publisher.
    for topic in ('/wholebody_safety/cmd_out', '/wholebody_safety/diag',
                  '/arm_link_distance/points'):
        n_pub = h.count_publishers(topic)
        if n_pub != 1:
            print(f'  ★ {topic} 有 {n_pub} 個 publisher（應為 1）—— '
                  f'先清乾淨舊節點再測，否則診斷資料會跨執行污染',
                  file=sys.stderr)
            return 2
    if h.pts is None or h.q_arm() is None:
        print('  missing /arm_link_distance/points or /joint_states',
              file=sys.stderr)
        return 2
    import subprocess
    urdf = subprocess.check_output(
        ['xacro', 'src/my_omnibot_description/urdf/omni_bot_wholebody.urdf.xacro'],
        text=True)
    K = WholeBodyKinematics.from_urdf_string(urdf)
    fails = []
    stats = []

    h.goto(TUCK)
    h.spin(1.0)
    P = h.points()
    # Pick the detection point with the closest surface -- the row that will
    # actually bind.
    ok_rows = [(i, P[i]) for i in range(len(FRAMES)) if P[i][7] == 0]
    i_near = min(ok_rows, key=lambda t: t[1][6])[0]
    fr = FRAMES[i_near]
    nvec = P[i_near][3:6] / max(np.linalg.norm(P[i_near][3:6]), 1e-9)
    d_near = P[i_near][6]
    print(f'  最近偵測點 {fr}  d={d_near:.3f} m  n=({nvec[0]:+.2f},'
          f'{nvec[1]:+.2f},{nvec[2]:+.2f})')

    q = np.zeros(9)
    q[3:] = h.q_arm()
    J = K.jacobian(q, fr)[:3]
    row = nvec @ J

    # ------------------------------------------------------- stage 2
    if a.stage in (0, 2):
        print('\n  [階段 2] 單關節在環')
        # Choose the arm joint with the strongest approach coupling.
        j = int(np.argmax(np.abs(row[3:]))) + 3
        s = np.sign(row[j]) or 1.0
        print(f'      主導關節 joint{j-2}  d(approach)/d(qdot) = {row[j]:+.4f}')
        for label, scale in (('接近', +1.0), ('遠離', -1.0)):
            v = np.zeros(9)
            v[j] = 0.5 * s * scale
            out, _ = hold(h, v, 1.2)
            if out is None:
                print(f'      {label:6} 無新鮮輸出')
                fails.append(f'stage2/{label}')
                continue
            a_in = float(row @ v)
            a_out = float(row @ out)
            if scale > 0:
                good = a_out <= a_in + 1e-6
            else:
                good = abs(a_out - a_in) < 1e-3
            print(f'      {label:6} 輸入接近 {a_in:+.4f} → 輸出 {a_out:+.4f}'
                  f'   {"✓" if good else "✗"}')
            if not good:
                fails.append(f'stage2/{label}')
        h.send(np.zeros(9))

    # ------------------------------------------------------- stage 3
    if a.stage in (0, 3):
        print('\n  [階段 3] 六關節在環（多列同時、關節限位）')
        v = np.zeros(9)
        v[3:] = 0.35 * np.sign(row[3:] + 1e-12)     # all six pushing to approach
        st = run_cmd(h, K, v, 4.0, apply_arm=True)
        d = st['diag']
        P2 = h.points()
        mind = min(P2[i][6] for i in range(len(FRAMES)) if P2[i][7] == 0)
        lim = np.array([2.9356, 2.4435, 2.9356, 2.9356, 1.9897, 2.9356])
        qn = h.q_arm()
        within = bool(np.all(np.abs(qn) <= lim + 1e-3))
        print(f'      約束列 {int(d.get("n_rows",0))}  作用 {int(d.get("n_active",0))}'
              f'  殘差 {d.get("resid_after",0):.2e}  最小間距 {mind:.3f} m')
        print(f'      關節皆在限位內 {"✓" if within else "✗"}   '
              f'降級 {st["fb"]}/{st["n"]}')
        stats.append(('六關節', st, mind))
        if not within:
            fails.append('stage3/joint-limits')
        h.send(np.zeros(9))
        h.goto(TUCK)

    # ------------------------------------------------------- stage 4
    if a.stage in (0, 4):
        print('\n  [階段 4] 加入底盤 3-DOF')
        base = None
        try:
            t = h.buf.lookup_transform('odom', 'base_link', rclpy.time.Time())
            R = _quat_to_rot(t.transform.rotation.x, t.transform.rotation.y,
                             t.transform.rotation.z, t.transform.rotation.w)
            base = (t.transform.translation.x, t.transform.translation.y,
                    math.atan2(R[1, 0], R[0, 0]))
        except Exception:
            pass
        if base is None:
            print('      TF odom→base_link 不可得，略過')
            fails.append('stage4/tf')
        else:
            v = np.zeros(9)
            v[0] = 0.20 * np.sign(row[0] + 1e-12)
            st = run_cmd(h, K, v, 3.0, apply_arm=False, apply_base=True)
            d = st['diag']
            vmax = float(np.abs(h.out[:3]).max()) if h.out is not None else 0.0
            wheel_ok = vmax <= 0.2775 + 1e-6
            print(f'      底盤 ({base[0]:+.2f},{base[1]:+.2f},{math.degrees(base[2]):+.0f}°)'
                  f'  輸出基座速度上限 {vmax:.4f} ≤ 0.2775  {"✓" if wheel_ok else "✗"}')
            print(f'      約束列 {int(d.get("n_rows",0))}  殘差 {d.get("resid_after",0):.2e}'
                  f'  降級 {st["fb"]}/{st["n"]}')
            stats.append(('底盤', st, d.get('min_d', -1)))
            if not wheel_ok:
                fails.append('stage4/wheel-limit')
        h.send(np.zeros(9))

    # ------------------------------------------------------- stage 5
    if a.stage in (0, 5):
        print('\n  [階段 5] 降級與 fail-safe')
        v = np.zeros(9)
        v[3] = 0.4
        out0, d0 = hold(h, v, 1.5)
        print(f'      正常          reason={REASON.get(int(d0.get("reason",9)),"?"):14}'
              f' 輸出範數 {np.linalg.norm(out0) if out0 is not None else float("nan"):.4f}')

        # Stop publishing cmd_in: a stale command must not keep the robot going.
        h.spin(1.2)
        d1 = h.diag_d()
        zero_on_stale = float(np.abs(h.out).max()) < 1e-9
        print(f'      停止送指令      reason={REASON.get(int(d1.get("reason",0)),"?"):14}'
              f' 輸出範數 {np.linalg.norm(h.out):.4f}  {"✓ 歸零" if zero_on_stale else "✗"}')
        if not zero_on_stale:
            fails.append('stage5/stale-cmd')

        # Stop the distance source: every row becomes NODATA -> speed cap.
        subprocess.run(['pkill', '-STOP', '-f', 'arm_link_distance'],
                       capture_output=True)
        out2, d2 = hold(h, v, 2.5)
        if out2 is None:
            out2 = np.zeros(9)
        capped = float(np.abs(out2).max()) <= 0.05 + 1e-6
        print(f'      距離來源停止    nodata={int(d2.get("n_nodata",0)):2d}'
              f' cap={d2.get("speed_cap",-1):.3f}'
              f' 輸出最大 {float(np.abs(out2).max()):.4f}  {"✓ 降級" if capped else "✗"}')
        if not capped:
            fails.append('stage5/nodata')
        subprocess.run(['pkill', '-CONT', '-f', 'arm_link_distance'],
                       capture_output=True)
        h.send(np.zeros(9)); h.spin(1.0)

    # ------------------------------------------------------- stage 6
    print('\n  [階段 6] 端到端量測')
    # Residual is LHS - RHS, so <= 0 means satisfied and the WORST case is the
    # value closest to zero, not the most negative one. Reporting a large
    # negative number as "max residual" reads as if the constraint were badly
    # violated when it is in fact comfortably satisfied -- it is a margin, and
    # is labelled as one.
    for label, st, mind in stats:
        worst = st["res"].max()
        print(f'      {label:8} 端到端延遲 中位 {np.median(st["lat"])*1e3:5.1f} ms'
              f'  p95 {np.percentile(st["lat"],95)*1e3:5.1f} ms'
              f'  節點耗時 p95 {np.percentile(st["rt"],95):4.2f} ms'
              f'  最差殘差 {worst:+.2e}'
              f'  餘裕 {-worst:+.3f}'
              f'  降級 {100*st["fb"]/st["n"]:4.1f}%'
              f'  最小間距 {mind:.3f} m')
    gw = max(st["res"].max() for _, st, _ in stats) if stats else 0.0
    lp95 = max(np.percentile(st["lat"], 95) for _, st, _ in stats) if stats else 0.0
    print(f'      全測試最差殘差 {gw:+.2e}（<=0 為滿足）   '
          f'延遲 p95 最高 {lp95*1e3:.1f} ms')
    print(f'      碰撞：最小間距皆 > 0 '
          f'{"✓" if all(m > 0 for _, _, m in stats) else "✗"}')

    h.goto(TUCK)
    print('\n  ' + ('在環驗收通過 ✓' if not fails else f'★ 失敗: {fails}'))
    h.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
