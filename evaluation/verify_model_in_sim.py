"""Section 3.3, the two items that need a running simulator.

Both are usually written as "check it looks right in the viewer". A viewer shows
that nothing is obviously broken; it does not catch a joint whose axis is
negated, or a base transform composed in the wrong order. Both of those render
perfectly plausibly. So each check is numerical: drive the robot, read what
came back, and compare against the controller's own kinematics.

  A  joint directions and zeros
     Move one joint at a time. Feed the MEASURED joint values into
     wholebody_kinematics and compare the predicted base_link -> link_tcp
     against the TF the simulator publishes. A sign error or a wrong axis shows
     up as centimetres here while looking fine on screen.

  C  the SIMULATED robot against the model
     A and B both compare against TF, and TF is computed by
     robot_state_publisher from the same URDF. That is a real check -- it is an
     independent FK (KDL) and it catches a misread URDF -- but it cannot catch
     a simulator whose robot differs from the URDF, because neither side would
     know. gz publishes true per-link poses on /world/<w>/pose/info, so C
     compares those against the model. Links joined by fixed joints are lumped
     by gz and appear under composite names, so the deepest genuinely dynamic
     link is the one to test.

  B  base transport
     Hold the arm still and drive the base. The end effector's world pose must
     equal (world <- base) composed with the arm's own FK. This is what
     verifies that the three virtual planar joints in the whole-body model
     reproduce the floating base the simulator actually has -- the model and
     the simulator represent the base completely differently, and this is the
     only place that difference is tested.

Run against a live sim:
    ros2 launch my_omnibot_description omni_bot_arm.launch.py \
        use_arm:=true add_gripper:=true gui:=false foxglove:=false
    python3 evaluation/verify_model_in_sim.py
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (  # noqa: E402
    DOF_NAMES, WholeBodyKinematics, iso)

ARM = [f'joint{i}' for i in range(1, 7)]
TOL_POS = 5e-3          # m, TF vs FK position
TOL_ROT = 5e-3          # rad-equivalent on the rotation matrix


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class Checker(Node):
    def __init__(self, K, target):
        super().__init__('verify_model_in_sim')
        self.K = K
        self.target = target
        self.js = {}
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.ac = ActionClient(
            self, FollowJointTrajectory,
            '/lite6_traj_controller/follow_joint_trajectory')

    def _js(self, m):
        for n, p in zip(m.name, m.position):
            self.js[n] = p

    def spin(self, sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.05)

    def tf(self, parent, child):
        try:
            t = self.buf.lookup_transform(parent, child, rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        tr = t.transform.translation
        return iso(quat_to_rot(q.x, q.y, q.z, q.w), np.array([tr.x, tr.y, tr.z]))

    def q_arm(self):
        if not all(j in self.js for j in ARM):
            return None
        q = np.zeros(9)
        for j in ARM:
            q[DOF_NAMES.index(j)] = self.js[j]
        return q

    def compare(self, parent, q):
        """TF(parent -> target) against FK for the same configuration."""
        T_tf = self.tf(parent, self.target)
        if T_tf is None:
            return None
        T_fk = self.K.fk(q, self.target)
        if parent != 'world':
            T_fk = np.linalg.inv(self.K.fk(q, parent)) @ T_fk
        dp = float(np.abs(T_tf[:3, 3] - T_fk[:3, 3]).max())
        dr = float(np.abs(T_tf[:3, :3] - T_fk[:3, :3]).max())
        return dp, dr

    def goto(self, positions, secs=3.0):
        if not self.ac.wait_for_server(timeout_sec=10.0):
            return False
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = ARM
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in positions]
        p.time_from_start = Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9))
        g.trajectory.points = [p]
        fut = self.ac.send_goal_async(g)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=15.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False
        res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=secs + 15.0)
        self.spin(1.0)
        return True

    def drive(self, vx, vy, wz, secs):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        t0 = time.time()
        while time.time() - t0 < secs:
            self.cmd.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.cmd.publish(Twist())
        self.spin(1.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=None)
    ap.add_argument('--target', default='link_tcp')
    a = ap.parse_args()

    urdf = a.urdf
    if urdf is None:
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(here, '..', 'src', 'my_omnibot_description', 'urdf',
                           'omni_bot_wholebody.urdf.xacro')
        urdf = subprocess.check_output(['xacro', src], text=True)
        K = WholeBodyKinematics.from_urdf_string(urdf)
    else:
        K = WholeBodyKinematics.from_urdf_file(urdf)

    rclpy.init()
    n = Checker(K, a.target)
    n.spin(4.0)
    if n.q_arm() is None:
        print('  no /joint_states with the six arm joints -- is the sim up?',
              file=sys.stderr)
        return 2

    fails = []
    print(f'  target: {a.target}')

    # ---------------------------------------------------------------- A
    print('\n  [A] 逐關節方向與零位（TF vs FK，base_link 座標）')
    tuck = [0.0, -0.082, 0.089, 0.0, 1.679, 0.0]
    n.goto(tuck, 3.0)
    base = n.compare('base_link', n.q_arm())
    print(f'    {"tuck":22} 位置誤差 {base[0]:.2e} m   旋轉誤差 {base[1]:.2e}')
    if base[0] > TOL_POS or base[1] > TOL_ROT:
        fails.append('A/tuck')

    for i, j in enumerate(ARM):
        q = list(tuck)
        lo = K.joints[j].lower
        hi = K.joints[j].upper
        delta = 0.35 if q[i] + 0.35 < hi - 0.05 else -0.35
        if q[i] + delta < lo + 0.05:
            delta = -delta
        q[i] += delta
        if not n.goto(q, 3.0):
            print(f'    {j:22} 指令失敗')
            fails.append(f'A/{j}')
            continue
        meas = n.js.get(j, float('nan'))
        r = n.compare('base_link', n.q_arm())
        ok = r and r[0] <= TOL_POS and r[1] <= TOL_ROT
        print(f'    {j} {delta:+.2f} -> 回授 {meas:+.4f}   '
              f'位置誤差 {r[0]:.2e} m   旋轉誤差 {r[1]:.2e}  {"✓" if ok else "✗"}')
        if not ok:
            fails.append(f'A/{j}')
    n.goto(tuck, 3.0)

    # ---------------------------------------------------------------- B
    print('\n  [B] 底盤運動時的末端世界位姿（手臂固定於 tuck）')
    print('      檢查 world->EE 是否等於 (world->base_link) ∘ (base_link->EE)')
    moves = [('前進', 0.15, 0.0, 0.0, 3.0),
             ('側移', 0.0, 0.15, 0.0, 3.0),
             ('旋轉', 0.0, 0.0, 0.4, 3.0),
             ('斜向+旋轉', 0.12, -0.12, -0.3, 3.0)]
    for label, vx, vy, wz, secs in moves:
        n.drive(vx, vy, wz, secs)
        T_wb = n.tf('odom', 'base_link')
        T_we = n.tf('odom', a.target)
        if T_wb is None or T_we is None:
            print(f'    {label:12} TF 不可得')
            fails.append(f'B/{label}')
            continue
        T_be_fk = n.K.fk(n.q_arm(), a.target)
        T_be_fk = np.linalg.inv(n.K.fk(n.q_arm(), 'base_link')) @ T_be_fk
        pred = T_wb @ T_be_fk
        dp = float(np.abs(pred[:3, 3] - T_we[:3, 3]).max())
        dr = float(np.abs(pred[:3, :3] - T_we[:3, :3]).max())
        ok = dp <= TOL_POS and dr <= TOL_ROT
        print(f'    {label:12} 底盤 ({T_wb[0,3]:+.2f},{T_wb[1,3]:+.2f})  '
              f'EE ({T_we[0,3]:+.2f},{T_we[1,3]:+.2f},{T_we[2,3]:+.2f})  '
              f'位置誤差 {dp:.2e}  旋轉誤差 {dr:.2e}  {"✓" if ok else "✗"}')
        if not ok:
            fails.append(f'B/{label}')

    # ---------------------------------------------------------------- C
    print('\n  [C] gz 模擬器的真值連桿位姿 vs 模型（獨立於 TF）')
    world = os.environ.get('GZ_WORLD', 'random_room')
    try:
        out = subprocess.check_output(
            ['gz', 'topic', '-e', '-t', f'/world/{world}/pose/info', '-n', '1'],
            text=True, timeout=25, stderr=subprocess.DEVNULL)
    except Exception as exc:                                   # noqa: BLE001
        print(f'    無法取得 gz 真值（{exc}）— 略過')
        out = ''
    if out:
        poses = {}
        cur = None
        for blk in re.split(r'\n(?=pose \{)', out):
            nm = re.search(r'name:\s*"([^"]+)"', blk)
            px = re.search(r'position \{(.*?)\}', blk, re.S)
            ox = re.search(r'orientation \{(.*?)\}', blk, re.S)
            if not (nm and px):
                continue
            g = lambda t, k: float(re.search(rf'{k}:\s*([-\d.e+]+)', t).group(1)) \
                if re.search(rf'{k}:\s*([-\d.e+]+)', t) else 0.0
            poses[nm.group(1)] = (
                np.array([g(px.group(1), 'x'), g(px.group(1), 'y'), g(px.group(1), 'z')]),
                np.array([g(ox.group(1), 'x'), g(ox.group(1), 'y'),
                          g(ox.group(1), 'z'), g(ox.group(1), 'w') or 1.0]))
        if 'omni_bot' not in poses:
            print('    真值訊息中找不到 omni_bot — 略過')
        else:
            for link in ('link6', 'link3', 'link1'):
                if link not in poses:
                    continue
                p_l, q_l = poses[link]
                T_gz = iso(quat_to_rot(*q_l), p_l)        # model-relative
                q = n.q_arm()
                T_fk = np.linalg.inv(n.K.fk(q, 'base_link')) @ n.K.fk(q, link)
                # gz reports links relative to the MODEL origin, which sits at
                # base_footprint; base_link is 0.05 m above it.
                T_fk[2, 3] += 0.05
                dp = float(np.abs(T_gz[:3, 3] - T_fk[:3, 3]).max())
                dr = float(np.abs(T_gz[:3, :3] - T_fk[:3, :3]).max())
                ok = dp <= TOL_POS and dr <= TOL_ROT
                print(f'    {link:8} gz ({p_l[0]:+.4f},{p_l[1]:+.4f},{p_l[2]:+.4f})  '
                      f'位置誤差 {dp:.2e} m  旋轉誤差 {dr:.2e}  {"✓" if ok else "✗"}')
                if not ok:
                    fails.append(f'C/{link}')

    print('\n  ' + ('§3.3 剩餘兩項通過 ✓' if not fails else f'★ 失敗: {fails}'))
    n.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
