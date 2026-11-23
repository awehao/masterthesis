"""Play the 5B pre-grasp runs into the running Gazebo simulation.

The offline acceptance (verify_pregrasp.py) produces, for each starting pose, a
joint trajectory that has already been through the link safety filter. This
sends those trajectories to lite6_traj_controller so the same runs can be
watched on the real model, with the real meshes, in the simulator.

What this is and is not
-----------------------
This is a REPLAY. The trajectory was computed offline against the scene model,
and Gazebo is being asked to follow it. The safety filter is not in the loop
here: nothing is being sensed, and nothing would stop the arm if the simulated
world disagreed with the scene model. In-loop verification -- the filter reading
live sensor data and issuing commands inside the simulator -- is a separate,
still outstanding step, and the difference matters when quoting results.

The base is not driven. 5B fixes it, and it has already been placed at the
pre-grasp standoff with gz's set_pose service; the arm is the only thing moving.

    python3 evaluation/play_5b_gazebo.py <expanded.urdf> [--start N] [--speed S]
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')

import rclpy  # noqa: E402
from builtin_interfaces.msg import Duration  # noqa: E402
from rclpy.node import Node  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_detection_points import _iso  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import (  # noqa: E402
    ARM_JOINTS, TCP, min_jerk, plan_pregrasp)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, filter_velocity)
import verify_pregrasp as VP  # noqa: E402
import verify_self_collision as VSC  # noqa: E402

TOPIC = '/lite6_traj_controller/joint_trajectory'


def build_runs(urdf, world, n_starts):
    """Re-run the offline acceptance and keep each filtered joint trajectory."""
    xml = open(urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(world)
    clouds = VSC.link_clouds(xml, max_pts=250)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    cfg = SafetyConfig()
    n = len(K.dof_names)

    adj, rigid = set(), {}

    def find(x):
        while rigid.get(x, x) != x:
            x = rigid[x]
        return x
    for j in K.joints.values():
        adj.add(frozenset((j.parent, j.child)))
        if j.jtype not in ('revolute', 'prismatic', 'continuous'):
            x, y = find(j.parent), find(j.child)
            if x != y:
                rigid[x] = y
    names = [x for x in clouds if x in K.parent_of or x == 'base_link']
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if frozenset((x, y)) not in adj and find(x) != find(y)
             and frozenset((x, y)) != frozenset(('uflite_finger1', 'uflite_finger2'))]
    from scipy.spatial import cKDTree

    def self_clear(q):
        w = {}
        for nm in names:
            T = K.fk(q, nm)
            w[nm] = (T[:3, :3] @ clouds[nm].T).T + T[:3, 3]
        return min(float(cKDTree(w[x]).query(w[y], k=1)[0].min()) for x, y in pairs)

    def env_clear(q):
        worst = math.inf
        for nm in ('link2', 'link3', 'link4', 'link5', 'link6', 'uflite_gripper_link'):
            if nm not in clouds:
                continue
            T = K.fk(q, nm)
            P = (T[:3, :3] @ clouds[nm].T).T + T[:3, 3]
            for p in P[::12]:
                worst = min(worst, VP.nearest(obs, p)[0])
        return worst

    box = min((o for o in obs if o.kind == 'box'),
              key=lambda o: float(o.T_world_link[2, 3]))
    c = box.T_world_link[:3, 3]
    face = (np.array([1.0, 0.0, 0.0]) if box.size[0] <= box.size[1]
            else np.array([0.0, 1.0, 0.0]))
    half = 0.5 * float(box.size @ np.abs(face))
    p_base = c + face * (half + 0.55)
    q_base = np.array([p_base[0], p_base[1], math.atan2(-face[1], -face[0])])
    p_des = c + face * (half + 0.12)
    p_des[2] = float(np.clip(0.55, c[2] - 0.25, c[2] + 0.25))
    zc = -face
    xc = np.array([0.0, 0.0, 1.0]); xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = _iso(np.column_stack([xc, np.cross(zc, xc), zc]), p_des)

    rng = np.random.default_rng(0)
    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    starts = [np.array(VP.TUCK)] + [rng.uniform(lo6 * 0.5, hi6 * 0.5)
                                    for _ in range(n_starts - 1)]

    runs = []
    for si, q0a in enumerate(starts):
        q = np.zeros(n); q[:3] = q_base; q[idx] = q0a
        plan = plan_pregrasp(K, q, T_des, self_clear, env_clear)
        if not plan.ok:
            print(f'  起始 {si}: 規劃失敗，略過（{plan.reason[:44]}）')
            continue
        T, qg = plan.duration, plan.q_goal[idx]
        v_prev, v_prev2, qc = np.zeros(n), None, q0a.copy()
        traj = [qc.copy()]
        for k in range(int(math.ceil((T + 1.5) / cfg.dt))):
            tt = k * cfg.dt
            qd = min_jerk(q0a, qg, T, tt)[1] if tt <= T else np.zeros(6)
            qref = min_jerk(q0a, qg, T, min(tt, T))[0]
            v_in = np.zeros(n); v_in[idx] = qd + 2.0 * (qref - qc)
            qf = np.zeros(n); qf[:3] = q_base; qf[idx] = qc
            pts = []
            for fr in VP.FRAMES:
                p = K.fk(qf, fr)[:3, 3]
                d, vv = VP.nearest(obs, p)
                pts.append(DetectionPoint(fr, p, vv / max(d, 1e-9), d, STATUS_OK))
            a_prev = ((v_prev - v_prev2) / cfg.dt) if v_prev2 is not None else None
            r = filter_velocity(K, qf, v_in, pts, cfg, v_prev=v_prev,
                                a_prev=a_prev, dt=cfg.dt)
            qc = qc + r.v[idx] * cfg.dt
            traj.append(qc.copy())
            v_prev2, v_prev = v_prev, r.v.copy()
        qf = np.zeros(n); qf[:3] = q_base; qf[idx] = qc
        Tc = K.fk(qf, TCP)
        err = float(np.linalg.norm(T_des[:3, 3] - Tc[:3, 3])) * 1e3
        runs.append(dict(start=si, q=np.array(traj), dt=cfg.dt, err_mm=err,
                         reached=err < 10.0))
        print(f'  起始 {si}: {len(traj)} 個點，末端誤差 {err:6.1f} mm  '
              f'{"到達" if err < 10.0 else "被屏障擋住"}')
    return runs


def send(node, pub, q_list, dt, t0=1.0):
    msg = JointTrajectory()
    msg.joint_names = list(ARM_JOINTS)
    for i, q in enumerate(q_list):
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        tsec = t0 + i * dt
        p.time_from_start = Duration(sec=int(tsec),
                                     nanosec=int(round((tsec % 1.0) * 1e9)))
        msg.points.append(p)
    pub.publish(msg)
    return t0 + len(q_list) * dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--start', type=int, default=None,
                    help='play only this starting pose')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--loop', action='store_true')
    a = ap.parse_args()

    print('  重算離線 5B 軌跡…')
    runs = build_runs(a.urdf, a.world, a.n)
    if a.start is not None:
        runs = [r for r in runs if r['start'] == a.start]
    if not runs:
        print('  沒有可播放的軌跡')
        return 1

    rclpy.init()
    node = Node('play_5b')
    pub = node.create_publisher(JointTrajectory, TOPIC, 10)
    print(f'\n  等待 {TOPIC} 的訂閱者…')
    for _ in range(100):
        if pub.get_subscription_count() > 0:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print('  ★ 沒有訂閱者 —— lite6_traj_controller 沒在跑？')
        node.destroy_node(); rclpy.shutdown(); return 1
    print(f'  訂閱者 {pub.get_subscription_count()} 個\n')

    dt = runs[0]['dt'] / max(a.speed, 1e-3)
    while True:
        for r in runs:
            print(f'  ▶ 起始 {r["start"]}  '
                  f'{"到達" if r["reached"] else "被屏障擋住"}'
                  f'（末端誤差 {r["err_mm"]:.1f} mm）')
            # move to the starting pose first, slowly, so the jump from
            # wherever the arm happens to be is not itself a violent command
            end = send(node, pub, [r['q'][0]], dt, t0=2.5)
            t_end = time.time() + 2.5 + 0.4
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
            end = send(node, pub, r['q'], dt, t0=0.4)
            t_end = time.time() + end + 0.8
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
        if not a.loop:
            break
    print('\n  播放結束')
    node.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
