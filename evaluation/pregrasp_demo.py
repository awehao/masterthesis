"""Pre-grasp delivery: plan it offline, then actually drive it in Gazebo.

Stage 5B has so far produced numbers. This produces the motion those numbers
were about: from a fixed start pose, the gripper moves to a pose in front of the
box face, stops, and holds, with the link safety layer in the loop the whole
time. The point is something that can be watched, so it records video, and the
arrival is reported as measured error against the commanded pose rather than as
"the plan finished".

Two stages, and the second does not begin unless the first passes:

  offline   IK for the pre-grasp pose, a minimum-jerk path from the start, and
            then the WHOLE path checked -- joint position/velocity/acceleration
            limits, self-clearance, and clearance to the box at 25 waypoints. A
            start and a goal that are both clear say nothing about the arc
            between them, and the arc is what the arm traverses.

  online    the path is TRACKED, not replayed: the command is a feedforward
            joint velocity plus a proportional correction on the measured
            position, published to /wholebody_safety/cmd_in. Everything goes
            through the barrier, the adapter and the watchdog gate, so the
            safety layer can and will alter it. If the barrier holds the arm
            short of the target, that shows up as arrival error, which is the
            honest outcome -- not something to be planned around.

Self-clearance here is the SAMPLED figure (point clouds, ~250 points per link).
It is a screen, not a certificate: exact triangle-to-triangle distance costs
minutes per pose (`evaluation/mesh_distance.py`) and is run afterwards on the
single tightest waypoint, which is reported separately.

    python3 evaluation/pregrasp_demo.py --stand 0.24
    python3 evaluation/pregrasp_demo.py --stand 0.24 --check-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, _HERE)

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE            # noqa: E402
from ammr_wholebody_mpc.arm_detection_points import _iso        # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import (                   # noqa: E402
    ARM_JOINTS, min_jerk, plan_pregrasp, rot_error)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import verify_pregrasp as VP                                    # noqa: E402
import verify_self_collision as VSC                             # noqa: E402

# The simulator spawns the model root 0.05 m above the world origin, so the
# kinematic model's frame and the world frame differ by exactly this. Every
# target given in world coordinates is shifted once, here, rather than at each
# use -- a translation leaves Jacobians untouched but not positions.
MODEL_Z = 0.05


def _expand(path):
    return (subprocess.check_output(['xacro', path], text=True)
            if path.endswith('.xacro') else open(path).read())


def build_scene(xml, world):
    """Kinematics, obstacles, and the two clearance functions the planner needs."""
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(world)
    clouds = VSC.link_clouds(xml, max_pts=250)

    adj, rigid = set(), {}

    def find(x):
        while rigid.get(x, x) != x:
            x = rigid[x]
        return x

    for j in K.joints.values():
        adj.add(frozenset((j.parent, j.child)))
        if j.jtype not in ('revolute', 'prismatic', 'continuous'):
            a, b = find(j.parent), find(j.child)
            if a != b:
                rigid[a] = b
    names = [x for x in clouds if x in K.parent_of or x == 'base_link']
    # The exclusion list is the one the self-collision acceptance already uses.
    # It is NOT edited to drop whichever pair turns out to be closest: removing
    # a pair because it became the answer is choosing the result.
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if frozenset((x, y)) not in adj and find(x) != find(y)
             and frozenset((x, y)) != frozenset(('uflite_finger1',
                                                 'uflite_finger2'))]
    from scipy.spatial import cKDTree

    def world_pts(q, nm):
        T = K.fk(q, nm)
        return (T[:3, :3] @ clouds[nm].T).T + T[:3, 3]

    def self_clear(q):
        w = {nm: world_pts(q, nm) for nm in names}
        return min(float(cKDTree(w[x]).query(w[y], k=1)[0].min())
                   for x, y in pairs)

    def env_clear(q):
        worst = math.inf
        which = None
        for nm in names:
            P = world_pts(q, nm) + np.array([0.0, 0.0, MODEL_Z])
            for p in P[::6]:
                d = VP.nearest(obs, p)[0]
                if d < worst:
                    worst, which = d, nm
        return worst if which is None else worst

    return K, obs, self_clear, env_clear, pairs


def target_pose(obs, stand):
    """Pre-grasp pose in front of the box's near face, in MODEL coordinates.

    The face chosen is the one the arm is on: the box is approached from -x,
    where the base sits, so the pre-grasp point is at (face_x - stand) and the
    tool +z axis points at +x, into the face.
    """
    box = min((o for o in obs if o.kind == 'box'),
              key=lambda o: float(o.T_world_link[2, 3]))
    c = box.T_world_link[:3, 3]
    face_x = float(c[0]) - 0.5 * float(box.size[0])
    p_world = np.array([face_x - stand, float(c[1]), 0.55])
    zc = np.array([1.0, 0.0, 0.0])              # tool +z into the face
    xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    R = np.column_stack([xc, np.cross(zc, xc), zc])
    p_model = p_world - np.array([0.0, 0.0, MODEL_Z])
    return box, face_x, p_world, _iso(R, p_model)


def offline(a):
    xml = _expand(a.urdf)
    K, obs, self_clear, env_clear, pairs = build_scene(xml, a.world)
    n = len(K.dof_names)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    box, face_x, p_world, T_des = target_pose(obs, a.stand)

    # base_x / base_y / base_theta are REAL joints in the whole-body model, so
    # parking the base is a change to q, not a change to the model. 5B fixes the
    # base but never said it sits at the origin: with the base at the origin the
    # pre-grasp point ends up 0.20 m from the shoulder, inside the region where
    # position-plus-orientation IK stops converging, and every plan fails for a
    # reason that has nothing to do with the safety layer.
    q0 = np.zeros(n)
    q0[:3] = [a.base_x, a.base_y, a.base_yaw]
    q0[idx] = a.start
    print(f'  箱體 {box.name} 中心 ({box.T_world_link[0,3]:.2f}, '
          f'{box.T_world_link[1,3]:.2f}, {box.T_world_link[2,3]:.2f}) '
          f'尺寸 {np.round(box.size,2).tolist()}；近面 x = {face_x:.3f}')
    print(f'  起始關節角 {np.round(a.start,4).tolist()}')
    print(f'  預抓取目標（world）({p_world[0]:.3f}, {p_world[1]:.3f}, '
          f'{p_world[2]:.3f})，離面 {a.stand:.3f} m，工具 +z 指向箱面')
    sh = K.fk(q0, 'link1')[:3, 3] + np.array([0, 0, MODEL_Z])
    print(f'  肩部到目標 {np.linalg.norm(p_world - sh):.3f} m（臂展約 0.44 m）')

    plan = plan_pregrasp(K, q0, T_des, self_clear, env_clear,
                         min_margin=a.min_margin, n_check=a.n_check,
                         n_seeds=a.n_seeds)
    print(f'\n  IK {"成功" if plan.ik.ok else "失敗"}；'
          f'路徑檢查 {"通過" if plan.ok else "未通過：" + plan.reason}')
    for k, v in sorted(plan.checks.items()):
        print(f'    {k:22} {v}')
    if not plan.ok:
        return None, None, None, None

    T_goal = K.fk(plan.q_goal, a.tcp)
    pe = float(np.linalg.norm(T_goal[:3, 3] - T_des[:3, 3]))
    re = float(np.linalg.norm(rot_error(T_goal[:3, :3], T_des[:3, :3])))
    print(f'  目標關節角 {np.round(plan.q_goal[idx],4).tolist()}')
    print(f'  IK 殘差 位置 {pe*1000:.2f} mm 姿態 {math.degrees(re):.2f}°；'
          f'時長 {plan.duration:.2f} s')

    # Whole-path limits and clearance, reported rather than only thresholded.
    ts = np.linspace(0.0, plan.duration, a.n_report)
    sc, ec, vmax, amax = [], [], 0.0, 0.0
    for t in ts:
        q, qd, qdd = min_jerk(q0, plan.q_goal, plan.duration, float(t))
        sc.append(self_clear(q))
        ec.append(env_clear(q))
        vmax = max(vmax, float(np.abs(qd[idx]).max()))
        amax = max(amax, float(np.abs(qdd[idx]).max()))
    sc, ec = np.array(sc), np.array(ec)
    lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
    qs = np.array([min_jerk(q0, plan.q_goal, plan.duration, float(t))[0][idx]
                   for t in ts])
    pos_ok = bool((qs >= lo).all() and (qs <= hi).all())
    print(f'\n  全路徑（{a.n_report} 個取樣點）')
    print(f'    關節位置在安全限位內      {pos_ok}')
    print(f'    速度峰值 {vmax:.3f} rad/s（限 {LITE6_SAFE.max_velocity.min():.2f}）'
          f'   加速度峰值 {amax:.3f} rad/s² （限 {LITE6_SAFE.max_acceleration.min():.2f}）')
    print(f'    自間距（取樣，非證明）最小 {sc.min()*1000:.2f} mm  於 t={ts[sc.argmin()]:.2f}s')
    print(f'    對箱體間距（取樣）    最小 {ec.min()*1000:.2f} mm  於 t={ts[ec.argmin()]:.2f}s')
    worst_q, _, _ = min_jerk(q0, plan.q_goal, plan.duration, float(ts[sc.argmin()]))
    return plan, T_des, p_world, dict(
        pos_ok=pos_ok, vmax=vmax, amax=amax,
        self_min=float(sc.min()), self_min_t=float(ts[sc.argmin()]),
        env_min=float(ec.min()), env_min_t=float(ts[ec.argmin()]),
        worst_q=[float(x) for x in worst_q[idx]],
        ik_pos_err=pe, ik_rot_err=re, duration=float(plan.duration))


def record_start(out, topic):
    """Record the SIMULATOR's camera, never the screen.

    The first two attempts used ffmpeg's x11grab on the Gazebo window's screen
    region. x11grab captures whatever is composited there, so when another
    window sat in front -- which it did, twice -- that window is what was
    written to the file, personal data included. There is no way to make a
    screen region safe on a desktop someone is using. A camera sensor in the
    world can only see the simulated scene.
    """
    if not topic:
        return None
    print(f'  錄影開始（模擬器內相機 {topic}）→ {out}')
    return subprocess.Popen(
        [sys.executable, os.path.join(_HERE, 'record_gz_camera.py'),
         '--topic', topic, '--out', out],
        cwd=_ROOT, start_new_session=True)


def record_stop(p):
    if p is None:
        return
    import signal
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
        p.wait(timeout=30)
    except Exception:                                           # noqa: BLE001
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:                                       # noqa: BLE001
            pass


def online(a, plan, T_des, p_world, checks):
    import rclpy
    from std_msgs.msg import Float64MultiArray
    import barrier_probe as BP

    rclpy.init()
    args = argparse.Namespace(
        urdf=a.urdf, report_frame=a.report_frame, tcp=a.tcp, speed=0.0,
        phase_s=0.0, ramp_s=1.0, rate=a.rate, damping=0.05,
        joint_vmax=a.joint_vmax, abort_d=a.abort_d, phases=[],
        out='', home_kp=a.home_kp, home=None, home_only=False)
    p = BP.Probe(args)
    print('\n  等待 /joint_states 與距離資料…', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20 and (p.q is None or p.rows is None):
        rclpy.spin_once(p, timeout_sec=0.1)
    if p.q is None or p.rows is None:
        print('  堆疊沒起來，不下命令。', file=sys.stderr)
        rclpy.try_shutdown()
        return None

    idx = p.idx
    q0 = np.zeros(p.n)
    q0[idx] = a.start
    if not p.home(a.start, tol=a.home_tol):
        rclpy.try_shutdown()
        return None
    time.sleep(1.0)

    rec = record_start(a.video, a.camera_topic)
    log = []
    T = plan.duration * a.time_scale
    print(f'\n── 追蹤預抓取路徑 ── 時長 {T:.2f} s（時間縮放 {a.time_scale:.2f}）',
          flush=True)
    t_start = time.monotonic()
    try:
        while True:
            deadline = time.monotonic() + 1.0 / a.rate
            while time.monotonic() < deadline:
                rclpy.spin_once(p, timeout_sec=0.002)
            why = p._guard()
            if why:
                p._stop()
                print(f'  中止：{why}', flush=True)
                break
            p._measure_tcp()
            t = time.monotonic() - t_start
            if t > T + a.settle_s:
                break
            tc = min(t, T)
            # T already carries the stretch, so min_jerk's own derivative is
            # the stretched one. Dividing it again would halve the feedforward
            # and leave the proportional term to make up the difference.
            q_ref, qd_ref, _ = min_jerk(q0, plan.q_goal, T, tc)
            # Feedforward plus proportional correction on the MEASURED position.
            # Pure feedforward integrates its own error, and the barrier is
            # expected to alter the command, so the tracker has to be able to
            # see that it was altered.
            err = q_ref[idx] - p.q
            v = np.zeros(p.n)
            v[idx] = np.clip(qd_ref[idx] + a.track_kp * err,
                             -a.joint_vmax, a.joint_vmax)
            if t >= T:
                v[idx] = np.clip(a.track_kp * err, -a.joint_vmax, a.joint_vmax)
            m = Float64MultiArray()
            m.data = [float(x) for x in v]
            p.pub.publish(m)
            log.append(dict(
                t=t, q=[float(x) for x in p.q],
                q_ref=[float(x) for x in q_ref[idx]],
                cmd_in=[float(x) for x in v],
                cmd_out=None if p.cmd_out is None else [float(x) for x in p.cmd_out],
                resid_in=p.r_in, resid_out=p.r_out, n_binding=len(p.binding),
                min_d_model=p.min_d, clearance_ub=p.ub,
                clearance_ub_age=p.ub_age,
                tcp_speed=p.tcp_speed))
    finally:
        p._stop()
        time.sleep(1.5)
        for _ in range(40):
            rclpy.spin_once(p, timeout_sec=0.02)
        record_stop(rec)

    # Arrival, measured from TF against the pose that was COMMANDED.
    pos, _ = p._tcp()
    Tm = p.tf_buffer.lookup_transform(a.report_frame, a.tcp, rclpy.time.Time())
    r = Tm.transform.rotation
    R_cur = BP._quat_to_rot(r.x, r.y, r.z, r.w)
    pos_err = float(np.linalg.norm(pos - p_world))
    rot_err_deg = math.degrees(float(np.linalg.norm(
        rot_error(R_cur, T_des[:3, :3]))))
    q_err = float(np.abs(p.q - plan.q_goal[idx]).max())
    print(f'\n  到達（由 TF 量測，非由計畫宣告）')
    print(f'    TCP 位置 ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  '
          f'目標 ({p_world[0]:.4f}, {p_world[1]:.4f}, {p_world[2]:.4f})')
    print(f'    位置誤差 {pos_err*1000:.2f} mm    姿態誤差 {rot_err_deg:.2f}°    '
          f'關節最大誤差 {q_err*1000:.1f} mrad')
    print(f'    停穩時模型下界 {p.min_d*1000:.1f} mm，獨立上界 {p.ub*1000:.1f} mm，'
          f'TCP 速度 {p.tcp_speed*1000:.2f} mm/s')
    result = dict(pos_err=pos_err, rot_err_deg=rot_err_deg, q_err=q_err,
                  tcp=[float(x) for x in pos],
                  target=[float(x) for x in p_world],
                  min_d_model=p.min_d, clearance_ub=p.ub,
                  tcp_speed_final=p.tcp_speed, checks=checks, log=log)
    rclpy.try_shutdown()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='src/my_omnibot_description/urdf/'
                                      'omni_bot_wholebody.urdf.xacro')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/arm_barrier_test.sdf')
    ap.add_argument('--report-frame', default='world')
    # link_tcp, because that is the frame solve_ik drives. Measuring arrival at
    # uflite_gripper_link instead reported an 83.6 mm "error" that was purely
    # the fixed offset between the two frames.
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--stand', type=float, default=0.24,
                    help='pre-grasp standoff from the box face, m')
    ap.add_argument('--start', nargs=6, type=float, default=[0, 0, 0.30, 0, 0, 0])
    ap.add_argument('--base-x', type=float, default=0.0)
    ap.add_argument('--base-y', type=float, default=0.0)
    ap.add_argument('--base-yaw', type=float, default=0.0)
    ap.add_argument('--min-margin', type=float, default=0.005)
    ap.add_argument('--n-check', type=int, default=25)
    ap.add_argument('--n-seeds', type=int, default=12)
    ap.add_argument('--n-report', type=int, default=41)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--track-kp', type=float, default=2.0)
    ap.add_argument('--home-kp', type=float, default=1.2)
    ap.add_argument('--home-tol', type=float, default=0.012)
    ap.add_argument('--joint-vmax', type=float, default=0.5)
    ap.add_argument('--abort-d', type=float, default=0.02)
    ap.add_argument('--time-scale', type=float, default=5.0,
                    help='stretch the planned duration; slow is easier to watch '
                         'and keeps the approach speed well inside the barrier')
    ap.add_argument('--settle-s', type=float, default=3.0)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--video', default='evaluation/results/pregrasp_demo.mp4')
    ap.add_argument('--camera-topic', default='/demo_cam',
                    help='simulator camera to record; empty disables recording')
    ap.add_argument('--out', default='evaluation/results/pregrasp_demo.json')
    a = ap.parse_args()

    print('══ 離線：IK、路徑與全路徑檢查')
    plan, T_des, p_world, checks = offline(a)
    if plan is None:
        print('\n  離線檢查未通過，不進 Gazebo。', file=sys.stderr)
        return 1
    if a.check_only:
        return 0

    print('\n══ 線上：經安全層追蹤到預抓取位姿')
    res = online(a, plan, T_des, p_world, checks)
    if res is None:
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(args={k: (list(v) if isinstance(v, list) else v)
                         for k, v in vars(a).items()}, **res),
              open(a.out, 'w'), ensure_ascii=False)
    print(f'\n  結果寫入 {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
