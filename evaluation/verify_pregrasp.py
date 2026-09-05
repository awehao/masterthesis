"""Acceptance for 5B: reach a pre-grasp pose with a fixed base, and hold it.

Runs the whole chain offline against the scene model -- IK, trajectory, the
link safety filter, and the limit set -- over many legal starting poses, and
reports the numbers the plan asks for:

    reach rate over several legal starts
    TCP position and orientation error
    minimum environment and self-collision clearance
    position / velocity / acceleration / jerk violation rates
    worst constraint residual and safety_override count
    settling: does it hold, or oscillate

The target is a stand-in handle taken from the scene model (Gazebo truth): a
pose a fixed standoff in front of one face of a known box. Nothing is grasped
and nothing is touched -- the run stops at the pre-grasp pose and holds.

    python3 evaluation/verify_pregrasp.py <expanded.urdf> [--n 12]
"""
from __future__ import annotations

import argparse
import math
import re
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import (  # noqa: E402
    ARM_JOINTS, TCP, min_jerk, plan_pregrasp, rot_error)
from ammr_wholebody_mpc.arm_detection_points import (  # noqa: E402
    Obstacle, _closest_local, _inv, _iso, _rpy_to_rot)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, filter_velocity)
import verify_self_collision as VSC  # noqa: E402

FRAMES = ['detect0_1', 'detect0_2', 'detect1', 'detect2_1', 'detect2_2',
          'detect2_3', 'detect3_1', 'detect3_2', 'detect4_1', 'detect4_2',
          'detect5', 'detect6']
TUCK = [0.0, -0.082, 0.089, 0.0, 1.679, 0.0]


def obstacles_from_world(path: str):
    sdf = open(path).read()
    out = []
    for m in re.finditer(r'<model name="(obs_\d+)">(.*?)</model>', sdf, re.S):
        body = m.group(2)
        pose = re.search(r'<pose>([-\d.eE\s]+)</pose>', body)
        if not pose:
            continue
        v = [float(x) for x in pose.group(1).split()]
        o = Obstacle(name=m.group(1), model='', kind='')
        box = re.search(r'<box><size>([^<]+)</size>', body)
        cyl = re.search(r'<cylinder><radius>([\d.]+)</radius>\s*<length>([\d.]+)', body)
        if box:
            o.kind = 'box'
            o.size = np.array([float(x) for x in box.group(1).split()])
        elif cyl:
            o.kind = 'cylinder'
            o.radius, o.height = float(cyl.group(1)), float(cyl.group(2))
        else:
            continue
        o.T_world_link = _iso(_rpy_to_rot(*v[3:6]), np.array(v[:3]))
        o.T_link_collision = np.eye(4)
        out.append(o)
    return out


def nearest(obs, p):
    best_d, best_v = math.inf, None
    for o in obs:
        T = o.T_world_link @ o.T_link_collision
        loc = (_inv(T) @ np.append(p, 1.0))[:3]
        surf = (T @ np.append(_closest_local(o, loc), 1.0))[:3]
        v = surf - p
        d = float(np.linalg.norm(v))
        if d < best_d:
            best_d, best_v = d, v
    return best_d, best_v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--stand', type=float, default=0.12,
                    help='pre-grasp standoff from the box face, m')
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = obstacles_from_world(a.world)
    clouds = VSC.link_clouds(xml, max_pts=250)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    cfg = SafetyConfig()
    rng = np.random.default_rng(0)
    n = len(K.dof_names)

    # Rigid groups / adjacency, as in the self-collision acceptance.
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
        return min(float(cKDTree(w[x]).query(w[y], k=1)[0].min())
                   for x, y in pairs)

    def env_clear(q):
        worst = math.inf
        for nm in ('link2', 'link3', 'link4', 'link5', 'link6',
                   'uflite_gripper_link'):
            if nm not in clouds:
                continue
            T = K.fk(q, nm)
            P = (T[:3, :3] @ clouds[nm].T).T + T[:3, 3]
            for p in P[::12]:
                worst = min(worst, nearest(obs, p)[0])
        return worst

    # ---- place the base, then pick the target ----------------------------
    # 5B fixes the base but does not say it sits at the origin: reaching a
    # handle is stage C's job to drive to. Putting the base at the origin left
    # every target ~1.9 m away against a 0.44 m reach, so all eight plans failed
    # for a reason that has nothing to do with IK, trajectory or safety.
    #
    # The base is therefore parked in front of the target box at a standoff
    # that puts the pre-grasp point inside the arm's workspace, and held there.
    box = min((o for o in obs if o.kind == 'box'),
              key=lambda o: float(o.T_world_link[2, 3]))
    c = box.T_world_link[:3, 3]
    # Approach the box along its SHORTER horizontal axis, so the standoff
    # geometry is not dominated by a 2.5 m face.
    face = (np.array([1.0, 0.0, 0.0]) if box.size[0] <= box.size[1]
            else np.array([0.0, 1.0, 0.0]))
    half = 0.5 * float(box.size @ np.abs(face))
    stand = a.stand
    # Standoff chosen from the measured workspace, not guessed. The shoulder
    # sits 0.11 m ahead of the base origin, so a 0.12 m standoff puts the target
    # ~0.32 m from it. A sweep of the reachable set shows position-plus-
    # orientation IK converging from 0.25 m outward and failing at 0.20 m -- the
    # arm cannot fold that tightly while also pointing the tool at the face. The
    # first attempt used 0.42 m, which left the target 0.19 m away, and all
    # eight plans failed for that reason alone.
    #
    # So the base retreats by exactly as much as the standoff grows. Moving the
    # pre-grasp point away from the BOX while leaving the base put would move it
    # TOWARD the shoulder -- 0.20 m of standoff would leave 0.241 m of reach,
    # back in the region where IK stops converging, and the re-run would fail
    # for a reason that has nothing to do with the safety layer it is meant to
    # test. Holding shoulder-to-target fixed keeps the two questions separate.
    base_off = 0.55 + (stand - 0.12)     # base origin to face
    p_base = c + face * (half + base_off)
    yaw = math.atan2(-face[1], -face[0])  # face the box
    q_base = np.array([p_base[0], p_base[1], yaw])

    p_des = c + face * (half + stand)
    p_des[2] = float(np.clip(0.55, c[2] - 0.25, c[2] + 0.25))
    zc = -face                             # tool +z points at the face
    xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = _iso(np.column_stack([xc, np.cross(zc, xc), zc]), p_des)
    print(f'  目標箱體 {box.name}  中心 ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  '
          f'尺寸 {np.round(box.size,2)}')
    print(f'  底盤停放 ({q_base[0]:.2f},{q_base[1]:.2f},{math.degrees(q_base[2]):.0f}°)'
          f'  離面 {base_off:.2f} m（固定不動）')
    print(f'  預抓取位姿 ({p_des[0]:.3f},{p_des[1]:.3f},{p_des[2]:.3f})  '
          f'離面 {stand:.2f} m')
    q_probe = np.zeros(n); q_probe[:3] = q_base; q_probe[idx] = TUCK
    sh = K.fk(q_probe, 'link1')[:3, 3]
    print(f'  肩部到目標 {np.linalg.norm(p_des - sh):.3f} m（臂展約 0.44 m）')

    # ---- run from several legal starts -----------------------------------
    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    reached = 0
    pe_l, re_l, tset_l = [], [], []
    minenv, minself = math.inf, math.inf
    v_bad = a_bad = j_bad = p_bad = 0
    n_steps_tot = 0
    worst_resid = -math.inf
    n_override = 0
    R = {k: -math.inf for k in ('barrier','position','velbox','jerk','pre_fb')}
    n_unres = 0
    n_fb = 0
    osc = []
    starts = [np.array(TUCK)] + [rng.uniform(lo6 * 0.5, hi6 * 0.5)
                                 for _ in range(a.n - 1)]

    n_ikfail = 0
    fail_reasons = []
    for si, q0a in enumerate(starts):
        q = np.zeros(n)
        q[:3] = q_base
        q[idx] = q0a
        plan = plan_pregrasp(K, q, T_des, self_clear, env_clear)
        if not plan.ok:
            n_ikfail += 1
            fail_reasons.append(plan.reason)
            continue
        T = plan.duration
        qg = plan.q_goal[idx]
        v_prev = np.zeros(n)
        v_prev2 = None
        qc = q0a.copy()
        traj = []
        steps = int(math.ceil((T + 1.5) / cfg.dt))
        for k in range(steps):
            t = k * cfg.dt
            qd_ref = min_jerk(q0a, qg, T, t)[1] if t <= T else np.zeros(6)
            # feedback so tracking error does not accumulate
            qref = min_jerk(q0a, qg, T, min(t, T))[0]
            v_in = np.zeros(n)
            v_in[idx] = qd_ref + 2.0 * (qref - qc)
            qf = np.zeros(n)
            qf[:3] = q_base
            qf[idx] = qc
            pts = []
            for fr in FRAMES:
                p = K.fk(qf, fr)[:3, 3]
                d, vv = nearest(obs, p)
                pts.append(DetectionPoint(fr, p, vv / max(d, 1e-9), d, STATUS_OK))
            a_prev = ((v_prev - v_prev2) / cfg.dt) if v_prev2 is not None else None
            r = filter_velocity(K, qf, v_in, pts, cfg, v_prev=v_prev,
                                a_prev=a_prev, dt=cfg.dt)
            n_override += int(r.safety_override)
            n_unres += int(r.unresolved)
            n_fb += int(r.fallback)
            worst_resid = max(worst_resid, r.max_resid_after)
            R['barrier'] = max(R['barrier'], r.resid_barrier)
            R['position'] = max(R['position'], r.resid_position)
            R['velbox'] = max(R['velbox'], r.resid_velbox)
            R['jerk'] = max(R['jerk'], r.resid_jerk)
            R['pre_fb'] = max(R['pre_fb'], r.resid_before_fallback)
            qc = qc + r.v[idx] * cfg.dt
            traj.append((qc.copy(), r.v[idx].copy()))
            v_prev2 = v_prev
            v_prev = r.v.copy()
            n_steps_tot += 1
            p_bad += int(np.any(qc < lo6 - 1e-6) or np.any(qc > hi6 + 1e-6))
            v_bad += int(np.any(np.abs(r.v[idx]) > LITE6_SAFE.max_velocity + 1e-6))
            if a_prev is not None:
                acc = (r.v[idx] - v_prev2[idx]) / cfg.dt
                a_bad += int(np.any(np.abs(acc) > LITE6_SAFE.max_acceleration + 1e-3))
            qf2 = np.zeros(n); qf2[:3] = q_base; qf2[idx] = qc
            minself = min(minself, self_clear(qf2))
            minenv = min(minenv, env_clear(qf2))

        qf = np.zeros(n)
        qf[:3] = q_base
        qf[idx] = qc
        Tc = K.fk(qf, TCP)
        pe = float(np.linalg.norm(T_des[:3, 3] - Tc[:3, 3]))
        re_ = float(np.linalg.norm(rot_error(Tc[:3, :3], T_des[:3, :3])))
        print(f'      起始 {si}: 末端誤差 {pe*1000:7.1f} mm  '
              f'{math.degrees(re_):6.2f}°   T={T:.2f}s')
        if pe < 0.01 and re_ < 0.05:
            reached += 1
            pe_l.append(pe)
            re_l.append(re_)
            tail = np.array([t[0] for t in traj[-20:]])
            osc.append(float(np.abs(np.diff(tail, axis=0)).max()))
            tset_l.append(T)

    print(f'\n  起始姿態 {len(starts)} 組')
    print(f'  到達率            {reached}/{len(starts)}  '
          f'({100*reached/len(starts):.0f}%)')
    if n_ikfail:
        print(f'  規劃失敗          {n_ikfail}  例：{fail_reasons[0][:70]}')
    if pe_l:
        print(f'  TCP 位置誤差      中位 {np.median(pe_l)*1000:.2f} mm   '
              f'最大 {max(pe_l)*1000:.2f} mm')
        print(f'  TCP 姿態誤差      中位 {math.degrees(np.median(re_l)):.2f}°   '
              f'最大 {math.degrees(max(re_l)):.2f}°')
        print(f'  到達後保持        末 20 步最大關節變化 '
              f'{math.degrees(max(osc)):.4f}°  '
              f'{"✓ 穩定" if max(osc) < 1e-3 else "★ 有殘餘運動"}')
        print(f'  軌跡時間          中位 {np.median(tset_l):.2f} s')
    print(f'  最小環境間距      {minenv:.4f} m')
    print(f'  最小自碰撞間距    {minself:.4f} m')
    print(f'  違反率（{n_steps_tot} 步）位置 {p_bad}  速度 {v_bad}  加速度 {a_bad}')
    print(f'  最大約束殘差      {worst_resid:+.2e}')
    print(f'  ── 依類別拆分（<=0 為滿足）')
    for k, lab in (('barrier','屏障'),('position','關節位置'),
                   ('velbox','速度／加速度盒'),('jerk','jerk'),
                   ('pre_fb','fallback 前')):
        print(f'     {lab:16} {R[k]:+.3e}')
    print(f'  fallback {n_fb}   unresolved {n_unres}   '
          f'safety_override {n_override}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
