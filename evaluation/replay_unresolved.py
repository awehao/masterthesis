"""Replay the unresolved cycles of 5B and decide WHY they are unresolved.

The 5B run leaves a worst barrier residual of +4.22e-02 on three cycles. That
number alone cannot say whether the projection failed to find a solution that
exists, or whether no solution exists at all -- and the two call for opposite
fixes. This script separates them.

Method
------
Step 1 replays the acceptance loop bit-for-bit (same seed, same starts, same
scene) and, on every cycle the filter reports unresolved, dumps a complete
snapshot: q, v_prev, a_prev, v_in, dt, every constraint row with its class and
row id, the per-point d_i / n_i / d_stop_i, the output before and after
fallback, and the worst residual per class.

Step 2 hands the SAME rows to an independent solver -- scipy's HiGHS simplex,
which shares no code and no algorithm with the successive projection in the
filter -- as a minimum-violation LP:

    min s   s.t.   A_barrier v <= b_barrier + s
                   A_hardware v <= b_hardware        (never relaxed)
                   s >= 0

s* is then the smallest barrier violation that is achievable at all while the
hardware limits hold. Reading it:

    s* ~ 0                  a feasible v exists; the projection did not find
                            it. Solver execution problem.
    s* ~ filter residual    no v does better; the filter is already at the
                            optimum and the conflict is physical.
    0 < s* < residual       both: real conflict, and the projection is also
                            leaving something on the table.

The barrier is asked twice, because the filter's fallback does not target the
original bound. It relaxes to b' = max(b, 0) -- "do not approach any further"
rather than "retreat at the rate the barrier wants" -- and the reported
residual is measured against b'. Testing only the original b would answer a
question the filter is not asking.

    python3 evaluation/replay_unresolved.py <expanded.urdf> [--n 8]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import (  # noqa: E402
    ARM_JOINTS, min_jerk, plan_pregrasp)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_NODATA, STATUS_OK, STATUS_STALE, DetectionPoint, SafetyConfig,
    _box_rows, _brake_along, _jerk_rows, _joint_limit_rows, filter_velocity)
import verify_pregrasp as VP  # noqa: E402
import verify_self_collision as VSC  # noqa: E402


def build_blocks(K, q, v_in, pts, cfg, v_prev, a_prev, dt):
    """Rebuild the filter's rows, keeping each class separate.

    The filter concatenates them and loses the boundary between the velocity
    box and the acceleration box, which _box_rows emits interleaved. Here they
    are kept apart so a residual can be attributed to one or the other.
    """
    n = len(K.dof_names)
    Ab, bb, cap, meta = [], [], np.inf, []
    for pt in pts:
        if pt.status == STATUS_NODATA:
            cap = min(cap, cfg.nodata_speed_cap)
            continue
        J = K.jacobian(q, pt.frame)[:3]
        row = pt.n @ J
        d_eff = pt.d
        if pt.status == STATUS_STALE:
            d_eff = pt.d - pt.age * cfg.stale_obstacle_speed
            cap = min(cap, cfg.stale_speed_cap)
        v_app = max(0.0, float(row @ v_in))
        a_br = (max(_brake_along(row, cfg, len(row)), cfg.brake_floor)
                if cfg.use_jacobian_brake else cfg.a_brake)
        d_stop = (cfg.d0 + v_app * cfg.tau
                  + v_app * v_app / (2.0 * max(a_br, 1e-3)) + cfg.eps)
        Ab.append(row)
        bb.append(cfg.alpha * (d_eff - d_stop))
        meta.append(dict(frame=pt.frame, d=float(pt.d), d_eff=float(d_eff),
                         n=[float(x) for x in pt.n], d_stop=float(d_stop),
                         v_app=float(v_app), a_brake=float(a_br),
                         status=int(pt.status), occluded=bool(pt.occluded),
                         kind='barrier'))
        if pt.occluded:
            Ab.append(row)
            bb.append(cfg.blind_approach_cap)
            meta.append(dict(frame=pt.frame, kind='blind_cap',
                             d=float(pt.d), d_stop=float(d_stop)))

    Aj, bj = _joint_limit_rows(K, q, cfg)
    Ax, bx = _box_rows(cfg, n, cap, v_prev, dt)
    Ak, bk = (_jerk_rows(cfg, n, v_prev, a_prev, dt)
              if cfg.enforce_jerk else ([], []))

    # Velocity box alone, i.e. what _box_rows would emit with no v_prev. The
    # difference between this and Ax/bx is exactly the acceleration bound.
    Axv, bxv = _box_rows(cfg, n, cap, None, dt)
    return dict(Ab=np.array(Ab).reshape(-1, n), bb=np.array(bb),
                Aj=np.array(Aj).reshape(-1, n), bj=np.array(bj),
                Ax=np.array(Ax).reshape(-1, n), bx=np.array(bx),
                Axv=np.array(Axv).reshape(-1, n), bxv=np.array(bxv),
                Ak=np.array(Ak).reshape(-1, n), bk=np.array(bk),
                cap=cap, meta=meta)


def min_violation(A_soft, b_soft, A_hard, b_hard, n):
    """min s  s.t.  A_soft v <= b_soft + s,  A_hard v <= b_hard,  s >= 0.

    Variables are [v (n), s (1)]. HiGHS, not projection: an independent answer
    to whether the soft block can be met at all.
    """
    ns = A_soft.shape[0]
    nh = A_hard.shape[0]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    blocks, rhs = [], []
    if ns:
        blocks.append(np.hstack([A_soft, -np.ones((ns, 1))]))
        rhs.append(b_soft)
    if nh:
        blocks.append(np.hstack([A_hard, np.zeros((nh, 1))]))
        rhs.append(b_hard)
    A_ub = np.vstack(blocks)
    b_ub = np.concatenate(rhs)
    bounds = [(None, None)] * n + [(0.0, None)]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not r.success:
        return dict(ok=False, status=r.message, s=None, v=None)
    return dict(ok=True, s=float(r.x[-1]), v=[float(x) for x in r.x[:n]],
                status='optimal')


def hard_feasible(A_hard, b_hard, n):
    """Is the hardware set alone satisfiable? It should always be."""
    if not A_hard.shape[0]:
        return dict(ok=True, s=0.0)
    return min_violation(A_hard, b_hard, np.zeros((0, n)), np.zeros(0), n)


def brake_boundary(q, v, cfg, K, idx):
    """Acceleration-aware position stopping check, per arm joint.

    A joint moving toward its limit needs qdot^2 <= 2 a_max d_limit to be able
    to stop before reaching it. The one-step position row in the filter does
    not check this: it only forbids OVERSHOOTING WITHIN THE NEXT STEP, so a
    joint can be inside the row and still be committed to passing the limit
    because no achievable deceleration can arrest it in the distance left.
    """
    out = []
    lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
    amax = LITE6_SAFE.max_acceleration
    for k, j in enumerate(idx):
        qk, vk = float(q[j]), float(v[j])
        if vk > 0:
            d_lim = hi[k] - qk
        elif vk < 0:
            d_lim = qk - lo[k]
        else:
            d_lim = min(hi[k] - qk, qk - lo[k])
        need = vk * vk
        have = 2.0 * amax[k] * max(d_lim, 0.0)
        out.append(dict(joint=ARM_JOINTS[k], q=qk, v=vk, d_limit=float(d_lim),
                        qdot2=float(need), two_a_d=float(have),
                        stoppable=bool(need <= have + 1e-12)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default='evaluation/results/unresolved_snapshots.json')
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(a.world)
    clouds = VSC.link_clouds(xml, max_pts=250)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    cfg = SafetyConfig()
    rng = np.random.default_rng(0)
    n = len(K.dof_names)

    # Same scene setup as the acceptance, reused rather than restated so the
    # replay cannot drift from the run it is explaining.
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
                worst = min(worst, VP.nearest(obs, p)[0])
        return worst

    box = min((o for o in obs if o.kind == 'box'),
              key=lambda o: float(o.T_world_link[2, 3]))
    c = box.T_world_link[:3, 3]
    face = (np.array([1.0, 0.0, 0.0]) if box.size[0] <= box.size[1]
            else np.array([0.0, 1.0, 0.0]))
    half = 0.5 * float(box.size @ np.abs(face))
    stand, base_off = 0.12, 0.55
    p_base = c + face * (half + base_off)
    yaw = math.atan2(-face[1], -face[0])
    q_base = np.array([p_base[0], p_base[1], yaw])
    p_des = c + face * (half + stand)
    p_des[2] = float(np.clip(0.55, c[2] - 0.25, c[2] + 0.25))
    zc = -face
    xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = VP._iso(np.column_stack([xc, np.cross(zc, xc), zc]), p_des)

    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    starts = [np.array(VP.TUCK)] + [rng.uniform(lo6 * 0.5, hi6 * 0.5)
                                    for _ in range(a.n - 1)]

    snaps = []
    n_cycles = 0
    for si, q0a in enumerate(starts):
        q = np.zeros(n)
        q[:3] = q_base
        q[idx] = q0a
        plan = plan_pregrasp(K, q, T_des, self_clear, env_clear)
        if not plan.ok:
            continue
        T = plan.duration
        qg = plan.q_goal[idx]
        v_prev = np.zeros(n)
        v_prev2 = None
        qc = q0a.copy()
        steps = int(math.ceil((T + 1.5) / cfg.dt))
        for k in range(steps):
            t = k * cfg.dt
            qd_ref = min_jerk(q0a, qg, T, t)[1] if t <= T else np.zeros(6)
            qref = min_jerk(q0a, qg, T, min(t, T))[0]
            v_in = np.zeros(n)
            v_in[idx] = qd_ref + 2.0 * (qref - qc)
            qf = np.zeros(n)
            qf[:3] = q_base
            qf[idx] = qc
            pts = []
            for fr in VP.FRAMES:
                p = K.fk(qf, fr)[:3, 3]
                d, vv = VP.nearest(obs, p)
                pts.append(DetectionPoint(fr, p, vv / max(d, 1e-9), d, STATUS_OK))
            a_prev = ((v_prev - v_prev2) / cfg.dt) if v_prev2 is not None else None
            r = filter_velocity(K, qf, v_in, pts, cfg, v_prev=v_prev,
                                a_prev=a_prev, dt=cfg.dt)
            n_cycles += 1

            if r.unresolved:
                B = build_blocks(K, qf, v_in, pts, cfg, v_prev, a_prev, cfg.dt)
                A_hw = np.vstack([B['Aj'], B['Ax']])
                b_hw = np.concatenate([B['bj'], B['bx']])
                bb_relaxed = np.maximum(B['bb'], 0.0)

                lp_strict = min_violation(B['Ab'], B['bb'], A_hw, b_hw, n)
                lp_relax = min_violation(B['Ab'], bb_relaxed, A_hw, b_hw, n)
                lp_hw = hard_feasible(A_hw, b_hw, n)
                # Same question with jerk added back, to show whether jerk was
                # the binding thing or an innocent bystander.
                if B['Ak'].shape[0]:
                    A_hwj = np.vstack([A_hw, B['Ak']])
                    b_hwj = np.concatenate([b_hw, B['bk']])
                    lp_relax_jerk = min_violation(B['Ab'], bb_relaxed,
                                                  A_hwj, b_hwj, n)
                else:
                    lp_relax_jerk = dict(ok=True, s=None, status='no jerk rows')

                def worst(A, b, v):
                    return float((A @ v - b).max()) if A.shape[0] else 0.0
                v_out = r.v
                snaps.append(dict(
                    start=si, step=k, t=float(t), dt=float(cfg.dt),
                    q=[float(x) for x in qf], v_in=[float(x) for x in v_in],
                    v_prev=[float(x) for x in v_prev],
                    a_prev=(None if a_prev is None
                            else [float(x) for x in a_prev]),
                    v_out=[float(x) for x in v_out],
                    filter=dict(n_rows=r.n_rows, n_active=r.n_active,
                                iters=r.iters, fallback=r.fallback,
                                override=r.safety_override,
                                resid_after=float(r.max_resid_after),
                                resid_before=float(r.max_resid_before),
                                resid_before_fallback=float(r.resid_before_fallback),
                                resid_barrier=float(r.resid_barrier),
                                resid_position=float(r.resid_position),
                                speed_cap=(None if not np.isfinite(r.speed_cap)
                                           else float(r.speed_cap))),
                    resid_out=dict(
                        barrier_strict=worst(B['Ab'], B['bb'], v_out),
                        barrier_relaxed=worst(B['Ab'], bb_relaxed, v_out),
                        position=worst(B['Aj'], B['bj'], v_out),
                        velbox=worst(B['Axv'], B['bxv'], v_out),
                        accbox=worst(B['Ax'], B['bx'], v_out),
                        jerk=worst(B['Ak'], B['bk'], v_out)),
                    lp=dict(barrier_strict=lp_strict, barrier_relaxed=lp_relax,
                            barrier_relaxed_with_jerk=lp_relax_jerk,
                            hardware_only=lp_hw),
                    rows=dict(n_barrier=int(B['Ab'].shape[0]),
                              n_position=int(B['Aj'].shape[0]),
                              n_box=int(B['Ax'].shape[0]),
                              n_jerk=int(B['Ak'].shape[0])),
                    points=B['meta'],
                    brake_boundary=brake_boundary(qf, v_out, cfg, K, idx),
                    A_barrier=[[float(x) for x in row] for row in B['Ab']],
                    b_barrier=[float(x) for x in B['bb']],
                    b_barrier_relaxed=[float(x) for x in bb_relaxed],
                    A_position=[[float(x) for x in row] for row in B['Aj']],
                    b_position=[float(x) for x in B['bj']],
                    A_box=[[float(x) for x in row] for row in B['Ax']],
                    b_box=[float(x) for x in B['bx']],
                    A_jerk=[[float(x) for x in row] for row in B['Ak']],
                    b_jerk=[float(x) for x in B['bk']],
                ))

            qc = qc + r.v[idx] * cfg.dt
            v_prev2 = v_prev
            v_prev = r.v.copy()

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(n_cycles=n_cycles, n_unresolved=len(snaps), snaps=snaps),
              open(a.out, 'w'), indent=1)

    print(f'  週期總數 {n_cycles}   unresolved {len(snaps)}')
    print(f'  快照寫入 {a.out}\n')
    for s in snaps:
        print(f'  ── 起始 {s["start"]} 第 {s["step"]} 步  t={s["t"]:.2f}s')
        f = s['filter']
        print(f'     濾波器: rows {s["rows"]} active {f["n_active"]} '
              f'iters {f["iters"]} fallback {f["fallback"]} '
              f'override {f["override"]}')
        ro = s['resid_out']
        print(f'     輸出殘差: 屏障(原) {ro["barrier_strict"]:+.3e}  '
              f'屏障(放寬) {ro["barrier_relaxed"]:+.3e}')
        print(f'               位置 {ro["position"]:+.3e}  '
              f'速度盒 {ro["velbox"]:+.3e}  加速度盒 {ro["accbox"]:+.3e}  '
              f'jerk {ro["jerk"]:+.3e}')
        lp = s['lp']
        def fmt(x):
            return 'INFEASIBLE' if not x['ok'] else (
                'n/a' if x['s'] is None else f'{x["s"]:+.3e}')
        print(f'     獨立 LP 最小違反量 s*:')
        print(f'        硬體限制單獨          {fmt(lp["hardware_only"])}')
        print(f'        屏障(原) + 硬體       {fmt(lp["barrier_strict"])}')
        print(f'        屏障(放寬) + 硬體     {fmt(lp["barrier_relaxed"])}')
        print(f'        屏障(放寬) + 硬體+jerk {fmt(lp["barrier_relaxed_with_jerk"])}')
        gap = None
        if lp['barrier_relaxed']['ok'] and lp['barrier_relaxed']['s'] is not None:
            gap = ro['barrier_relaxed'] - lp['barrier_relaxed']['s']
            print(f'        濾波器殘差 - s*      {gap:+.3e}  '
                  f'{"← 投影未達最佳" if gap > 1e-6 else "← 濾波器已達最佳"}')
        ns = [x for x in s['brake_boundary'] if not x['stoppable']]
        if ns:
            print(f'     位置制動邊界不足: '
                  + ', '.join(f'{x["joint"]}(qdot²={x["qdot2"]:.3f} > '
                              f'2ad={x["two_a_d"]:.3f})' for x in ns))
        else:
            print(f'     位置制動邊界: 全部關節可停住')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
