"""Offline: at the stalled instant, can a CONSTRAINED single-step solve move the arm?

Why this exists
---------------
The obstacle run stopped 225.9 mm short with the base blocked and the arm
commanded to zero, and it would be easy to write that up as "a one-step
controller cannot reassign the task, only a multi-step whole-body MPC can".
That claim is not supported by the evidence. What the run shows is that THIS
architecture stalls: the task is solved first, with a strong posture preference
and no knowledge of the active obstacle constraints, and only afterwards is the
answer projected onto the safe set. A one-step solver can also put the task,
the collision constraints and the hardware limits into ONE problem and demote
the posture to a secondary objective. Whether that is enough here is a question
with an answer, and this measures it instead of assuming it.

Two solves on exactly the same state, the same constraint set, and the same
hardware limits:

  A  task solve then projection      the pipeline as it runs today
  B  one constrained solve           the same task, the same constraints, but
                                     inside the same problem, with the posture
                                     term as a secondary objective

The number that matters is the directional derivative of the end-effector
position error along the commanded velocity,

    d|e|/dt = -(e/|e|) . J_v v,

negative meaning the error is being reduced. A solve that returns a feasible v
with d|e|/dt < 0 has found a reassignment AT THIS INSTANT, with no prediction.
If neither does, the question moves to the local Jacobian and the active set:
an IK solution existing somewhere does not mean a feasible descent direction
exists here.

    python3 evaluation/compare_stall_solvers.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, _HERE)

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                    # noqa: E402
from ammr_wholebody_mpc.arm_link_geometry import (                      # noqa: E402
    arm_link_names, nearest_points, sample_links_certified)
from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS, rot_error       # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (                # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, _box_rows, _joint_limit_rows,
    _project, _rows_from_points)
import verify_pregrasp as VP                                            # noqa: E402


def _expand(path):
    return (subprocess.check_output(['xacro', path], text=True)
            if path.endswith('.xacro') else open(path).read())


def build_rows(K, q, xml, world, cfg, v_in, samples, link_names):
    """The barrier rows the distance node would produce for this exact state."""
    obs = VP.obstacles_from_world(world)
    pts = []
    for li, nm in enumerate(link_names):
        T = K.fk(q, nm)
        S = samples[nm]
        W = (S.points @ T[:3, :3].T) + T[:3, 3]
        from ammr_wholebody_mpc.arm_link_geometry import obstacle_distances
        d, v, _ = obstacle_distances(W, obs)
        dmin = float(d.min())
        sel = np.nonzero(d <= dmin + S.rho)[0]
        sel = sel[np.argsort(d[sel])]
        for k in sel:
            nv = v[k] / max(abs(float(d[k])), 1e-9)
            pts.append(DetectionPoint(
                frame=nm, p=W[k], n=nv, d=float(d[k]), status=STATUS_OK,
                age=0.0, occluded=False, offset=S.points[k], rho=float(S.rho)))
    return pts


def constraints(K, q, pts, cfg, v_in, v_prev, dt):
    Ab, bb, cap, _ = _rows_from_points(K, q, pts, cfg, v_in)
    Aj, bj = _joint_limit_rows(K, q, cfg)
    Ax, bx = _box_rows(cfg, len(K.dof_names), cap, v_prev, dt)
    A = np.array(Ab + Aj + Ax)
    b = np.array(bb + bj + bx)
    return A, b, len(Ab)


def solve_qp(H, g, A, b, n):
    """min 1/2 v'Hv + g'v  s.t. Av <= b, via OSQP."""
    import osqp
    from scipy import sparse
    P = sparse.csc_matrix((H + H.T) / 2.0)
    Ac = sparse.csc_matrix(A)
    lo = np.full(len(b), -np.inf)
    m = osqp.OSQP()
    m.setup(P=P, q=g, A=Ac, l=lo, u=b, verbose=False,
            eps_abs=1e-9, eps_rel=1e-9, max_iter=40000, polish=True)
    r = m.solve()
    ok = r.info.status_val in (1, 2)          # solved / solved inaccurate
    return (np.asarray(r.x, float) if ok else None), r.info.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='evaluation/results/wholebody_pregrasp.json')
    ap.add_argument('--urdf', default='src/my_omnibot_description/urdf/'
                                      'omni_bot_wholebody.urdf.xacro')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/arm_barrier_test.sdf')
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--out', default='evaluation/results/stall_solver_compare.json')
    a = ap.parse_args()

    rec = json.load(open(a.log))
    last = rec['log'][-1]
    args = rec['args']
    q = np.zeros(9)
    q[:3] = last['base']
    xml = _expand(a.urdf)
    K = WholeBodyKinematics.from_urdf_string(xml)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    q[idx] = last['q']
    q_pref = np.array(args['posture'], float)
    target = np.array(args['target'], float)

    zc = np.array([1.0, 0.0, 0.0]); xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc; xc /= np.linalg.norm(xc)
    R_des = np.column_stack([xc, np.cross(zc, xc), zc])

    T = K.fk(q, a.tcp)
    e_p = target - T[:3, 3]
    e_r = rot_error(T[:3, :3], R_des)
    ne = float(np.linalg.norm(e_p))
    u = e_p / max(ne, 1e-12)
    J = K.jacobian(q, a.tcp)
    print(f'  停滯狀態  底盤 {np.round(q[:3], 4).tolist()}')
    print(f'            手臂 {np.round(q[idx], 4).tolist()}')
    print(f'  位置誤差 {ne*1000:.2f} mm  方向 {np.round(u, 3).tolist()}')
    print(f'  姿態誤差 {np.degrees(np.linalg.norm(e_r)):.4f}°')

    print('\n  重建屏障列（與距離節點同一套認證取樣與帶選擇）…', flush=True)
    samples = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    link_names = arm_link_names(xml)
    cfg = SafetyConfig(dt=1.0 / args['rate'])
    v_prev = np.zeros(9)

    # ---- A: task solve, then projection (the pipeline as it runs) ----------
    w = np.ones(9); w[:3] = args['base_weight']
    lam, mu = args['damping'], args['mu_post']
    lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
    m_ = 0.35
    push = (np.maximum(0.0, m_ - (q[idx] - lo))
            - np.maximum(0.0, m_ - (hi - q[idx]))) / m_
    v_post = np.zeros(9)
    v_post[idx] = args['kp_post'] * (q_pref - q[idx]) + args['k_limit'] * push
    S = np.zeros((9, 9)); S[idx, idx] = 1.0

    ep_c = e_p * min(1.0, args['v_task'] / max(ne, 1e-12))
    nr = float(np.linalg.norm(e_r))
    er_c = e_r * min(1.0, args['w_task'] / max(nr, 1e-12))
    e_task = np.concatenate([args['kp_p'] * ep_c, args['kp_r'] * er_c])
    H_a = J.T @ J + mu * mu * S + lam * lam * np.diag(1.0 / (w * w))
    v_task = np.linalg.solve(H_a, J.T @ e_task + mu * mu * (S @ v_post))

    pts = build_rows(K, q, xml, a.world, cfg, v_task, samples, link_names)
    A, b, n_bar = constraints(K, q, pts, cfg, v_task, v_prev, cfg.dt)
    v_a, _ = _project(A, b, v_task.copy(), cfg.proj_iters, cfg.relax)
    print(f'\n  屏障列 {n_bar}，總約束 {len(A)}')

    def rate(v):
        return float(-(u @ (J[:3] @ v)))          # d|e|/dt, 負為縮小

    print('\n══ A：任務求解 → 安全濾波（現行管線）')
    print(f'    任務解 v   底盤 {np.round(v_task[:3], 5).tolist()}  手臂 {np.round(v_task[idx], 5).tolist()}')
    print(f'    投影後 v   底盤 {np.round(v_a[:3], 5).tolist()}  手臂 {np.round(v_a[idx], 5).tolist()}')
    print(f'    d|e|/dt = {rate(v_a)*1000:+.4f} mm/s   最大違反 {float((A @ v_a - b).max())*1000:+.4f}')

    # ---- B: one constrained solve, posture demoted -------------------------
    print('\n══ B：單次約束式求解（任務、碰撞、硬體限制同一問題；臂形為次要目標）')
    print(f"    {'mu_post':>8} {'狀態':>8} {'底盤 vy':>9} {'手臂 |max|':>10} {'d|e|/dt mm/s':>13} {'違反':>10}")
    rows = []
    for mu_b in (args['mu_post'], 0.3, 0.1, 0.03, 0.01, 0.0):
        H = J.T @ J + mu_b * mu_b * S + lam * lam * np.diag(1.0 / (w * w))
        g = -(J.T @ e_task + mu_b * mu_b * (S @ v_post))
        v_b, st = solve_qp(H, g, A, b, 9)
        if v_b is None:
            print(f'    {mu_b:8.3f} {st:>8}'); continue
        viol = float((A @ v_b - b).max())
        rows.append(dict(mu=mu_b, v=[float(x) for x in v_b],
                         rate=rate(v_b), viol=viol))
        print(f'    {mu_b:8.3f} {"ok":>8} {v_b[1]:+9.5f} {np.abs(v_b[idx]).max():10.5f} '
              f'{rate(v_b)*1000:+13.4f} {viol*1000:+10.4f}')

    best = min(rows, key=lambda r: r['rate']) if rows else None
    print('\n  ── 判讀 ──')
    print(f'    A（現行）d|e|/dt = {rate(v_a)*1000:+.4f} mm/s')
    if best:
        print(f'    B（最佳 mu={best["mu"]}）d|e|/dt = {best["rate"]*1000:+.4f} mm/s')
        print(f'      手臂 {np.round(np.array(best["v"])[idx], 5).tolist()}')
        print(f'      底盤 {np.round(np.array(best["v"])[:3], 5).tolist()}')
        if best['rate'] < -1e-6:
            print('    → 在同一組安全與硬體限制下，單次約束式求解**能**產生縮小末端誤差的動作。')
            print('      本停滯點的重新分配不需要多步預測。')
        else:
            print('    → 單次約束式求解也找不到下降方向。問題在局部 Jacobian／作用約束，')
            print('      而不只是任務分配；IK 有解不等於此處存在可行的局部下降方向。')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(state=dict(q=[float(x) for x in q], err=ne,
                              n_barrier=int(n_bar), n_con=int(len(A))),
                   A_rate=rate(v_a), A_v=[float(x) for x in v_a], B=rows),
              open(a.out, 'w'), ensure_ascii=False)
    print(f'\n  結果寫入 {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
