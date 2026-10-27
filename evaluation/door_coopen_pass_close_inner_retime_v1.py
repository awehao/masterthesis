#!/usr/bin/env python3
"""內側把手通過中拉關串接 v1 的重新時間化修正（Codex r42 §1）：舊 RB3.retime 以位移絕對值核速度、以單段時長核接點加速度，漏掉方向反轉
（static A→I 接點第 5 關節 +0.9999 → −0.6377 rad/s，中點時距加速度約 29.3 rad/s² ＞ 19.984），且未納入門角速率。
本版（只改時間化，不改構型路徑）：
  時間參數化＝經過點版「線性段＋拋物線混合」（LSPB）：每段有號速度 v_k＝Δ_k／dt_k，dt_k 由各軸速度上限決定；接點 j 混合時長
    τ_j＝max_i |v_{k+1,i} − v_{k,i}| / a_i（底盤 x、y 以接點偏航轉到車體座標），混合內 p(t)＝p_j＋v_k(t−t_j)＋Δv(t−t_j＋τ/2)²/(2τ)；
    起訖由靜止混合。若某段兩端混合半長和 ＞ dt_k 則拉長該段（×1.1）並重算，直到成立。混合使軌跡不精確經過接點（偏差 Δv·τ/8），故全部取樣重核。
  速度上限：底盤車體 x、y 0.035255 m/s、偏航 0.1999 rad/s；手臂 0.9999 rad/s；加速度 底盤 0.5 m/s²、偏航 2.0 rad/s²、手臂 19.984 rad/s²（同 RB3）；
    門角（P 段由持握帶動）≤ 0.6 °/s＝已登錄開門參考上限 r_max_dps（bench_6 config），遠低於保護值 door_abort_rate 0.05 rad/s（不放寬）；
    門角加速度無登錄值 ⇒ 不設門檻、只報；手指開度 ≤ finger_open／close_ramp_s＝0.0089 m/s，不混合（位置命令）。
  20 Hz（控制週期 0.05 s）取樣後核：有號差分速度、二階差分加速度（底盤以取樣點偏航轉車體座標）、門速；每取樣點淨空以 door_coopen_pass_close_inner_v1.check
    （階段＝該時刻所在段的終點列階段）重核；抓持段（G 閉爪完成後、P、R_open 張爪前）核「當時 TCP 對當時內側把手」相對位姿與 T_HG 的誤差：
    門檻＝已登錄 drift_abort_m 5 mm／drift_rot_abort 2°，另對照 grip_tol_m 2 mm。重疊指標與整機清空在取樣上重算。
只處理 static（Codex r42：先做一組 static 名目閉迴路；有門餘轉的情境需另核把手追蹤）。輸出含 20 Hz 參考（供 W-GMPC 名目追蹤）。

    python3 -u evaluation/door_coopen_pass_close_inner_retime_v1.py   # 寫 results/door/door_coopen_pass_close_inner_retime_v1_static.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_geom_v3 as G3                              # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_inner_v1_static.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_inner_retime_v1_static.json')
C = PI.C
VB, WB, AB, AW = 0.035255, 0.1999, 0.5, 2.0
VA, AA = 0.9999, 19.984
VQ = math.radians(0.6)                       # r_max_dps
VF = 0.0089 / 1.0                            # finger_open / close_ramp_s
DT_CTRL = 0.05
DRIFT_M, DRIFT_RAD, GRIP_TOL = 0.005, math.radians(2.0), 0.002
VMAX = np.array([VB, VB, WB] + [VA] * 6)
AMAX = np.array([AB, AB, AW] + [AA] * 6)


def body(v, yaw):
    """世界座標 (vx, vy, w, arm...) → 車體座標。"""
    o = v.copy()
    o[0] = math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
    o[1] = -math.sin(yaw) * v[0] + math.cos(yaw) * v[1]
    return o


def seg_dt(P, Q, F, k):
    d = P[k + 1] - P[k]
    yaw = 0.5 * (P[k][2] + P[k + 1][2])
    db = np.abs(body(d, yaw))
    return max(float(np.max(db / VMAX)), abs(Q[k + 1] - Q[k]) / VQ, abs(F[k + 1] - F[k]) / VF, 1e-3)


def plan(P, Q, F):
    n = len(P)
    dt = np.array([seg_dt(P, Q, F, k) for k in range(n - 1)])
    for it in range(5000):
        v = [(P[k + 1] - P[k]) / dt[k] for k in range(n - 1)]
        vq = [(Q[k + 1] - Q[k]) / dt[k] for k in range(n - 1)]
        vext = [np.zeros(9)] + v + [np.zeros(9)]
        tau = np.zeros(n)
        for j in range(n):
            dv = body(vext[j + 1] - vext[j], float(P[j][2]))
            tau[j] = float(np.max(np.abs(dv) / AMAX))
        bad = [k for k in range(n - 1) if 0.5 * (tau[k] + tau[k + 1]) > dt[k] * (1 - 1e-9)]
        if not bad:
            return dt, tau, v, vq, it
        for k in bad:
            dt[k] *= 1.1
    raise RuntimeError('時間化未收斂')


def sample(P, Q, F, dt, tau, v, vq, stages):
    n = len(P)
    tw = np.concatenate([[tau[0] / 2], tau[0] / 2 + np.cumsum(dt)])           # 接點時刻（起點前留半個混合）
    T_end = tw[-1] + tau[-1] / 2
    vext = [np.zeros(9)] + v + [np.zeros(9)]
    qext = [0.0] + vq + [0.0]
    out = []
    for t in np.arange(0.0, T_end + 1e-9, DT_CTRL):
        j = int(np.argmin(np.abs(tw - t)))
        if abs(t - tw[j]) <= tau[j] / 2 and tau[j] > 0:
            s = t - tw[j] + tau[j] / 2
            x = P[j] + vext[j] * (t - tw[j]) + (vext[j + 1] - vext[j]) * s * s / (2 * tau[j])
            q = Q[j] + qext[j] * (t - tw[j]) + (qext[j + 1] - qext[j]) * s * s / (2 * tau[j])
        else:
            k = j if t > tw[j] else j - 1                                       # 所在段 k：tw[k] ≤ t ≤ tw[k+1]
            k = min(max(k, 0), n - 2)
            x = P[k] + v[k] * (t - tw[k])
            q = Q[k] + vq[k] * (t - tw[k])
        k = int(np.searchsorted(tw, t, side='right')) - 1
        k = min(max(k, 0), n - 2)
        a = min(max((t - tw[k]) / dt[k], 0.0), 1.0)
        f = F[k] + (F[k + 1] - F[k]) * a
        out.append({'t': float(t), 'x': x, 'q': float(q), 'qf': float(f), 'stage': stages[k + 1], 'seg': k})
    return out, tw


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    src = json.load(open(SRC))
    rows = src['rows']
    P = [np.array(r['x'], float) for r in rows]
    Q = [float(r['q']) for r in rows]
    F = [float(r['qf']) for r in rows]
    stages = [r['stage'] for r in rows]
    for k in range(1, len(P)):                                                  # 偏航連續化
        P[k][2] = P[k - 1][2] + math.remainder(P[k][2] - P[k - 1][2], 2 * math.pi)
    dt, tau, v, vq, iters = plan(P, Q, F)
    S, tw = sample(P, Q, F, dt, tau, v, vq, stages)
    X = np.array([s['x'] for s in S])
    Qs = np.array([s['q'] for s in S])
    # 20 Hz 有號差分核
    V = np.diff(X, axis=0) / DT_CTRL
    Vb = np.array([body(V[i], float(0.5 * (X[i, 2] + X[i + 1, 2]))) for i in range(len(V))])
    A = np.diff(V, axis=0) / DT_CTRL
    Ab = np.array([body(A[i], float(X[i + 1, 2])) for i in range(len(A))])
    vr = np.max(np.abs(Vb) / VMAX, axis=0)
    ar = np.max(np.abs(Ab) / AMAX, axis=0)
    qd = np.abs(np.diff(Qs)) / DT_CTRL
    qdd = np.abs(np.diff(Qs, 2)) / DT_CTRL ** 2
    # 抓持段：G 最後一列（閉爪完成）之後到 R_open 第一列（開始張爪）之前
    i_g_end = max(i for i, s in enumerate(stages) if s == 'G')
    i_ro = min(i for i, s in enumerate(stages) if s == 'R_open')
    t_hold0, t_hold1 = tw[i_g_end], tw[i_ro]
    THG = PI.THG
    bad, err_p, err_r, mins = [], 0.0, 0.0, {'nonfinger': math.inf, 'self': math.inf, 'base': math.inf}
    for i, s in enumerate(S):
        if i % 200 == 0:
            TJ.cool()
        ok, why, info = PI.check(s['x'], s['q'], s['qf'], s['stage'])
        if not ok:
            bad.append({'t': s['t'], 'stage': s['stage'], 'why': why})
        else:
            for k_ in mins:
                mins[k_] = min(mins[k_], float(info[k_][0]))
        if t_hold0 <= s['t'] <= t_hold1:
            Th = G3.handle_pose_inner(PI.D3, math.radians(s['q']))
            Tr = np.linalg.inv(Th) @ C.K.fk([float(z) for z in s['x']], 'link_tcp')
            E = np.linalg.inv(THG) @ Tr
            ep = float(np.linalg.norm(Th[:3, :3] @ E[:3, 3]))
            er = float(math.acos(max(-1.0, min(1.0, (np.trace(E[:3, :3]) - 1) / 2))))
            s['grip_err'] = (ep, er)
            err_p, err_r = max(err_p, ep), max(err_r, er)
        W = C.world([float(z) for z in s['x']], list(C.CL), s['qf'])
        s['ymin_body'] = float(min(min(float(Pp[:, 1].min()) for Pp in W.values()), s['x'][1] - C.RB))
    after_p = [s for s in S if s['t'] >= tw[i_g_end]]
    pre = [s for s in after_p if s['stage'] == 'P' and s['x'][1] < PI.Y_CLEAR]
    bc = next((s for s in after_p if s['x'][1] >= PI.Y_CLEAR), None)
    wc = next((s for s in after_p if s['ymin_body'] >= PI.WALL_FAR), None)
    names = ['base_x', 'base_y', 'base_yaw'] + [f'j{i + 1}' for i in range(6)]
    res = {'schema': 'door_coopen_pass_close_inner_retime/1', 'source': os.path.relpath(SRC, HERE), 'scenario': 'static',
           'method': 'LSPB 經過點版；20 Hz 取樣全部重核', 'limits': {'v': dict(zip(names, VMAX.tolist())), 'a': dict(zip(names, AMAX.tolist())),
                                                              'door_rate_deg_s': 0.6, 'door_abort_rate_rad_s': 0.05, 'finger_m_s': VF,
                                                              'grip_drift_m': DRIFT_M, 'grip_drift_rad': DRIFT_RAD, 'grip_tol_m': GRIP_TOL},
           'iters': iters, 't_end_s': S[-1]['t'], 'n_samples': len(S),
           'stage_t_start': {st: next((float(tw[i - 1]) for i, s in enumerate(stages) if s == st and i > 0), None)
                             for st in ('M', 'A', 'I', 'G', 'P', 'R_open', 'R_slide', 'R_stow')},
           'max_ratio_v': dict(zip(names, vr.tolist())), 'max_ratio_a': dict(zip(names, ar.tolist())),
           'door_rate_max_deg_s': float(np.degrees(qd.max())), 'door_acc_max_deg_s2_report_only': float(np.degrees(qdd.max())),
           'clearance_fail': bad[:30], 'n_clearance_fail': len(bad), 'min_clear_m': mins,
           'grip_err_max': {'pos_m': err_p, 'rot_rad': err_r, 'ok_drift': bool(err_p <= DRIFT_M and err_r <= DRIFT_RAD), 'within_grip_tol': bool(err_p <= GRIP_TOL),
                            'window_s': [float(t_hold0), float(t_hold1)]},
           'overlap_pass_close': {'q_at_P_start': next(s['q'] for s in S if s['stage'] == 'P'),
                                  'closing_deg_before_base_clear': (max(s['q'] for s in pre) - min(s['q'] for s in pre)) if pre else 0.0,
                                  'duration_overlap_s': (pre[-1]['t'] - pre[0]['t']) if len(pre) > 1 else 0.0,
                                  'base_clear': {'t_s': bc['t'], 'stage': bc['stage'], 'q_deg': bc['q']} if bc else None},
           'whole_robot_clear': {'t_s': wc['t'] if wc else None, 'stage': wc['stage'] if wc else None, 'final_ymin_body_m': S[-1]['ymin_body'],
                                 'final_ok': bool(S[-1]['ymin_body'] >= PI.WALL_FAR)},
           'reference_20hz': {'cols': ['t', 'x9', 'q_door_deg', 'q_finger', 'stage'],
                              'rows': [[s['t'], [float(z) for z in s['x']], s['q'], s['qf'], s['stage']] for s in S]},
           'wall_s': None}
    res['wall_s'] = time.time() - t0w
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    print(json.dumps({k: v_ for k, v_ in res.items() if k != 'reference_20hz'}, ensure_ascii=False, default=float)[:3500], flush=True)


if __name__ == '__main__':
    main()
