#!/usr/bin/env python3
"""窗口協調修正版 bottle_w1_dyn2b2_g1 最後失敗拍的短分析（Codex 20261101_122313；只用既有紀錄重解，不開實跑）。
1 不可行來源：最後失敗拍與前兩拍，用候選同一設定（base_box＋數值接受容差）重解，攔截送進 GI 的耦合 QP（P,q,A,l,u），依欄分底盤（δξ＋CBF 鬆弛）與手臂（u_a），
  本趟無三維交叉列 ⇒ 兩組硬限制可分開。用 HiGHS（scipy linprog）分別判全體、底盤列、手臂列的可行性；不可行的那組再逐一拿掉子群組（底盤：速度／加速度／輪速／輪加速度／窗口／CBF／鬆弛下界；
  手臂：速度／加速度／實測關節限位／設定點限位），看拿掉哪一群就恢復可行。全體可行而求解器說不可行 ⇒ 數值求解問題。
2 抬升誤差拆解（這幾拍的實測）：每個控制週期 TCP 位移拆成底盤貢獻與手臂貢獻，投影到「朝抬升位姿」方向；列出命令中達速度上限的關節、設定點離限位的距離。
3 停止期間窗口：分列求解器回傳（退回 hold 解的預測）、控制器處置、實際施加（在途＋減速）的底盤命令、實測底盤位置對窗口的餘裕。
    python3 evaluation/bottle_w1_box2_infeas_diag.py
"""
import dataclasses
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_box2_ctrl as BX2                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

RUN = 'bottle_w1_dyn2b2_g1'
D = os.path.join(HERE, 'runs', RUN)


def feasible(A, l, u):
    """l ≤ A x ≤ u 是否可行（x 無界）；回傳 HiGHS 狀態碼與訊息（0＝找到可行解，2＝不可行）。"""
    A = np.asarray(A, float)
    fu, fl = np.isfinite(u), np.isfinite(l)
    A_ub = np.vstack([A[fu], -A[fl]])
    b_ub = np.r_[u[fu], -l[fl]]
    r = linprog(np.zeros(A.shape[1]), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * A.shape[1], method='highs')
    return int(r.status), str(r.message)[:80]


def capture(kw, cfg):
    got = []
    orig = GW.qp_dual_active

    def spy(P, q, A, l, u, *a, **k):
        got.append((P, q, A, l, u))
        return orig(P, q, A, l, u, *a, **k)
    GW.qp_dual_active = spy
    try:
        r = GW.GMPC(cfg).solve(**kw)
    finally:
        GW.qp_dual_active = orig
    return r, got


def groups(A, l, u, nz, n_arm, N):
    """依建構順序切列群組。"""
    A = np.asarray(A.todense()) if hasattr(A, 'todense') else np.asarray(A)
    base = np.flatnonzero(np.abs(A[:, nz:]).sum(axis=1) == 0)
    arm = np.flatnonzero(np.abs(A[:, :nz]).sum(axis=1) == 0)
    mixed = np.setdiff1d(np.arange(A.shape[0]), np.union1d(base, arm))
    Nm = 3 * N
    gb, i = {}, 0
    for name, n in (('base_vel', Nm), ('base_acc', Nm), ('wheel_v', 4 * N), ('wheel_a', 4 * N), ('box', 2 * N)):
        gb[name] = base[i:i + n]
        i += n
    rest = base[i:]
    single = np.array([r for r in rest if np.count_nonzero(A[r]) == 1 and l[r] == 0.0 and np.isinf(u[r])], int)
    gb['slack_lb'] = single
    gb['cbf'] = np.setdiff1d(rest, single)
    ga, j = {}, 0
    ga['arm_vel'], j = arm[j:j + n_arm], j + n_arm
    ga['arm_acc'], j = arm[j:j + n_arm], j + n_arm
    rem = arm[j:]
    blk = rem.reshape(N, 2, 6) if len(rem) == N * 12 else None
    ga['arm_joint_lim'] = blk[:, 0, :].ravel() if blk is not None else rem
    ga['arm_setpoint_lim'] = blk[:, 1, :].ravel() if blk is not None else np.zeros(0, int)
    return A, base, arm, mixed, gb, ga


def analyze(t_want):
    Z = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    i = int(np.argmin(np.abs(Z['t'] - t_want)))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    cfgc = C.make_cfg(ident)
    cfgc.arm_model.bias = Z['bias'][i]
    arm = dict(mode='active', K=C.S0_K(), tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfgc.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
               lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
    box = ((float(rep['window'][0]), float(rep['window'][1])), (float(B.Y_TUBE[0]), float(B.Y_TUBE[1])))
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION, base_box=box, wheel_fit_boundary_tol=BX2.TOL_WHEEL)
    kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=np.asarray(Z['u_prev'][i][:3], float), arm=arm,
              obstacles=json.loads(str(Z['obs_json'][i])))
    r, got = capture(kw, nav)
    N, n_arm = nav.N, 6 * nav.N
    cpl = [g for g in got if g[2].shape[1] > 3 * N + n_arm - 1]
    out = dict(t=float(Z['t'][i]), phase=BR.PHASES[int(Z['phase'][i])], reproduce=float(np.abs(np.asarray(r.u_opt, float) - Z['u'][i][:3]).max()),
               outer=str(r.accept_action), fallback=(r.arm_check or {}).get('fallback'), coupled_status=(r.arm_check or {}).get('coupled_status'))
    if not cpl:
        out['note'] = '未攔截到耦合 QP'
        return out
    P, q, A, l, u = cpl[-1]
    nz = A.shape[1] - n_arm
    A, base, armr, mixed, gb, ga = groups(A, np.asarray(l, float), np.asarray(u, float), nz, n_arm, N)
    l, u = np.asarray(l, float), np.asarray(u, float)
    out['dims'] = dict(cols=A.shape[1], nz=nz, n_arm=n_arm, rows=A.shape[0], base_rows=len(base), arm_rows=len(armr), mixed_rows=len(mixed),
                       groups={k: len(v) for k, v in {**gb, **ga}.items()})
    out['full'] = feasible(A, l, u)
    out['base'] = feasible(A[np.ix_(base, np.arange(nz))], l[base], u[base])
    out['arm'] = feasible(A[np.ix_(armr, np.arange(nz, nz + n_arm))], l[armr], u[armr])
    drops = {}
    if out['base'][0] == 2:
        for g, rows in gb.items():
            keep = np.setdiff1d(base, rows)
            drops['base−' + g] = feasible(A[np.ix_(keep, np.arange(nz))], l[keep], u[keep])[0]
    if out['arm'][0] == 2:
        for g, rows in ga.items():
            keep = np.setdiff1d(armr, rows)
            drops['arm−' + g] = feasible(A[np.ix_(keep, np.arange(nz, nz + n_arm))], l[keep], u[keep])[0]
            if g in ('arm_joint_lim', 'arm_setpoint_lim') and len(rows):          # 哪一步、哪個關節的限位最緊
                v = []
                for rr in rows:
                    v.append(float(min(u[rr], -l[rr]) if np.isfinite(u[rr]) and np.isfinite(l[rr]) else np.nan))
    out['drop_one'] = drops
    # 手臂限位的初值餘裕（第一步，未加 u 的部分 c＝實測／設定點預測）：lo_j − c ≤ M u ≤ hi_j − c
    lj = np.asarray(arm['lo'], float) + arm['margin']
    hj = np.asarray(arm['hi'], float) - arm['margin']
    out['joint_now'] = dict(q=np.round(Z['qp'][i][3:], 4).tolist(), s=np.round(Z['sp'][i], 4).tolist(), lo=np.round(lj, 4).tolist(), hi=np.round(hj, 4).tolist(),
                            margin_q_min=float(np.min(np.minimum(Z['qp'][i][3:] - lj, hj - Z['qp'][i][3:]))), margin_s_min=float(np.min(np.minimum(Z['sp'][i] - lj, hj - Z['sp'][i]))),
                            u_arm_prev=np.round(Z['u_prev'][i][3:], 4).tolist(), v_arm=float(arm['v_arm']), a_arm=float(arm['a_arm']))
    return out


def lift_and_window():
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    K = C.S0_K()
    t = c['t']
    Q = np.column_stack([c[k] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
    rows = []
    t_lift = rep['t_lift_start']
    # 抬升目標：抬升開始時的抓取位姿上抬 lift_m（同 BottleTask.T_lift；用報告事件時刻的實測 TCP 近似抓取位姿，只供方向）
    i0 = int(np.argmin(np.abs(t - t_lift)))
    T0 = K.fk(Q[i0], 'link_tcp')
    goal = T0[:3, 3] + np.array([0.0, 0.0, 0.05])
    for tt in np.arange(21.6, 22.25, 0.05):
        a, b = int(np.argmin(np.abs(t - tt))), int(np.argmin(np.abs(t - (tt + 0.05))))
        pa = K.fk(Q[a], 'link_tcp')[:3, 3]
        pb = K.fk(Q[b], 'link_tcp')[:3, 3]
        qb_base = np.r_[Q[b][:3], Q[a][3:]]
        qb_arm = np.r_[Q[a][:3], Q[b][3:]]
        d_base = K.fk(qb_base, 'link_tcp')[:3, 3] - pa
        d_arm = K.fk(qb_arm, 'link_tcp')[:3, 3] - pa
        e = goal - pa
        en = e / max(np.linalg.norm(e), 1e-9)
        sat = [j + 1 for j in range(6) if abs(c[f's{j + 1}'][b] - c[f's{j + 1}'][a]) / 0.05 >= 0.99 * B.W_ARM['v_arm']]
        rows.append(dict(t=round(float(tt), 2), err_mm=round(float(np.linalg.norm(e)) * 1e3, 2), toward_goal_mm=dict(total=round(float((pb - pa) @ en) * 1e3, 3), base=round(float(d_base @ en) * 1e3, 3),
                         arm=round(float(d_arm @ en) * 1e3, 3)), base_xy_mm=np.round((Q[b][:2] - Q[a][:2]) * 1e3, 2).tolist(), sat_joints=sat))
    win = rep['window']
    k = np.flatnonzero(t >= 22.1)
    stop = dict(x_we=win[1], y_tube=list(B.Y_TUBE),
                measured=[dict(t=round(float(t[j]), 2), x=round(float(c['base_x'][j]), 4), y=round(float(c['base_y'][j]), 4), margin_x_mm=round((win[1] - c['base_x'][j]) * 1e3, 2),
                               cmd=[round(float(c['base_cmd_vx'][j]), 4), round(float(c['base_cmd_vy'][j]), 4)]) for j in k[::10][:30]])
    stop['max_x_after'] = float(c['base_x'][k].max())
    stop['min_margin_x_mm'] = float((win[1] - c['base_x'][k]).min() * 1e3)
    stop['y_range_after'] = [float(c['base_y'][k].min()), float(c['base_y'][k].max())]
    return rows, stop


def main():
    res = dict(run=RUN, ticks=[analyze(tt) for tt in (22.10, 22.15, 22.20)])
    for o in res['ticks']:
        print(f"{o['t']:.2f} {o['phase']}：重現 {o['reproduce']:.1e}；外層 {o['outer']}／{o['fallback']}；耦合 {o.get('coupled_status')}；全體 {o.get('full')}；底盤 {o.get('base')}；手臂 {o.get('arm')}；拿掉一組 {o.get('drop_one')}")
        print(f"      維度 {o.get('dims')}；關節 {o.get('joint_now')}")
    res['lift'], res['stop_window'] = lift_and_window()
    for r in res['lift']:
        print('   抬升', r)
    sw = res['stop_window']
    print(f"   停止期間：x_we {sw['x_we']}，停止後實測最大 x {sw['max_x_after']:.4f}（最小餘裕 {sw['min_margin_x_mm']:.2f} mm），y 範圍 {np.round(sw['y_range_after'], 4).tolist()}")
    for m in sw['measured'][:12]:
        print('     ', m)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', f'w1_box2_infeas_diag_{RUN}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
