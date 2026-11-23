#!/usr/bin/env python3
"""同一 SHROW 堆疊預測組的 CBF 鬆弛短分析（Codex 20261101_134541；只讀紀錄，不實跑、不改權重／時域／追蹤器）。
eps0 定義：gmpc_wb 第 0 步 CBF 鬆弛，加在線性化連續 CBF 不等式 ḣ ≥ −α·h 上（h＝|p−p_o|²−r_eff²），單位 m²/s，不是距離。
本任務 cbf_static_slack_scale＝1、cbf_hard_k0_static＝False ⇒ 不分塊：每步一個鬆弛，動態與靜態障礙列共用。
每案取交會時窗（t_meet ±3 s）內 eps0 最大的求解紀錄（s1_cbf.npz）：
  1 同輸入重解重現（命令、eps0）；
  2 逐障礙算原解的 CBF 列殘差（無鬆弛時），指出用到鬆弛的障礙、時域步，與該障礙的估計位置／速度對照讀回真值（命令生效時刻）；
  3 必要性：同輸入、同限制（速度、加速度、輪級列、CBF 列不帶鬆弛）解可行性 LP；可行 ⇒ 鬆弛是代價取捨，不可行 ⇒ 維持可行所必需。
     第一拍 Shield 一致列：第一案有實際掃描紀錄（scan_log）一併加入；第二案無掃描紀錄，另以不加此列的結果陳述（其可行集只會更大）。
    python3 evaluation/bottle_w1_sr_slack_diag.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import test_bottle_w1_box2 as TB2                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402

RUNS = [('w1_dyn_pass1', 'bottle_w1_dyn1sr_g1'), ('w1_dyn_overtake1', 'bottle_w1_dyn2sr_g1')]


def u0_rows_at(run, t):
    p = os.path.join(HERE, 'runs', run, 'scan_log.npz')
    if not os.path.exists(p):
        return None
    S = np.load(p)
    k = np.flatnonzero(S['t'] <= t + 1e-9)
    if not len(k):
        return None
    i = int(k[-1])
    sh = SR.make_shield()
    pts = SR.scan_points(sh, S['ranges'][i])
    A, b, info = SR.shield_rows(sh, pts, (t - float(S['t'][i])) + SR.AGE_PAD)
    return A, b, dict(info, t_scan=float(S['t'][i]))


def feas(cfg, kw, u0=None):
    """可行性 LP：min 0 s.t. 速度／加速度（含輪級）列、CBF 列（無鬆弛）、第一拍列（若有）。"""
    N = cfg.N
    xi = np.asarray(kw['xi_ref_win'], float)
    A_c, l_c, u_c = GW._build_constraints(cfg, xi, np.asarray(kw['xi_prev'], float))
    A_c = A_c.toarray()
    A_b, l_b, _, _ = GW._build_cbf_horizon(cfg, kw['X_now'], kw['X_ref_win'], xi, kw['obstacles'], N)
    A_b = A_b.toarray()[:, :3 * N]
    G, h = [], []
    fin_u, fin_l = np.isfinite(u_c), np.isfinite(l_c)
    G += [A_c[fin_u], -A_c[fin_l], -A_b]
    h += [u_c[fin_u], -l_c[fin_l], -l_b]
    if u0 is not None:
        Au = np.zeros((len(u0[1]), 3 * N))
        Au[:, :3] = u0[0]
        G.append(Au)
        h.append(u0[1] - u0[0] @ xi[0])
    res = linprog(np.zeros(3 * N), A_ub=np.vstack(G), b_ub=np.concatenate(h), bounds=[(None, None)] * (3 * N), method='highs')
    return res.status == 0, int(res.status), str(res.message)[:80]


def per_obstacle(cfg, kw, delta):
    N = cfg.N
    xi = np.asarray(kw['xi_ref_win'], float)
    out = []
    for i, o in enumerate(kw['obstacles']):
        A_b, l_b, _, _ = GW._build_cbf_horizon(cfg, kw['X_now'], kw['X_ref_win'], xi, [o], N)
        if A_b.shape[0] == 0:
            continue
        r = A_b.toarray()[:, :3 * N] @ np.asarray(delta, float).ravel() - l_b
        out.append(dict(i=i, static=bool(o.get('static', 'margin' in o)), x=o['x'], y=o['y'], vx=o.get('vx', 0.0), vy=o.get('vy', 0.0), radius=o['radius'], margin=o.get('margin'),
                        resid_min=float(r.min()), n_neg=int((r < -1e-9).sum()), n_rows=int(len(r))))
    return out


def main():
    cfg = TB2.nav(None, 0.0)
    res = dict(eps0_def='第 0 步 CBF 鬆弛，加在 ḣ ≥ −α·h（h＝|p−p_o|²−r_eff²）上，單位 m²/s；本任務動靜障礙共用（不分塊）', cbf_alpha=cfg.cbf_alpha, slack_weight=cfg.cbf_slack_weight, eps0_scale=cfg.cbf_eps0_scale)
    for case, run in RUNS:
        CA = DC.CASES[case]
        Zc = np.load(os.path.join(HERE, 'runs', run, 's1_cbf.npz'), allow_pickle=True)
        m = (Zc['t'] >= CA['t_meet'] - 3) & (Zc['t'] <= CA['t_meet'] + 3)
        idx = np.flatnonzero(m)
        i = int(idx[np.argmax(Zc['eps0'][idx])])
        t = float(Zc['t'][i])
        TB2.RUN = run
        kw, _, tt, ph = TB2.record(t)
        u0 = u0_rows_at(run, t)                                         # 實跑時這一拍有第一拍列（有掃描紀錄者重建同一組列）
        u0r = None if u0 is None else u0[:2]
        r0 = GW.GMPC(cfg).solve(**kw, u0_rows=u0r)
        u_rec = np.asarray(Zc['u'][i][:3], float)
        rep_err = float(np.abs(np.asarray(r0.u_opt) - u_rec).max())
        r_in = GW.GMPC(cfg).solve(**dict(kw, arm=dict(kw['arm'], _inner=True)), u0_rows=u0r)
        delta = np.asarray(r_in.delta_xi_all, float)
        po = per_obstacle(cfg, kw, delta)
        used = [o for o in po if o['n_neg'] > 0]
        Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
        col = {str(k): Z['rows'][:, j] for j, k in enumerate(Z['cols'])}
        t_eff = t + B.C0.D_STEPS * B.C0.PHYS if hasattr(B, 'C0') else t + 0.08
        k = int(np.argmin(np.abs(col['t'] - t_eff)))
        truth = dict(x=float(col['dyn_x'][k]), y=float(col['dyn_y'][k]), vx=float(col['dyn_vx'][k]), vy=float(col['dyn_vy'][k]))
        dyn_est = [o for o in po if not o['static']]
        f_no, st_no, msg_no = feas(cfg, kw)
        f_u0 = feas(cfg, kw, u0[:2]) if u0 is not None else None
        res[case] = dict(run=run, t=t, phase=ph, eps0_rec=float(Zc['eps0'][i]), eps0_resolve=float(r0.eps0), resolve_cmd_err=rep_err, used=used, dyn_est=dyn_est, truth_at_t_eff=truth,
                         feasible_no_slack=f_no, lp_status=st_no, lp_msg=msg_no, feasible_with_u0=None if f_u0 is None else f_u0[0], u0_info=None if u0 is None else u0[2])
        print(f"[{case}] {run} 最大鬆弛 t {t:.2f}（{ph}）：eps0 紀錄 {Zc['eps0'][i]:.3f}／重解 {r0.eps0:.3f} m²/s，命令重現差 {rep_err:.1e}（重解含第一拍列：{u0 is not None}）")
        for o in used:
            print(f"   用到鬆弛：{'靜態' if o['static'] else '動態'}點 ({o['x']:.3f},{o['y']:.3f}) v ({o['vx']:.3f},{o['vy']:.3f}) r {o['radius']:.3f} 邊距 {o['margin']}；殘差最小 {o['resid_min']:.3f}，負列 {o['n_neg']}/{o['n_rows']}")
        for o in dyn_est:
            print(f"   動態估計 ({o['x']:.3f},{o['y']:.3f}) v ({o['vx']:.3f},{o['vy']:.3f}) r {o['radius']:.3f}；真值（生效時刻 {t_eff:.2f}）({truth['x']:.3f},{truth['y']:.3f}) v ({truth['vx']:.3f},{truth['vy']:.3f})")
        print(f"   無鬆弛可行性 LP（速度、加速度、輪級、CBF）：{'可行 ⇒ 鬆弛是代價取捨' if f_no else '不可行 ⇒ 維持可行所必需'}（{msg_no}）；"
              f"加第一拍列：{None if f_u0 is None else ('可行' if f_u0[0] else '不可行')}{'' if u0 is not None else '（第二案無掃描紀錄）'}")
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_sr_slack_diag.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
