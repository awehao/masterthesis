#!/usr/bin/env python3
"""窗口協調候選 bottle_w1_dyn2b_g1 的輪級後處理診斷（Codex 20261101_095306 第 1 點；只用既有紀錄重解，不開實跑）。
停止前約 0.5 s 的每個控制週期，用 compact 紀錄（s1_cbf.npz）與候選相同的設定（CLOSE／LIFT 時加 base_box）重解耦合問題，列出：
重現誤差（重解回傳 vs 實際命令）、QP 原始底盤命令（xi_ref[0]＋δξ_0）、後處理命令與動作（as_is／wheel_scaled／wheel_projected）、λ、
原始命令的輪速與輪加速度原單位殘差（rad/s、rad/s²；正值＝違反）、前一筆命令 xi_prev 本身是否已超出輪速上限。
QP 列與後處理使用的 xi_prev、dt、輪參數在 gmpc_wb.solve 內是同一個變數／同一個 cfg，這裡另外逐項比對數值。
    python3 evaluation/bottle_w1_box_wheel_diag.py [bottle_w1_dyn2b_g1] [t0] [t1]
"""
import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'bottle_w1_dyn2b_g1'
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 21.0
    t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 21.55
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    win = rep['window']
    box = ((float(win[0]), float(win[1])), (float(B.Y_TUBE[0]), float(B.Y_TUBE[1])))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    K = C.S0_K()
    cfg = C.make_cfg(ident)
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)
    W = GW.wheel_matrix(nav)
    r_w = nav.wheel_radius
    out = []
    for i in np.flatnonzero((Z['t'] >= t0 - 1e-9) & (Z['t'] <= t1 + 1e-9)):
        ph = BR.PHASES[int(Z['phase'][i])]
        on = ph in ('CLOSE', 'LIFT')
        m = GW.GMPC(dataclasses.replace(nav, base_box=box) if on else nav)
        cfg.arm_model.bias = Z['bias'][i]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
        xp = np.asarray(Z['u_prev'][i][:3], float)
        kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=xp, arm=arm)
        obs = json.loads(str(Z['obs_json'][i]))
        r = m.solve(obstacles=obs, **kw)
        # 耦合內層（不經 hold／退回）：直接看耦合解的原始底盤命令與後處理
        ri = m.solve(obstacles=obs, **dict(kw, arm=dict(arm, _inner=True)))
        raw = np.asarray(Z['xi_ref'][i][0], float) + np.asarray(ri.delta_xi_all, float).reshape(-1, 3)[0]
        w_raw = W @ raw / r_w
        a_raw = W @ (raw - xp) / (r_w * nav.dt)
        w_prev = W @ xp / r_w
        fit, lam = GW.fit_to_wheels(np.clip(raw, nav.u_min, nav.u_max), xp, nav, nav.dt)
        row = dict(t=round(float(Z['t'][i]), 3), phase=ph, box=on, reproduce=float(np.abs(np.asarray(r.u_opt, float) - Z['u'][i][:3]).max()),
                   outer_action=str(r.accept_action), outer_fallback=(r.arm_check or {}).get('fallback'), inner_status=str(ri.status), inner_action=str(ri.accept_action),
                   inner_scale=None if ri.accept_scale is None else float(ri.accept_scale), raw=raw.round(5).tolist(), inner_u=np.asarray(ri.u_opt, float).round(5).tolist(),
                   lam_recomputed=lam, w_resid=float(np.max(np.abs(w_raw)) - nav.wheel_w_max), a_resid=float(np.max(np.abs(a_raw)) - nav.wheel_a_max),
                   prev_w_resid=float(np.max(np.abs(w_prev)) - nav.wheel_w_max), u_clip_changed=bool(np.any(np.abs(np.clip(raw, nav.u_min, nav.u_max) - raw) > 0)))
        out.append(row)
        print(f"{row['t']:6.2f} {ph:6s} 窗口 {int(on)} 重現 {row['reproduce']:.1e} 外層 {row['outer_action']}/{row['outer_fallback']} 內層 {row['inner_status']}/{row['inner_action']} λ {row['inner_scale']} "
              f"原始 {row['raw']} 輪速殘差 {row['w_resid']:+.2e} 輪加速度殘差 {row['a_resid']:+.2e} 前一筆輪速殘差 {row['prev_w_resid']:+.2e} 速度框夾 {row['u_clip_changed']}")
    print(f"參數：dt {nav.dt}、輪半徑 {r_w}、ω_max {nav.wheel_w_max}、α_max {nav.wheel_a_max}、輪距 L {nav.wheel_base_L}、fit_mode {getattr(nav, 'wheel_fit_mode', None)}、enforce {nav.wheel_enforce_output}")
    json.dump(dict(run=run, box=box, rows=out, params=dict(dt=nav.dt, r=r_w, w_max=nav.wheel_w_max, a_max=nav.wheel_a_max, L=nav.wheel_base_L)),
              open(os.path.join(HERE, 'results', 'bottle', f'w1_box_wheel_diag_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
