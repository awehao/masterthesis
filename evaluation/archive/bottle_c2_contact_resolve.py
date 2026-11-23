#!/usr/bin/env python3
"""C2 接觸段協調候選的關鍵拍重解核對（Codex 20261102_070114：B1 v3 後，用 C2 既有紀錄少量重解，通過才申請一趟；不實跑、非物理）。
候選：接觸段（INSERT 起到離桌成立）把底盤逐軸加速度上限 a_max 收緊為 A_CONTACT，放在同一個 QP 內與動態／靜態 CBF 共同求解
（GMPCConfig.a_max 就是 QP 的 |u_k − u_{k−1}| ≤ a_max·dt 列，不是求解後夾命令）；QP 無解或用到 slack 時照既有流程（退回／停止），不另放寬。
來源：runs/bottle_motm_multi_c2_1/s1_cbf.npz（每控制拍的 QP 輸入：量測構型、設定點、上一命令、參考窗、TCP 參考、障礙清單）。
每拍三種解：
  orig     原 nav 設定（a_max＝A_BASE）＝重現核對（對紀錄命令的最大差；紀錄的 QP 另含 Shield 第一拍列與持瓶包圍盒列，本工具未重建，差異照報）；
  contact  a_max＝A_CONTACT，上一命令＝紀錄值（逐拍獨立）；
  chain    a_max＝A_CONTACT，上一命令＝前一拍 chain 解（構型與參考仍用紀錄值；串接近似，不是閉迴路）。
報：底盤命令、相鄰控制拍逐軸變化率、狀態、slack eps0、動態障礙 h（k＝0）、無 slack 殘差、輸出處置（accept_action）。
    python3 evaluation/bottle_c2_contact_resolve.py
註（Codex 20261102_213316）：本工具命令先取四位才算重現差與串接，5e-5 是四捨五入造成的（全精度 1e-11）；串接不更新構型。後續核對改用 bottle_c2_contact_tube_check.py，本工具與舊結果原樣保留。
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

RUN = 'bottle_motm_multi_c2_1'
WIN = (20.5, 21.3)
CANDS = dict(c03=np.array([0.3, 0.2, 0.4]), c05=np.array([0.5, 0.35, 0.7]))
OUT = os.path.join(HERE, 'results', 'bottle', 'c2_contact_resolve.json')


def main():
    Z = np.load(os.path.join(HERE, 'runs', RUN, 's1_cbf.npz'))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    K = C.S0_K()
    cfg = C.make_cfg(ident)
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)
    solvers = dict(orig=GW.GMPC(nav), **{k: GW.GMPC(dataclasses.replace(nav, a_max=v)) for k, v in CANDS.items()})
    t = Z['t']
    idx = [i for i in range(len(t)) if WIN[0] <= t[i] <= WIN[1]]
    rows, prev_chain = [], {k: None for k in CANDS}
    for i in idx:
        cfg.arm_model.bias = Z['bias'][i]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
        ob = json.loads(str(Z['obs_json'][i]))
        base_kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], obstacles=ob)
        row = dict(t=round(float(t[i]), 3), phase=BR.PHASES[int(Z['phase'][i])], n_dyn=sum(1 for o in ob if not o.get('static')), u_log=np.round(Z['u'][i][:3], 4).tolist(),
                   u_prev_log=np.round(Z['u_prev'][i][:3], 4).tolist())

        def sol(name, xi_prev):
            r = solvers[name].solve(xi_prev=np.asarray(xi_prev, float), arm=dict(arm), **base_kw)
            return dict(u=np.round(np.asarray(r.u_opt, float), 4).tolist(), status=r.status, eps0=round(float(r.eps0), 6), min_h_dyn=round(float(r.min_h_dynamic), 4) if np.isfinite(r.min_h_dynamic) else None,
                        resid_noslack=round(float(r.cbf_resid_noslack), 5) if np.isfinite(r.cbf_resid_noslack) else None, accept=r.accept_action,
                        rate=np.round((np.asarray(r.u_opt, float) - np.asarray(xi_prev, float)) / B.DT, 3).tolist())
        row['orig'] = sol('orig', Z['u_prev'][i][:3])
        row['orig']['reproduce_err'] = round(float(np.abs(np.array(row['orig']['u']) - Z['u'][i][:3]).max()), 5)
        for k in CANDS:
            row[k] = sol(k, Z['u_prev'][i][:3])
            pc = prev_chain[k] if prev_chain[k] is not None else Z['u_prev'][i][:3]
            row[k + '_chain'] = sol(k, pc)
            prev_chain[k] = np.array(row[k + '_chain']['u'])
        rows.append(row)
    summ = {}
    for k in ['orig'] + list(CANDS) + [c + '_chain' for c in CANDS]:
        rr = [r_[k] for r_ in rows]
        summ[k] = dict(rate_abs_max=np.abs(np.array([x['rate'] for x in rr])).max(axis=0).round(3).tolist(), status=sorted({x['status'] for x in rr}),
                       eps0_max=max(x['eps0'] for x in rr), min_h_dyn=min((x['min_h_dyn'] for x in rr if x['min_h_dyn'] is not None), default=None),
                       accept=sorted({x['accept'] for x in rr}))
    summ['orig']['reproduce_err_max'] = max(r_['orig']['reproduce_err'] for r_ in rows)
    out = dict(run=RUN, window=WIN, cands={k: v.tolist() for k, v in CANDS.items()}, a_base=B.A_BASE.tolist(), rows=rows, summary=summ,
               note='重解輸入取自紀錄；未重建 Shield 第一拍列與持瓶包圍盒列（重現差照報）；chain 只串上一命令，構型／參考仍是紀錄值，不是閉迴路')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for r_ in rows:
        print(r_['t'], r_['phase'], 'n_dyn', r_['n_dyn'], 'log', r_['u_log'], '| orig', r_['orig']['u'], r_['orig']['reproduce_err'], '| c03', r_['c03']['u'], r_['c03']['eps0'], '| c03_chain', r_['c03_chain']['u'], r_['c03_chain']['status'], r_['c03_chain']['eps0'], r_['c03_chain']['min_h_dyn'])
    print(json.dumps(summ, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
