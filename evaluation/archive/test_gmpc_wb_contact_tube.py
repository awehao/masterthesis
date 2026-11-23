#!/usr/bin/env python3
"""gmpc_wb 接觸管道列（arm['contact_tube']，Codex 20261102_213316）的最小核對；資料＝C2 紀錄 20.85 s 那一拍的 QP 輸入。
1 管道極寬（e＝10 m）：列不作用，解與無列幾乎相同。
2 正常管道（e＝3 mm、前 10 步）：列數 30，線性化誤差逐軸都在 ±e 內，回傳未退回。
3 不可行走既有退回：21.00 s 紀錄狀態（預測 TCP 誤差 7.34 mm）、e＝3 mm、a_max＝A_CONTACT ⇒ coupled_not_solved，seq_consistent False。
  （20.85 s 誤差僅 0.89 mm 時連 e＝1e-5 m 都可解：k＝0 是 u_0 作用後第一個未來步，線性模型下手臂一步就補得回來；目前狀態不受列約束。）
4 底盤座標模式（tcp_rel_base）：管道不生效（無 ct 欄）。
預設 None 與 git HEAD 逐位元相同由 test_gmpc_wb_zero_clip.py 第 1 項涵蓋。
    python3 evaluation/test_gmpc_wb_contact_tube.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402


def main():
    Z = np.load(os.path.join(HERE, 'runs', 'bottle_motm_multi_c2_1', 's1_cbf.npz'))
    i0 = int(np.argmin(np.abs(Z['t'] - 20.85)))
    cfg = C.make_cfg(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))))
    K = C.S0_K()
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)

    def sol(ct, rel=False, i=i0, mpc=GW.GMPC(nav)):
        cfg.arm_model.bias = Z['bias'][i]
        ob = json.loads(str(Z['obs_json'][i]))
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=rel, qp_solver='gi', **B.W_ARM)
        if ct is not None:
            arm['contact_tube'] = ct
        return mpc.solve(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=Z['u_prev'][i][:3], obstacles=ob, arm=arm)
    r0 = sol(None)
    checks = {}
    rw = sol(dict(e=10.0, n=10))
    d = float(np.abs(np.asarray(rw.u_opt) - np.asarray(r0.u_opt)).max())
    checks['1_wide'] = (d <= 1e-6 and rw.arm_check['ct']['bind'] == 0 and rw.arm_check['ct']['rows'] == 30, f'命令差 {d:.2e}，bind {rw.arm_check["ct"]["bind"]}')
    rn = sol(dict(e=0.003, n=10))
    ct = rn.arm_check.get('ct')
    checks['2_normal'] = (ct is not None and ct['rows'] == 30 and np.abs(ct['e_lin']).max() <= 0.003 + 1e-6 and rn.arm_check.get('fallback') is None, f"狀態 {rn.status}，線性誤差最大 {np.abs(ct['e_lin']).max() * 1e3 if ct else None} mm，bind {ct['bind'] if ct else None}")
    rt = sol(dict(e=0.003, n=10), i=int(np.argmin(np.abs(Z['t'] - 21.00))), mpc=GW.GMPC(dataclasses.replace(nav, a_max=np.array([0.3, 0.2, 0.4]))))
    checks['3_infeasible'] = (rt.arm_check.get('fallback') == 'coupled_not_solved' and rt.arm_check.get('seq_consistent') is False, f"退回 {rt.arm_check.get('fallback')}（耦合狀態 {rt.arm_check.get('coupled_status')}）")
    rr = sol(dict(e=0.003, n=10), rel=True)
    checks['4_rel'] = ('ct' not in (rr.arm_check or {}), f"arm_check 含 ct：{'ct' in (rr.arm_check or {})}")
    for k, (ok, msg) in checks.items():
        print(('✓' if ok else '✗'), k, msg)
    ok = all(v[0] for v in checks.values())
    print('全過' if ok else '未過')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
