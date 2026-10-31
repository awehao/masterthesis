#!/usr/bin/env python3
"""等價效能核對：對 bottle_s1_4 的逐週期紀錄重解，比對底盤與手臂命令（等價性），並量求解耗時。不開 Isaac。
    python3 evaluation/bottle_s1_perf_check.py [RUN_ID]
"""
import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'bottle_s1_4'
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_cbf.npz'))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    K, cfg = C.S0_K(), C.make_cfg(ident)
    m = GW.GMPC(dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION))
    db, da, T, asm, gi = [], [], [], [], []
    for i in range(len(Z['t'])):
        cfg.arm_model.bias = Z['bias'][i]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
        obs = json.loads(str(Z['obs_json'][i]))
        t0 = time.perf_counter()
        r = m.solve(obstacles=obs, X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=Z['u_prev'][i][:3], arm=arm)
        T.append((time.perf_counter() - t0) * 1e3)
        db.append(float(np.abs(np.asarray(r.u_opt, float) - Z['u'][i][:3]).max()))
        da.append(float(np.abs(r.u_arm_seq[0] - Z['u'][i][3:]).max()))
        q_ = (r.arm_check or {}).get('qp') or {}
        asm.append(q_.get('asm_ms') or 0.0)
        gi.append(q_.get('gi_ms') or 0.0)
    T = np.array(T)
    print(f'{len(T)} 週期：底盤命令差峰 {max(db):.2e}、手臂命令差峰 {max(da):.2e}')
    print(f'solve ms p50 {np.percentile(T, 50):.1f} p95 {np.percentile(T, 95):.1f} p99 {np.percentile(T, 99):.1f} max {T.max():.1f}；>50 ms {(T > 50).sum()}；組裝 p50 {np.percentile(asm, 50):.1f} p95 {np.percentile(asm, 95):.1f}；GI p50 {np.percentile(gi, 50):.1f} p95 {np.percentile(gi, 95):.1f} max {np.max(gi):.1f}')
    json.dump(dict(run=run, n=len(T), base_max=max(db), arm_max=max(da), ms=dict(p50=float(np.percentile(T, 50)), p95=float(np.percentile(T, 95)), p99=float(np.percentile(T, 99)), max=float(T.max()))), open(os.path.join(HERE, 'results', 'bottle', 's1_perf_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
