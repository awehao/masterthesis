#!/usr/bin/env python3
"""S1 實跑後的 CBF 反事實重解（Codex 20261031_215150 (3)）：讀 s1_cbf.npz，逐週期先重解確認重現實際命令，再移除 CBF 重解比較命令差。實跑中不做，避免拖慢物理。
    python3 evaluation/bottle_s1_cbf_replay.py <RUN_ID>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402
import dataclasses                                      # noqa: E402

DU_BOUND = 1e-3
OBST_DEF = [dict(x=-2.0, y=0.0, radius=0.2 * 2.0 ** 0.5, margin=0.47, static=True)]


def replay(path, n_stride=1, obst=None, ident=None):
    Z = np.load(path)
    if ident is None:
        ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    K = C.S0_K()
    cfg = C.make_cfg(ident)
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)
    m_o, m_n = GW.GMPC(nav), GW.GMPC(nav)
    obs = OBST_DEF if obst is None else obst
    du, rep, ph = [], [], {}
    idx = range(0, len(Z['t']), n_stride)
    for i in idx:
        cfg.arm_model.bias = Z['bias'][i]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
        kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=Z['u_prev'][i][:3], arm=arm)
        r_o, r_n = m_o.solve(obstacles=obs, **kw), m_n.solve(obstacles=None, **kw)
        rep.append(float(np.abs(np.asarray(r_o.u_opt, float) - Z['u'][i][:3]).max()))
        d = float(np.linalg.norm(np.asarray(r_o.u_opt, float) - np.asarray(r_n.u_opt, float)))
        du.append(d)
        if d >= DU_BOUND and int(Z['cbf_active'][i]) > 0:
            k = BR.PHASES[int(Z['phase'][i])]
            ph[k] = ph.get(k, 0) + 1
    return dict(n=len(du), reproduce_max=max(rep), n_bound=sum(ph.values()), phase_bound=ph, du=du, du_max=max(du), eps0_max=float(Z['eps0'].max()), resid_ns_min=float(np.nanmin(Z['resid_ns'][np.isfinite(Z['resid_ns'])])) if np.isfinite(Z['resid_ns']).any() else None,
                resid_s_min=float(np.nanmin(Z['resid_s'][np.isfinite(Z['resid_s'])])) if np.isfinite(Z['resid_s']).any() else None)


def main():
    p = os.path.join(HERE, 'runs', sys.argv[1], 's1_cbf.npz')
    r = replay(p)
    r.pop('du')
    print(r)
    json.dump(r, open(os.path.join(os.path.dirname(p), 's1_cbf_replay.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
