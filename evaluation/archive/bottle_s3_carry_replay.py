#!/usr/bin/env python3
"""P1（Codex 20261101_033920）：s3_1 持瓶段車速鋸齒的同輸入重解，只定位、不改參數。
s1_cbf.npz 沒存三維約束輸入：以當時版本（gmpc_wb、bottle_s1_ctrl 雜湊須與 versions.sha256 相同）重建——夾爪與連桿包圍盒固定，
瓶包圍盒用抓持成立那一步的量測重建 T_rel（同 S0Judge：rel_pose(T_tcp, T_bottle)）。先重現原命令，重現不了就只報限制。
反事實（同一輸入、逐週期獨立）：拿掉三維約束、拿掉底盤 CBF、兩者都拿掉。只能說局部影響，不能證明閉迴路振盪原因。
S_LEAD：由 X_ref[0] 反推 s_ref，對照當步實測 x＋S_LEAD，報夾住比例（描述，不是根因）。
    python3 evaluation/bottle_s3_carry_replay.py bottle_s3_1
"""
import dataclasses
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0_logic as L                             # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

CHECK = ('src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py', 'evaluation/bottle_s1_ctrl.py', 'evaluation/bottle_s0b_ctrl.py', 'evaluation/bottle_s0_ctrl.py')


def flips(v):
    d = np.diff(v)
    return int((np.sign(d[1:]) * np.sign(d[:-1]) < 0).sum())


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    ver = {ln.split()[1]: ln.split()[0] for ln in open(os.path.join(D, 'versions.sha256')) if ln.strip()}
    bad = [f for f in CHECK if hashlib.sha256(open(os.path.join(WS, f), 'rb').read()).hexdigest() != ver.get(f)]
    if bad:
        print('目前檔案與實跑版本不同，不能重建：', bad)
        return 2
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    S = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): S['rows'][:, i] for i, k in enumerate(S['cols'])}
    Z = np.load(os.path.join(D, 's1_cbf.npz'))
    K = C.S0_K()
    # 抓持成立那一步的 T_rel（同 S0Judge）
    i_g = int(np.argmin(np.abs(c['t'] - rep['t_grasp'])))
    q9 = np.array([c[k][i_g] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
    qb = np.array([c['bottle_qw'][i_g], c['bottle_qx'][i_g], c['bottle_qy'][i_g], c['bottle_qz'][i_g]])
    w, x, y, z = qb
    Rb = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    Tb = np.eye(4)
    Tb[:3, :3], Tb[:3, 3] = Rb, [c['bottle_base_x'][i_g], c['bottle_base_y'][i_g], c['bottle_base_z'][i_g]]
    T_rel = L.rel_pose(K.fk(q9, 'link_tcp'), Tb)
    rb = rep['readback']
    box = [float(v) for v in rb['s2_box'].split(',')]
    arg = open(os.path.join(D, 'run.log'), errors='replace').readline()                 # 實跑參數（readback 沒存 s2_links）
    links = tuple(arg.split('--s2-links ')[1].split()[0].rstrip('］').split(',')) if '--s2-links ' in arg else ()
    ctl = SC.make(path=SC.PATH_P2, body_boxes=[(np.array(box[:3]), np.array(box[3:]))], body_links=tuple(sorted(links)), **SC.FAST)
    ctl.q_arm_ready, ctl._grip_corners, ctl.judge.T_rel = np.array(rb['q_ready'], float), None, T_rel
    extra = ctl._arm_extra()
    cfg = C.make_cfg(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))))
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)
    m = GW.GMPC(nav)
    ph = Z['phase'].astype(int)
    idx = [i for i in range(len(ph)) if BR.PHASES[ph[i]] == 'CARRY']
    out = {k: [] for k in ('full', 'no_body', 'no_cbf', 'none')}
    rep_max, lead, tt = 0.0, [], []
    for i in idx:
        cfg.arm_model.bias = Z['bias'][i]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i],
                   w_p=float(Z['w_p'][i]), lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
        kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=Z['u_prev'][i][:3])
        ob = json.loads(str(Z['obs_json'][i]))
        for k, a_, o_ in (('full', dict(arm, **extra), ob), ('no_body', arm, ob), ('no_cbf', dict(arm, **extra), None), ('none', arm, None)):
            r = m.solve(obstacles=o_, arm=a_, **kw)
            u = np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float) if r.u_arm_seq is not None else np.full(6, np.nan)]
            out[k].append(u)
            if k == 'full':
                rep_max = max(rep_max, float(np.abs(u - Z['u'][i]).max()))
        t = float(Z['t'][i])
        j = int(np.argmin(np.abs(c['t'] - t)))
        s_ref = float(Z['X_ref'][i][0][0, 2]) - float(c['v_b'][j]) * C.D_STEPS * C.PHYS
        lead.append(s_ref - float(c['base_x'][j]))
        tt.append(t)
    res = dict(run=run, n=len(idx), t=[tt[0], tt[-1]], reproduce_max=rep_max, T_rel_from_step=float(c['t'][i_g]), body_links=sorted(links))
    tt, lead = np.array(tt), np.array(lead)
    win = (tt >= 25.2) & (tt <= 34.3)                     # 先前觀察鋸齒的同一段（持瓶行進、未進出口區）
    res['window'] = [25.2, 34.3]
    for k, v in out.items():
        v = np.array(v)
        sp = np.hypot(v[:, 0], v[:, 1])
        st = lambda s_: dict(speed_min=round(float(s_.min()), 4), speed_max=round(float(s_.max()), 4), speed_mean=round(float(s_.mean()), 4), flips=flips(s_), big_steps=int((np.abs(np.diff(s_)) > 0.03).sum()), n=int(s_.size))   # noqa: E731
        res[k] = dict(all=st(sp), win=st(sp[win]))
    cl = np.abs(lead - B.S_LEAD) < 1e-6
    res['s_lead'] = dict(S_LEAD=B.S_LEAD, clamped_frac_all=round(float(cl.mean()), 3), clamped_frac_win=round(float(cl[win].mean()), 3), lead_min=round(float(lead.min()), 4), lead_max=round(float(lead.max()), 4))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 's3_carry_replay.json'), 'w'), ensure_ascii=False, indent=1)
    return 0 if rep_max < 1e-6 else 1


if __name__ == '__main__':
    sys.exit(main())
