#!/usr/bin/env python3
"""INSERT 段停止核對對齊（Codex 20261111_042529 第 1 項）：離線近似（設定點 s＝當下構型、在途命令＝當拍命令重複）得 4.5 mm，
物理線上預測 6.8–7.6 mm；本工具用實錄的實際狀態重算，查明差異來源。
資料：runs/<RUN>/s1_steps.npz（每物理步 0.01 s：q、手臂設定點 s1–s6、底盤命令、瓶實測位姿）、s1_report.json（相位事件、線上 clear_bottle）。
每個控制拍（INSERT 起到閉爪請求）：
  A 實錄對齊：q、s 取實錄；在途命令＝之後 D_STEPS 物理步實際生效的命令（手臂＝s 差分／PHYS、底盤＝實錄命令）；u_last＝最後一筆。
  B 離線近似（tpas_offline_check 的作法）：s＝q、在途＝當拍命令（由 A 的 u_last）重複。
  兩者都用既有 bottle_w1_compactcarry21_ctrl.stop_traj 跑 extra 0／1，核非手指連桿對實錄瓶圓柱（同 bottle_s1_ctrl._bottle_clear）。
另列：實測構型本身的非手指對瓶（d_meas）。
    python3 evaluation/insert_stop_align.py RUN OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_s1_geom as G  # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

CFG = C0.make_cfg(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))))


def bottle_T(c, i):
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([c['bottle_qx'][i], c['bottle_qy'][i], c['bottle_qz'][i], c['bottle_qw'][i]]).as_matrix()
    T[:3, 3] = [c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]]
    return T


def clear_nf(q9, Tb):
    Pl = np.vstack([P for n_, P in G.link_world(q9, G.Q_OPEN).items() if n_ not in G.C.FINGERS])
    Pb = (Tb[:3, :3].T @ (Pl - Tb[:3, 3]).T).T
    return float(G.cyl_dist(Pb, 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min())


def main():
    run, out = sys.argv[1], sys.argv[2]
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    t_ins = next(e['t'] for e in r['events'] if e['what'] == 'phase' and e.get('to') == 'INSERT')
    t_cl = next(e['t'] for e in r['events'] if e['what'] in ('close_request',) or (e['what'] == 'phase' and e.get('to') == 'CLOSE'))
    Q = np.c_[c['base_x'], c['base_y'], c['base_yaw'], np.column_stack([c[f'q{j}'] for j in range(1, 7)])]
    S = np.column_stack([c[f's{j}'] for j in range(1, 7)])
    UB = np.column_stack([c['base_cmd_vx'], c['base_cmd_vy'], c['base_cmd_wz']])
    UA = np.vstack([np.diff(S, axis=0) / C0.PHYS, np.zeros((1, 6))])        # 第 i 物理步實際生效的手臂速度命令
    U = np.column_stack([UB, UA])
    rows = []
    idx = np.flatnonzero((t >= t_ins - 1e-9) & (t <= t_cl + 1e-9))
    for i in idx[::C0.NPH]:
        if i + C0.D_STEPS >= len(t):
            break
        Tb = bottle_T(c, i)
        if not np.isfinite(Tb).all():
            continue
        q9, s6 = Q[i].copy(), S[i].copy()
        pipe = [U[i + k].copy() for k in range(C0.D_STEPS)]
        u_last = U[i + C0.D_STEPS - 1].copy()
        rec = dict(t=round(float(t[i]), 3), d_meas=round(clear_nf(q9, Tb), 5))
        for lab, s0, pp in (('A_aligned', s6, pipe), ('B_approx', q9[3:9].copy(), [u_last.copy() for _ in range(C0.D_STEPS)])):
            for extra in (0, 1):
                Tr = C21.stop_traj(q9, s0, pp, u_last, CFG, extra_ticks=extra)
                rec[f'{lab}_e{extra}'] = round(min(clear_nf(Tr[k], Tb) for k in range(0, len(Tr), 2)), 5)
        rec['s_minus_q_max'] = round(float(np.abs(s6 - q9[3:9]).max()), 5)
        rows.append(rec)
    keys = [k for k in rows[0] if k not in ('t',)]
    mins = {k: min(rr[k] for rr in rows) for k in keys}
    cb = r.get('clear_bottle')
    json.dump(dict(run=run, t_insert=t_ins, t_close=t_cl, n=len(rows), min=mins, online_clear_bottle=cb, rows=rows,
                   note='A＝實錄設定點與實際在途命令；B＝離線近似（s＝q、命令重複）；兩者同一 stop_traj；非手指對實錄瓶圓柱'),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(run, 'n', len(rows), {k: round(v * 1e3, 2) for k, v in mins.items() if k != 's_minus_q_max'}, 'mm; s−q 最大', round(mins['s_minus_q_max'], 4),
          '; 線上', None if not cb else {k: (round(cb[k] * 1e3, 2) if isinstance(cb.get(k), float) else cb.get(k)) for k in ('meas_min', 'pred_min', 'next_min')})


if __name__ == '__main__':
    main()
