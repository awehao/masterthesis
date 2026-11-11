#!/usr/bin/env python3
"""抓持成立前 0.5 s 到後 2 s 的逐物理步比較（Codex 20261112；只讀）。每趟：瓶相對 TCP 的完整相對位姿（位置三軸、相對閉爪請求時刻的轉角三軸）、
相對速度（逐步差分）、手指開度與開度速度、雙指力與其短時波動。輸出逐步序列與分段統計（不是只取中位數或稀疏點）。CPU 輕量：只取 250 步／趟。
    python3 evaluation/grasp_slip_window.py OUT.json RUN [RUN...]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C                                        # noqa: E402
from grasp_slip_compare import quat_R                             # noqa: E402

K = C.S0_K()


def rotvec(Rm):
    a = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    S = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {n: i for i, n in enumerate(list(S['cols']))}
    R = S['rows']
    t = R[:, c['t']]
    tg = float(r['t_grasp'])
    i0, i1 = int(np.searchsorted(t, tg - 0.5)), int(np.searchsorted(t, tg + 2.0))
    ig = int(np.searchsorted(t, tg))
    rel = []
    for i in range(i0, i1 + 1):
        q = np.r_[R[i, c['base_x']], R[i, c['base_y']], R[i, c['base_yaw']], [R[i, c[f'q{j}']] for j in range(1, 7)]]
        T = K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = quat_R(R[i, c['bottle_qw']], R[i, c['bottle_qx']], R[i, c['bottle_qy']], R[i, c['bottle_qz']])
        Tb[:3, 3] = [R[i, c['bottle_base_x']], R[i, c['bottle_base_y']], R[i, c['bottle_base_z']]]
        rel.append(np.linalg.inv(T) @ Tb)
    Tg = rel[ig - i0]
    tt = t[i0:i1 + 1] - tg
    pos = np.array([(x[:3, 3] - Tg[:3, 3]) * 1e3 for x in rel])
    rv = np.degrees(np.array([rotvec(Tg[:3, :3].T @ x[:3, :3]) for x in rel]))
    vel = np.vstack([np.zeros(3), np.diff(pos, axis=0) / np.diff(tt)[:, None]])
    f1, f2 = R[i0:i1 + 1, c['f_finger1_bottle']], R[i0:i1 + 1, c['f_finger2_bottle']]
    fq1, fq2, fc = R[i0:i1 + 1, c['finger1_q']], R[i0:i1 + 1, c['finger2_q']], R[i0:i1 + 1, c['finger_cmd']]
    ph = R[i0:i1 + 1, c['phase_index']].astype(int)
    zt = np.array([K.fk(np.r_[R[i, c['base_x']], R[i, c['base_y']], R[i, c['base_yaw']], [R[i, c[f'q{j}']] for j in range(1, 7)]], 'link_tcp')[2, 3] for i in range(i0, i1 + 1)])
    segs = {}
    for nm, a, b in (('抓持成立前 0.5 s', -0.5, 0.0), ('成立後 0–0.4 s（抬升前）', 0.0, 0.4), ('0.4–1.0 s（抬升）', 0.4, 1.0), ('1.0–2.0 s（抬升末／運送起）', 1.0, 2.0)):
        m = (tt >= a) & (tt < b)
        if not m.any():
            continue
        sp = np.linalg.norm(vel[m], axis=1)
        segs[nm] = dict(n=int(m.sum()), d_pos_mm=np.round(pos[m][-1] - pos[m][0], 3).tolist(), d_rot_deg=np.round(rv[m][-1] - rv[m][0], 3).tolist(), v_rel_max_mm_s=round(float(sp.max()), 2), v_rel_p95=round(float(np.percentile(sp, 95)), 2),
                        t_v_max=round(float(tt[m][int(sp.argmax())]), 3), f1=[round(float(f1[m].min()), 3), round(float(np.median(f1[m])), 3), round(float(f1[m].max()), 3)], f2=[round(float(f2[m].min()), 3), round(float(np.median(f2[m])), 3), round(float(f2[m].max()), 3)],
                        f_sum_min=round(float((f1[m] + f2[m]).min()), 3), f_imbalance_max=round(float(np.abs(f1[m] - f2[m]).max()), 3), f_std=[round(float(f1[m].std()), 3), round(float(f2[m].std()), 3)],
                        finger_q=[round(float(fq1[m].min()) * 1e3, 3), round(float(fq1[m].max()) * 1e3, 3)], dz_tcp_mm=round(float(zt[m][-1] - zt[m][0]) * 1e3, 2), phases=sorted(set(ph[m].tolist())))
    # 最大相對速度事件
    sp = np.linalg.norm(vel, axis=1)
    k = int(sp.argmax())
    return dict(run=run, t_grasp=tg, slip_max=r['slip_max_mm_deg'], segments=segs, peak=dict(t=round(float(tt[k]), 3), v_mm_s=np.round(vel[k], 2).tolist(), f1=round(float(f1[k]), 3), f2=round(float(f2[k]), 3), phase=int(ph[k])),
                series=dict(t=np.round(tt, 3).tolist(), pos_mm=np.round(pos, 3).tolist(), rot_deg=np.round(rv, 3).tolist(), f1=np.round(f1, 3).tolist(), f2=np.round(f2, 3).tolist(), finger_q_mm=np.round(fq1 * 1e3, 3).tolist(),
                            finger_cmd_mm=np.round(fc * 1e3, 3).tolist(), z_tcp=np.round(zt, 4).tolist(), phase=ph.tolist()))


res = dict(runs=[one(x) for x in sys.argv[2:]], missing='接觸點、法向、切向力不在紀錄內；瓶對 TCP 的絕對抓姿未做左右側別對齊')
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
for x in res['runs']:
    print('==', x['run'][12:], 'slip', [round(v, 2) for v in x['slip_max']], 'peak', x['peak'])
    for k, v in x['segments'].items():
        print('  ', k, {kk: v[kk] for kk in ('d_pos_mm', 'd_rot_deg', 'v_rel_max_mm_s', 't_v_max', 'f1', 'f2', 'f_sum_min', 'f_imbalance_max', 'f_std', 'dz_tcp_mm', 'phases')})
