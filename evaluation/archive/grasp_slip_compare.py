#!/usr/bin/env python3
"""抓持滑移的只讀比較（Codex 20261112）：多趟以「閉爪請求」對齊，列瓶相對 TCP 的完整相對位姿（位置三軸＋轉角）、相對速度、手指開度、雙指力時間序列。
只讀 s1_steps.npz／s1_report.json。紀錄沒有接觸點與切向力 ⇒ 不由指力大小推定摩擦裕度（明記缺欄）。
    python3 evaluation/grasp_slip_compare.py OUT.json RUN [RUN...]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C                                        # noqa: E402

K = C.S0_K()


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    S = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {n: i for i, n in enumerate(list(S['cols']))}
    R = S['rows']
    t = R[:, c['t']]
    tc, tg, tl = float(r['t_close']), float(r['t_grasp']), float(r['t_lift_ok'])
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'] + (':' + str(e['to']) if 'to' in e else ''), e['t'])
    tp = ev.get('place_phase:PARKED')

    def rel(i):
        q = np.r_[R[i, c['base_x']], R[i, c['base_y']], R[i, c['base_yaw']], [R[i, c[f'q{j}']] for j in range(1, 7)]]
        T = K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = quat_R(R[i, c['bottle_qw']], R[i, c['bottle_qx']], R[i, c['bottle_qy']], R[i, c['bottle_qz']])
        Tb[:3, 3] = [R[i, c['bottle_base_x']], R[i, c['bottle_base_y']], R[i, c['bottle_base_z']]]
        return np.linalg.inv(T) @ Tb
    i_g = int(np.searchsorted(t, tg))
    Tg = rel(i_g)
    rows = []
    offs = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.39, 1.6, 2.0, 2.5, 3.0, 3.42, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0] + ([tp - tc] if tp else [])
    for dt in offs:
        i = int(np.searchsorted(t, tc + dt))
        if i >= len(t) - 5:
            continue
        Tr = rel(i)
        Tr2 = rel(min(i + 5, len(t) - 1))
        d = np.linalg.inv(Tg) @ Tr
        ang = float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
        vrel = (Tr2[:3, 3] - Tr[:3, 3]) / max(t[min(i + 5, len(t) - 1)] - t[i], 1e-9)
        rows.append(dict(dt=round(float(dt), 2), phase=int(R[i, c['phase_index']]), rel_mm=np.round(Tr[:3, 3] * 1e3, 2).tolist(), d_from_grasp_mm=np.round((Tr[:3, 3] - Tg[:3, 3]) * 1e3, 2).tolist(), d_ang_deg=round(ang, 3),
                         v_rel_mm_s=np.round(vrel * 1e3, 2).tolist(), finger_q=[round(float(R[i, c['finger1_q']]), 5), round(float(R[i, c['finger2_q']]), 5)], finger_cmd=round(float(R[i, c['finger_cmd']]), 5),
                         f1=round(float(R[i, c['f_finger1_bottle']]), 3), f2=round(float(R[i, c['f_finger2_bottle']]), 3), f_table=round(float(R[i, c['f_bottle_table']]), 3),
                         base_v=round(float(np.hypot(R[min(i + 5, len(t) - 1), c['base_x']] - R[i, c['base_x']], R[min(i + 5, len(t) - 1), c['base_y']] - R[i, c['base_y']]) / max(t[min(i + 5, len(t) - 1)] - t[i], 1e-9)), 4)))
    m = (t >= tg) & (t <= (tp or t[-1]))
    return dict(run=run, t_close=tc, t_grasp=tg, t_lift_ok=tl, t_parked=tp, slip_max=r['slip_max_mm_deg'], success=r['success'], rows=rows,
                hold=dict(f1_min=round(float(R[m, c['f_finger1_bottle']].min()), 3), f2_min=round(float(R[m, c['f_finger2_bottle']].min()), 3), f1_med=round(float(np.median(R[m, c['f_finger1_bottle']])), 3),
                          f2_med=round(float(np.median(R[m, c['f_finger2_bottle']])), 3), finger_q_med=[round(float(np.median(R[m, c['finger1_q']])), 5), round(float(np.median(R[m, c['finger2_q']])), 5)]),
                missing='接觸點位置、切向力／摩擦力分量、接觸法向不在紀錄內；f_finger*_bottle 是合力大小')


res = dict(runs=[one(x) for x in sys.argv[2:]])
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
for x in res['runs']:
    print('==', x['run'], 'success', x['success'], 'slip', [round(v, 2) for v in x['slip_max']], 't_close', x['t_close'], 'hold', x['hold'])
    for r in x['rows']:
        print('  ', r['dt'], r['phase'], 'rel', r['rel_mm'], 'd', r['d_from_grasp_mm'], 'ang', r['d_ang_deg'], 'v', r['v_rel_mm_s'], 'fq', r['finger_q'], 'f', r['f1'], r['f2'], 'ft', r['f_table'], 'vb', r['base_v'])
