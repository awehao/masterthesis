#!/usr/bin/env python3
"""door_coopen_core_bench_6 末段持握餘裕的依據（只讀；Codex r34：選一項有依據的最小修正前的證據）。
1. 末段漂移分量（把手座標：x 沿桿、y 門扇法向、z 鉛直；analysis2 同法 E＝H_meas⁻¹·(T_tcp·G_T_H)），38–48° 每 1° 平均。
2. 底盤命令 vs 引導：求解端已發布列 stages.request.body[0:3] 與 coord.base_vref 的差（OPEN 窗口末 20 s）。
3. 閉合所需的最小修正（實測底盤位姿、實測手臂角、實測把手位姿 → 目標 TCP＝H·G_T_H⁻¹，位置＋姿態 6 維）：
   (a) 只動手臂（阻尼最小平方）的 Δq；(b) 全身 9 自由度加權最小範數（底盤 1 cm、關節 10 mrad 同權）的底盤位移與 Δq。
   門角 30／40／45／47／47.7°。
4. 底盤沿桿速度與沿桿誤差變化率的相關（0.5 s 窗口；排除「底盤沿桿運動直接拉出」的假設用）。
界線：運動學閉合計算，非動力學；不含接觸、摩擦、夾爪柔順；不判定唯一原因。

    python3 evaluation/door_coopen_core_bench_6_late_closure.py   # 寫 runs/door_coopen_core_bench_6/analysis/door_coopen_core_bench_6_late_closure.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_k2_check as KC                                    # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_6_late_closure.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')


def logR(Rm):
    a = math.acos(max(-1.0, min(1.0, (np.trace(Rm) - 1.0) / 2.0)))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])


def solve(K, x0, Tt, Winv, idx):
    x = x0.copy()
    for _ in range(100):
        Tq = K.fk(x, 'link_tcp')
        e = np.concatenate([Tt[:3, 3] - Tq[:3, 3], Tq[:3, :3] @ logR(Tq[:3, :3].T @ Tt[:3, :3])])
        if np.linalg.norm(e) < 1e-9:
            break
        J = np.asarray(K.jacobian(x, 'link_tcp'), float)[:, idx]
        x[idx] = x[idx] + Winv @ J.T @ np.linalg.solve(J @ Winv @ J.T + 1e-9 * np.eye(6), e)
    return x, float(np.linalg.norm(e))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    GTH = np.asarray(t['G_T_H'], float)
    t_open = next(e['t'] for e in t['policy_events'] if e.get('to') == 'OPEN')
    t_set = next(e['t'] for e in t['policy_events'] if e.get('to') == 'SETTLE')
    evd = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('T_WH'), list) and len(e['T_WH']) == 16:
            evd[e['step']] = np.asarray(e['T_WH'], float).reshape(4, 4)
    S = [s for s in r['steps'] if t_open <= s[c['sim_t']] < t_set and s[c['step']] in evd]
    rows = []
    for s in S:
        H = evd[s[c['step']]]
        E = np.linalg.inv(H) @ (KC._tcp(list(s[c['base_xyth']]), list(s[c['q_arm_meas']])) @ GTH)
        rows.append({'t': s[c['sim_t']], 'q': math.degrees(s[c['door_q_rad']]), 'e': (E[:3, 3] * 1e3).tolist(), 'bax': H[:3, 0].tolist(),
                     'b': list(s[c['base_xyth']])})
    comp = []
    for lo in range(38, 48):
        w = [x for x in rows if lo <= x['q'] < lo + 1]
        if w:
            e = np.array([x['e'] for x in w])
            comp.append({'q_bin_deg': [lo, lo + 1], 'n': len(w), 'e_mean_mm': e.mean(axis=0).tolist(), 'norm_mean_mm': float(np.linalg.norm(e, axis=1).mean()),
                         'norm_max_mm': float(np.linalg.norm(e, axis=1).max())})
    gd = []
    for x in sv['log']:
        if x.get('published') and t_set - 20.0 <= x['sim_t'] < t_set and (x.get('coord') or {}).get('base_vref') and ((x.get('stages') or {}).get('request') or {}).get('body'):
            gd.append(np.abs(np.asarray(x['stages']['request']['body'][:3]) - np.asarray(x['coord']['base_vref'])).tolist())
    G = np.array(gd)
    K = WholeBodyKinematics.from_urdf_file(URDF)
    clos = []
    Wb = np.diag([0.01 ** 2] * 3 + [0.01 ** 2] * 6)
    for qd in (30.0, 40.0, 45.0, 47.0, 47.7):
        s = min(S, key=lambda s: abs(math.degrees(s[c['door_q_rad']]) - qd))
        x0 = np.array(list(s[c['base_xyth']]) + list(s[c['q_arm_meas']]), float)
        Tt = evd[s[c['step']]] @ np.linalg.inv(GTH)
        xa, ra = solve(K, x0, Tt, np.eye(6), list(range(3, 9)))
        xw, rw = solve(K, x0, Tt, Wb, list(range(9)))
        clos.append({'q_deg': math.degrees(s[c['door_q_rad']]), 't': s[c['sim_t']],
                     'arm_only': {'dq_mrad': ((xa - x0)[3:] * 1e3).tolist(), 'dq_max_mrad': float(np.abs(xa - x0)[3:].max() * 1e3), 'resid': ra},
                     'whole_body': {'dbase_mm_mrad': [(xw - x0)[0] * 1e3, (xw - x0)[1] * 1e3, (xw - x0)[2] * 1e3], 'dq_mrad': ((xw - x0)[3:] * 1e3).tolist(),
                                    'dq_max_mrad': float(np.abs(xw - x0)[3:].max() * 1e3), 'resid': rw}})
    o = []
    for i in range(50, len(rows), 50):
        dt = rows[i]['t'] - rows[i - 50]['t']
        vb = (np.array(rows[i]['b'][:2]) - np.array(rows[i - 50]['b'][:2])) / dt
        o.append((rows[i]['q'], float(vb @ np.array(rows[i]['bax'][:2])) * 1e3, (rows[i]['e'][0] - rows[i - 50]['e'][0]) / dt))
    O = np.array(o)
    late = O[O[:, 0] >= 36]
    out = {'schema': 'door_coopen_core_bench_6_late_closure/1', 'drift_components_1deg': comp,
           'base_cmd_vs_vref_last20s': {'n': len(G), 'absdiff_mean': G.mean(axis=0).tolist() if len(G) else None, 'absdiff_max': G.max(axis=0).tolist() if len(G) else None},
           'closure_min_correction': clos,
           'base_along_bar_vs_ex_rate_corr': {'ge36deg': float(np.corrcoef(late[:, 1], late[:, 2])[0, 1]), 'all': float(np.corrcoef(O[:, 1], O[:, 2])[0, 1])},
           'note': '運動學閉合計算；不判定唯一原因'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for x in comp:
        print('  %d–%d° e 均 %s |e| 均 %.2f 最大 %.2f' % (x['q_bin_deg'][0], x['q_bin_deg'][1], np.round(x['e_mean_mm'], 2), x['norm_mean_mm'], x['norm_max_mm']))
    print('底盤命令 vs 引導 |差| 均', np.round(out['base_cmd_vs_vref_last20s']['absdiff_mean'], 6), '最大', np.round(out['base_cmd_vs_vref_last20s']['absdiff_max'], 6))
    for x in clos:
        print('  %.1f° 只動手臂 Δq 最大 %.1f mrad；全身 底盤 %s、Δq 最大 %.1f mrad' % (x['q_deg'], x['arm_only']['dq_max_mrad'],
              np.round(x['whole_body']['dbase_mm_mrad'], 2), x['whole_body']['dq_max_mrad']))
    print('相關', out['base_along_bar_vs_ex_rate_corr'])


if __name__ == '__main__':
    main()
