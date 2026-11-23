#!/usr/bin/env python3
"""K3-A 單趟診斷 v2（只讀實錄、不判定；依 Codex 20261021 door_k3a_1 結果裁定）。在**求解週期**（20 Hz，align_solver.json log）上對齊：
  - 抓取姿態誤差 E＝H(q)⁻¹·T_tcp·G_T_H_grasp：平移與旋轉向量，皆在把手座標（x 沿橫桿、y 門扇法向、z 鉛直＝夾合方向）；
  - 參考超前量 q_ref − q_meas；已發布目標 vs FK（把手座標）；
  - 手臂設定點 s_pred vs 實測 q_meas（各關節，solve_in）；偏移估計 d̂（d_hat_after）與更新狀態（offset_moving.ok／used／why）；
  - 求解器 err_p／err_r、請求速度；
  - 同時刻的兩指橫桿接觸、手指實測位置、腕力、門角速率（room_run 取 ≤ 該週期的最近一步）。
分窗：HOLD、OPEN→兩指力分裂（|f1−f2| > 2 N）、分裂→首次單指 < 0.5 N、首次 < 0.5 N→中止、中止後；各窗報起訖值、極值與各量首次越過門檻的時刻。
門檻只用於排序「何者先出現」，不是判準。

    python3 evaluation/door_k3a_diag2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3a_diag2.json 與 .png（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402

FINGERS = KC.FINGERS
THR = {'rot_x_deg': 0.1, 'rot_y_deg': 0.1, 'rot_z_deg': 0.1, 'off_x_mm': 0.5, 'off_z_mm': 0.3, 'd_hat_change_rad': 0.001,
       'sp_gap_change_rad': 0.002, 'lead_deg': 0.2, 'track_y_mm': 1.0, 'f_split_n': 2.0, 'wrist_rise_n': 2.0}


def rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    a = math.acos(c)
    if a < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(a))
    return w * a


def main():
    run = sys.argv[1]
    out_j = os.path.join(run, 'analysis', 'door_k3a_diag2.json')
    out_p = os.path.join(run, 'analysis', 'door_k3a_diag2.png')
    for p in (out_j, out_p):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    t_hold = next(e['t'] for e in ev if e['to'] == 'HOLD')
    t_open = next(e['t'] for e in ev if e['to'] == 'OPEN')
    ab = next((e for e in t['policy_events'] if 'abort' in e), None)
    t_ab = ab['t'] if ab else math.inf
    G = np.array(t['G_T_H'])
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tr_t = np.array([e['t'] for e in tr])
    st = r['steps']
    st_t = np.array([s[I['sim_t']] for s in st])
    rows = []
    for L in a['log']:
        tt = L['sim_t']
        if tt < t_hold - 0.5:
            continue
        k = int(np.searchsorted(st_t, tt, side='right') - 1)
        if k < 0:
            continue
        s = st[k]
        q = s[I['door_q_rad']]
        H = KC._H_of(q)
        T = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        E = np.linalg.inv(H) @ T @ G
        j = int(np.clip(np.searchsorted(tr_t, tt, side='right') - 1, 0, len(tr) - 1))
        Tg = np.array(tr[j]['target']).reshape(4, 4)
        tgt_err = H[:3, :3].T @ (T[:3, 3] - Tg[:3, 3])
        si = L.get('solve_in') or {}
        sp, qm = si.get('s_pred'), si.get('q_meas')
        gap = (np.array(sp) - np.array(qm[3:9])).tolist() if (sp is not None and qm is not None and len(qm) >= 9) else None
        om = L.get('offset_moving') or {}
        lc = s[I['leaf_contacts']] or {}
        fb = [lc.get(f, {}).get('bar', 0.0) for f in FINGERS]
        rows.append({'t': tt, 'q_deg': math.degrees(q), 'qdot': s[I['door_qdot_rad_s']],
                     'q_ref_deg': None if tr[j].get('q_ref') is None else math.degrees(tr[j]['q_ref']),
                     'lead_deg': None if tr[j].get('q_ref') is None else math.degrees(tr[j]['q_ref'] - q),
                     'grasp_off_mm': (E[:3, 3] * 1e3).tolist(), 'grasp_rot_deg': np.degrees(rotvec(E[:3, :3])).tolist(),
                     'tgt_minus_fk_mm': (-tgt_err * 1e3).tolist(), 'sp_minus_q_rad': gap,
                     'd_hat': om.get('d_hat_after'), 'offset_ok': om.get('ok'), 'offset_used': om.get('used'), 'offset_why': om.get('why'),
                     'err_p_mm': None if L.get('err_p') is None else L['err_p'] * 1e3, 'err_r_deg': None if L.get('err_r') is None else math.degrees(L['err_r']),
                     'request_arm': (L.get('request_world') or [None] * 9)[3:9], 'published': L.get('published'),
                     'f_bar': fb, 'q_finger_mm': [x * 1e3 for x in s[I['q_finger']]], 'wrist_n': s[I['wrist_force_n']]})
    t_split = next((x['t'] for x in rows if x['t'] >= t_open and abs(x['f_bar'][0] - x['f_bar'][1]) > THR['f_split_n']), None)
    t_low = next((x['t'] for x in rows if x['t'] >= t_open and min(x['f_bar']) < 0.5), None)
    wins = [('HOLD', t_hold, t_open), ('OPEN→split', t_open, t_split), ('split→first_low', t_split, t_low), ('first_low→abort', t_low, t_ab),
            ('post_abort', t_ab, math.inf)]
    base = next((x for x in rows if x['t'] >= t_open), rows[0])        # OPEN 第一個求解週期作基準

    def first_cross():
        out = {}
        b_dh = np.array(base['d_hat']) if base['d_hat'] is not None else None
        b_gap = np.array(base['sp_minus_q_rad']) if base['sp_minus_q_rad'] is not None else None
        for x in rows:
            if x['t'] < t_open:
                continue
            tests = {'rot_x_deg': abs(x['grasp_rot_deg'][0] - base['grasp_rot_deg'][0]), 'rot_y_deg': abs(x['grasp_rot_deg'][1] - base['grasp_rot_deg'][1]),
                     'rot_z_deg': abs(x['grasp_rot_deg'][2] - base['grasp_rot_deg'][2]),
                     'off_x_mm': abs(x['grasp_off_mm'][0] - base['grasp_off_mm'][0]), 'off_z_mm': abs(x['grasp_off_mm'][2] - base['grasp_off_mm'][2]),
                     'd_hat_change_rad': (float(np.abs(np.array(x['d_hat']) - b_dh).max()) if (x['d_hat'] is not None and b_dh is not None) else 0.0),
                     'sp_gap_change_rad': (float(np.abs(np.array(x['sp_minus_q_rad']) - b_gap).max()) if (x['sp_minus_q_rad'] is not None and b_gap is not None) else 0.0),
                     'lead_deg': abs(x['lead_deg'] or 0.0), 'track_y_mm': abs(x['tgt_minus_fk_mm'][1]),
                     'f_split_n': abs(x['f_bar'][0] - x['f_bar'][1]), 'wrist_rise_n': x['wrist_n'] - base['wrist_n']}
            for k, v in tests.items():
                if k not in out and v > THR[k]:
                    out[k] = {'t': x['t'], 'value': v, 'q_deg': x['q_deg']}
        return dict(sorted(out.items(), key=lambda kv: kv[1]['t']))

    def summ(lo, hi):
        w = [x for x in rows if lo is not None and x['t'] >= lo and (hi is None or x['t'] < hi)]
        if not w:
            return None

        def ex(key, i=None):
            v = [(x[key] if i is None else x[key][i]) for x in w if x[key] is not None]
            return None if not v else {'start': v[0], 'end': v[-1], 'min': min(v), 'max': max(v)}
        o = {'n': len(w), 't0': w[0]['t'], 't1': w[-1]['t'], 'q_deg': ex('q_deg'), 'lead_deg': ex('lead_deg'), 'wrist_n': ex('wrist_n'),
             'f1': ex('f_bar', 0), 'f2': ex('f_bar', 1), 'finger1_mm': ex('q_finger_mm', 0), 'finger2_mm': ex('q_finger_mm', 1),
             'err_p_mm': ex('err_p_mm')}
        for i, n in enumerate('xyz'):
            o[f'grasp_off_{n}_mm'] = ex('grasp_off_mm', i)
            o[f'grasp_rot_{n}_deg'] = ex('grasp_rot_deg', i)
            o[f'tgt_minus_fk_{n}_mm'] = ex('tgt_minus_fk_mm', i)
        if w[0]['d_hat'] is not None and w[-1]['d_hat'] is not None:
            o['d_hat_start'], o['d_hat_end'] = w[0]['d_hat'], w[-1]['d_hat']
        if w[0]['sp_minus_q_rad'] is not None and w[-1]['sp_minus_q_rad'] is not None:
            o['sp_minus_q_start'], o['sp_minus_q_end'] = w[0]['sp_minus_q_rad'], w[-1]['sp_minus_q_rad']
        o['offset_used_counts'] = {}
        for x in w:
            key = f"{x['offset_ok']}|{x['offset_used']}"
            o['offset_used_counts'][key] = o['offset_used_counts'].get(key, 0) + 1
        return o
    res = {'run': os.path.basename(os.path.normpath(run)), 't_hold': t_hold, 't_open': t_open, 't_split': t_split, 't_first_low': t_low, 'abort': ab,
           'thresholds_for_ordering_only': THR, 'first_crossings_after_open': first_cross(),
           'windows': {n: summ(lo, hi if math.isfinite(hi) else None) for n, lo, hi in wins},
           'solver_offset_free': a.get('offset_free'), 'solver_args_offset': {k: a['args'].get(k) for k in ('offset_free', 'offset_moving', 'offset_tau_s', 'offset_gate_rad', 'offset_max_rad')},
           'note': '只讀診斷。E＝H(q)⁻¹·T_tcp·G_T_H_grasp（模型把手；實錄無 prim 位姿）。tgt_minus_fk＝已發布目標 − FK（把手座標）。'
                   'sp_minus_q＝求解器輸入的手臂設定點預測 − 實測關節。room 取 ≤ 週期時刻的最近物理步。',
           'rows': rows}
    os.makedirs(os.path.dirname(out_j), exist_ok=True)
    json.dump(res, open(out_j, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tt = np.array([x['t'] for x in rows])
    fig, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
    ax[0].plot(tt, [x['f_bar'][0] for x in rows], label='finger1-bar N')
    ax[0].plot(tt, [x['f_bar'][1] for x in rows], label='finger2-bar N')
    ax[0].plot(tt, [x['wrist_n'] for x in rows], label='wrist N')
    for i, n in enumerate('xyz'):
        ax[1].plot(tt, [x['grasp_rot_deg'][i] for x in rows], label=f'grasp rot about handle {n} (deg)')
    for i, n in enumerate('xyz'):
        ax[2].plot(tt, [x['tgt_minus_fk_mm'][i] for x in rows], label=f'target - FK {n} (mm)')
    ax[2].plot(tt, [np.nan if x['lead_deg'] is None else x['lead_deg'] * 10 for x in rows], 'k--', label='lead x10 (deg)')
    gaps = np.array([x['sp_minus_q_rad'] if x['sp_minus_q_rad'] is not None else [np.nan] * 6 for x in rows])
    for jn in range(6):
        ax[3].plot(tt, gaps[:, jn] * 1e3, label=f'j{jn + 1} setpoint - meas (mrad)')
    dh = np.array([x['d_hat'] if x['d_hat'] is not None else [np.nan] * 6 for x in rows])
    for jn in range(6):
        ax[4].plot(tt, dh[:, jn] * 1e3, label=f'j{jn + 1} d_hat (mrad)')
    ax[4].set_xlabel('sim time (s)')
    for a_ in ax:
        a_.axvline(t_open, color='k', lw=0.8)
        for tv, c in ((t_split, 'orange'), (t_low, 'm'), (t_ab, 'r')):
            if tv is not None and math.isfinite(tv):
                a_.axvline(tv, color=c, lw=0.8)
        a_.legend(fontsize=6, loc='upper left', ncol=2)
        a_.grid(alpha=0.3)
    ax[0].set_title(f"{res['run']}: OPEN (black), split (orange), first<0.5N (magenta), abort (red)")
    fig.tight_layout()
    fig.savefig(out_p, dpi=120)
    print(json.dumps({k: res[k] for k in ('t_open', 't_split', 't_first_low', 'first_crossings_after_open')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
