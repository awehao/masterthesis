#!/usr/bin/env python3
"""K3-A 單趟診斷（只讀實錄，不判定）：OPEN 起到中止／結束的兩指橫桿接觸、手指開度、門角與參考角、抓取偏移與追蹤誤差（把手座標）、腕力。

    python3 evaluation/door_k3a_diag.py <run_dir>   # 寫 <run_dir>/analysis/door_k3a_diag.json 與 .png（已存在則拒絕覆寫）
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


def main():
    run = sys.argv[1]
    out_j = os.path.join(run, 'analysis', 'door_k3a_diag.json')
    out_p = os.path.join(run, 'analysis', 'door_k3a_diag.png')
    for p in (out_j, out_p):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    t_open = next(e['t'] for e in ev if e['to'] == 'OPEN')
    ab = next((e for e in t['policy_events'] if 'abort' in e), None)
    t_hold = next(e['t'] for e in ev if e['to'] == 'HOLD')
    G = np.array(t['G_T_H'])
    Ginv = np.linalg.inv(G)
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tts = np.array([e['t'] for e in tr])
    rows = []
    for s in r['steps']:
        tt = s[I['sim_t']]
        if tt < t_hold:
            continue
        q = s[I['door_q_rad']]
        H = KC._H_of(q)
        T = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        off = (np.linalg.inv(H) @ T)[:3, 3] - Ginv[:3, 3]
        k = int(np.clip(np.searchsorted(tts, tt, side='right') - 1, 0, len(tr) - 1))
        Tg = np.array(tr[k]['target']).reshape(4, 4)
        te = H[:3, :3].T @ (T[:3, 3] - Tg[:3, 3])
        lc = s[I['leaf_contacts']] or {}
        rows.append({'t': tt, 'q_deg': math.degrees(q), 'qdot': s[I['door_qdot_rad_s']], 'q_ref_deg': None if tr[k].get('q_ref') is None else math.degrees(tr[k]['q_ref']),
                     'f_bar': [lc.get(f, {}).get('bar', 0.0) for f in FINGERS], 'q_finger_mm': [x * 1e3 for x in s[I['q_finger']]],
                     'grasp_off_mm': (off * 1e3).tolist(), 'track_err_mm': (te * 1e3).tolist(), 'wrist_n': s[I['wrist_force_n']]})
    first_low = next((x for x in rows if x['t'] >= t_open and min(x['f_bar']) < 0.5), None)
    first_split = next((x for x in rows if x['t'] >= t_open and abs(x['f_bar'][0] - x['f_bar'][1]) > 2.0), None)
    pre_abort = [x for x in rows if t_open <= x['t'] <= (ab['t'] if ab else math.inf)]
    res = {'run': os.path.basename(os.path.normpath(run)), 't_hold': t_hold, 't_open': t_open, 'abort': ab,
           'first_finger_force_split_over_2N': first_split, 'first_finger_below_0.5N': first_low,
           'pre_abort': {'q_max_deg': max(x['q_deg'] for x in pre_abort), 'wrist_max_n': max(x['wrist_n'] for x in pre_abort),
                         'grasp_off_max_mm': [max(abs(x['grasp_off_mm'][i]) for x in pre_abort) for i in range(3)],
                         'track_err_max_mm': [max(abs(x['track_err_mm'][i]) for x in pre_abort) for i in range(3)],
                         'q_finger_max_mm': max(max(x['q_finger_mm']) for x in pre_abort)},
           'post': {'q_max_deg': max(x['q_deg'] for x in rows), 'q_finger_max_mm': max(max(x['q_finger_mm']) for x in rows)},
           'note': '把手座標：x 沿橫桿、y 門扇法向、z 鉛直＝夾合方向。偏移＝實測 TCP 相對模型把手、扣除夾持成立時的抓取關係；追蹤誤差＝實測 TCP − 當下已發布目標。'
                   '實錄無逐步把手 prim 位姿，僅能用模型把手。',
           'rows_every_10': rows[::10]}
    os.makedirs(os.path.dirname(out_j), exist_ok=True)
    json.dump(res, open(out_j, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tt = np.array([x['t'] for x in rows])
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(tt, [x['q_deg'] for x in rows], label='door q (measured)')
    ax[0].plot(tt, [np.nan if x['q_ref_deg'] is None else x['q_ref_deg'] for x in rows], '--', label='q_ref')
    ax[0].set_ylabel('deg')
    ax[1].plot(tt, [x['f_bar'][0] for x in rows], label='finger1-bar')
    ax[1].plot(tt, [x['f_bar'][1] for x in rows], label='finger2-bar')
    ax[1].axhline(0.5, color='gray', lw=0.8, ls=':')
    ax[1].set_ylabel('N')
    for i, n in enumerate('xyz'):
        ax[2].plot(tt, [x['grasp_off_mm'][i] for x in rows], label=f'grasp offset {n}')
    ax[2].set_ylabel('mm')
    ax[3].plot(tt, [x['q_finger_mm'][0] for x in rows], label='finger1 pos')
    ax[3].plot(tt, [x['q_finger_mm'][1] for x in rows], label='finger2 pos')
    ax[3].plot(tt, [x['wrist_n'] / 10 for x in rows], label='wrist force / 10 (N)')
    ax[3].set_ylabel('mm / daN')
    ax[3].set_xlabel('sim time (s)')
    for a in ax:
        a.axvline(t_open, color='k', lw=0.8)
        if ab:
            a.axvline(ab['t'], color='r', lw=0.8)
        a.legend(fontsize=7, loc='upper left')
        a.grid(alpha=0.3)
    ax[0].set_title(f"{res['run']}: OPEN at {t_open:.2f} s (black), abort {ab['abort'] if ab else '-'} (red)")
    fig.tight_layout()
    fig.savefig(out_p, dpi=120)
    print(json.dumps({k: res[k] for k in ('abort', 'first_finger_force_split_over_2N', 'first_finger_below_0.5N', 'pre_abort', 'post')},
                     ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
