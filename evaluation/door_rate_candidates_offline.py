#!/usr/bin/env python3
"""門速量測候選的離線比較（Codex 20261023_233900 第 3 節；只讀；**不更換線上保護**）。

比較對象（逐物理步，room_run.json）：
  raw      執行端 door_qdot_rad_s（RigidPrim −ω_z；原策略 door_abort_rate 與長窗觀察器現用）
  bdN      因果後向差分 (q_k − q_{k−N}) / (t_k − t_{k−N})，N＝1、5、10（0.01／0.05／0.1 s）；
           只用過去樣本；窗口內缺步、步號不連續、時間不遞增或非有限 ⇒ 該步「無值」（不補零）；
           開頭 N 步亦無值（初始化行為）。
逐段（OPEN 起至策略中止、閂鎖後 0–1 s、1 s–窗末）報：|值| 峰值、峰值時刻、超過 0.05 rad/s（原 door_abort_rate）的首次時刻與步數、無值步數。
界線：
  - 線上任務節點以 50 Hz 接收狀態（約每 2 物理步一筆），線上因果差分的實際間距與可用樣本不同於本表；
  - 前向差分需要未來樣本，線上不可用，故不列；
  - 本表只描述數值，**不據此判定保護保守、失效或可替換**。

    python3 evaluation/door_rate_candidates_offline.py <out_json> <run_dir> [...]
"""
import json
import math
import os
import sys

import numpy as np

TH = 0.05
NS = (1, 5, 10)


def bd(t, q, steps, N):
    v = np.full(len(q), np.nan)
    for k in range(N, len(q)):
        seg_s, seg_t, seg_q = steps[k - N:k + 1], t[k - N:k + 1], q[k - N:k + 1]
        if not (np.all(np.diff(seg_s) == 1) and np.all(np.diff(seg_t) > 0) and np.all(np.isfinite(seg_q))):
            continue
        v[k] = (q[k] - q[k - N]) / (t[k] - t[k - N])
    return v


def seg_stats(t, x, a, b):
    m = (t >= a - 1e-9) & (t < b - 1e-9)
    xv, tv = x[m], t[m]
    fin = np.isfinite(xv)
    if not fin.any():
        return {'n': int(m.sum()), 'n_no_value': int((~fin).sum())}
    ax = np.abs(np.where(fin, xv, 0.0))
    i = int(np.argmax(ax))
    over = fin & (np.abs(np.where(fin, xv, 0.0)) > TH)
    return {'n': int(m.sum()), 'n_no_value': int((~fin).sum()), 'peak_abs': float(ax[i]), 'peak_t': float(tv[i]),
            'first_over_t': float(tv[np.argmax(over)]) if over.any() else None, 'n_over': int(over.sum()),
            'mean_abs': float(np.mean(np.abs(xv[fin])))}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    res = {'schema': 'door_rate_candidates_offline/1', 'threshold_rad_s': TH, 'candidates': ['raw'] + [f'bd{N}' for N in NS], 'runs': [],
           'note': '只讀離線比較；不更換線上保護；不判定保守或失效；線上 50 Hz 取樣與本表逐物理步不同。'}
    for run in sys.argv[2:]:
        r = json.load(open(os.path.join(run, 'room_run.json')))
        tk = json.load(open(os.path.join(run, 'task.json')))
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        S = r['steps']
        t = np.array([s[I['sim_t']] for s in S], float)
        q = np.array([s[I['door_q_rad']] if s[I['door_q_rad']] is not None else math.nan for s in S], float)
        steps = np.array([s[I['step']] for s in S])
        raw = np.array([s[I['door_qdot_rad_s']] if s[I['door_qdot_rad_s']] is not None else math.nan for s in S], float)
        ev = {e['to']: e['t'] for e in tk['policy_events'] if 'to' in e}
        ab = next((e['t'] for e in tk['policy_events'] if 'abort' in e), None)
        lat = [e['k2_abort_latched']['sim_t'] for e in r['events'] if 'k2_abort_latched' in e]
        L = lat[0] if lat else None
        segs = {}
        if 'OPEN' in ev and ab is not None:
            segs['OPEN_to_policy_abort'] = (ev['OPEN'], ab)
        if L is not None:
            segs['post_0_1s'] = (L, L + 1.0)
            segs['post_1s_end'] = (L + 1.0, t[-1] + 1e-6)
        C = {'raw': raw, **{f'bd{N}': bd(t, q, steps, N) for N in NS}}
        R = {'run': os.path.basename(os.path.normpath(run)), 'n_steps': len(S), 'n_step_gaps': int(np.sum(np.diff(steps) != 1)),
             'segments': {nm: {k: seg_stats(t, v, a, b) for k, v in C.items()} for nm, (a, b) in segs.items()},
             'init_no_value_steps': {k: int(np.sum(~np.isfinite(v[:20]))) for k, v in C.items()}}
        res['runs'].append(R)
        print('==', R['run'], '缺步', R['n_step_gaps'], '開頭無值', R['init_no_value_steps'])
        for nm, d in R['segments'].items():
            print(f'  {nm:22s} ' + '  '.join(f"{k}: 峰 {v.get('peak_abs', float('nan')):.4f} 超 {v.get('n_over')} 首 {v.get('first_over_t')}" for k, v in d.items()))
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
