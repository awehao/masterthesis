#!/usr/bin/env python3
"""door_clear_door_fast 對 door_wbopt.clear_door 的逐構型等值核對（env／base／self／shell_bar 四欄逐值、who 欄位）。
構型：full_2 開門密集參考全部取樣；隨機擾動（底盤 ±5 cm、偏航 ±3°、手臂 ±0.15 rad、門角 0–90°）；手指開度 0.0017 與 0.0089 兩種；
另含退出段（第三趟終態附近）構型。量測兩者每次呼叫耗時。
    python3 evaluation/door_clear_door_fast_check.py OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as F
rng = np.random.default_rng(20261113)
P = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'plan.npz'))
T, phib = P['T_WH_bind'], float(P['phi_bind'])
cfgs = [x for x in P['Xd']]
for _ in range(1200):
    x = P['Xd'][rng.integers(len(P['Xd']))].copy()
    x[:2] += rng.uniform(-0.05, 0.05, 2); x[2] += math.radians(rng.uniform(-3, 3)); x[3:9] += rng.uniform(-0.15, 0.15, 6); x[9] = math.radians(rng.uniform(0, 90))
    cfgs.append(x)
L3 = np.load(os.path.join(HERE, 'runs', 'm8house_door_open_3', 'log.npz'))['rows']
for r in L3[L3[:, 1] >= 5][::10]:
    cfgs.append(np.r_[r[24:27], r[16:22], r[2]])
res = dict(n=0, max_abs_diff={k: 0.0 for k in ('env', 'base', 'self', 'shell_bar')}, who_mismatch=0, t_orig_ms=[], t_fast_ms=[])
for fq in (0.00174, 0.0089):
    WB.set_model(phi0=phib, finger_q=fq)
    for x in cfgs:
        t0 = time.perf_counter(); a = WB.clear_door(x, T); t1 = time.perf_counter(); b = F.clear_door_fast(x, T); t2 = time.perf_counter()
        res['t_orig_ms'].append((t1 - t0) * 1e3); res['t_fast_ms'].append((t2 - t1) * 1e3)
        for k in res['max_abs_diff']:
            res['max_abs_diff'][k] = max(res['max_abs_diff'][k], abs(a[k] - b[k]) if np.isfinite(a[k]) or np.isfinite(b[k]) else 0.0)
        res['who_mismatch'] += int(a['who'] != b['who'])
        res['n'] += 1
out = dict(n=res['n'], max_abs_diff=res['max_abs_diff'], who_mismatch=res['who_mismatch'], orig_ms=dict(p50=round(float(np.median(res['t_orig_ms'])), 3), mean=round(float(np.mean(res['t_orig_ms'])), 3)),
           fast_ms=dict(p50=round(float(np.median(res['t_fast_ms'])), 3), mean=round(float(np.mean(res['t_fast_ms'])), 3)))
out['equal'] = bool(max(res['max_abs_diff'].values()) == 0.0)
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(out)
