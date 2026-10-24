#!/usr/bin/env python3
"""底盤「實測速度 − 套用命令」偏差逐相位統計（只讀；W1c／W1d 暫停期分析的附表，Codex 20261025_013926 (a)）。

對每趟每個任務相位：每 0.1 s 取世界位姿位移轉本體得實測速度，減去同 0.1 s 內 base_cmd_body 平均，取中位數。
用來看偏差是否在接觸門之前（UNFOLD／APPROACH，PARK_HOLD）就已存在、是否隨命令大小改變（加性 vs 比例）。單趟描述，不推因果。
輸出：results/door/base_cmd_offset_phases.json。

    python3 evaluation/door_base_cmd_offset_phases.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ('door_k3m_pause_p4b_1', 'door_wide_w1b_30_1', 'door_wide_w1c_30_1', 'door_wide_w1d_30_1')
OUT = os.path.join(HERE, 'results', 'door', 'base_cmd_offset_phases.json')


def body(v, y):
    c, s = math.cos(y), math.sin(y)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'base_cmd_offset_phases/1', 'method': '0.1 s 世界位姿位移轉本體 − 同窗 base_cmd_body 平均；逐相位中位數（mm/s）', 'runs': {}}
    for R in RUNS:
        run = os.path.join(HERE, 'runs', R)
        r = json.load(open(os.path.join(run, 'room_run.json')))
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        t = json.load(open(os.path.join(run, 'task.json')))
        ph = [(e['to'], e['t']) for e in t['policy_events'] if 'to' in e]
        t_end = next((e['t'] for e in t['policy_events'] if e.get('abort') or e.get('to') == 'DONE'), r['steps'][-1][1])
        out = []
        for k, (nm, a) in enumerate(ph):
            b = ph[k + 1][1] if k + 1 < len(ph) else t_end
            w = [x for x in r['steps'] if a <= x[I['sim_t']] <= b]
            if len(w) < 20:
                continue
            d, cm, vv = [], [], []
            for i in range(10, len(w), 10):
                x, p = w[i], w[i - 10]
                dt = x[I['sim_t']] - p[I['sim_t']]
                bx, bp = x[I['base_xyth']], p[I['base_xyth']]
                v = body([(bx[0] - bp[0]) / dt, (bx[1] - bp[1]) / dt], bx[2])
                c = np.mean([y[I['base_cmd_body']][:2] for y in w[i - 10:i]], 0)
                d.append((v - c) * 1e3)
                cm.append(c * 1e3)
                vv.append(v * 1e3)
            d, cm, vv = np.array(d), np.array(cm), np.array(vv)
            m = np.median(d, 0)
            out.append({'phase': nm, 't': [a, b], 'n_windows': len(d), 'v_minus_cmd_median': m.tolist(), 'v_minus_cmd_norm': float(np.hypot(*m)),
                        'v_minus_cmd_iqr_norm': float(np.hypot(*(np.percentile(d, 75, 0) - np.percentile(d, 25, 0)))),
                        'cmd_median': np.median(cm, 0).tolist(), 'cmd_norm_p95': float(np.percentile(np.hypot(cm[:, 0], cm[:, 1]), 95)),
                        'v_median': np.median(vv, 0).tolist()})
        res['runs'][R] = out
        print(R)
        for o in out:
            print(f"  {o['phase']:10s} {o['t'][0]:6.2f}-{o['t'][1]:6.2f} v−cmd {np.round(o['v_minus_cmd_median'], 2)} |{o['v_minus_cmd_norm']:.2f}| "
                  f"IQR {o['v_minus_cmd_iqr_norm']:.2f}  cmd {np.round(o['cmd_median'], 2)} p95|cmd| {o['cmd_norm_p95']:.2f}")
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
