#!/usr/bin/env python3
"""K3-A-B 承載平衡觀察報告（只讀、描述性、一趟介入觀察；規格 K3_finger_balance_compliance_spec.md §7）。與 K3-A 任務判定分開。

事前登錄的量（窗口以來源物理步對齊：OPEN 步＝task trace 第一列 phase＝OPEN 的 step；觸發步＝第一列 ABORTED 的 step；
SETTLE 起點＝第一列 phase＝SETTLE 的 step；主窗＝[OPEN 步, min(觸發步, SETTLE 步))）：
  1. 逐步 F1、F2、e、ẽ、Δ（task.json balance.records）；
  2. 首次任一指 < 0.5 N 的物理步與門角（room leaf_contacts，主窗內；無 ⇒ None）；
  3. 主窗內 F1、F2 最小值；4. 主窗內 |F2 − F1| ≤ 2 N 的步數比例；5. Δ 飽和累計時間（sat 步數 × 0.01 s）；
  6. 對照 run（無柔順）以同一定義列 2–4；7. Δ 與求解輸入 d̂ 的同窗時序（快照時刻、d̂ 六軸、該時刻最後一筆 Δ）。
不判定、不推因果；不宣稱根因已定位，也不把承載平衡結果當成任務通過。

    python3 evaluation/door_k3ab_report.py <run_dir> [對照 run_dir ...]   # 寫 <run_dir>/analysis/door_k3ab_report.json（已存在則拒絕覆寫）
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


def step_of(task, phase):
    return next((int(e['step']) for e in task.get('trace') or [] if e.get('phase') == phase), None)


def window_stats(task, room):
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    s_open, s_trig, s_set = step_of(task, 'OPEN'), step_of(task, 'ABORTED'), step_of(task, 'SETTLE')
    ends = [x for x in (s_trig, s_set) if x is not None]
    s_end = min(ends) if ends else math.inf
    rows = [s for s in room['steps'] if s_open is not None and s_open <= s[I['step']] < s_end]
    f = []
    for s in rows:
        lc = s[I['leaf_contacts']]
        f.append(None if lc is None else [(lc.get(b) or {}).get('bar', 0.0) for b in FINGERS])
    val = [(s, x) for s, x in zip(rows, f) if x is not None]
    low = next(((s, x) for s, x in val if min(x) < 0.5), None)
    F = np.array([x for _, x in val]) if val else np.zeros((0, 2))
    return {'steps': {'OPEN': s_open, 'SETTLE': s_set, 'trigger': s_trig}, 'n_steps': len(rows), 'n_na': len(rows) - len(val),
            'first_low': None if low is None else {'step': low[0][I['step']], 'q_deg': math.degrees(low[0][I['door_q_rad']]),
                                                   'steps_since_open': low[0][I['step']] - s_open, 'f': low[1]},
            'f1_min': None if not len(F) else float(F[:, 0].min()), 'f2_min': None if not len(F) else float(F[:, 1].min()),
            'frac_abs_e_le_2N': None if not len(F) else float((np.abs(F[:, 1] - F[:, 0]) <= 2.0).mean())}


def dhat_series(run, records):
    p = os.path.join(run, 'align_solver.json')
    if not os.path.exists(p):
        return None
    a = json.load(open(p))
    rt = [(r['sim_t'], r['delta']) for r in records]
    out = []
    for x in a.get('log', []):
        si = x.get('solve_in') or {}
        if si.get('offset_d_hat') is None or si.get('snap_sim_t_exact') is None:
            continue
        ts = si['snap_sim_t_exact']
        if not rt or ts < rt[0][0]:
            continue
        d = [v for t_, v in rt if t_ <= ts + 1e-9]
        out.append({'snap_t': ts, 'd_hat': si['offset_d_hat'], 'delta_m': d[-1] if d else None})
    return out


def main():
    run, refs = sys.argv[1], sys.argv[2:]
    out = os.path.join(run, 'analysis', 'door_k3ab_report.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    bal = t.get('balance') or {}
    recs = bal.get('records') or []
    res = {'run': os.path.basename(os.path.normpath(run)), 'abort': t.get('abort'), 'final_phase': t.get('final_phase'),
           'window': window_stats(t, r), 'balance': {'s0': bal.get('s0'), 'n_steps': len(recs), 'stopped': bal.get('stopped'),
                                                     'delta_final_m': bal.get('delta_final_m'),
                                                     'delta_max_abs_m': max([abs(x['delta']) for x in recs] or [0.0]),
                                                     'saturated_s': 0.01 * sum(1 for x in recs if x.get('sat')),
                                                     'records': recs},
           'delta_vs_dhat': dhat_series(run, recs), 'references': [],
           'note': '一趟介入觀察（描述性）。承載平衡觀察與 K3-A 任務判定分開；不宣稱根因已定位；回授方向是依幾何提出的候選方向。'}
    for ref in refs:
        t2 = json.load(open(os.path.join(ref, 'task.json')))
        r2 = json.load(open(os.path.join(ref, 'room_run.json')))
        res['references'].append({'run': os.path.basename(os.path.normpath(ref)), 'window': window_stats(t2, r2)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('run', 'abort', 'final_phase', 'window', 'references')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
