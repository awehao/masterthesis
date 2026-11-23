#!/usr/bin/env python3
"""door_coopen_core_g3_1（接近段 w_a 0.01）與 door_coopen_core_g_1（原值 0.05）對照（只讀）。
每趟：APPROACH 起訖、閘控暫停／續進事件（各暫停持續時間）、s 終值；求解已發布輪的手臂命令（request_body[3:]）在 APPROACH 內的
  五點移動平均殘差最大值、一階差方向反轉次數／秒（|Δ| > 1e-6）、|u| 最大；暫停期間（固定目標）垂直誤差 trace 每 0.5 s。
指標名稱沿 Codex r13 更正：「五點移動平均殘差」「命令一階差方向反轉」，不解讀為物理振動。

    python3 evaluation/door_coopen_core_g3_1_compare.py   # 寫 runs/door_coopen_core_g3_1/analysis/door_coopen_core_g3_1_compare.json（不覆寫）
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'door_coopen_core_g3_1', 'analysis', 'door_coopen_core_g3_1_compare.json')


def one(run):
    d = os.path.join(HERE, 'runs', run)
    t = json.load(open(os.path.join(d, 'task.json')))
    sv = json.load(open(os.path.join(d, 'align_solver.json')))
    tr = [json.loads(ln)['row'] for ln in open(os.path.join(d, 'task_trace.jsonl'))]
    tA0 = next(r['t'] for r in tr if r['phase'] == 'APPROACH')
    tA1 = next((r['t'] for r in tr if r['t'] > tA0 and r['phase'] != 'APPROACH'), tr[-1]['t'])
    ev = [e for e in t['policy_events'] if 'approach_gate_pause' in e or 'approach_gate_resume' in e]
    pauses = []
    for e in ev:
        if e.get('approach_gate_pause'):
            pauses.append({'t_pause': e['t'], 's': e['s'], 'perp_mm': e['perp_m'] * 1e3, 't_resume': None})
        elif pauses:
            pauses[-1]['t_resume'] = e['t']
    for p in pauses:
        p['dur_s'] = (p['t_resume'] if p['t_resume'] is not None else tA1) - p['t_pause']
    rows = [x for x in sv['log'] if x.get('published') and x.get('request_body') and tA0 <= x['sim_t'] < tA1]
    A = np.array([x['request_body'][3:9] for x in rows], float)
    tt = np.array([x['sim_t'] for x in rows])
    sm = np.array([np.convolve(A[:, j], np.ones(5) / 5, 'same') for j in range(6)]).T
    hf = np.abs(A - sm)[2:-2].max(axis=0)
    d1 = np.diff(A, axis=0)
    sg = np.sign(np.where(np.abs(d1) > 1e-6, d1, 0.0))
    flips = int(sum(np.sum(sg[1:, j] * sg[:-1, j] < 0) for j in range(6)))
    perp = [(r['t'], r['gate']['perp_m'] * 1e3) for r in tr if tA0 <= r['t'] < tA1 and isinstance(r.get('gate'), dict) and r['gate'].get('perp_m') is not None]
    return {'t_A0': tA0, 't_A1': tA1, 'abort': t.get('abort'), 's_end': next((r['s'] for r in reversed(tr) if r['t'] < tA1), None), 'pauses': pauses,
            'arm_cmd': {'n': len(A), 'dur_s': float(tt[-1] - tt[0]), 'ma5_resid_max_mrad_s': (hf * 1e3).tolist(), 'ma5_resid_max_all_mrad_s': float(hf.max() * 1e3),
                        'dir_flips_total': flips, 'dir_flips_per_s': flips / float(tt[-1] - tt[0]), 'abs_max_rad_s': np.abs(A).max(axis=0).tolist()},
            'perp_mm_0p5s': [(round(a, 2), round(b, 3)) for a, b in perp[::8]]}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    out = {'schema': 'door_coopen_core_g3_1_compare/1', 'g_1_w_a_0.05': one('door_coopen_core_g_1'), 'g3_1_w_a_0.01': one('door_coopen_core_g3_1')}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('g_1_w_a_0.05', 'g3_1_w_a_0.01'):
        r = out[k]
        print(f"{k}：APPROACH {r['t_A0']:.2f}–{r['t_A1']:.2f}、s 終 {r['s_end']}、中止 {r['abort']}")
        for p in r['pauses']:
            print(f"   暫停 {p['t_pause']:.2f}（s {p['s']:.4f}、垂直 {p['perp_mm']:.2f} mm）→ 續進 {p['t_resume']}（{p['dur_s']:.2f} s）")
        a = r['arm_cmd']
        print(f"   手臂命令：五點平均殘差 max {a['ma5_resid_max_all_mrad_s']:.2f} mrad/s、方向反轉 {a['dir_flips_total']}（{a['dir_flips_per_s']:.2f}/s）、|u| max {max(a['abs_max_rad_s']):.3f}")


if __name__ == '__main__':
    main()
