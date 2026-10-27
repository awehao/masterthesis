#!/usr/bin/env python3
"""door_coopen_core_bench_6（bench9：聯合配置）與 bench_3、bench_4、bench_5 的比較（door_coopen_core_bench_5_compare.py 的版本化複本；四趟）。
以下為 bench_5 版說明。
door_coopen_core_bench_5（bench7：bench6 控制配置＋停頓診斷）與 bench_3、bench_4 的比較（door_coopen_core_bench_4_compare.py 的版本化複本；三趟）。
以下為 bench_4 版說明。
door_coopen_core_bench_4（bench6：開門段恢復偏移估計更新）對 door_coopen_core_bench_3 的比較（只讀；Codex r28：先比**共同門角區間**，新增段另報）。
窗口：各趟 OPEN 起至「下一相位或中止事件」取早者（同判定器 r12 的 OPEN 窗口）；另報 bench_4 OPEN 之後到暫停生效（SETTLE 等）的更新段。
共同區間：[0, min(兩趟 OPEN 窗口最大門角)) 以 5° 分箱；bench_4 超出 bench_3 最大門角的部分列「新增段」。每箱報：
  1. 持握漂移（analysis2 同法：E＝H_meas⁻¹·(T_tcp·G_T_H)，位置範數平均／最大、把手座標 z 分量平均）；
  2. 兩指承載（room finger_contact_n 每指平均、最小、< 0.5 N 步數）；
  3. 命令粗糙度（applied_arm 逐物理步一階差分 RMS，六軸最大者）；
  4. 求解端 d̂ j2（offset_interval.d_after）平均。
另報：相位、中止、最大門角、offset_hold（trigger、時刻、d_frozen、n_upd_before）、OPEN／OPEN 後到凍結的 updated 數、d̂ 是否觸限。

    python3 evaluation/door_coopen_core_bench_4_compare.py   # 寫 runs/door_coopen_core_bench_4/analysis/door_coopen_core_bench_4_compare.json（不覆寫）
"""
import bisect
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_check_core12 as J12                       # noqa: E402
import door_k2_check as KC                                   # noqa: E402

RUNS = {k: os.path.join(HERE, 'runs', f'door_coopen_core_{k}') for k in ('bench_3', 'bench_4', 'bench_5', 'bench_6')}
OUT = os.path.join(RUNS['bench_6'], 'analysis', 'door_coopen_core_bench_6_compare.json')
BIN = 5.0


def load(run):
    ld = (lambda f: json.load(open(os.path.join(run, f))))
    r, t, sv = ld('room_run.json'), ld('task.json'), ld('align_solver.json')
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    t0, t1 = J12._open_window(t)
    t1 = float('inf') if t1 is None else t1
    op = [s for s in r['steps'] if t0 - 1e-9 <= s[c['sim_t']] < t1]
    GTH = np.asarray(t['G_T_H'], float)
    evd = {}
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('T_WH'), list) and len(e['T_WH']) == 16:
            evd[e['step']] = np.asarray(e['T_WH'], float).reshape(4, 4)
    rows = []
    prev_u = None
    for s in op:
        q = math.degrees(s[c['door_q_rad']])
        u = s[c['applied_arm']]
        du = None if (prev_u is None or not isinstance(u, list)) else np.asarray(u, float) - prev_u
        prev_u = np.asarray(u, float) if isinstance(u, list) else None
        d = ez = None
        if s[c['step']] in evd:
            E = np.linalg.inv(evd[s[c['step']]]) @ (KC._tcp(list(s[c['base_xyth']]), list(s[c['q_arm_meas']])) @ GTH)
            d, ez = float(np.linalg.norm(E[:3, 3])) * 1e3, float(E[2, 3]) * 1e3
        f = s[c['finger_contact_n']]
        rows.append({'t': s[c['sim_t']], 'q': q, 'd': d, 'ez': ez, 'f': f if isinstance(f, list) and len(f) == 2 else None, 'du': du})
    ts = [x['t'] for x in rows]
    dh = []
    for x in sv['log']:
        oi = x.get('offset_interval')
        if isinstance(oi, dict) and oi.get('d_after') and t0 - 1e-9 <= x['sim_t'] < t1:
            i = min(len(rows) - 1, max(0, bisect.bisect_left(ts, x['sim_t'])))
            dh.append({'t': x['sim_t'], 'q': rows[i]['q'], 'j2': oi['d_after'][1], 'used': oi.get('used'), 'd': oi['d_after']})
    ev = t['policy_events']
    h = (sv.get('stats') or {}).get('offset_hold')
    upd_after_open = [x for x in sv['log'] if isinstance(x.get('offset_interval'), dict) and x['offset_interval'].get('used') == 'updated'
                      and x['sim_t'] >= t1 - 1e-9 and (h is None or x['sim_t'] < h['sim_t'] - 1e-9)]
    dmax = float(sv['args'].get('offset_max_rad', 0.05))
    D = np.asarray([x['d'] for x in dh], float) if dh else np.zeros((0, 6))
    summ = {'phases': [(e['t'], e['to']) for e in ev if 'to' in e], 'abort': [e for e in ev if e.get('abort')],
            'open_window': [t0, None if t1 == float('inf') else t1], 'q_open_max_deg': max(x['q'] for x in rows) if rows else None,
            'offset_hold': None if not isinstance(h, dict) else {k: h.get(k) for k in ('sim_t', 'trigger', 'd_frozen', 'n_upd_before')},
            'n_updated_open': sum(1 for x in dh if x['used'] == 'updated'), 'n_updated_after_open_before_freeze': len(upd_after_open),
            'd_hat_open': {'min': D.min(axis=0).tolist() if len(D) else None, 'max': D.max(axis=0).tolist() if len(D) else None,
                           'hit_clip': bool(len(D) and np.any(np.abs(D) >= dmax - 1e-12))},
            'n_updates_report': (sv.get('offset_free') or {}).get('n_updates')}
    return rows, dh, summ


def bin_stats(rows, dh, lo, hi):
    w = [x for x in rows if lo <= x['q'] < hi]
    if not w:
        return None
    dd = [x['d'] for x in w if x['d'] is not None]
    F = np.asarray([x['f'] for x in w if x['f'] is not None], float)
    dU = np.asarray([x['du'] for x in w if x['du'] is not None], float)
    j2 = [x['j2'] for x in dh if lo <= x['q'] < hi]
    return {'n': len(w), 'drift_mean_mm': float(np.mean(dd)) if dd else None, 'drift_max_mm': float(np.max(dd)) if dd else None,
            'ez_mean_mm': float(np.mean([x['ez'] for x in w if x['ez'] is not None])) if dd else None,
            'finger_mean_n': F.mean(axis=0).tolist() if len(F) else None, 'finger_min_n': F.min(axis=0).tolist() if len(F) else None,
            'finger_below_0p5': (F < 0.5).sum(axis=0).tolist() if len(F) else None,
            'arm_du_rms_max': float(np.sqrt((dU ** 2).mean(axis=0)).max()) if len(dU) else None,
            'd_hat_j2_mean': float(np.mean(j2)) if j2 else None}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    D = {k: load(v) for k, v in RUNS.items()}
    qc = min(D[k][2]['q_open_max_deg'] for k in D)
    edges = [x * BIN for x in range(int(math.floor(qc / BIN)) + 1)] + [qc]
    common = []
    for lo, hi in zip(edges, edges[1:]):
        if hi > lo:
            common.append({'q_bin_deg': [lo, hi], **{k: bin_stats(D[k][0], D[k][1], lo, hi) for k in D}})
    new = {k: ({'q_range_deg': [qc, D[k][2]['q_open_max_deg']], **bin_stats(D[k][0], D[k][1], qc, D[k][2]['q_open_max_deg'] + 1e-9)}
               if D[k][2]['q_open_max_deg'] > qc + 1e-9 else None) for k in D}
    out = {'schema': 'door_coopen_core_bench_6_compare/1', 'q_common_max_deg': qc, 'summary': {k: D[k][2] for k in D}, 'common_bins': common,
           'new_segment_beyond_common': new,
           'note': '共同門角區間逐箱比較；新增段另報；窗口＝OPEN 起至下一相位或中止事件'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in D:
        s = D[k][2]
        print(k, '門角最大 %.2f°' % s['q_open_max_deg'], '中止', [a.get('why') or a.get('abort') for a in s['abort']], 'hold', s['offset_hold'] and
              {kk: s['offset_hold'][kk] for kk in ('sim_t', 'trigger')}, 'OPEN 更新', s['n_updated_open'], 'OPEN 後到凍結更新', s['n_updated_after_open_before_freeze'],
              '觸限', s['d_hat_open']['hit_clip'])
    print('共同區間 0–%.2f°' % qc)
    f = (lambda v, p='%.2f': '—' if v is None else p % v)
    for b in common:
        print('  %5.1f–%5.1f°  漂移均 %s  最大 %s  ez %s  指均 %s  臂差分 RMS %s  d̂j2 %s' % (b['q_bin_deg'][0], b['q_bin_deg'][1],
              '/'.join(f(b[k] and b[k]['drift_mean_mm']) for k in D), '/'.join(f(b[k] and b[k]['drift_max_mm']) for k in D),
              '/'.join(f(b[k] and b[k]['ez_mean_mm']) for k in D), '/'.join(str(b[k] and [round(v, 2) for v in b[k]['finger_mean_n']]) for k in D),
              '/'.join(f(b[k] and b[k]['arm_du_rms_max'], '%.5f') for k in D), '/'.join(f(b[k] and b[k]['d_hat_j2_mean'], '%.4f') for k in D)))
    for k, v in new.items():
        if v:
            print('新增段', k, json.dumps(v, default=float)[:400])


if __name__ == '__main__':
    main()
