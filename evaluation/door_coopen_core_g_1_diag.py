#!/usr/bin/env python3
"""door_coopen_core_g_1 中止診斷（只讀；timeout:APPROACH）：閘控時間線、s 進度、各暫停區間內垂直誤差的收斂（目標靜止時）。
**界線**：只描述；「目標靜止時側向誤差收斂到非零值、且隨 s 減小而增大」為觀察，未確認來源（求解器取捨、約束或其他）。

    python3 evaluation/door_coopen_core_g_1_diag.py   # 寫 runs/door_coopen_core_g_1/analysis/door_coopen_core_g_1_diag.json（不覆寫）
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_g_1_diag.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t = json.load(open(os.path.join(RUN, 'task.json')))
    tr = [r for r in t['trace'] if r.get('phase') == 'APPROACH' and (r.get('gate') or {}).get('info')]
    pauses, cur = [], None
    for r in tr:
        g = r['gate']
        i = g['info']
        if g['paused']:
            if cur is None:
                cur = {'t0': r['t'], 's': i['s'], 'perp0_mm': i['perp_m'] * 1e3, 'perp_min_mm': i['perp_m'] * 1e3, 'perp_series': []}
            cur['perp_min_mm'] = min(cur['perp_min_mm'], i['perp_m'] * 1e3)
            cur['perp_series'].append([r['t'], i['perp_m'] * 1e3])
            cur['t1'], cur['perp1_mm'] = r['t'], i['perp_m'] * 1e3
        elif cur is not None:
            pauses.append(cur)
            cur = None
    if cur is not None:
        pauses.append(cur)
    s0 = tr[0]['gate']['info']['s'] if tr else None
    res = {'schema': 'door_coopen_core_g_1_diag/1', 'gate_stats': t.get('approach_gate_stats'),
           'events': [e for e in t['policy_events'] if 'approach_gate' in str(e) or 'abort' in e or e.get('to') in ('ALIGN_HOLD', 'APPROACH')],
           's_start': s0, 's_end': tr[-1]['gate']['info']['s'] if tr else None, 't_approach': [tr[0]['t'], tr[-1]['t']] if tr else None,
           'pauses': [{k: v for k, v in p.items() if k != 'perp_series'} | {'duration_s': p['t1'] - p['t0']} for p in pauses],
           'pause_series': [p['perp_series'] for p in pauses],
           'perp_vs_s_moving': [[r['gate']['info']['s'], r['gate']['info']['perp_m'] * 1e3, r['gate']['info']['along_m'] * 1e3] for r in tr[::10] if not r['gate']['paused']],
           'observation': '目標靜止（暫停）時垂直誤差收斂緩慢且到非零值：s 0.0348 處 3.2 s 由 3.0 降到 ≤ 2 mm；s 0.0224 處 4.2 s 由 3.5 降到 2.3 mm 未達 2 mm；推進中垂直誤差隨 s 減小而增大（0.4 → 3.5 mm）'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for p in res['pauses']:
        print(f"暫停 s={p['s']:.4f}：{p['t0']:.2f}–{p['t1']:.2f} s（{p['duration_s']:.1f} s），垂直 {p['perp0_mm']:.1f} → 最小 {p['perp_min_mm']:.1f} → 末 {p['perp1_mm']:.1f} mm")
    print('s', res['s_start'], '→', res['s_end'], 'stats', res['gate_stats'])


if __name__ == '__main__':
    main()
