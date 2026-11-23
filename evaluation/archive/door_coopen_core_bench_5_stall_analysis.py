#!/usr/bin/env python3
"""door_coopen_core_bench_5 停頓定位（只讀；Codex r29：回報停頓落在哪個區段）。
來源：執行端 exec_timing.npz（ev9r3：每輪 8 標記單調牆鐘、GC 各代起訖）、room_run exec_timing 摘要；任務端 task.json trace（wall_mono＝本輪起點、
state_age_wall）與 task_trace.jsonl（enq_wall_mono）。兩端時鐘皆 time.monotonic（同系統時鐘）。
報告：
  1. 執行端每輪總耗時（本輪起點→下一輪起點）> 0.1 s 的輪：步號、模擬時刻、各區段耗時、與該輪時間重疊的 GC（generation、耗時）。
  2. 任務端：state_age_wall > 0.09 s 的列；以「本輪起點 − state_age_wall」推得最後一次收到狀態的牆鐘，對照執行端該步附近的步後發布時刻
     （該步 after_world_step 至下一輪起點之間）⇒ 傳輸／回呼延遲上界的估計（只是時序配對，不是逐訊息追蹤）。
  3. 全程各區段分位數、GC 各代次數與最大耗時、最慢 GC。

    python3 evaluation/door_coopen_core_bench_5_stall_analysis.py [run_dir]   # 寫 <run>/analysis/door_coopen_core_bench_5_stall_analysis.json（不覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exec_timing_diag as X                                 # noqa: E402

SLOW_S = 0.1


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'runs', 'door_coopen_core_bench_5')
    out_p = os.path.join(run, 'analysis', 'door_coopen_core_bench_5_stall_analysis.json')
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    z = np.load(os.path.join(run, 'exec_timing.npz'))
    T, meta, G = z['T'], z['meta'], z['G']
    st = X.StepTimer(cap=len(T) or 1)
    st.T[:len(T)], st.meta[:len(T)], st.row = T, meta, len(T) - 1
    st.G = G if len(G) else np.full((1, 6), np.nan)
    st.n_gc = len(G)
    S = st.sections()
    tot = np.full(len(T), np.nan)
    tot[:-1] = T[1:, 0] - T[:-1, 0]
    gcs = st.gc_events()
    slow = []
    for i in np.where(np.nan_to_num(tot) > SLOW_S)[0]:
        w0, w1 = T[i, 0], T[i + 1, 0]
        ov = [{'wall_start': g[0], 'dur_s': g[1], 'generation': g[2], 'collected': g[3]} for g in gcs if g[0] < w1 and g[0] + g[1] > w0]
        slow.append({'row': int(i), 'step': float(meta[i, 0]), 'sim_t': float(meta[i, 1]), 'wall_start': float(w0), 'total_s': float(tot[i]),
                     'sections_s': {s: (None if not np.isfinite(S[i, j]) else float(S[i, j])) for j, s in enumerate(X.SECTIONS)},
                     'dominant_section': X.SECTIONS[int(np.nanargmax(np.nan_to_num(S[i], nan=-1)))], 'gc_overlap': ov})
    t = json.load(open(os.path.join(run, 'task.json')))
    task_slow = []
    for e in t.get('trace') or []:
        a = e.get('state_age_wall')
        if isinstance(a, (int, float)) and a > 0.09 and isinstance(e.get('wall_mono'), (int, float)) and e.get('t', 0) > 1.0:
            rx = e['wall_mono'] - a
            k = int(np.searchsorted(T[:, 0], rx)) - 1
            task_slow.append({'sim_t': e['t'], 'phase': e.get('phase'), 'state_age_wall': a, 'loop_wall': e['wall_mono'], 'last_state_rx_wall': rx,
                              'exec_row_at_rx': k, 'exec_step_at_rx': (None if k < 0 else float(meta[k, 0])),
                              'exec_rows_during_age': int(np.sum((T[:, 0] > rx) & (T[:, 0] <= e['wall_mono'])))})
    ab = [x for x in t.get('policy_events') or [] if x.get('abort')]
    out = {'schema': 'door_coopen_core_bench_5_stall_analysis/1', 'abort': ab, 'n_rows': int(len(T)), 'exec_slow_rows': slow, 'task_stale_rows': task_slow,
           'summary': json.load(open(os.path.join(run, 'room_run.json'))).get('exec_timing'),
           'gc_top': sorted([{'wall_start': g[0], 'dur_s': g[1], 'generation': g[2], 'collected': g[3], 'row': g[4]} for g in gcs],
                            key=lambda x: -x['dur_s'])[:10],
           'note': '時序配對；不是逐訊息追蹤。exec_rows_during_age＝任務端兩次收到狀態之間執行端前進的輪數（> 0 而任務端未收到 ⇒ 傳輸／回呼側）'}
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(out, open(out_p, 'w'), ensure_ascii=False, indent=1, default=float)
    print('中止', ab)
    print('執行端 > %.1f s 的輪：%d' % (SLOW_S, len(slow)))
    for x in slow[:10]:
        print('  step %d sim %.2f total %.3f s 主區段 %s %s GC %s' % (x['step'], x['sim_t'], x['total_s'], x['dominant_section'],
              {k: round(v, 4) for k, v in x['sections_s'].items() if v is not None and v > 0.005}, [(g['generation'], round(g['dur_s'], 4)) for g in x['gc_overlap']]))
    print('任務端 state_age > 0.09 s：')
    for x in task_slow[:10]:
        print('  sim %.2f age %.3f 執行端在期間前進 %d 輪（收到時執行端步 %s）' % (x['sim_t'], x['state_age_wall'], x['exec_rows_during_age'], x['exec_step_at_rx']))
    print('最慢 GC', [(g['generation'], round(g['dur_s'], 4), g['row']) for g in out['gc_top'][:5]])


if __name__ == '__main__':
    main()
