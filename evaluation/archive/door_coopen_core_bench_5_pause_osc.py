#!/usr/bin/env python3
"""door_coopen_core_bench_5 暫停緩移的補充對時（只讀；Codex r30 §3 的同一短分析）：
  1. 各相位窗口的持握振盪：j6 反作用力 Fz 標準差與主頻（FFT，dt 0.01 s；50 Hz＝逐步交替）、finger1 力標準差、j2 去線性趨勢後的標準差與峰峰值。
  2. 停止觀察段（STOPPED → DONE／FAIL）與先前通過暫停的 door_wide_w1f_60_3（同為暫停期手指上限 20 N）對照：手臂 0.5 s 窗口速度最大／p99、
     j2 去趨勢峰峰、j2 線性斜率、兩指合力平均、finger1 標準差。
界線：描述性對照；w1f_60_3 為不同抓法與門角，不是同條件對照；不判定原因。

    python3 evaluation/door_coopen_core_bench_5_pause_osc.py   # 寫 runs/door_coopen_core_bench_5/analysis/door_coopen_core_bench_5_pause_osc.json（不覆寫）
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_5')
REF = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_5_pause_osc.json')
SEGS = {'OPEN_60_65': (60.0, 65.0), 'OPEN_95_100': (95.0, 100.0), 'SETTLE_101.5_102.9': (101.5, 102.9), 'PAUSE_20N_103.1_104.7': (103.1, 104.7),
        'LOWSPEED_104.75_107.1': (104.75, 107.1), 'STOPPED_107.19_108.04': (107.19, 108.04)}


def load(run):
    r = json.load(open(os.path.join(run, 'room_run.json')))
    ev = {}
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        e = json.loads(ln)
        ev[e['step']] = e
    return r, {k: i for i, k in enumerate(r['steps_cols'])}, ev


def detr(t, y):
    return y - np.polyval(np.polyfit(t - t[0], y, 1), t - t[0])


def stop_stats(run):
    r, c, _ = load(run)
    o = json.load(open(os.path.join(run, 'align_solver.json')))['stats']['k3m_pause']['supervisor_outcome']
    ts = o['transitions']['STOPPED']
    te = o['transitions'].get('DONE') or o['transitions'].get('FAIL')
    S = [s for s in r['steps'] if ts <= s[c['sim_t']] <= te]
    T = np.array([s[c['sim_t']] for s in S])
    Q = np.array([s[c['q_arm_meas']] for s in S], float)
    F = np.array([s[c['finger_contact_n']] for s in S], float)
    w = []
    for i in range(len(T)):
        k = int(np.searchsorted(T, T[i] - 0.5 - 1e-6))
        if T[i] - T[k] > 0.4:
            w.append(float((np.abs(Q[i] - Q[k]) / (T[i] - T[k])).max()))
    d = detr(T, Q[:, 1])
    return {'state': o['state'], 'stopped_t': ts, 'end_t': te, 'arm_win_max': max(w), 'arm_win_p99': float(np.percentile(w, 99)),
            'j2_detrended_pp_mrad': float(d.max() - d.min()) * 1e3, 'j2_slope_rad_s': float(np.polyfit(T - T[0], Q[:, 1], 1)[0]),
            'f_sum_mean_n': float(F.sum(axis=1).mean()), 'f1_std_n': float(F[:, 0].std()),
            'pause_finger_effort_set': ((r.get('finger_effort') or {}).get('pause') or {}).get('set')}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r, c, ev = load(RUN)
    seg = {}
    for k, (a, b) in SEGS.items():
        S = [s for s in r['steps'] if a <= s[c['sim_t']] < b and s[c['step']] in ev]
        t = np.array([s[c['sim_t']] for s in S])
        fz = np.array([ev[s[c['step']]]['j6_reaction_wrench'][2] for s in S], float)
        f1 = np.array([s[c['finger_contact_n']][0] for s in S], float)
        d = detr(t, np.array([s[c['q_arm_meas']][1] for s in S], float))
        sp = np.abs(np.fft.rfft(fz - fz.mean()))
        fr = np.fft.rfftfreq(len(fz), 0.01)
        seg[k] = {'n': len(S), 'fz_std_n': float(fz.std()), 'fz_dominant_hz': float(fr[1:][sp[1:].argmax()]), 'f1_std_n': float(f1.std()),
                  'j2_detrended_std_mrad': float(d.std()) * 1e3, 'j2_detrended_pp_mrad': float(d.max() - d.min()) * 1e3}
    out = {'schema': 'door_coopen_core_bench_5_pause_osc/1', 'segments': seg,
           'stop_compare': {'bench_5': stop_stats(RUN), 'door_wide_w1f_60_3': stop_stats(REF)},
           'note': '描述性對照；w1f_60_3 為不同抓法與門角；不判定原因'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, v in seg.items():
        print('%-24s Fz std %.2f N（主頻 %.1f Hz） f1 std %.2f N  j2 去趨勢 std %.3f／峰峰 %.3f mrad' % (k, v['fz_std_n'], v['fz_dominant_hz'], v['f1_std_n'],
              v['j2_detrended_std_mrad'], v['j2_detrended_pp_mrad']))
    for k, v in out['stop_compare'].items():
        print(k, json.dumps(v, default=float))


if __name__ == '__main__':
    main()
