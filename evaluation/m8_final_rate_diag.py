#!/usr/bin/env python3
"""最終命令（/cmd_vel）變化率超限事件的成因配對（Codex 20261110_221334：以 Shield 週期序號精確配對，先不混入控制修改）。只讀紀錄。
每筆事件：找輸出值相同的 /shield/diag 週期 k 與 k−1，報 Shield 是否改寫（active、輸出−輸入）、輸入差（＝選擇器送出的差）、
週期間隔（接收時間）、同期 /cmd_vel_pre_mux（平滑器）在兩週期間的訊息數與值；並統計導航節點輸出（/cmd_vel_pre_shield）長間隔的週期性。
分類：shield_rewrite（active 或輸出≠輸入）｜selector_stale（兩週期間平滑器有新值但選擇器送舊值或漏拍）｜record_burst（錄製端同時收到、時間換算失真）｜other。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_final_rate_diag.py RUN_ID
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402

A_LIM = np.array([1.5, 1.0, 2.0])


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/shield/diag', '/cmd_vel', '/cmd_vel_pre_mux', '/cmd_vel_pre_shield'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def ts(r):
        return float(np.interp(r, rec, od[:, 0]))
    post = [(r, ts(r), np.array([m.linear.x, m.linear.y, m.angular.z])) for r, m in B['/cmd_vel']]
    sd = [(r, np.asarray(m.data, float)) for r, m in B['/shield/diag']]
    sdr = np.array([x[0] for x in sd])
    pm = [(r, np.array([m.linear.x, m.linear.y, m.angular.z])) for r, m in B['/cmd_vel_pre_mux']]
    pmr = np.array([x[0] for x in pm])
    ev = []
    for i in range(1, len(post)):
        dt = post[i][1] - post[i - 1][1]
        if dt <= 1e-3:
            continue
        a = np.abs(post[i][2] - post[i - 1][2]) / dt
        if not (a > A_LIM * 1.05).any():
            continue
        j = int(np.argmin(np.abs(sdr - post[i][0])))
        cand = [k for k in range(max(1, j - 3), min(len(sd), j + 4)) if np.abs(sd[k][1][8:11] - post[i][2]).max() < 1e-5]
        k = cand[0] if cand else j
        d0, d1 = sd[k - 1][1], sd[k][1]
        gap = float(sdr[k] - sdr[k - 1])
        inb = (pmr > sdr[k - 1] - 0.06) & (pmr < sdr[k])
        pm_vals = [p[1].round(3).tolist() for p, ok in zip(pm, inb) if ok]
        rewrite = bool(d1[1] > 0.5 or np.abs(d1[8:11] - d1[5:8]).max() > 1e-6)
        burst = bool(np.sum(np.abs(sdr - sdr[k]) < 1e-3) > 1 or gap < 0.005)
        stale = bool(np.allclose(d0[5:8], sd[k - 2][1][5:8]) and gap > 0.075) if k >= 2 else False
        stale = stale or bool(sum(1 for _ in pm_vals) >= 2 and gap > 0.075)
        kind = 'shield_rewrite' if rewrite else ('record_burst' if burst else ('selector_stale' if stale or (k >= 2 and sdr[k - 1] - sdr[k - 2] > 0.075) else 'other'))
        ev.append(dict(t_sim=round(post[i][1], 3), dt_msg=round(dt, 3), acc=a.round(2).tolist(), axes_over=[bool(x) for x in a > A_LIM * 1.05],
                       cycles=[int(d0[0]), int(d1[0])], cycle_gap_s=round(gap, 3), prev_gap_s=round(float(sdr[k - 1] - sdr[k - 2]), 3) if k >= 2 else None,
                       shield_active=[int(d0[1]), int(d1[1])], d_in=(d1[5:8] - d0[5:8]).round(3).tolist(), d_out=(d1[8:11] - d0[8:11]).round(3).tolist(),
                       in_prev2=sd[k - 2][1][5:8].round(3).tolist() if k >= 2 else None, in_prev=d0[5:8].round(3).tolist(), in_now=d1[5:8].round(3).tolist(),
                       smoother_between=pm_vals, kind=kind))
    r = np.array([x[0] for x in B['/cmd_vel_pre_shield']])
    g = np.diff(r)
    big = np.flatnonzero(g > 0.08)
    own = [(round(ts(r[i]), 3), round(float(g[i]) * 1e3, 1)) for i in big if ((pmr > r[i]) & (pmr < r[i + 1])).any()]
    tt = np.array([x[0] for x in own])
    out = dict(run=run, n_events=len(ev), events=ev,
               selector_gaps=dict(n=len(own), list=own, period_s=None if len(tt) < 3 else round(float(np.median(np.diff(tt))), 2),
                                  median_gap_ms=None if not own else float(np.median([x[1] for x in own])),
                                  note='/cmd_vel_pre_shield（導航節點選擇器輸出）間隔 > 80 ms 且期間平滑器仍有輸出＝節點本身漏拍'),
               note='事件以 /cmd_vel 相鄰訊息差分（bag 接收時間換算模擬時間）找出；再以 /shield/diag 週期序號配對')
    json.dump(out, open(os.path.join(D, 'm8_final_rate_diag.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'selector_gaps'}, ensure_ascii=False, indent=1, default=float))
    print('選擇器漏拍', out['selector_gaps']['n'], '次；週期中位', out['selector_gaps']['period_s'], 's；間隔中位', out['selector_gaps']['median_gap_ms'], 'ms')


if __name__ == '__main__':
    main()
