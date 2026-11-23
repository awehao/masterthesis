#!/usr/bin/env python3
"""S0c 逐步紀錄獨立重算：實測下降速度、各禁止通道峰值與首個非零時刻、觸發時 TCP 高度、觸發後額外位移、停止尾段手臂／底盤速度。
    python3 evaluation/bottle_probe_analyze.py <RUN_ID>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C                              # noqa: E402


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 'probe_steps.npz'))
    R, cols = Z['rows'], [str(c) for c in Z['cols']]
    rep = json.load(open(os.path.join(D, 'probe_report.json')))
    c = {k: R[:, i] for i, k in enumerate(cols)}
    t = c['t']
    K = C.S0_K()
    Q = np.c_[c['base_x'], c['base_y'], c['base_yaw'], *[c[f'q{i}'] for i in range(1, 7)]]
    z = np.array([K.fk(q, 'link_tcp')[2, 3] for q in Q])
    td, ts = rep['t_descend'], (rep['stop'] or {}).get('t')
    print(f"probe_pass {rep['probe_pass']}（碼 {rep['probe_exit']}）第一阻點 {rep['stop']}；t_descend {td}；步數 {len(t)}")
    if td is not None:
        m = (t >= td + 1.0) & (t <= (ts if ts else t[-1]) - 0.2)
        if m.sum() > 2:
            print(f'實測 TCP 下降速度 {-np.polyfit(t[m], z[m], 1)[0] * 1e3:.3f} mm/s（參考 {rep["v_down"] * 1e3:.1f}）')
    if ts is not None:
        i = int(np.searchsorted(t, ts))
        print(f'觸發時 t {ts:.2f} TCP z {z[i]:.5f}；下降量 {(z[np.searchsorted(t, td)] - z[i]) * 1e3:.2f} mm；參考進給 {rep["feed_at_stop_m"] * 1e3:.2f} mm；設定點額外位移見下')
        print(f'觸發後 TCP 額外下降 {(z[i] - z[i:].min()) * 1e3:.3f} mm；末 TCP z {z[-1]:.5f}')
        for w, a_, b_ in (('0–0.5 s', 0.0, 0.5), ('0.5–1 s', 0.5, 1.0), ('1 s 後', 1.0, 9.9)):
            m2 = (t >= ts + a_) & (t < ts + b_)
            if m2.sum() > 1:
                dq = np.abs(np.diff(Q[m2, 3:], axis=0)).max() / 0.01
                vb = np.hypot(*np.diff(Q[m2, :2], axis=0).T).max() / 0.01
                print(f'  停止尾段 {w}：手臂峰 {dq:.5f} rad/s、底盤峰 {vb:.5f} m/s')
    print('停止有效', rep['stop_effect'])
    for k in cols:
        if k.startswith('forb_'):
            v = c[k]
            nz = np.where(np.nan_to_num(v) > 0)[0]
            print(f"  {k:<22} 峰 {np.nanmax(v):.3f} N、>0 步 {len(nz)}、首 t {t[nz[0]] if len(nz) else None}、≥0.05 N 步 {int((np.nan_to_num(v) >= 0.05).sum())}、NaN {int(np.isnan(v).sum())}")
    print('脈衝', rep['forbidden_pulses'], '穿透估計峰', rep['pen_max_mm'], 'mm')
    print(f"瓶水平位移峰 {np.hypot(c['bottle_base_x'] - c['bottle_base_x'][0], c['bottle_base_y'] - c['bottle_base_y'][0]).max() * 1e3:.3f} mm；底盤漂移 {np.hypot(c['base_x'] - c['base_x'][0], c['base_y'] - c['base_y'][0]).max() * 1e3:.2f} mm")


if __name__ == '__main__':
    sys.exit(main())
