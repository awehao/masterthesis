#!/usr/bin/env python3
"""實跑命令鏈的手臂命令變化率（Codex 20261102_174123：用實跑紀錄補算，不用假世界數值代替）。
來源：runs/<run>/s1_cbf.npz 的 u（每控制拍 GMPC 求解輸出，底盤 3＋手臂 6）與 t（任務時刻）。變化率＝相鄰控制拍 |Δu_arm|／Δt（rad/s²），對照 A_ARM。
兩趟：bottle_rb1_1（固定 V_HIGH）、bottle_rb1_gaze_1（視向）。視窗：
  接近段＝U 開始到漸變起點（blend_start）；事件窗＝各事件 ±0.25 s：視向起步（gaze ev.t_first_offset，只有視向趟）、U、H、R、DONE。
每窗報各關節最大值與發生時刻；接近段另報 p95 與「> 2 rad/s²」拍數。
    python3 evaluation/bottle_gaze_cmd_rate.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C0                             # noqa: E402

RUNS = ('bottle_rb1_1', 'bottle_rb1_gaze_1')
OUT = os.path.join(HERE, 'results', 'bottle', 'gaze_cmd_rate.json')
HALF = 0.25


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    t, U = np.asarray(Z['t'], float), np.asarray(Z['u'], float)[:, 3:]
    dt = np.diff(t)
    ok = dt > 1e-6
    du = np.abs(np.diff(U, axis=0))[ok] / dt[ok][:, None]
    tm = t[1:][ok]
    ev = {e[1]: float(e[0]) for e in r['vhigh_seq']['log']}
    tb = next(e['t'] for e in r['events'] if e['what'] == 'blend_start')
    gz = (r.get('gaze') or {}).get('ev') or {}
    if gz.get('t_first_offset') is not None:
        ev['gaze_onset'] = float(gz['t_first_offset'])

    def win(lo, hi):
        m = (tm >= lo - 1e-9) & (tm <= hi + 1e-9)
        if not m.any():
            return None
        d = du[m]
        k = d.argmax(axis=0)
        return dict(n=int(m.sum()), max=d.max(axis=0).round(3).tolist(), t_max=[round(float(tm[m][i]), 3) for i in k], u_abs_max=np.abs(U[1:][ok][m]).max(axis=0).round(4).tolist())
    ap = win(ev['U'], tb)
    m = (tm >= ev['U']) & (tm <= tb)
    ap.update(p95=np.percentile(du[m], 95, axis=0).round(3).tolist(), n_over_2=[int((du[m][:, j] > 2.0).sum()) for j in range(6)])
    return dict(run=run, t_blend=tb, events=ev, approach=ap, at_event={k: win(v - HALF, v + HALF) for k, v in ev.items()}, dt_median=float(np.median(dt)))


def main():
    out = dict(A_ARM=C0.A_ARM, half_window_s=HALF, note='GMPC 每控制拍輸出的手臂命令（s1_cbf.npz u[3:]）相鄰拍差分／Δt；不是設定點網格二階差分', runs={k: one(k) for k in RUNS})
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for k, o in out['runs'].items():
        print('==', k, '漸變起點', o['t_blend'], '事件', o['events'])
        a = o['approach']
        print('   接近段 max', a['max'], 't', a['t_max'], 'p95', a['p95'], '>2', a['n_over_2'], '|u|max', a['u_abs_max'])
        for e, w in o['at_event'].items():
            print(f'   {e:11s} ±{HALF}s max', None if w is None else w['max'], 't', None if w is None else w['t_max'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
