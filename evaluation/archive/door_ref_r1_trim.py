#!/usr/bin/env python3
"""R1 截短（銜接包）：參考段 4 只留收臂段（前 --rows 列），之後交導航由 GMPC 邊走邊轉；路徑與 R2 計畫不變。離線以實際監看核對截短段與轉向接 R2 的淨空。
    python3 evaluation/door_ref_r1_trim.py --rows 90 --out evaluation/results/door/door_regrasp_ref_pb60j.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60h.json'))
    ap.add_argument('--rows', type=int, default=90)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    P = json.load(open(a.src))
    s = P['segments']['4']
    n0 = len(s['t'])
    for k, v in s.items():
        if isinstance(v, list) and len(v) == n0:
            s[k] = v[:a.rows + 1]
    P['r1_trim'] = {'rows': a.rows, 'n0': n0, 'src': os.path.basename(a.src)}
    b, arm = np.asarray(s['base'], float), np.asarray(s['arm'], float)
    stow = np.asarray(P['stow'], float)
    print(f"R1 {n0}→{a.rows + 1} 列（{n0 * P['dt']:.1f}→{(a.rows + 1) * P['dt']:.1f} s）；末位姿 {b[-1].round(3).tolist()}；手臂對收攏 {np.abs(arm[-1] - stow).max():.4f} rad")
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    V = np.asarray(P['m4r']['R2']['vertices'], float)
    q = math.radians(float(P['segments']['8']['q_door_deg'][0]) if False else 60.0)
    x = b[-1].copy()
    path = np.vstack([x[:2], V])
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    t, s_, st, wn, mn, dt = 0.0, 0.0, 0, 0, {}, 0.05
    v_cmd, y = 0.0, float(x[2])
    while s_ < L[-1] and t < 60.0:
        t += dt
        v_cmd = min(0.28, v_cmd + 0.5 * dt)
        s_ += v_cmd * dt
        p = np.array([np.interp(s_, L, path[:, 0]), np.interp(s_, L, path[:, 1])])
        pa = np.array([np.interp(min(s_ + 0.4, L[-1]), L, path[:, 0]), np.interp(min(s_ + 0.4, L[-1]), L, path[:, 1])])
        h = math.atan2(*(pa - p)[::-1])
        y += float(np.clip(math.atan2(math.sin(h - y), math.cos(h - y)), -1.0 * dt, 1.0 * dt))
        dec, det = mon.step(t, np.r_[p, y, stow], q, 0.0089, 0.1, None)
        st += dec == 'stop'
        wn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v['who'], round(t, 1))
    print(f'R2 邊走邊轉（運動學）{t:.1f} s、路長 {L[-1]:.2f} m：stop {st} warn {wn} {mn}')
    if st:
        raise SystemExit('離線監看有 stop，不寫檔')
    json.dump(P, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out)


if __name__ == '__main__':
    raise SystemExit(main())
