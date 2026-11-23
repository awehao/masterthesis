#!/usr/bin/env python3
"""R1 時間壓縮（提速包；路徑不變）：參考包 pb60h 的 mode 4（R1 近原地轉）依因子重取樣，其餘段原樣；偏航峰值受 w_cap 0.1 rad/s 限制。
    python3 evaluation/door_ref_r1_retime.py --factor 1.3 --out evaluation/results/door/door_regrasp_ref_pb60i.json
"""
import argparse
import json
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
    ap.add_argument('--factor', type=float, default=1.3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    P = json.load(open(a.src))
    s = P['segments']['4']
    n0 = len(s['t'])
    n1 = int(np.ceil((n0 - 1) / a.factor)) + 1
    u = np.minimum(np.arange(n1) * a.factor, n0 - 1)
    i0 = np.minimum(np.floor(u).astype(int), n0 - 2)
    w = u - i0
    def itp(v):
        v = np.asarray(v, float)
        ww = w.reshape((-1,) + (1,) * (v.ndim - 1))
        return (v[i0] * (1 - ww) + v[i0 + 1] * ww).tolist()
    out = dict(s)
    dt = float(P['dt'])
    out['t'] = (float(s['t'][0]) + np.arange(n1) * dt).tolist()
    for k in ('tcp', 'base', 'arm', 'qf', 'q_door_deg'):
        out[k] = itp(s[k])
    out['base_v'] = (np.asarray(itp(s['base_v'])) * a.factor).tolist()
    out['src_index'] = [s['src_index'][int(i)] for i in np.round(u).astype(int)]
    out['stage'] = [s['stage'][int(i)] for i in np.round(u).astype(int)]
    P['segments']['4'] = out
    P['r1_retime'] = {'factor': a.factor, 'n0': n0, 'n1': n1, 'src': os.path.basename(a.src)}
    b = np.asarray(out['base'], float)
    w_ = np.diff(np.unwrap(b[:, 2])) / dt
    v_ = np.hypot(*(np.diff(b[:, :2], axis=0) / dt).T)
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    t, st, wn, mn = 0.0, 0, 0, {}
    arm, qd, qf = np.asarray(out['arm'], float), np.radians(np.asarray(out['q_door_deg'], float)), np.asarray(out['qf'], float)
    for i in range(n1):
        t += dt
        dec, det = mon.step(t, np.r_[b[i], arm[i]], float(qd[i]), float(np.max(qf[i])), 0.1, None)
        st += dec == 'stop'
        wn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v['who'])
    print(f'R1 {n0}→{n1} 列（{n0 * dt:.1f}→{n1 * dt:.1f} s）w 峰 {np.abs(w_).max():.4f} v 峰 {v_.max():.4f} 手臂列差 {np.abs(np.diff(arm, axis=0)).max():.5f}；監看 stop {st} warn {wn} {mn}')
    if st:
        raise SystemExit('離線監看有 stop，不寫檔')
    json.dump(P, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out)


if __name__ == '__main__':
    raise SystemExit(main())
