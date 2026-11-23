#!/usr/bin/env python3
"""返程 A 段 S4（推開）重時間化（Codex 20261031 條件核可）：沿 S4 原列序列（x＝底盤＋手臂、q_deg）以同一進度參數 τ 重時間化，
固定 20 Hz 輸出；dτ/dt＝r(τ)，r ≥ 1、門速 r·|q̇₀| ≤ --door-rate、底盤 r·v₀ ≤ --v-max，頭尾 --keep-s 內 r＝1（原速、原端點），
dr/dt ≤ min(門／線／角加速度上限 ÷ 原速)（前後向掃描）。其他段（含 B、S8）列不動。輸出新包，不覆寫來源。
離線核：輸出檔 20 Hz 差分的本體速度、手臂速率、門速與加速度；S3→S4→S5 接點；幾何監看（stop／warn 皆 0 才寫檔）。
    python3 evaluation/door_return_ref_s4_retime.py --out evaluation/results/door/door_return_ref_v3.json
"""
import argparse
import hashlib
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

V_BOX, W_BOX, ARM_BOX = 0.035255, 0.1999, 0.9999


def rates(X, q, dt):
    d = np.diff(X, axis=0) / dt
    c, s = np.cos(X[:-1, 2]), np.sin(X[:-1, 2])
    vb = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]]
    return vb, np.abs(d[:, 3:]).max(1), np.diff(q) / dt, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(HERE, 'results', 'door', 'door_return_ref_v2.json'))
    ap.add_argument('--seg', default='A')
    ap.add_argument('--stage', default='S4')
    ap.add_argument('--door-rate', type=float, default=1.0)        # °/s
    ap.add_argument('--v-max', type=float, default=0.0125)
    ap.add_argument('--a-door', type=float, default=0.5)           # °/s²
    ap.add_argument('--a-lin', type=float, default=0.02)
    ap.add_argument('--a-ang', type=float, default=0.3)
    ap.add_argument('--keep-s', type=float, default=1.5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    R = json.load(open(a.src))
    dt = float(R['dt'])
    rows = R[a.seg]
    idx = [i for i, r in enumerate(rows) if r['stage'].startswith(a.stage)]
    i0, i1 = idx[0], idx[-1]
    if idx != list(range(i0, i1 + 1)):
        raise SystemExit('段列不連續')
    X = np.array([r['x'] for r in rows[i0:i1 + 1]], float)
    q = np.array([r['q_deg'] for r in rows[i0:i1 + 1]], float)
    n = len(X)
    tau_rows = np.arange(n) * dt
    T0 = float(tau_rows[-1])
    vb0, _, qd0, d0 = rates(X, q, dt)
    v0 = np.r_[np.hypot(d0[:, 0], d0[:, 1]), np.hypot(d0[-1, 0], d0[-1, 1])]
    w0 = np.r_[np.abs(d0[:, 2]), abs(d0[-1, 2])]
    q0 = np.r_[np.abs(qd0), abs(qd0[-1])]
    tg = np.linspace(0.0, T0, int(T0 / 0.001) + 1)
    vg, wg, qg = (np.interp(tg, tau_rows, z) for z in (v0, w0, q0))
    with np.errstate(divide='ignore'):
        rmax = np.minimum(np.where(qg > 1e-9, a.door_rate / qg, np.inf), np.where(vg > 1e-9, a.v_max / vg, np.inf))
        racc = np.minimum.reduce([np.where(qg > 1e-9, a.a_door / qg, np.inf), np.where(vg > 1e-9, a.a_lin / vg, np.inf), np.where(wg > 1e-9, a.a_ang / wg, np.inf)])
    rmax = np.maximum(1.0, np.minimum(rmax, 5.0))
    rmax[(tg < a.keep_s) | (tg > T0 - a.keep_s)] = 1.0
    racc = np.minimum(racc, 5.0)
    r = rmax.copy()
    dtg = tg[1] - tg[0]
    for k in range(1, len(tg)):
        r[k] = min(r[k], math.sqrt(r[k - 1] ** 2 + 2 * racc[k] * dtg))
    for k in range(len(tg) - 2, -1, -1):
        r[k] = min(r[k], math.sqrt(r[k + 1] ** 2 + 2 * racc[k] * dtg))
    tn = np.r_[0.0, np.cumsum(0.5 * (1.0 / r[1:] + 1.0 / r[:-1]) * dtg)]
    Tn = float(tn[-1])
    m = int(math.ceil(Tn / dt - 1e-9))
    tau_s = np.interp(np.minimum(np.arange(0, m + 1) * dt, Tn), tn, tg)
    if abs(tau_s[0]) > 1e-12 or abs(tau_s[-1] - T0) > 1e-9:
        raise SystemExit('首末樣本未落在原端點')
    new = []
    for ts in tau_s:
        j = min(int(round(ts / dt)), n - 1)
        rw = dict(rows[i0 + j])
        rw['x'] = [float(np.interp(ts, tau_rows, X[:, c])) for c in range(X.shape[1])]
        rw['q_deg'] = float(np.interp(ts, tau_rows, q))
        new.append(rw)
    out_rows = rows[:i0] + new + rows[i1 + 1:]
    Xn = np.array([r_['x'] for r_ in out_rows], float)
    qn = np.array([r_['q_deg'] for r_ in out_rows], float)
    vb, armr, qdn, dn = rates(Xn, qn, dt)
    acc = np.abs(np.diff(dn, axis=0)) / dt
    j0, j1 = i0, i0 + len(new) - 1
    seg = slice(j0, j1)
    print(f'{a.seg} {a.stage}：{n * dt:.2f}→{len(new) * dt:.2f} s（r 峰 {r.max():.3f}）；首末列與原同：{np.allclose(Xn[j0], X[0]) and np.allclose(Xn[j1], X[-1])}')
    print(f'  20 Hz 差分（新 {a.stage}）：本體 |vx| {np.abs(vb[seg, 0]).max():.4f} |vy| {np.abs(vb[seg, 1]).max():.4f} |ω| {np.abs(vb[seg, 2]).max():.4f}；臂速 {armr[seg].max():.4f}；'
          f'門速 {np.abs(qdn[seg]).max():.3f}°/s（原 {np.abs(qd0).max():.3f}）')
    print(f'  加速度（新 {a.stage} 內部）：x {acc[j0:j1 - 1, 0].max():.4f} y {acc[j0:j1 - 1, 1].max():.4f} ω {acc[j0:j1 - 1, 2].max():.4f}；'
          f'門 {np.abs(np.diff(qdn[seg]) / dt).max():.3f}°/s²')
    for lab, k in (('S3→S4', j0), ('S4→S5', j1)):
        print(f'  接點 {lab} 列 {k}：差分 v_body {np.round(vb[k - 1], 4).tolist()}→{np.round(vb[k], 4).tolist()}；門速 {qdn[k - 1]:.3f}→{qdn[k]:.3f}；臂速 {armr[k - 1]:.4f}→{armr[k]:.4f}')
    if np.abs(vb[:, :2]).max() > V_BOX or np.abs(vb[:, 2]).max() > W_BOX or armr.max() > ARM_BOX:
        raise SystemExit('差分速度超框，不寫檔')
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    tt, nst, nwn, mn = 0.0, 0, 0, {}
    for k in range(j0, j1 + 1):
        tt += dt
        dec, det = mon.step(tt, Xn[k], math.radians(qn[k]), 0.0016, 0.1, out_rows[k]['target'])
        nst += dec == 'stop'
        nwn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 2), v['who'], k - j0)
    print(f'  幾何監看（新 {a.stage}，門角取參考、手指 0.0016）：stop {nst} warn {nwn}；最小 d_pred {mn}')
    if nst or nwn:
        raise SystemExit('離線監看有 stop／warn，不寫檔')
    R[a.seg] = out_rows
    R[f'{a.stage.lower()}_retime'] = {'src': os.path.basename(a.src), 'src_sha256': hashlib.sha256(open(a.src, 'rb').read()).hexdigest(), 'seg': a.seg,
                                      'T_src': n * dt, 'T_new': len(new) * dt, 'r_peak': float(r.max()), 'door_rate': a.door_rate, 'v_max': a.v_max,
                                      'a_door': a.a_door, 'a_lin': a.a_lin, 'a_ang': a.a_ang, 'keep_s': a.keep_s, 'rows_src': n, 'rows_new': len(new),
                                      'offline': {'stop': nst, 'warn': nwn, 'min': {k: list(v) for k, v in mn.items()}}}
    json.dump(R, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out, hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:16])


if __name__ == '__main__':
    raise SystemExit(main())
