#!/usr/bin/env python3
"""R3v2（Codex 20261031 條件核可）：段 41 前導（tau_pre_end 之前）不動；尾段沿**原完整位姿序列**（x、y、偏航、手臂）以同一進度參數 τ 重時間化：
新時間 t、dτ/dt＝r(τ)，r ≥ 1（不比原來慢）、r·v₀(τ) ≤ --v-max、r·|ω₀(τ)| ≤ --w-pk，尾段頭尾 --keep-s 內 r＝1（保留原接點的加減速形狀），
dr/dt ≤ min(A_LIN/v₀, A_ANG/|ω₀|)（前後向掃描）。前饋 base_v＝r·base_v₀(τ)（鏈鎖律）。另寫 m4r.r3_w_cap（只在 R3 生效）。
離線核：輸出檔 20 Hz 差分的本體速度／加速度、前導→尾段與尾段→mode 5 接點、前饋與差分一致性、幾何監看（stop／warn 皆 0 才寫檔）。
    python3 evaluation/door_ref_r3_retime2.py --out evaluation/results/door/door_regrasp_ref_pb60n.json
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
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

W_BOX, V_BOX = 0.1999, 0.035255


def body_rates(b, dt):
    d = np.diff(b, axis=0) / dt
    c, s = np.cos(b[:-1, 2]), np.sin(b[:-1, 2])
    vb = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]]
    return vb, np.abs(np.diff(d, axis=0)) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60m.json'))
    ap.add_argument('--v-max', type=float, default=0.030)
    ap.add_argument('--w-pk', type=float, default=0.15)
    ap.add_argument('--a-lin', type=float, default=0.02)
    ap.add_argument('--a-ang', type=float, default=0.3)
    ap.add_argument('--keep-s', type=float, default=1.5)
    ap.add_argument('--r3-w-cap', type=float, default=0.19)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    if not (a.v_max < V_BOX and a.w_pk <= a.r3_w_cap < W_BOX):
        raise SystemExit('上限須在框內且 w_pk ≤ r3_w_cap')
    P = json.load(open(a.src))
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    dt = float(P['dt'])
    s = P['segments']['41']
    n0 = len(s['t'])
    i0 = int(round(float(P['m4r']['r3_pre']['tau_pre_end']) / dt))
    B = np.asarray(s['base'], float)
    B[:, 2] = np.unwrap(B[:, 2])
    A = np.asarray(s['arm'], float)
    BV = np.asarray(s['base_v'], float)
    tau_rows = np.arange(n0 - i0) * dt                                         # 尾段原時間（列 i0 為 τ＝0）
    Tt = float(tau_rows[-1])
    v0 = np.hypot(BV[i0:, 0], BV[i0:, 1])
    w0 = np.abs(BV[i0:, 2])
    # 細格 τ 上做前後向掃描（r² 的變化率 ≤ 2·dr/dt 上限）
    tg = np.linspace(0.0, Tt, int(Tt / 0.001) + 1)
    vg, wg = np.interp(tg, tau_rows, v0), np.interp(tg, tau_rows, w0)
    with np.errstate(divide='ignore'):
        rmax = np.minimum(np.where(vg > 1e-9, a.v_max / vg, np.inf), np.where(wg > 1e-9, a.w_pk / wg, np.inf))
    rmax = np.maximum(1.0, np.minimum(rmax, 10.0))
    rmax[(tg < a.keep_s) | (tg > Tt - a.keep_s)] = 1.0
    with np.errstate(divide='ignore'):
        racc = np.minimum(np.where(vg > 1e-9, a.a_lin / vg, np.inf), np.where(wg > 1e-9, a.a_ang / wg, np.inf))
    racc = np.minimum(racc, 5.0)
    r = rmax.copy()
    dtg = tg[1] - tg[0]
    for k in range(1, len(tg)):
        r[k] = min(r[k], math.sqrt(r[k - 1] ** 2 + 2 * racc[k] * dtg))
    for k in range(len(tg) - 2, -1, -1):
        r[k] = min(r[k], math.sqrt(r[k + 1] ** 2 + 2 * racc[k] * dtg))
    tn = np.r_[0.0, np.cumsum(0.5 * (1.0 / r[1:] + 1.0 / r[:-1]) * dtg)]       # 新時間 t(τ)
    Tn = float(tn[-1])
    m = int(math.ceil(Tn / dt - 1e-9))
    t_s = np.minimum(np.arange(1, m + 1) * dt, Tn)
    tau_s = np.interp(t_s, tn, tg)
    r_s = np.interp(tau_s, tg, r)
    if abs(tau_s[-1] - Tt) > 1e-9:
        raise SystemExit('末樣本未落在原終點')
    keep = {k: list(v[:i0 + 1]) for k, v in s.items() if isinstance(v, list) and len(v) == n0}
    for j, (ts, rs) in enumerate(zip(tau_s, r_s)):
        x = np.array([np.interp(ts, tau_rows, B[i0:, c]) for c in range(3)])
        q = np.array([np.interp(ts, tau_rows, A[i0:, c]) for c in range(A.shape[1])])
        bv = np.array([np.interp(ts, tau_rows, BV[i0:, c]) for c in range(3)]) * rs
        keep['base'].append([float(v) for v in x])
        keep['arm'].append([float(v) for v in q])
        keep['base_v'].append([float(v) for v in bv])
        keep['tcp'].append([float(v) for v in K.fk(np.r_[x, q], 'link_tcp').reshape(-1)])
        keep['t'].append(keep['t'][-1] + dt)
        ir = min(int(round(ts / dt)) + i0, n0 - 1)
        for k in ('qf', 'q_door_deg', 'src_index', 'stage'):
            keep[k].append(s[k][ir])
    keep['base_v'][-1] = list(BV[-1])                                          # 終點與原包相同（原末列前饋）
    for k, v in keep.items():
        s[k] = v
    Bn = np.asarray(s['base'], float)
    vb, acc = body_rates(Bn, dt)
    ff = np.asarray(s['base_v'], float)
    d_ff = np.abs(np.diff(Bn, axis=0) / dt - ff[1:]).max(0)
    j0 = i0
    print(f'段 41：{n0 * dt:.2f}→{len(s["t"]) * dt:.2f} s（尾段 {Tt:.2f}→{Tn:.2f} s；r 峰 {r.max():.3f}）')
    print(f'  20 Hz 差分本體速度峰 |vx| {np.abs(vb[:, 0]).max():.4f} |vy| {np.abs(vb[:, 1]).max():.4f} |ω| {np.abs(vb[:, 2]).max():.4f}（框 {V_BOX}/{W_BOX}）；'
          f'世界加速度峰 x {acc[:, 0].max():.4f} y {acc[:, 1].max():.4f} ω {acc[:, 2].max():.4f}')
    print(f'  前饋 vs 差分 最大差 {d_ff.round(5).tolist()}；前導→尾段接點 列 {j0}：差分 v {np.round(vb[j0 - 1], 4).tolist()}→{np.round(vb[j0], 4).tolist()}→{np.round(vb[j0 + 1], 4).tolist()}')
    s5 = P['segments']['5']
    b5, a5 = np.asarray(s5['base'][0], float), np.asarray(s5['arm'][0], float)
    print(f'  尾段→mode 5：位姿差 {np.abs(Bn[-1] - b5).max():.2e}、手臂差 {np.abs(np.asarray(s["arm"][-1]) - a5).max():.2e}；尾段末 5 列差分 v {np.round(vb[-5:], 4).tolist()}；'
          f'mode 5 首列前饋 {np.round(s5["base_v"][0], 4).tolist() if "base_v" in s5 else None}')
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    q = math.radians(float(P['q_ref_deg']))
    tt, nst, nwn, mn = 0.0, 0, 0, {}
    An = np.asarray(s['arm'], float)
    for i in range(len(Bn)):
        tt += dt
        dec, det = mon.step(tt, np.r_[Bn[i], An[i]], q, 0.0089, 0.1, None)
        nst += dec == 'stop'
        nwn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v['who'], i)
    print(f'  幾何監看（整段 41）：stop {nst} warn {nwn}；最小 d_pred {mn}')
    if nst or nwn:
        raise SystemExit('離線監看有 stop／warn，不寫檔')
    if np.abs(vb[:, :2]).max() > V_BOX or np.abs(vb[:, 2]).max() > W_BOX:
        raise SystemExit('差分速度超框，不寫檔')
    P['m4r']['r3_w_cap'] = a.r3_w_cap
    P['r3_retime2'] = {'src': os.path.basename(a.src), 'src_sha256': hashlib.sha256(open(a.src, 'rb').read()).hexdigest(), 'i0': i0, 'T_tail_src': Tt, 'T_tail': Tn,
                       'r_peak': float(r.max()), 'v_max': a.v_max, 'w_pk': a.w_pk, 'a_lin': a.a_lin, 'a_ang': a.a_ang, 'keep_s': a.keep_s, 'r3_w_cap': a.r3_w_cap,
                       'offline': {'stop': nst, 'warn': nwn, 'min': {k: list(v) for k, v in mn.items()}}}
    json.dump(P, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out, hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:16])


if __name__ == '__main__':
    raise SystemExit(main())
