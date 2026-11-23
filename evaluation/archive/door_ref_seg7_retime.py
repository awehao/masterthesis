#!/usr/bin/env python3
"""去程 RG7（換抓參考包段 7，拉關）重時間化（Codex 20261031 條件核可）：沿段 7 原列序列（底盤、手臂、q_door_deg）以同一進度參數 τ 重時間化，
固定 dt 輸出；dτ/dt＝r(τ)，r ≥ 1、門速 r·|q̇₀| ≤ --door-rate、底盤 r·v₀ ≤ --v-max，頭尾 --keep-s 內 r＝1（原速、原端點），
dr/dt ≤ min(門／線／角加速度上限 ÷ 原速)。TCP 由同一構型 FK 重算；qf／src_index／stage 取最近原列。其他段不動，輸出新包、不覆寫來源。
離線核：**經 RefRuntime3.at() 實際輸出**的目標與前饋（段 7 前饋＝相鄰構型差分）之速度／加速度、6→7→8 接點、終點門角、幾何監看（stop／warn 皆 0 才寫檔）。
    python3 evaluation/door_ref_seg7_retime.py --out evaluation/results/door/door_regrasp_ref_pb60o.json
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
import door_regrasp_policy3 as P3                      # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

V_BOX, W_BOX, ARM_BOX = 0.035255, 0.1999, 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60n.json'))
    ap.add_argument('--seg', default='7')
    ap.add_argument('--door-rate', type=float, default=1.0)
    ap.add_argument('--v-max', type=float, default=0.0125)
    ap.add_argument('--a-door', type=float, default=0.5)
    ap.add_argument('--a-lin', type=float, default=0.02)
    ap.add_argument('--a-ang', type=float, default=0.3)
    ap.add_argument('--keep-s', type=float, default=1.5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    P = json.load(open(a.src))
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    dt = float(P['dt'])
    s = P['segments'][a.seg]
    n0 = len(s['t'])
    B = np.asarray(s['base'], float)
    yaw_wrapped = B[:, 2].copy()
    B[:, 2] = np.unwrap(B[:, 2])
    Ar = np.asarray(s['arm'], float)
    q = np.asarray(s['q_door_deg'], float)
    tau_rows = np.arange(n0) * dt
    T0 = float(tau_rows[-1])
    d = np.diff(B, axis=0) / dt
    v0 = np.r_[np.hypot(d[:, 0], d[:, 1]), 0.0]
    w0 = np.r_[np.abs(d[:, 2]), 0.0]
    q0 = np.r_[np.abs(np.diff(q)) / dt, 0.0]
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
    m = max(1, int(round(Tn / dt)))                                       # 新時間均勻縮放 Tn→m·dt，使末樣本正好落在原終點（不留部分步長）
    tau_s = np.interp(np.arange(0, m + 1) * (Tn / m), tn, tg)
    if abs(tau_s[0]) > 1e-12 or abs(tau_s[-1] - T0) > 1e-9:
        raise SystemExit('首末樣本未落在原端點')
    keep_wrap = np.allclose(B[:, 2], yaw_wrapped)
    out = {k: [] for k in s}
    for j, ts in enumerate(tau_s):
        ir = min(int(round(ts / dt)), n0 - 1)
        bx = [float(np.interp(ts, tau_rows, B[:, c])) for c in range(3)]
        if not keep_wrap:
            bx[2] = math.atan2(math.sin(bx[2]), math.cos(bx[2]))
        ar = [float(np.interp(ts, tau_rows, Ar[:, c])) for c in range(Ar.shape[1])]
        out['t'].append(j * dt + float(s['t'][0]))
        out['base'].append(bx)
        out['arm'].append(ar)
        out['q_door_deg'].append(float(np.interp(ts, tau_rows, q)))
        out['tcp'].append([float(v_) for v_ in K.fk(np.r_[bx, ar], 'link_tcp').reshape(-1)])
        for k in ('qf', 'src_index', 'stage'):
            out[k].append(s[k][ir])
    for k in s:
        if k not in out or not out[k]:
            raise SystemExit(f'段 {a.seg} 欄位 {k} 未處理')
    P['segments'][a.seg] = out
    sn = out
    # 經實際 runtime 取目標與前饋
    rt = P3.RefRuntime3(P, K)
    ok, why, _ = rt.start(int(a.seg), sn['base'][0], sn['arm'][0], sn['q_door_deg'][0], np.asarray(sn['tcp'][0]).reshape(4, 4))
    if not ok:
        raise SystemExit(f'runtime start 失敗：{why}')
    L = rt.length_s(int(a.seg))
    FF, XB = [], []
    for tt in np.arange(0.0, L + 1e-9, dt):
        _, xb, ff = rt.at(int(a.seg), tt)
        FF.append(ff)
        XB.append(xb)
    FF, XB = np.asarray(FF), np.asarray(XB)
    c, s_ = np.cos(XB[:, 2]), np.sin(XB[:, 2])
    vb = np.c_[c * FF[:, 0] + s_ * FF[:, 1], -s_ * FF[:, 0] + c * FF[:, 1], FF[:, 2]]
    acc = np.abs(np.diff(FF, axis=0)) / dt
    arm = np.abs(np.diff(np.asarray(sn['arm']), axis=0)).max(1) / dt
    qd = np.diff(np.asarray(sn['q_door_deg'])) / dt
    print(f'段 {a.seg}：{n0 * dt:.2f}→{len(sn["t"]) * dt:.2f} s（r 峰 {r.max():.3f}）；runtime 段長 {L:.2f} s；首末列與原同：'
          f'{np.allclose(sn["base"][0], s["base"][0]) and np.allclose(sn["base"][-1], s["base"][-1]) and np.allclose(sn["arm"][-1], s["arm"][-1])}；終點門角 {sn["q_door_deg"][-1]:.4f}（原 {s["q_door_deg"][-1]:.4f}）')
    print(f'  runtime 前饋（本體）|vx| {np.abs(vb[:, 0]).max():.4f} |vy| {np.abs(vb[:, 1]).max():.4f} |ω| {np.abs(vb[:, 2]).max():.4f}；臂速 {arm.max():.4f}；門速 {np.abs(qd).max():.3f}°/s（原 {q0.max():.3f}）')
    print(f'  前饋變化率 x {acc[:, 0].max():.4f} y {acc[:, 1].max():.4f} ω {acc[:, 2].max():.4f}；99% 分位 {np.percentile(acc, 99, axis=0).round(4).tolist()}；門加速度峰 {np.abs(np.diff(qd) / dt).max():.3f}°/s²')
    s6, s8 = P['segments']['6'], P['segments']['8']
    for lab, xa, xb_ in (('6末→7首', np.r_[s6['base'][-1], s6['arm'][-1]], np.r_[sn['base'][0], sn['arm'][0]]), ('7末→8首', np.r_[sn['base'][-1], sn['arm'][-1]], np.r_[s8['base'][0], s8['arm'][0]])):
        dd = xa - xb_
        dd[2] = math.atan2(math.sin(dd[2]), math.cos(dd[2]))
        print(f'  接點 {lab}：構型差 {np.abs(dd).max():.2e}')
    print(f'  runtime 前饋首 {np.round(FF[0], 4).tolist()}、末前 {np.round(FF[-2], 4).tolist()}、末 {np.round(FF[-1], 4).tolist()}')
    if np.abs(vb[:, :2]).max() > V_BOX or np.abs(vb[:, 2]).max() > W_BOX or arm.max() > ARM_BOX:
        raise SystemExit('前饋超框，不寫檔')
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    tt, nst, nwn, mn = 0.0, 0, 0, {}
    for i in range(len(sn['t'])):
        tt += dt
        dec, det = mon.step(tt, np.r_[sn['base'][i], sn['arm'][i]], math.radians(sn['q_door_deg'][i]), float(sn['qf'][i]), 0.1, 'in')
        nst += dec == 'stop'
        nwn += dec == 'warn'
        for c_, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c_, (9,))[0]:
                mn[c_] = (round(v['d_pred'] * 1e3, 2), v['who'], i)
    print(f'  幾何監看（新段 {a.seg}，目標 in、手指取 qf）：stop {nst} warn {nwn}；最小 d_pred {mn}')
    if nst or nwn:
        raise SystemExit('離線監看有 stop／warn，不寫檔')
    P[f'seg{a.seg}_retime'] = {'src': os.path.basename(a.src), 'src_sha256': hashlib.sha256(open(a.src, 'rb').read()).hexdigest(), 'T_src': n0 * dt,
                               'T_new': len(sn['t']) * dt, 'r_peak': float(r.max()), 'door_rate': a.door_rate, 'v_max': a.v_max, 'a_door': a.a_door,
                               'a_lin': a.a_lin, 'a_ang': a.a_ang, 'keep_s': a.keep_s, 'offline': {'stop': nst, 'warn': nwn, 'min': {k: list(v) for k, v in mn.items()}}}
    json.dump(P, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out, hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:16])


if __name__ == '__main__':
    raise SystemExit(main())
