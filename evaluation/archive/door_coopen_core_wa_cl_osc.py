#!/usr/bin/env python3
"""w_a 候選的**振盪核對**（離線；補 door_coopen_core_wa_cl.py：該檔續進即停，候選只跑 1.95 s，混入暫態）。
同一起點、受控體、時序與參數（由 door_coopen_core_wa_cl 匯入，不另定義）；情境 A、C；w_a ∈ {0.05, 0.01}；**固定跑滿剩餘預算 4.14 s，續進後不停**。
逐輪記已發布命令 u（求解失敗輪單列、沿用上一筆已套用）與實際 TCP 誤差。指標分「全段 [0, 4.14)」與「末 2 s [2.14, 4.14)」：
  - 手臂命令：逐軸 |u| 最大、高頻幅值（減去 5 點移動平均後的 |殘差| 最大，去端點）、一階差號變次數（|Δ| > 1e-6）。
  - 由已發布命令的 FFT 取各軸主頻（去均值；只描述）。
  - TCP：垂直誤差末 2 s 的平均、最大、去線性趨勢後標準差；沿軸、姿態末值。
不據此宣稱 Isaac 振盪行為。

    python3 evaluation/door_coopen_core_wa_cl_osc.py   # 寫 results/door/door_coopen_core_wa_cl_osc_r1.json（不覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_core_wa_cl as W                           # noqa: E402
from door_coopen_core_wa_cl import S, task_error, KC, WholeBodyKinematics   # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_core_wa_cl_osc_r1.json')


def run_fixed(K, init, b_true, w_a, t_len):
    cfg = init['cfg'](w_a)
    q, s = init['q'].copy(), init['s'].copy()
    T, ax = init['T'], init['ax']
    dtp = cfg.arm_model.phys_dt
    n_phys = int(round(cfg.dt / dtp))
    pubq = [init['u_prev'].copy()] * W.N_PUB
    pipe = [init['u_prev'].copy()] * W.N_CMD
    u_prev = init['u_prev'].copy()
    U_warm = None
    ts, P, E_, fails = [], [], [], []
    t = 0.0
    while t < t_len - 1e-9:
        e = task_error(K, q, T, 'link_tcp')
        p = -np.asarray(e[:3])
        al = float(p @ ax)
        E_.append((al, float(np.linalg.norm(p - al * ax)), float(np.linalg.norm(e[3:]))))
        qs, ss = q.copy(), s.copy()
        for uu in list(pipe) + list(pubq):
            qs, ss = S.plant_phys_step(qs, ss, uu, cfg, 1)
        r = S.solve_sp(K, S.make_z(qs, ss), u_prev, T, cfg, U_warm=U_warm)
        if r.ok:
            u, U_warm = np.asarray(r.u0, float), r.U
        else:
            fails.append(round(t, 4))
            u, U_warm = u_prev.copy(), None
        ts.append(t)
        P.append(u.copy())
        for _ in range(n_phys):
            pubq.append(u.copy())
            pipe.append(pubq.pop(0))
            u_app = pipe.pop(0)
            q, s = S.plant_phys_step(q, s, u_app, cfg, 1, base_bias=b_true)
            u_prev = u_app.copy()
        t = round(t + cfg.dt, 9)
    return np.asarray(ts), np.asarray(P), np.asarray(E_), fails, cfg.dt


def metrics(ts, P, E_, lo, hi, dt):
    m = (ts >= lo - 1e-9) & (ts < hi - 1e-9)
    A = P[m, 3:]
    sm = np.array([np.convolve(A[:, j], np.ones(5) / 5, 'same') for j in range(6)]).T
    hf = np.abs((A - sm)[2:-2]).max(axis=0) if len(A) > 6 else np.full(6, np.nan)
    d1 = np.diff(A, axis=0)
    sg = np.sign(np.where(np.abs(d1) > 1e-6, d1, 0.0))
    flips = [int(np.sum(sg[1:, j] * sg[:-1, j] < 0)) for j in range(6)]
    f = np.fft.rfftfreq(len(A), dt)
    dom = [float(f[1 + np.argmax(np.abs(np.fft.rfft(A[:, j] - A[:, j].mean()))[1:])]) for j in range(6)]
    pe = E_[m, 1] * 1e3
    tt = ts[m]
    det = pe - np.polyval(np.polyfit(tt, pe, 1), tt)
    return {'interval_s': [lo, hi], 'n': int(m.sum()), 'arm_u_absmax': np.abs(A).max(axis=0).tolist(), 'arm_hf_amp': hf.tolist(),
            'arm_hf_amp_max': float(np.nanmax(hf)), 'arm_sign_flips': flips, 'arm_sign_flips_total': int(sum(flips)), 'arm_dom_freq_hz': dom,
            'perp_mm_mean': float(pe.mean()), 'perp_mm_max': float(pe.max()), 'perp_detrended_std_mm': float(det.std()),
            'end': {'along_mm': float(E_[m, 0][-1] * 1e3), 'perp_mm': float(pe[-1]), 'rot_rad': float(E_[m, 2][-1])}}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    sv = json.load(open(os.path.join(W.RUN, 'align_solver.json')))
    r = json.load(open(os.path.join(W.RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    row = min([x for x in sv['log'] if (x.get('solve_in') or {}).get('U_sol') is not None], key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - W.T_SNAP_Q))
    si = row['solve_in']
    st = min(r['steps'], key=lambda s_: abs(s_[c['sim_t']] - si['snap_sim_t_exact']))
    idn = json.load(open(W.IDENT))
    a = sv['args']
    a_H = np.asarray(json.load(open(W.GB))['args']['a_H'], float)

    def mkcfg(w_a):
        am = S.ArmSetpointModel(alpha=idn['alpha'], bias=si['arm_bias'], phys_dt=idn['phys_dt_measured_s'])
        cfg = S.WGMPCConfigSP(N=int(a['N']), dt=1.0 / float(a['rate']), tcp='link_tcp', arm_model=am,
                              w_s=float(a['w_s']), w_a=float(w_a), w_s_arm=float(a['w_s_arm']))
        cfg.base_hold = tuple(float(x) for x in row['park_hold']['u_hold'])
        return cfg
    u_hold = np.asarray(row['park_hold']['u_hold'], float)
    t0 = float(si['snap_sim_t_exact'])
    init = {'cfg': mkcfg, 'q': np.asarray(si['q_meas'], float), 's': np.asarray(si['s_meas'], float), 'u_prev': np.asarray(si['u_prev'], float),
            'T': np.asarray(si['T_cyc'], float).reshape(4, 4), 'ax': KC._H_of(st[c['door_q_rad']])[:3, :3] @ a_H}
    t_len = W.T_ABORT - t0
    K = WholeBodyKinematics.from_urdf_file(W.URDF)
    out = {'schema': 'door_coopen_core_wa_cl_osc/1', 'snap_sim_t': t0, 't_len_s': t_len, 'cases': {}}
    for nm, bt in (('A_unrealized_hold', -u_hold), ('C_fixed_hold_realized', np.zeros(3))):
        for w_a in W.W_A:
            ts, P, E_, fails, dt = run_fixed(K, init, bt, w_a, t_len)
            res = {'n_fail': len(fails), 'fail_t': fails, 'all': metrics(ts, P, E_, 0.0, t_len, dt), 'last2s': metrics(ts, P, E_, t_len - 2.0, t_len, dt),
                   'series': {'t': ts.tolist(), 'u_arm': P[:, 3:].tolist(), 'along_perp_rot': E_.tolist()}}
            out['cases'][f'{nm}/w_a={w_a}'] = res
            for k_ in ('all', 'last2s'):
                g = res[k_]
                print(f"{nm} w_a {w_a} {k_}：高頻幅值 max {g['arm_hf_amp_max'] * 1e3:.2f} mrad/s、號變 {g['arm_sign_flips_total']}、|u| max {max(g['arm_u_absmax']):.3f}、"
                      f"主頻 {g['arm_dom_freq_hz'][2]:.2f} Hz；垂直 mean {g['perp_mm_mean']:.2f} max {g['perp_mm_max']:.2f} 去趨勢 std {g['perp_detrended_std_mm']:.3f} mm；"
                      f"末 沿軸 {g['end']['along_mm']:.2f} 垂直 {g['end']['perp_mm']:.2f} 姿態 {g['end']['rot_rad']:.4f}；失敗 {res['n_fail']}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
