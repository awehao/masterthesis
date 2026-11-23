#!/usr/bin/env python3
"""TO_PRE 途中追蹤誤差診斷（Codex 20261112_124807；只讀既有紀錄，不跑模擬）。
  1 時間對齊：同時刻誤差 e(t)=|TCP(t)−ref(t)|；時間位移誤差 e_δ=|TCP(t)−ref(t−δ)|，掃 δ；路徑（輪廓）誤差＝實測 TCP 到整條參考 TCP 曲線的最近距離。
    起點接續：重算時刻的參考起點（qa＝當下實測關節）與實測 TCP 差；參考在命令生效時刻（t0＝重算＋8 步）才開始。
  2 參考可行／命令可行：參考關節速度峰值對 R1.V_ARM；含設定點模型的所需設定點速率 ṡ＝q̇_ref＋τ·q̈_ref（一階模型 τ＝dt_phys/α）對命令上限；實際設定點速率飽和時間。
    python3 evaluation/door_approach_track_diag.py RUN OUT.json"""
import json
import os
import sys

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_approach_ctrl as DA                                    # noqa: E402

K, R1, R2, B, C0 = DA.K, DA.R1, DA.R2, DA.B, DA.C0


def main():
    run, out = sys.argv[1:3]
    D = os.path.join(HERE, 'runs', run)
    R = np.load(os.path.join(D, 'log.npz'))['rows']
    res = json.load(open(os.path.join(D, 'result.json')))
    c = res['controller']
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    alpha = np.asarray(ident['alpha'], float)
    tau = DA.PHYS / alpha                                          # 一階設定點模型時間常數（每關節）
    t_rep, T2 = c['replan']['t'], c['replan']['T_to_pre']
    k_rep = int(round(t_rep / DA.PHYS))
    qa, qb, bw = R[k_rep, 5:11].copy(), np.array(c['replan']['q_pre']), np.array(c['replan']['base_at_replan'])
    t0 = t_rep + DA.D_STEPS * DA.PHYS

    def qref(t):
        p, dp = DA._mj((t - t0) / T2)
        v = dp / T2 if t0 <= t <= t0 + T2 else 0.0
        return qa + p * (qb - qa), v * (qb - qa)

    def tcp_ref(t):
        return (DA._tb(bw) @ K.fk(np.r_[0.0, 0.0, 0.0, qref(t)[0]], 'link_tcp'))[:3, 3]
    ks = np.arange(k_rep, int(round((t0 + T2) / DA.PHYS)) + 1)
    tt = R[ks, 0]
    tcp = np.array([K.fk(R[k, 2:11], 'link_tcp')[:3, 3] for k in ks])
    # 1 時間位移與路徑誤差
    sweep = []
    for d in np.arange(0.0, 0.401, 0.01):
        e = np.array([np.linalg.norm(tcp[i] - tcp_ref(tt[i] - d)) for i in range(len(ks))])
        sweep.append(dict(delta_s=round(float(d), 2), max_mm=round(float(e.max()) * 1e3, 2), p95_mm=round(float(np.percentile(e, 95)) * 1e3, 2)))
    best = min(sweep, key=lambda r: r['max_mm'])
    curve = np.array([tcp_ref(t) for t in np.arange(t0, t0 + T2 + 1e-9, 0.002)])
    path = np.array([np.linalg.norm(curve - p, axis=1).min() for p in tcp])
    e0 = np.array([np.linalg.norm(tcp[i] - tcp_ref(tt[i])) for i in range(len(ks))])
    # 關節空間同樣分解
    qm = R[ks, 5:11]
    ej0 = np.abs(qm - np.array([qref(t)[0] for t in tt])).max()
    ejb = np.abs(qm - np.array([qref(t - best['delta_s'])[0] for t in tt])).max()
    # 2 參考可行／命令可行
    ts = np.arange(t0, t0 + T2 + 1e-9, 0.001)
    qd = np.array([qref(t)[1] for t in ts])
    qdd = np.gradient(qd, ts, axis=0)
    s_rate = qd + tau[None, :] * qdd                                # 一階模型下逐關節所需設定點速率
    dq = np.abs(qb - qa)
    u_act = np.abs(np.diff(R[:, 11:17], axis=0)) / DA.PHYS          # 實際設定點速率（紀錄的 arm_target 差分）
    win = (R[1:, 0] >= t_rep) & (R[1:, 0] <= t0 + T2 + 1.0)
    lim = float(C0.make_cfg(ident).arm_rate_max) if hasattr(C0.make_cfg(ident), 'arm_rate_max') else 1.0
    sat = (u_act.max(axis=1) >= lim - 1e-3) & win
    out_d = dict(run=run, segment=dict(t_replan=t_rep, t0_ref_start=round(t0, 3), T=T2, dq_abs=[round(float(v), 4) for v in dq], limiting_joint=int(dq.argmax()) + 1),
                 start_continuity=dict(tcp_ref_start_vs_measured_at_replan_mm=round(float(np.linalg.norm(tcp_ref(t0) - tcp[0])) * 1e3, 3), measured_arm_rate_at_replan_rad_s=round(float(np.abs(R[k_rep, 5:11] - R[k_rep - 5, 5:11]).max() / (5 * DA.PHYS)), 5),
                                       ref_rate_at_start=0.0, note='參考起點＝重算當下實測關節、零速；參考延後 8 步（命令生效時刻）才開始'),
                 same_time_error=dict(tcp_max_mm=round(float(e0.max()) * 1e3, 2), tcp_p95_mm=round(float(np.percentile(e0, 95)) * 1e3, 2), t_at_max=round(float(tt[e0.argmax()]), 2), joint_max_rad=round(float(ej0), 4)),
                 time_shift=dict(best=best, joint_max_rad_at_best=round(float(ejb), 4), sweep=sweep),
                 path_error=dict(max_mm=round(float(path.max()) * 1e3, 2), p95_mm=round(float(np.percentile(path, 95)) * 1e3, 2), note='實測 TCP 到參考 TCP 曲線的最近距離（不看時間）'),
                 reference_vs_limits=dict(tau_s=[round(float(v), 4) for v in tau], ref_joint_speed_peak=round(float(np.abs(qd).max()), 4), V_ARM_used_for_T=float(R1.V_ARM), A_ARM_used_for_T=float(R1.A_ARM), arm_rate_cmd_limit=lim,
                                          required_setpoint_rate_peak=round(float(np.abs(s_rate).max()), 4), ref_joint_accel_peak=round(float(np.abs(qdd).max()), 4), headroom_note='所需設定點速率＝q̇_ref＋τ·q̈_ref（一階設定點模型、不含 8 步延遲與追回誤差所需）'),
                 actual_saturation=dict(duration_s=round(float(sat.sum() * DA.PHYS), 2), first_t=None if not sat.any() else round(float(R[1:, 0][sat][0]), 2), last_t=None if not sat.any() else round(float(R[1:, 0][sat][-1]), 2), setpoint_rate_max=round(float(u_act[win].max()), 4)),
                 timeline=dict(ref_end=round(t0 + T2, 2), end_reached=c['end']['reached']['t'], approach_start_to_reached_s=round(c['end']['reached']['t'] - t_rep, 2), convergence_wait_s=round(c['end']['reached']['t'] - (t0 + T2), 2)))
    json.dump(out_d, open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out_d.items() if k != 'time_shift'}, ensure_ascii=False, indent=1))
    print('best shift', best, [s for s in out_d['time_shift']['sweep'] if s['delta_s'] in (0.0, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3)])


if __name__ == '__main__':
    main()
