#!/usr/bin/env python3
"""K1（低速介面放寬決策資料）步驟 4b：**離線閉迴路追蹤**（明確標示模型與延遲；不是 Isaac、不是實跑）。
受控對象：wgmpc_closed_loop_sim.plant_step ＝ 手臂設定點模型（free4 辨識 α、偏差）＋底盤運動學 q⁺＝q＋dt·B(θ)u（增益 1.0，
  rec8 實測迴歸 0.89／0.92／0.82 未套用）；無接觸、無 Isaac 積分器、無安全層／E2。
延遲：發布 0.6 週期＋命令 1.0 週期（D_state 1.6），求解端以發布歷史前推 1.6 週期（同 rt_43 求解端 delay_comp 設定）。
控制器：solve_sp 本體，權重同 rt_43 門段求解端（w_s 1e-3、w_s_arm 0.05、w_a 0.05、w_p 50、w_vref 1）；×k 時底盤框乘 k，
  w_bt、w_br、底盤 w_s、w_vref 乘 k²（有效權重不變，見 k1_resolve renorm）。
參考：R3＝results/k1/r3_x{k}.json 段 41 逐列（底盤、手臂、base_v 前饋；以列內插，未經 RefRuntime3）；
      S8＝door_ref_overlap_v1.retime 以 V_MAX×k 重產（手臂固定 S8 首列）。
引導：policy1._coord 同式 v＝ff＋k_lin·e（世界逐軸截斷 v_cap 0.035·k、w_cap 0.19·k）轉本體。
起始：參考首列構型、設定點＝手臂、u_prev＝0（兩倍率相同的起始暫態）。
指標：參考期間 TCP 位置誤差（對參考列 FK）、底盤位置／偏航誤差（對參考列）、參考結束後收斂到 5 mm 的時間（續跑 4 s）、求解失敗數、底盤命令峰值。
校對：rt_43 同段實測 TCP 對 T_cyc 誤差（底盤 >0.02 m/s）R3 p50 8.19、S8 p50 8.17 mm；本模型 ×1 為 9.2／10.3 mm（約高 1.2 倍）。
    python3 evaluation/k1_closed_loop.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import wgmpc_closed_loop_sim as CL                     # noqa: E402
import door_ref_overlap_v1 as OV                       # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

DT = 0.05
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
OUT = os.path.join(HERE, 'results', 'k1', 'closed_loop.json')
SETTLE_S = 4.0                                         # 參考結束後續跑（參考保持末列），量收斂到 5 mm 的時間


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def r3_ref(k):
    P = json.load(open(os.path.join(HERE, 'results', 'k1', f'r3_x{k:g}.json')))
    s = P['segments']['41']
    return np.asarray(s['base'], float), np.asarray(s['arm'], float), np.asarray(s['base_v'], float)


def s8_ref(k):
    V0 = OV.V_MAX
    OV.V_MAX = V0 * k
    try:
        R = json.load(open(os.path.join(HERE, 'results', 'door', 'door_return_ref_v1.json')))
        A = R['A']
        i8 = [i for i, r in enumerate(A) if r['stage'].startswith('S8')]
        ie = i8[0] + 800
        X = np.array([r['x'] for r in A])
        v_e = float(np.hypot(*(X[ie + 1, :2] - X[ie, :2]) / DT))
        w_e = float((X[ie + 1, 2] - X[ie, 2]) / DT)
        new, T, vc, wc = OV.retime(X[i8[0]:ie + 1, :2], (float(X[i8[0], 2]), float(X[ie, 2])), 0.0, v_e, 0.0, w_e)
    finally:
        OV.V_MAX = V0
    B = np.vstack([X[i8[0], :3], new])
    Ar = np.tile(X[i8[0], 3:], (len(B), 1))
    BV = np.vstack([np.diff(B, axis=0) / DT, np.zeros(3)])
    return B, Ar, BV


def run(K, ident, B, Ar, BV, k):
    n = len(B)
    tt = np.arange(n) * DT
    Bu = B.copy()
    Bu[:, 2] = np.unwrap(Bu[:, 2])

    def ref_at(t):
        b = np.array([np.interp(t, tt, Bu[:, c]) for c in range(3)])
        a = np.array([np.interp(t, tt, Ar[:, c]) for c in range(Ar.shape[1])])
        ff = np.array([np.interp(t, tt, BV[:, c]) for c in range(3)])
        return b, a, ff
    cfg = CL.make_cfg(ident, w_s=1e-3, w_a=0.05, w_s_arm=0.05)
    cfg.w_p, cfg.w_vref = 50.0, 1.0
    k2 = k * k
    cfg.v_base_lin *= k
    cfg.v_base_ang *= k
    cfg.w_bt *= k2
    cfg.w_br *= k2
    cfg.w_s_base = cfg.w_s * k2
    cfg.w_vref *= k2
    v_cap, w_cap = 0.035 * k, 0.19 * k
    log = {'vref': []}

    def T_fn(t):
        b, a, _ = ref_at(t)
        return K.fk(np.r_[b, a], 'link_tcp')

    def pre(t, q, c):
        b, _, ff = ref_at(t)
        x, y, th = float(q[0]), float(q[1]), float(q[2])
        e = np.array([b[0] - x, b[1] - y, wrap(b[2] - th)])
        vw = ff + e
        vw = np.array([np.clip(vw[0], -v_cap, v_cap), np.clip(vw[1], -v_cap, v_cap), np.clip(vw[2], -w_cap, w_cap)])
        cs, sn = math.cos(th), math.sin(th)
        c.base_vref = (cs * vw[0] + sn * vw[1], -sn * vw[0] + cs * vw[1], float(vw[2]))
        log['vref'].append(c.base_vref)
    b0, a0, _ = ref_at(0.0)
    q0 = np.r_[b0, a0]
    res = CL.run(cfg, K, q0, a0.copy(), T_fn, t_end=tt[-1] + SETTLE_S, reach_p=-1.0, reach_r=-1.0,
                 delay_cycles=1.0, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', pre_solve=pre)
    r = res['rec']
    Q = np.asarray(r['q'])
    T = np.asarray(r['t'])
    eb = np.array([np.hypot(*(ref_at(t)[0][:2] - q[:2])) for t, q in zip(T, Q)])
    ey = np.array([abs(wrap(ref_at(t)[0][2] - q[2])) for t, q in zip(T, Q)])
    ep = np.asarray(r['ep'])
    U = np.asarray(r['u_req'])
    pc = lambda z, p: round(float(np.percentile(z, p)) * 1e3, 2)          # noqa: E731
    Te = float(tt[-1])
    after = T > Te + 1e-9
    bad = np.where(after & (ep > 0.005))[0]
    settle = (0.0 if not len(bad) else float(T[bad[-1]] - Te + DT)) if after.any() else None
    if bad.size and bad[-1] == len(T) - 1:
        settle = None                                                       # 窗內未收斂
    on = ~after
    return {'k': k, 'T_ref_s': round(float(tt[-1]), 2), 'n_cycles': int(len(T)), 'n_fail': int(res['n_fail']),
            'tcp_err_mm': {'p50': pc(ep[on], 50), 'p95': pc(ep[on], 95), 'max': round(float(ep[on].max()) * 1e3, 2)},
            'settle_to_5mm_after_ref_end_s': settle, 'tcp_err_end_mm': round(float(ep[-1]) * 1e3, 2),
            'base_err_mm': {'p50': pc(eb[on], 50), 'p95': pc(eb[on], 95), 'max': round(float(eb[on].max()) * 1e3, 2)},
            'base_yaw_err_mrad': {'p95': pc(ey[on], 95), 'max': round(float(ey[on].max()) * 1e3, 2)},
            'u_base_peak': np.abs(U[:, :3]).max(0).round(4).tolist(), 'u_arm_peak': round(float(np.abs(U[:, 3:]).max()), 4),
            'box_lin_ang': [cfg.v_base_lin, cfg.v_base_ang]}


def main():
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ident = json.load(open(IDENT))
    out = {'model': '手臂設定點模型（free4）＋底盤運動學（增益 1.0）；延遲 發布 0.6＋命令 1.0 週期，補償 1.6；無接觸', 'cases': {}}
    for name, fn in (('R3', r3_ref), ('S8', s8_ref)):
        out['cases'][name] = []
        for k in (1.0, 1.5, 2.0):
            r = run(K, ident, *fn(k), k)
            out['cases'][name].append(r)
            print(f"{name} ×{k:g}：參考 {r['T_ref_s']} s；失敗 {r['n_fail']}/{r['n_cycles']}；TCP 誤差 mm {r['tcp_err_mm']}；末後收斂 {r['settle_to_5mm_after_ref_end_s']} s（末 {r['tcp_err_end_mm']} mm）；底盤位置誤差 mm {r['base_err_mm']}；"
                  f"偏航 mrad {r['base_yaw_err_mrad']}；u 底盤峰 {r['u_base_peak']} 手臂峰 {r['u_arm_peak']}", flush=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(OUT, WS))


if __name__ == '__main__':
    sys.exit(main())
