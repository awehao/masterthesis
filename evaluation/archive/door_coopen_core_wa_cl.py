#!/usr/bin/env python3
"""接近段單一成本候選 w_a 0.05 → 0.01 的合成閉迴路對照（離線；Codex 20261027_011742）。d̂_b 關閉；其餘同 door_coopen_core_db_cl.py：
起點 g_1 最接近 21.48 s 的已記錄輪（實際 21.50 s）之 q_meas、s_meas、u_prev、T_cyc（靜止）、arm_bias、u_hold；同節點參數（N 5、dt 0.05、
w_s 0.001、w_s_arm 0.05、tcp link_tcp）；cfg.base_hold＝u_hold（**固定，不逐輪依位姿更新停車伺服**）；D_pub 3＋D_cmd 5 物理步、延遲補償推 8 步。
情境（同前）：A 保持未實現（b_true＝−u_hold，主要對照）；C 固定保持命令被實現（b_true＝0，回歸；非完整停車閉迴路）。
w_a ∈ {0.05（原值）, 0.01（候選，單一試驗選值）}。不掃描、不追加其他權重。
報告：
  - 剩餘預算 4.14 s（21.50 → 25.64）內是否滿足完整續進條件（垂直 ≤ 2 mm、|沿軸| ≤ 5 mm、姿態 ≤ 0.02 rad 連續 0.5 s）；15 s 上限內續進時刻。
  - 求解失敗：次數、時刻；失敗輪沿用上一筆已套用命令（單列）；續進保持窗內若有失敗 ⇒ resume_clean False（不算成功）。
  - 限制比例：max |u|/vmax（底盤、手臂分列）、max |Δu|/(amax·dt)（逐輪已套用命令差）。
  - 三維有號位置誤差（世界，實際 − 目標）與沿軸、垂直、姿態時間列（每 0.1 s）。
  - 命令振盪：手臂六軸已發布命令一階差的號變次數（|Δ| > 1e-6 rad/s 才計）／秒、二階差 RMS；TCP 垂直誤差在續進前的回升次數（相鄰 0.1 s 點上升 > 0.02 mm）。
界線：合成受控體為固定保持命令＋常值底盤偏差；不代表 Isaac 結果。

    python3 evaluation/door_coopen_core_wa_cl.py   # 寫 results/door/door_coopen_core_wa_cl_r1.json（不覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp_db as S        # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import task_error         # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_core_wa_cl_r1.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
T_SNAP_Q, T_ABORT = 21.48, 25.64
N_PUB, N_CMD = 3, 5
PERP_RES, ALONG_RES, ROT_RES, HOLD_S = 0.002, 0.005, 0.02, 0.5
T_MAX = 15.0
W_A = (0.05, 0.01)


def run_case(K, init, b_true, w_a):
    cfg = init['cfg'](w_a)
    q, s = init['q'].copy(), init['s'].copy()
    T, ax = init['T'], init['ax']
    dtp = cfg.arm_model.phys_dt
    n_phys = int(round(cfg.dt / dtp))
    vmax, amax = np.asarray(cfg.vmax(), float), np.asarray(cfg.amax(), float)
    pubq = [init['u_prev'].copy()] * N_PUB
    pipe = [init['u_prev'].copy()] * N_CMD
    u_prev = init['u_prev'].copy()
    U_warm = None
    t_ok0, t_resume = None, None
    rows, pubs, fails = [], [], []
    du_ratio = 0.0
    t = 0.0
    while t < T_MAX - 1e-9:
        e = task_error(K, q, T, 'link_tcp')
        p = -np.asarray(e[:3])
        al = float(p @ ax)
        pe = float(np.linalg.norm(p - al * ax))
        ro = float(np.linalg.norm(e[3:]))
        rows.append({'t': round(t, 4), 'p_world_mm': (p * 1e3).tolist(), 'along_mm': al * 1e3, 'perp_mm': pe * 1e3, 'rot_rad': ro})
        if pe <= PERP_RES and abs(al) <= ALONG_RES and ro <= ROT_RES:
            t_ok0 = t if t_ok0 is None else t_ok0
            if t - t_ok0 >= HOLD_S - 1e-9 and t_resume is None:
                t_resume = (t, t_ok0)
                break
        else:
            t_ok0 = None
        qs, ss = q.copy(), s.copy()
        for uu in list(pipe) + list(pubq):
            qs, ss = S.plant_phys_step(qs, ss, uu, cfg, 1)
        r = S.solve_sp(K, S.make_z(qs, ss), u_prev, T, cfg, U_warm=U_warm)
        if r.ok:
            u, U_warm = np.asarray(r.u0, float), r.U
        else:
            fails.append(round(t, 4))
            u, U_warm = u_prev.copy(), None                     # 失敗輪沿用上一筆已套用命令（單列）
        pubs.append(u.copy())
        for _ in range(n_phys):
            pubq.append(u.copy())
            pipe.append(pubq.pop(0))
            u_app = pipe.pop(0)
            du_ratio = max(du_ratio, float(np.max(np.abs(u_app - u_prev) / (amax * dtp))))
            q, s = S.plant_phys_step(q, s, u_app, cfg, 1, base_bias=b_true)
            u_prev = u_app.copy()
        t = round(t + cfg.dt, 9)
    P = np.asarray(pubs)
    d1 = np.diff(P[:, 3:], axis=0)
    sgn = np.sign(np.where(np.abs(d1) > 1e-6, d1, 0.0))
    flips = int(sum(np.sum((sgn[1:, j] * sgn[:-1, j]) < 0) for j in range(6)))
    dur = len(P) * cfg.dt
    pr = [x['perp_mm'] for x in rows][::2]
    budget = T_ABORT - init['t_snap']
    clean = t_resume is not None and not any(t_resume[1] - 1e-9 <= f_ <= t_resume[0] + 1e-9 for f_ in fails)
    return {'w_a': w_a, 't_resume_s': None if t_resume is None else t_resume[0], 'resume_hold_window_s': None if t_resume is None else [t_resume[1], t_resume[0]],
            'resume_within_budget': bool(clean and t_resume[0] <= budget + 1e-9), 'resume_clean': clean, 'budget_s': budget,
            'n_fail': len(fails), 'fail_t': fails,
            'u_over_vmax_max': {'base': float(np.max(np.abs(P[:, :3]) / vmax[:3])), 'arm': float(np.max(np.abs(P[:, 3:]) / vmax[3:]))},
            'du_applied_over_amax_dt_max': du_ratio,
            'arm_cmd_sign_flips_per_s': flips / dur, 'arm_cmd_d2_rms': float(np.sqrt(np.mean(np.diff(P[:, 3:], 2, axis=0) ** 2))) if len(P) > 2 else None,
            'perp_rebounds_0p1s': int(sum(1 for a_, b_ in zip(pr, pr[1:]) if b_ - a_ > 0.02)),
            'perp_mm_at': {str(x): next((r_['perp_mm'] for r_ in rows if r_['t'] >= x - 1e-9), None) for x in (0.0, 1.0, 2.0, 3.0, 4.0)},
            'final': rows[-1], 'rows_0p1s': rows[::2]}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    row = min([x for x in sv['log'] if (x.get('solve_in') or {}).get('U_sol') is not None], key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - T_SNAP_Q))
    si = row['solve_in']
    st = min(r['steps'], key=lambda s_: abs(s_[c['sim_t']] - si['snap_sim_t_exact']))
    idn = json.load(open(IDENT))
    a = sv['args']
    a_H = np.asarray(json.load(open(GB))['args']['a_H'], float)

    def mkcfg(w_a):
        am = S.ArmSetpointModel(alpha=idn['alpha'], bias=si['arm_bias'], phys_dt=idn['phys_dt_measured_s'])
        cfg = S.WGMPCConfigSP(N=int(a['N']), dt=1.0 / float(a['rate']), tcp='link_tcp', arm_model=am,
                              w_s=float(a['w_s']), w_a=float(w_a), w_s_arm=float(a['w_s_arm']))
        cfg.base_hold = tuple(float(x) for x in row['park_hold']['u_hold'])
        return cfg
    assert abs(float(a['w_a']) - W_A[0]) < 1e-12
    u_hold = np.asarray(row['park_hold']['u_hold'], float)
    init = {'cfg': mkcfg, 'q': np.asarray(si['q_meas'], float), 's': np.asarray(si['s_meas'], float), 'u_prev': np.asarray(si['u_prev'], float),
            'T': np.asarray(si['T_cyc'], float).reshape(4, 4), 'ax': KC._H_of(st[c['door_q_rad']])[:3, :3] @ a_H, 't_snap': float(si['snap_sim_t_exact'])}
    K = WholeBodyKinematics.from_urdf_file(URDF)
    out = {'schema': 'door_coopen_core_wa_cl/1', 'snap_sim_t': init['t_snap'], 'budget_s': T_ABORT - init['t_snap'], 'u_hold': u_hold.tolist(),
           'fixed': {'N': a['N'], 'w_s': a['w_s'], 'w_s_arm': a['w_s_arm'], 'base_bias': None, 'base_hold': 'fixed u_hold'}, 'cases': {}}
    for nm, bt in (('A_unrealized_hold', -u_hold), ('C_fixed_hold_realized', np.zeros(3))):
        for w_a in W_A:
            res = run_case(K, init, bt, w_a)
            out['cases'][f'{nm}/w_a={w_a}'] = res
            f = res['final']
            print(f"{nm} w_a {w_a}：續進 {res['t_resume_s']} s（預算 {res['budget_s']:.2f} s 內 {res['resume_within_budget']}，clean {res['resume_clean']}）；"
                  f"垂直 " + '、'.join(f"{k_}s {v:.2f}" for k_, v in res['perp_mm_at'].items() if v is not None)
                  + f" mm；終 p_world {[round(x, 2) for x in f['p_world_mm']]} mm 姿態 {f['rot_rad']:.4f}；失敗 {res['n_fail']}；"
                  f"|u|/vmax 底盤 {res['u_over_vmax_max']['base']:.3f} 手臂 {res['u_over_vmax_max']['arm']:.3f}；|Δu|/(a·dt) {res['du_applied_over_amax_dt_max']:.3f}；"
                  f"號變 {res['arm_cmd_sign_flips_per_s']:.1f}/s、d2 rms {res['arm_cmd_d2_rms']:.2e}；垂直回升 {res['perp_rebounds_0p1s']}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
