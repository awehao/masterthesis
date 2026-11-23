#!/usr/bin/env python3
"""d̂_b 合成閉迴路比較（離線、不跑 Isaac；Codex 20261027_010258 §3-2）。
起點：door_coopen_core_g_1 第二次暫停入口（最接近 21.48 s 的已記錄求解輪）之 q_meas、s_meas、u_prev、T_cyc（目標靜止）、arm_bias、park_hold.u_hold。
求解：wgmpc_core_sp_db.solve_sp，參數同節點（N 5、dt 0.05、w_s 0.001、w_a 0.05、w_s_arm 0.05、tcp link_tcp）；cfg.base_hold＝u_hold（每輪不變）。
時序：量測 → 發布 3 物理步（D_pub 0.6 週期）→ 生效再 5 步（D_cmd 1.0）；延遲補償以已發布歷史推 8 步（d_state 1.6），
  推進用 plant_phys_step(…, base_bias＝d̂_b)（與節點同一式）。手臂受控體＝同一設定點模型與同一 arm_bias（手臂模型視為正確）。
底盤受控體：每物理步 u＋b_true。情境：
  A unrealized_hold：b_true＝−u_hold（g_1 實測：保持命令 (0.24, −0.89 mm/s, −2.38 mrad/s) 而底盤 4.1 s 只移 0.05 mm）。
  C zero_bias：b_true＝0（保持命令照實現）——回歸：估計應近零、與無補償相近。
  D wrong_estimate：b_true＝−u_hold，但求解端固定採用 d̂_b＝＋u_hold 方向的錯誤值（反號；不經估計器）——反例，看受損程度。
方法：none（旗標關，＝原核心）／est（door_base_bias_est 因果估計，登錄常數，冷啟動：前 N_MIN 個區間 none）。
閘門（r2 續進條件，不改）：垂直 ≤ 2 mm、|沿軸| ≤ 5 mm、姿態 ≤ 0.02 rad 連續 0.5 s。期限：g_1 該暫停入口到中止的剩餘 4.16 s；另報 15 s 內是否達成。
界線：合成受控體只含一個常值底盤偏差；不代表 Isaac 會同樣達成，只比較同一合成條件下兩種預測模型。

    python3 evaluation/door_coopen_core_db_cl.py   # 寫 results/door/door_coopen_core_db_cl_r1.json（不覆寫）
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
import door_base_bias_est as E                               # noqa: E402
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_core_db_cl_r1.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
T_SNAP, T_ABORT = 21.48, 25.64
N_PUB, N_CMD = 3, 5
PERP_RES, ALONG_RES, ROT_RES, HOLD_S = 0.002, 0.005, 0.02, 0.5
T_MAX = 15.0


def decompose(K, q, T, ax):
    e = task_error(K, q, T, 'link_tcp')
    p = -np.asarray(e[:3])
    al = float(p @ ax)
    return al, float(np.linalg.norm(p - al * ax)), float(np.linalg.norm(e[3:]))


def run_case(K, init, b_true, method, b_fixed=None):
    cfg = init['cfg']()
    q, s = init['q'].copy(), init['s'].copy()
    T, ax = init['T'], init['ax']
    dtp = cfg.arm_model.phys_dt
    n_phys = int(round(cfg.dt / dtp))
    est = E.BaseBiasEstimator(dtp)
    est.set_enabled(True)
    k = 0
    est.on_pose(0.0, q[:3])
    pubq = [init['u_prev'].copy()] * N_PUB          # 尚未發布
    pipe = [init['u_prev'].copy()] * N_CMD          # 已發布、尚未生效
    hist_eff = []                                   # 依生效順序的命令（延遲補償用：未來 8 步會生效者）
    u_prev = init['u_prev'].copy()
    U_warm = None
    t_ok0, t_resume = None, None
    rows = []
    n_fail = 0
    t = 0.0
    while t < T_MAX - 1e-9:
        al, pe, ro = decompose(K, q, T, ax)
        b_hat, why, _ = est.estimate(t) if method == 'est' else (None, method, None)
        if method == 'fixed':
            b_hat, why = np.asarray(b_fixed, float), 'fixed'
        rows.append({'t': round(t, 4), 'along_mm': al * 1e3, 'perp_mm': pe * 1e3, 'rot_rad': ro,
                     'b_hat': None if b_hat is None else b_hat.tolist(), 'why': why, 'base_xy_mm': ((q[:2] - init['q'][:2]) * 1e3).tolist()})
        if pe <= PERP_RES and abs(al) <= ALONG_RES and ro <= ROT_RES:
            t_ok0 = t if t_ok0 is None else t_ok0
            if t - t_ok0 >= HOLD_S - 1e-9 and t_resume is None:
                t_resume = t
                break
        else:
            t_ok0 = None
        # 延遲補償：之後 8 步生效的命令＝管線內 5 筆＋待發布 3 筆（皆已知）
        inflight = list(pipe) + list(pubq)
        qs, ss = q.copy(), s.copy()
        for uu in inflight:
            qs, ss = S.plant_phys_step(qs, ss, uu, cfg, 1, base_bias=b_hat)
        cfg.base_bias = None if b_hat is None else tuple(float(x) for x in b_hat)
        r = S.solve_sp(K, S.make_z(qs, ss), u_prev, T, cfg, U_warm=U_warm)
        if r.ok:
            u, U_warm = np.asarray(r.u0, float), r.U
        else:
            n_fail += 1
            u, U_warm = u_prev.copy(), None
        for _ in range(n_phys):
            pubq.append(u.copy())
            pipe.append(pubq.pop(0))
            u_app = pipe.pop(0)
            est.on_cmd(k, round(k * dtp, 9), u_app[:3], 0, 1)
            q, s = S.plant_phys_step(q, s, u_app, cfg, 1, base_bias=b_true)
            hist_eff.append(u_app)
            k += 1
            if k % 2 == 0:
                est.on_pose(round(k * dtp, 9), q[:3])
            u_prev = u_app.copy()
        t = round(t + cfg.dt, 9)
    budget = T_ABORT - T_SNAP
    return {'t_resume_s': t_resume, 'resume_within_budget': (t_resume is not None and t_resume <= budget + 1e-9), 'budget_s': budget,
            'n_fail': n_fail, 'perp_mm_at': {str(x): next((r_['perp_mm'] for r_ in rows if r_['t'] >= x - 1e-9), None) for x in (0.0, 1.0, 2.0, 4.0, 8.0)},
            'final': rows[-1], 'est_rejects': dict(est.rejects), 'rows': rows[::2]}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    row = min([x for x in sv['log'] if (x.get('solve_in') or {}).get('U_sol') is not None], key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - T_SNAP))
    si = row['solve_in']
    st = min(r['steps'], key=lambda s_: abs(s_[c['sim_t']] - si['snap_sim_t_exact']))
    idn = json.load(open(IDENT))
    a = sv['args']
    a_H = np.asarray(json.load(open(GB))['args']['a_H'], float)

    def mkcfg():
        am = S.ArmSetpointModel(alpha=idn['alpha'], bias=si['arm_bias'], phys_dt=idn['phys_dt_measured_s'])
        cfg = S.WGMPCConfigSP(N=int(a['N']), dt=1.0 / float(a['rate']), tcp='link_tcp', arm_model=am,
                              w_s=float(a['w_s']), w_a=float(a['w_a']), w_s_arm=float(a['w_s_arm']))
        cfg.base_hold = tuple(float(x) for x in row['park_hold']['u_hold'])
        return cfg
    u_hold = np.asarray(row['park_hold']['u_hold'], float)
    init = {'cfg': mkcfg, 'q': np.asarray(si['q_meas'], float), 's': np.asarray(si['s_meas'], float), 'u_prev': np.asarray(si['u_prev'], float),
            'T': np.asarray(si['T_cyc'], float).reshape(4, 4), 'ax': KC._H_of(st[c['door_q_rad']])[:3, :3] @ a_H, 'u_hold': u_hold}
    K = WholeBodyKinematics.from_urdf_file(URDF)
    cases = {'A_unrealized_hold': (-u_hold, ('none', 'est')), 'C_zero_bias': (np.zeros(3), ('none', 'est')), 'D_wrong_estimate': (-u_hold, ('fixed',))}
    out = {'schema': 'door_coopen_core_db_cl/1', 'snap_sim_t': si['snap_sim_t_exact'], 'u_hold': u_hold.tolist(),
           'gate': {'perp_resume_m': PERP_RES, 'along_resume_m': ALONG_RES, 'rot_resume_rad': ROT_RES, 'hold_s': HOLD_S},
           'estimator': {'WIN_S': E.WIN_S, 'N_MIN': E.N_MIN, 'MAX_AGE_S': E.MAX_AGE_S, 'CAP': list(E.CAP)}, 'cases': {}}
    for nm, (bt, methods) in cases.items():
        for m in methods:
            res = run_case(K, init, bt, m, b_fixed=(u_hold if m == 'fixed' else None))
            out['cases'][f'{nm}/{m}'] = res
            f = res['final']
            print(f"{nm}/{m}：續進 {res['t_resume_s']} s（預算 {res['budget_s']:.2f} s 內 {res['resume_within_budget']}）；垂直 "
                  + '、'.join(f"{k_}s {v:.2f}" for k_, v in res['perp_mm_at'].items() if v is not None)
                  + f" mm；終 沿軸 {f['along_mm']:.2f} mm 姿態 {f['rot_rad']:.4f}；底盤位移 {f['base_xy_mm']}；d̂_b {f['b_hat']}；失敗 {res['n_fail']}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
