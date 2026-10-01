#!/usr/bin/env python3
"""`wgmpc_core_sp`（含手臂設定點的增廣狀態核心）的離線核對。

不開模擬器、不用 GPU。對照基準是 free4 的實錄與原核心 `wgmpc_core`。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core as C                    # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

RUN = os.environ.get('WG4_RUN',
                     'evaluation/runs/wgmpc_wg2_free4_182234')
IDENT = os.environ.get('WG4_IDENT',
                       'evaluation/results/wgmpc_arm_sp_ident_free4.json')
FAIL = []


def ck(name, ok, detail=''):
    print(f'  {name:62s} {"ok" if ok else "**錯**"}  {detail}', flush=True)
    if not ok:
        FAIL.append(name)


def main() -> int:
    ident = json.load(open(IDENT))
    am = S.ArmSetpointModel(alpha=ident['alpha'], bias=ident['bias_rad'],
                            phys_dt=ident['phys_dt_measured_s'])
    cfg = S.WGMPCConfigSP(N=5, dt=0.05, arm_model=am)

    print('T1  閉式連乘 == 顯式逐物理步連乘')
    P, Q, G, h, kp = cfg.composed()
    ck('控制步含物理步數 kp', kp == 5, f'kp={kp}')
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        x0 = rng.standard_normal(6); s0 = rng.standard_normal(6)
        u = rng.standard_normal(6)
        x, s = x0.copy(), s0.copy()
        for _ in range(kp):
            x = x + am.alpha * (s - x) + am.bias
            s = s + u * am.phys_dt
        worst = max(worst, float(np.abs(P * x0 + Q * s0 + G * u + h - x).max()),
                    float(np.abs(s0 + u * cfg.dt - s).max()))
    ck('200 組隨機輸入的最大差', worst < 1e-12, f'{worst:.2e}')
    ck('G 不等於 dt_c（**不可把 α 直接當 50 ms 係數**）',
       abs(G[0] - cfg.dt) > 1e-3,
       f'G={G[0]:.6f}  dt_c={cfg.dt}  比值 {cfg.dt/G[0]:.2f}×')

    print('\nT2  退化核對：α→1、b=0 應退回「落後一個物理步的理想模型」')
    am1 = S.ArmSetpointModel(alpha=np.full(6, 1 - 1e-12), bias=np.zeros(6),
                             phys_dt=0.01)
    cfg1 = S.WGMPCConfigSP(N=5, dt=0.05, arm_model=am1)
    P1, Q1, G1, h1, _ = cfg1.composed()
    ck('P→0、Q→1', abs(P1[0]) < 1e-9 and abs(Q1[0] - 1) < 1e-9,
       f'P={P1[0]:.2e} Q={Q1[0]:.9f}')
    ck('G→(kp−1)·dt_p = 0.04（**比理想的 0.05 少一個物理步**）',
       abs(G1[0] - 0.04) < 1e-9, f'G={G1[0]:.9f}')

    print('\nT3  仿射模型：手臂與設定點區塊**嚴格線性**（無線性化誤差）')
    K = WholeBodyKinematics.from_urdf_file(
        'evaluation/models/omni_bot_wholebody_expanded.urdf')
    rng = np.random.default_rng(3)
    wa, wb, wc = 0.0, 0.0, 0.0
    for _ in range(40):
        z0 = np.concatenate([rng.standard_normal(3) * 0.2,
                             rng.standard_normal(6) * 0.5,
                             rng.standard_normal(6) * 0.5])
        U = (rng.standard_normal((cfg.N, S.NU))
             * cfg.vmax()[None, :] * 0.8)
        # **在展開點上評估仿射預測恆等於非線性 rollout**（由 c_k 的定義與
        # 歸納法保證），所以必須拿一個**偏離 nominal** 的 U 來評估，
        # 否則底盤那一項也會是零而看不出線性化誤差。
        # 附帶結論：原核心的 `lin_error` 診斷正是在 nominal 上算的，
        # 因此它恆為數值量級 —— 那個欄位量不到線性化誤差。
        Z = S.rollout_sp(z0, U, cfg)
        A_l, B_l, c_l = [], [], []
        for k in range(cfg.N):
            a, b, c = S.affine_model_sp(Z[k], U[k], cfg)
            A_l.append(a); B_l.append(b); c_l.append(c)
        Phi, Gam, gam = S.build_prediction_sp(A_l, B_l, c_l)
        U2 = U + rng.standard_normal(U.shape) * cfg.vmax()[None, :] * 0.2
        Z2 = S.rollout_sp(z0, U2, cfg)
        za = (Phi @ z0 + Gam @ U2.reshape(-1) + gam).reshape(cfg.N, S.NZ)
        wa = max(wa, float(np.abs(za[:, 3:S.NQ] - Z2[1:, 3:S.NQ]).max()))
        wb = max(wb, float(np.abs(za[:, S.NQ:] - Z2[1:, S.NQ:]).max()))
        wc = max(wc, float(np.abs(za[:, :3] - Z2[1:, :3]).max()))
    ck('手臂實測區塊 仿射 == 非線性', wa < 1e-10, f'max {wa:.2e}')
    ck('設定點區塊   仿射 == 非線性', wb < 1e-10, f'max {wb:.2e}')
    ck('底盤區塊仍有線性化誤差（**不應為零**，與原核心同源）',
       wc > 1e-9, f'max {wc:.2e}　（在偏離 nominal 的 U 上評估）')

    print('\nT4  輸入防護')
    try:
        S.make_z(np.zeros(9), [0.0] * 5 + [float("nan")])
        ck('make_z 拒絕非有限設定點', False, '**未拒絕**')
    except ValueError as e:
        ck('make_z 拒絕非有限設定點', '不以實測關節角代替' in str(e), 'ValueError')
    try:
        S.ArmSetpointModel(phys_dt=0.01).compose(0.055)
        ck('compose 拒絕非整數倍 dt', False, '**未拒絕**')
    except ValueError as e:
        ck('compose 拒絕非整數倍 dt', '整數倍' in str(e), 'ValueError')
    try:
        S.ArmSetpointModel(alpha=np.full(6, 1.5))
        ck('α 必須在 (0,1)', False, '**未拒絕**')
    except ValueError:
        ck('α 必須在 (0,1)', True, 'ValueError')

    print('\nT5  從 free4 的實際狀態求解（與原核心同一組目標與權重）')
    w = json.load(open(os.path.join(RUN, 'wg2_out.json')))
    sim = json.load(open(os.path.join(RUN, 'sim', 'wb_run.json')))
    ci = {c: i for i, c in enumerate(sim['log_cols'])}
    L = np.array(sim['log'], float)
    t = L[:, ci['t']]
    pub = [x for x in w['log'] if x.get('published')]
    tcp = w['args']['tcp']
    T_des = np.eye(4)
    T_des[:3, 3] = np.array(w['target_tcp'], float)
    T_des[:3, :3] = K.fk(np.array(w['start_q'], float), tcp)[:3, :3]

    def q_at(k):
        return np.array([L[k, ci['base_x']], L[k, ci['base_y']],
                         L[k, ci['base_yaw']]]
                        + [L[k, ci[f'joint{i}_act']] for i in range(1, 7)])

    def s_at(k):
        return np.array([L[k, ci[f'joint{i}_sp']] for i in range(1, 7)])

    k_mid = int(np.argmin(np.abs(t - pub[len(pub) // 2]['sim_t'])))
    q0, s0 = q_at(k_mid), s_at(k_mid)
    u_prev = np.array((pub[len(pub) // 2]['stages']['applied'] or {})['v'],
                      float)
    z0 = S.make_z(q0, s0)
    r_sp = S.solve_sp(K, z0, u_prev, T_des, cfg)
    cfg_old = C.WGMPCConfig(N=5, dt=0.05)
    r_old = C.solve(K, q0, u_prev, T_des, cfg_old)
    ck('增廣核心求解成功', bool(r_sp.ok),
       f'stop={r_sp.sqp_stop_reason} 殘差 {r_sp.max_residual:.2e} '
       f'n_sqp={r_sp.n_sqp_used} {r_sp.timing_ms.get("total")} ms')
    ck('原核心（基準）求解成功', bool(r_old.ok),
       f'stop={r_old.sqp_stop_reason} {r_old.timing_ms.get("total")} ms')
    ck('控制維度仍是 9（設定點**不是**新增致動自由度）',
       r_sp.u0 is not None and r_sp.u0.shape == (9,), str(r_sp.u0.shape))
    ck('約束區塊含設定點與實測兩組限位',
       {'setpoint_position', 'measured_position'} <= set(
           r_sp.residual_by_block),
       ','.join(sorted(r_sp.residual_by_block)))
    ck('線性化誤差：手臂與設定點為數值量級',
       r_sp.lin_error['max_abs_arm'] < 1e-10
       and r_sp.lin_error['max_abs_setpoint'] < 1e-10,
       f"arm {r_sp.lin_error['max_abs_arm']:.1e} "
       f"sp {r_sp.lin_error['max_abs_setpoint']:.1e} "
       f"base_xy {r_sp.lin_error['max_abs_base_xy']:.1e}")
    ck('候選參數狀態隨結果回報', r_sp.arm_model_status == '候選模型參數，未定版',
       r_sp.arm_model_status)

    print('\nT6  設定點限位：界的形式正確，且回傳解一律遵守')
    from ammr_wholebody_mpc.arm_limits import LITE6_SAFE
    hi5 = LITE6_SAFE.upper[4] - cfg.joint_margin

    def rows_for(z0_, U_):
        Zr = S.rollout_sp(z0_, U_, cfg)
        A_l, B_l, c_l = [], [], []
        for k in range(cfg.N):
            a, b, c = S.affine_model_sp(Zr[k], U_[k], cfg)
            A_l.append(a); B_l.append(b); c_l.append(c)
        Ph, Gm, gm = S.build_prediction_sp(A_l, B_l, c_l)
        return S.build_constraints_sp(z0_, U_, np.zeros(9), cfg,
                                      cfg.delta_max, Ph, Gm, gm)

    # T6a 界的**形式**：hi 應恰等於設定點剩餘餘量，列係數應是前 k 步各 dt。
    # 這是對應執行端 `_apply` 硬失效的那一道；形式對不對與 QP 好不好解無關。
    s_hot = s0.copy(); s_hot[4] = hi5 - 0.004
    q_cold = q0.copy(); q_cold[3 + 4] = 0.0
    Am, lo, hi, bl = rows_for(S.make_z(q_cold, s_hot), np.zeros((cfg.N, S.NU)))
    i0, _ = bl['setpoint_position']
    okf = True
    for st in range(cfg.N):
        r = i0 + st * S.NS + 4
        okf &= abs(hi[r] - 0.004) < 1e-9
        exp = np.zeros(cfg.N * S.NU)
        for jj in range(st + 1):
            exp[jj * S.NU + 3 + 4] = cfg.dt
        okf &= bool(np.allclose(Am[r], exp))
    ck('設定點限位列：hi == 剩餘餘量、係數 == 前 k 步各 dt', okf,
       f'hi={hi[i0+4]:.6f}（餘量 0.004）')
    # 實測那組的界**不同**（係數帶 P/Q/G 的複合），確認兩組不是同一條
    j0, _ = bl['measured_position']
    ck('實測限位列與設定點限位列**不同**',
       not np.allclose(Am[i0 + 4], Am[j0 + 4]),
       f'measured hi step1 {hi[j0+4]:.5f} vs setpoint {hi[i0+4]:.5f}')

    # T6b 回傳解一律遵守設定點界（掃幾個餘量；不可解的照實記，不當成通過）
    print('    餘量掃描（d = 設定點距上界的餘量）')
    n_ok = n_resp = 0
    for d in (0.5, 0.2, 0.1, 0.05, 0.02, 0.004):
        sh = s0.copy(); sh[4] = hi5 - d
        r = S.solve_sp(K, S.make_z(q_cold, sh), np.zeros(9), T_des, cfg)
        if r.ok:
            n_ok += 1
            resp = bool((r.S_pred[1:, 4] <= hi5 + 1e-7).all())
            n_resp += resp
            print(f'      d={d:6.3f}  ok  預測設定點 max {r.S_pred[1:,4]:.0f}'
                  if False else
                  f'      d={d:6.3f}  ok   max s5 {r.S_pred[1:,4].max():.6f}'
                  f' <= {hi5:.6f}  {"遵守" if resp else "**越界**"}')
        else:
            print(f'      d={d:6.3f}  **{r.sqp_stop_reason}**'
                  f'  qp_status={r.qp_status[-1] if r.qp_status else "-"}')
    ck('所有**成功回傳**的解都遵守設定點界', n_ok > 0 and n_resp == n_ok,
       f'{n_resp}/{n_ok} 遵守')

    # T6c 兩組限位都必須保留：**蘊含不成立**（Howard 的反例，已獨立重現）
    print('    反例：b ≠ 0 時 x⁺ 不是 x 與 s 的凸組合 ⇒ 實測界不被設定點界蘊含')
    jj = 1                                  # joint2，偏置最大
    al_j = float(cfg.arm_model.alpha[jj]); b_j = float(cfg.arm_model.bias[jj])
    Uu = 0.0                                # 以收緊後上限為原點
    s_in = Uu - 0.5e-3
    x = s_in
    for _ in range(kp):
        x = x + al_j * (s_in - x) + b_j     # 命令 u = 0 ⇒ 設定點不變
    print(f'      α={al_j:.5f}  b={b_j:.4e}  穩態偏移 b/α = {b_j/al_j:+.4e} rad')
    print(f'      起始 x = s = U − 0.5 mrad、u = 0'
          f' ⇒ 五個物理步後 x = U {1e3*x:+.3f} mrad')
    ck('**反例成立**：設定點界成立而實測界不成立 ⇒ 兩組限位都要保留',
       s_in <= Uu and x > Uu,
       f's = U{1e3*s_in:+.1f} mrad（界內）、x = U{1e3*x:+.3f} mrad（**越界**）'
       '　⇒ 先前的「蘊含」論證已撤回')
    # 閉式複合映射必須給同一個值（否則反例與模型不是同一件事）
    Pj, Qj, Gj, hj, _ = cfg.composed()
    x_cf = Pj[jj] * s_in + Qj[jj] * s_in + Gj[jj] * 0.0 + hj[jj]
    ck('閉式複合映射重現同一反例', abs(x_cf - x) < 1e-15,
       f'閉式 {1e3*x_cf:+.3f} mrad　差 {abs(x_cf-x):.1e}')

    # T6d 列縮放前後的可解性：**同一批六個餘量案例**
    print('    列縮放前後的六個限位案例（殘差一律以未縮放原單位檢查）')
    import dataclasses
    cfg_ns = dataclasses.replace(cfg, row_scaling=False)
    print(f"      {'d (rad)':>9} {'未縮放':>26} {'列縮放':>26} {'縮放後殘差':>12}")
    n_fix = 0
    for d in (0.5, 0.2, 0.1, 0.05, 0.02, 0.004):
        sh = s0.copy(); sh[4] = hi5 - d
        z = S.make_z(q_cold, sh)
        r_ns = S.solve_sp(K, z, np.zeros(9), T_des, cfg_ns)
        r_rs = S.solve_sp(K, z, np.zeros(9), T_des, cfg)
        if (not r_ns.ok) and r_rs.ok:
            n_fix += 1
        print(f'      {d:9.3f} {r_ns.sqp_stop_reason:>26} '
              f'{r_rs.sqp_stop_reason:>26} '
              f'{(f"{r_rs.max_residual:.1e}" if r_rs.ok else "-"):>12}')
    ck('列縮放修復了原本無解的案例', n_fix > 0, f'{n_fix} 個案例由無解轉為有解')
    ck('列縮放後六個案例全部有解', all(
        S.solve_sp(K, S.make_z(q_cold, (lambda a: (a.__setitem__(4, hi5 - d),
                                                   a)[1])(s0.copy())),
                   np.zeros(9), T_des, cfg).ok
        for d in (0.5, 0.2, 0.1, 0.05, 0.02, 0.004)), '六個餘量')
    # 縮放係數範圍與等價性
    r_chk = S.solve_sp(K, S.make_z(q_cold, s_hot), np.zeros(9), T_des, cfg)
    ck('殘差以**未縮放原單位**檢查且 <= r_tol',
       r_chk.ok and r_chk.max_residual <= cfg.r_tol,
       f'max_residual {r_chk.max_residual:.2e} <= r_tol {cfg.r_tol:.0e}')
    ck('列縮放係數為有限正值', r_chk.row_scale_range[0] > 0
       and np.isfinite(r_chk.row_scale_range[1]),
       f'd ∈ [{r_chk.row_scale_range[0]:.3f}, {r_chk.row_scale_range[1]:.3f}]')
    ck('r_tol 與 accepted_status 未被放寬',
       cfg.r_tol == C.WGMPCConfig().r_tol
       and cfg.accepted_status == ('solved',),
       f'r_tol {cfg.r_tol:.0e}、accepted_status {cfg.accepted_status}')

    print('\nT7  用 free4 實錄比較兩個模型的**一控制步**預測落差')
    # 兩者都用同一段實際套用的命令；比較預測 TCP 誤差變化與實際
    rows = []
    for x in pub:
        ap = x['stages'].get('applied') or {}
        u, ts = ap.get('v'), ap.get('sim_t')
        if u is None or ts is None:
            continue
        k0 = int(np.argmin(np.abs(t - ts)))
        k1 = int(np.argmin(np.abs(t - (ts + cfg.dt))))
        if abs(t[k0] - ts) > 0.02 or abs(t[k1] - (ts + cfg.dt)) > 0.02:
            continue
        if not np.isfinite(s_at(k0)).all():
            continue
        u = np.asarray(u, float)
        q_a = q_at(k1)
        e_act = C.task_error(K, q_a, T_des, tcp)
        q_i = C.step(q_at(k0), u, cfg.dt)
        e_i = C.task_error(K, q_i, T_des, tcp)
        z_n = S.step_sp(S.make_z(q_at(k0), s_at(k0)), u, cfg)
        e_n = S.task_error_sp(K, z_n, T_des, tcp)
        rows.append([np.linalg.norm(e_i[:3] - e_act[:3]),
                     np.linalg.norm(e_n[:3] - e_act[:3]),
                     np.linalg.norm(e_i[3:] - e_act[3:]),
                     np.linalg.norm(e_n[3:] - e_act[3:])])
    R = np.array(rows)
    print(f'    配對週期 n={len(R)}')
    for nm, a, b in (('位置 m ', 0, 1), ('姿態 rad', 2, 3)):
        print(f'    {nm} 預測誤差 p50：原核心 {np.percentile(R[:,a],50):.5f}'
              f'　增廣 {np.percentile(R[:,b],50):.5f}'
              f'　p95：{np.percentile(R[:,a],95):.5f} / '
              f'{np.percentile(R[:,b],95):.5f}')
    ck('位置一步預測誤差下降', np.percentile(R[:, 1], 50)
       < np.percentile(R[:, 0], 50),
       f'{np.percentile(R[:,0],50)/max(np.percentile(R[:,1],50),1e-12):.1f}× 改善')
    ck('姿態一步預測誤差下降', np.percentile(R[:, 3], 50)
       < np.percentile(R[:, 2], 50),
       f'{np.percentile(R[:,2],50)/max(np.percentile(R[:,3],50),1e-12):.1f}× 改善')

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。**這是模型與求解器的離線核對，不是閉迴路成果**；')
        print('到達與保持要另行安排同目標的物理對照。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
