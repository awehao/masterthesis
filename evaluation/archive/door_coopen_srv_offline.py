#!/usr/bin/env python3
"""固定終點伺服建立持握的離線可行性（Codex r17 §3：標準交棒後，自 Q_PRE 以同一 W-GMPC 直接伺服到 T(H0, s＝0)；不跑 Isaac）。
起點：停位（coopen_core_params_48_r6 start.stand）＋ Q_PRE（coopen_core_h0_r4 q_pre，s 0.10 m），設定點＝實測、u_prev＝0。
目標：T(H0, 0)＝H0·T_HG_full（coopen_core_grasp_basis_r4，與任務節點同一式），門角 0、固定。
求解：wgmpc_core_sp（原核心，ADM9 所用），參數同 g2 求解節點（N 5、dt 0.05、w_s 0.001、w_a 0.05、w_s_arm 0.05、tcp link_tcp）；cfg.base_hold＝(0,0,0)。
時序：D_pub 3＋D_cmd 5 物理步；延遲補償以已發布歷史推 8 步（同 door_coopen_core_wa_cl）。手臂設定點模型 α、b＝free4 辨識值。
受控體情境：
  S0 nominal：手臂受控體＝同一模型同一 b；底盤靜止（保持伺服抵銷漂移的理想化）。
  S1 sag_unmodeled：受控體 j2 另有穩態偏移 −0.027 rad（iso_1 交棒前實測；求解端未知、本離線不模擬線上偏移觀測 ⇒ 最壞情況）。
每控制週期：TCP 位置／姿態誤差、沿軸／垂直分量；淨空 TJ.GEO（door_coopen_traj_v1，與 h0／build 同一檢查器；手指張開 C.FINGER_OPEN）：
  nonfinger、finger、self、base。門檻沿用 build 的停位構型檢查：5、5、5、30 mm（不新增）。
准入：位置 < 2 mm 且姿態 ≤ 0.02 rad 連續 0.5 s（同 init_grasp2）；期限 15 s。另報 TCP 最大速度、|u|/vmax。
界線：不模擬 adapter／E2 與執行端低速界限、線上偏移觀測、底盤漂移與保持伺服動態、接觸力學；只回答「同一核心與延遲下，固定終點伺服的路徑是否保持幾何淨空並在期限內准入」。

    python3 evaluation/door_coopen_srv_offline.py   # 寫 results/door/door_coopen_srv_offline_r1.json（不覆寫）
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
from ammr_wholebody_mpc import wgmpc_core_sp as S           # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_coopen_reach_v1 as RV                            # noqa: E402
import door_coopen_traj_v1 as TJ                             # noqa: E402
import door_k2_check as KC                                   # noqa: E402

R = os.path.join(HERE, 'results', 'door')
OUT = os.path.join(R, 'door_coopen_srv_offline_r1.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
N_PUB, N_CMD = 3, 5
GRIP_TOL, ROT_TOL, SETTLE, T_MAX = 0.002, 0.02, 0.5, 15.0
THR = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005, 'base': 0.030}
SAG_J2 = -0.027


def run_case(K, cfg, q0, s0, T, ax, b_extra):
    dtp = cfg.arm_model.phys_dt
    n_phys = int(round(cfg.dt / dtp))
    cfgp = S.WGMPCConfigSP(**{k: getattr(cfg, k) for k in ('N', 'dt', 'tcp', 'w_s', 'w_a', 'w_s_arm')},
                           arm_model=S.ArmSetpointModel(alpha=cfg.arm_model.alpha, bias=np.asarray(cfg.arm_model.bias, float) + b_extra, phys_dt=dtp))
    cfgp.base_hold = cfg.base_hold
    q, s = q0.copy(), s0.copy()
    u_prev = np.zeros(S.NU)
    pubq, pipe = [np.zeros(S.NU)] * N_PUB, [np.zeros(S.NU)] * N_CMD
    U_warm, t, t0, t_admit = None, 0.0, None, None
    rows, fails = [], []
    vmax = np.asarray(cfg.vmax(), float)
    p_prev = None
    while t < T_MAX - 1e-9:
        Tt = K.fk(q, 'link_tcp')
        e = Tt[:3, 3] - T[:3, 3]
        al = float(e @ ax)
        Rr = T[:3, :3].T @ Tt[:3, :3]
        rot = math.acos(max(-1.0, min(1.0, (float(np.trace(Rr)) - 1.0) / 2.0)))
        ep = float(np.linalg.norm(e))
        g = TJ.GEO(list(q[:3]) + list(q[3:]), 0.0, RV.C.FINGER_OPEN)
        v_tcp = None if p_prev is None else float(np.linalg.norm(Tt[:3, 3] - p_prev)) / cfg.dt
        p_prev = Tt[:3, 3].copy()
        rows.append({'t': round(t, 4), 'err_p_mm': ep * 1e3, 'along_mm': al * 1e3, 'perp_mm': float(np.linalg.norm(e - al * ax)) * 1e3, 'rot_rad': rot,
                     'clear_mm': {k: float(g[k][0]) * 1e3 for k in THR}, 'who': {k: g[k][1] for k in THR}, 'v_tcp_mps': v_tcp})
        if ep < GRIP_TOL and rot <= ROT_TOL:
            t0 = t if t0 is None else t0
            if t - t0 >= SETTLE - 1e-9 and t_admit is None:
                t_admit = t
                break
        else:
            t0 = None
        qs, ss = q.copy(), s.copy()
        for uu in list(pipe) + list(pubq):
            qs, ss = S.plant_phys_step(qs, ss, uu, cfg, 1)
        r = S.solve_sp(K, S.make_z(qs, ss), u_prev, T, cfg, U_warm=U_warm)
        if r.ok:
            u, U_warm = np.asarray(r.u0, float), r.U
        else:
            fails.append(round(t, 4))
            u, U_warm = u_prev.copy(), None
        rows[-1]['u_over_vmax'] = float(np.max(np.abs(u) / vmax))
        for _ in range(n_phys):
            pubq.append(u.copy())
            pipe.append(pubq.pop(0))
            u_app = pipe.pop(0)
            q, s = S.plant_phys_step(q, s, u_app, cfgp, 1)
            u_prev = u_app.copy()
        t = round(t + cfg.dt, 9)
    mins = {k: min(r_['clear_mm'][k] for r_ in rows) for k in THR}
    argm = {k: min(rows, key=lambda r_: r_['clear_mm'][k]) for k in THR}
    viol = {k: [r_['t'] for r_ in rows if r_['clear_mm'][k] < THR[k] * 1e3] for k in THR}
    return {'t_admit_s': t_admit, 'n_fail': len(fails), 'fail_t': fails,
            'clear_min_mm': mins, 'clear_min_at': {k: {'t': argm[k]['t'], 'who': argm[k]['who'][k], 'err_p_mm': argm[k]['err_p_mm']} for k in THR},
            'clear_violation_t': {k: v[:20] for k, v in viol.items()}, 'n_clear_violation': {k: len(v) for k, v in viol.items()},
            'v_tcp_max_mps': max((r_['v_tcp_mps'] or 0.0) for r_ in rows), 'u_over_vmax_max': max(r_.get('u_over_vmax', 0.0) for r_ in rows),
            'final': rows[-1], 'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    prm = json.load(open(os.path.join(R, 'coopen_core_params_48_r6.json')))
    h0 = json.load(open(os.path.join(R, 'coopen_core_h0_r4.json')))
    gb = json.load(open(os.path.join(R, 'coopen_core_grasp_basis_r4.json')))
    idn = json.load(open(IDENT))
    stand = np.asarray(prm['start']['stand'], float)
    q_pre = np.asarray(h0['q_pre'], float)
    T_HG = np.asarray(gb['args']['T_HG_full'], float)
    a_H = np.asarray(gb['args']['a_H'], float)
    H0 = KC._H_of(0.0)
    T = H0 @ T_HG
    ax = H0[:3, :3] @ a_H
    K = WholeBodyKinematics.from_urdf_file(URDF)
    am = S.ArmSetpointModel(alpha=idn['alpha'], bias=idn['bias_rad'], phys_dt=idn['phys_dt_measured_s'])
    cfg = S.WGMPCConfigSP(N=5, dt=0.05, tcp='link_tcp', arm_model=am, w_s=0.001, w_a=0.05, w_s_arm=0.05)
    cfg.base_hold = (0.0, 0.0, 0.0)
    q0 = np.concatenate([stand, q_pre])
    alpha = np.asarray(idn['alpha'], float).reshape(-1) * np.ones(6)
    b_sag = np.zeros(6)
    b_sag[1] = alpha[1] * SAG_J2                              # 穩態 x − s ＝ b/α ⇒ 額外 b＝α·偏移
    out = {'schema': 'door_coopen_srv_offline/1', 'start': {'stand': stand.tolist(), 'q_pre': q_pre.tolist(), 's_pre_m': h0['s_pre_m']},
           'target': T.tolist(), 'thresholds_mm': {k: v * 1e3 for k, v in THR.items()}, 'cases': {}}
    for nm, bx in (('S0_nominal', np.zeros(6)), ('S1_sag_unmodeled', b_sag)):
        res = run_case(K, cfg, q0, q_pre.copy(), T, ax, bx)
        out['cases'][nm] = res
        f = res['final']
        print(f"{nm}：准入 {res['t_admit_s']} s；失敗 {res['n_fail']}；淨空最小（mm）" + '、'.join(f"{k} {v:.2f}（{res['clear_min_at'][k]['who']} @ {res['clear_min_at'][k]['t']} s）" for k, v in res['clear_min_mm'].items())
              + f"；違反輪數 {res['n_clear_violation']}；TCP 最大速度 {res['v_tcp_max_mps']:.3f} m/s；|u|/vmax {res['u_over_vmax_max']:.2f}；"
              f"終 誤差 {f['err_p_mm']:.2f} mm（沿軸 {f['along_mm']:.2f}、垂直 {f['perp_mm']:.2f}）姿態 {f['rot_rad']:.4f}", flush=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
