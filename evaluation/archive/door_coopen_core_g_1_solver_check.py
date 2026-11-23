#!/usr/bin/env python3
"""door_coopen_core_g_1 求解紀錄分析（只讀；Codex 20261027_005251 §2）：第二次固定目標等待期間（暫停，目標 T(s 0.0224) 靜止）的早／中／晚三個快照。
每個快照（align_solver.json log 中最接近 22.0／23.8／25.4 s 的已發布輪）：
  1. 重建**原本那次求解**：起點 z0＝make_z(solve_in.q_pred, solve_in.s_pred)（延遲補償後），手臂模型 α＝free4、bias＝solve_in.arm_bias（當輪 α⊙d̂），
     WGMPCConfigSP 同節點參數（N 5、dt 0.05、w_s 0.001、w_a 0.05、w_s_arm 0.05、tcp link_tcp），以 solve_in.U_sol 經 rollout_sp 展開；
     對當輪目標 T_cyc 算 TCP 誤差（task_error：位置世界、姿態本體），位置分解為沿接近軸（世界 H_R·a_H，H 取當時門角）、垂直；報 k＝0（起點）、1（首步）、5（終端）。
  2. 時間對齊：預測步 k 對應 sim 時刻 snap_sim_t ＋ d_state_cycles·dt ＋ k·dt；在 room_run 取最接近該時刻的實測（base_xyth、q_arm_meas）算同一目標的誤差，
     與預測並列（不拿未來預測對當下量測）。
  3. 命令：stages.request（求解輸出 u0）vs stages.applied（執行端套用）逐軸差；stages.modified 是否為 null；u0 相對速度框（v_base_lin、v_base_ang、v_arm）的比例；
     U_sol 相鄰步差相對加速度框（a·dt）的比例。只報觀察，不宣稱機制。
**界線**：重建以記錄的求解輸入與同一核心函式；若記錄不足即標示，不另做閉迴路重算。

    python3 evaluation/door_coopen_core_g_1_solver_check.py   # 寫 runs/door_coopen_core_g_1/analysis/door_coopen_core_g_1_solver_check.json（不覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S           # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import task_error        # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_g_1_solver_check.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
SNAPS = (22.0, 23.8, 25.4)


def decompose(e, ax):
    p = -np.asarray(e[:3])                                   # task_error 位置＝目標−實際；改成 實際−目標
    along = float(p @ ax)
    return {'along_mm': along * 1e3, 'perp_mm': float(np.linalg.norm(p - along * ax)) * 1e3, 'rot_rad': float(np.linalg.norm(e[3:]))}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    cols = {c: i for i, c in enumerate(r['steps_cols'])}
    steps = [(s[cols['sim_t']], s) for s in r['steps']]
    idn = json.load(open(IDENT))
    a_H = np.asarray(json.load(open(GB))['args']['a_H'], float)
    K = WholeBodyKinematics.from_urdf_file(URDF)
    args = sv['args']
    out = {'schema': 'door_coopen_core_g_1_solver_check/1', 'node_args': {k: args.get(k) for k in ('N', 'rate', 'w_s', 'w_a', 'w_s_arm', 'tcp', 'reach_pos_m', 'reach_rot_rad')},
           'snapshots': []}
    logs = [x for x in sv['log'] if x.get('published') and (x.get('solve_in') or {}).get('U_sol') is not None]
    for ts in SNAPS:
        row = min(logs, key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - ts))
        si = row['solve_in']
        am = S.ArmSetpointModel(alpha=idn['alpha'], bias=si['arm_bias'], phys_dt=idn['phys_dt_measured_s'])
        cfg = S.WGMPCConfigSP(N=int(args['N']), dt=1.0 / float(args['rate']), tcp=args.get('tcp', 'link_tcp'), arm_model=am,
                              w_s=float(args['w_s']), w_a=float(args['w_a']), w_s_arm=float(args['w_s_arm']))
        z0 = S.make_z(si['q_pred'], si['s_pred'])
        U = np.asarray(si['U_sol'], float)
        Z = S.rollout_sp(z0, U, cfg)
        T_cyc = np.asarray(si['T_cyc'], float).reshape(4, 4)
        t_snap = float(si['snap_sim_t_exact'])
        d_state = float(row['delay_comp']['d_state_cycles'])
        ks = (0, 1, len(U))
        pred, meas = {}, {}
        for k in ks:
            tk = t_snap + d_state * cfg.dt + k * cfg.dt
            st = min(steps, key=lambda x: abs(x[0] - tk))[1]
            H = KC._H_of(st[cols['door_q_rad']])
            ax = H[:3, :3] @ a_H
            pred[k] = decompose(task_error(K, Z[k][:9], T_cyc, cfg.tcp), ax)
            qm = np.asarray(list(st[cols['base_xyth']]) + list(st[cols['q_arm_meas']]), float)
            meas[k] = dict(decompose(task_error(K, qm, T_cyc, cfg.tcp), ax), t=st[cols['sim_t']], t_target=tk)
        req = np.asarray(row['stages']['request']['body'], float)
        app = np.asarray((row['stages'].get('applied') or {}).get('v') or [np.nan] * 9, float)
        vmax = np.asarray(cfg.vmax(), float)
        amax = np.asarray(cfg.amax(), float)
        dU = np.abs(np.diff(np.vstack([np.asarray(si['u_prev'], float)[None, :], U]), axis=0))
        snap = {'snap_sim_t': t_snap, 'd_state_cycles': d_state, 'pred': pred, 'meas_aligned': meas,
                'pred_terminal_minus_start_perp_mm': pred[len(U)]['perp_mm'] - pred[0]['perp_mm'],
                'request_minus_applied_absmax': float(np.nanmax(np.abs(req - app))), 'modified': row['stages'].get('modified'),
                'u0_over_vmax_max': float(np.max(np.abs(req) / vmax)), 'dU_over_amax_dt_max': float(np.max(dU / (amax * cfg.dt))),
                'err_p_logged_mm': row['err_p'] * 1e3, 'in_tol_logged': row.get('in_tol'), 'sqp_stop': row.get('sqp_stop')}
        out['snapshots'].append(snap)
        print(f"快照 {t_snap:.2f} s：預測 垂直 起點 {pred[0]['perp_mm']:.2f} → 首步 {pred[1]['perp_mm']:.2f} → 終端 {pred[len(U)]['perp_mm']:.2f} mm"
              f"（沿軸 {pred[0]['along_mm']:.2f}→{pred[len(U)]['along_mm']:.2f}；姿態 {pred[0]['rot_rad']:.4f}→{pred[len(U)]['rot_rad']:.4f}）；"
              f"對齊實測 垂直 {meas[0]['perp_mm']:.2f}／{meas[1]['perp_mm']:.2f}／{meas[len(U)]['perp_mm']:.2f} mm；"
              f"request−applied {snap['request_minus_applied_absmax']:.2e}、modified {snap['modified']}、u0/vmax {snap['u0_over_vmax_max']:.3f}、ΔU/(a·dt) {snap['dU_over_amax_dt_max']:.3f}", flush=True)
    out['note'] = '只讀重建；預測誤差非零不等於已證明成本取捨，接近零不等於受控體模型錯；有限時域、時序與座標對齊見各欄'
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
