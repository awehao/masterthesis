#!/usr/bin/env python3
"""door_coopen_core_g_1：第二次固定目標等待的一步殘差＋時域敏感度（只讀；Codex 20261027_005850）。

更正（相對 door_coopen_core_g_1_solver_check.py；該檔與輸出保留）：
  - 其 25.40 s 快照的終端實測取自 25.73 s（任務 25.64 s 已中止）⇒「實現約 5%」「越晚差距越大」撤回；
  - stages 為 unpaired_latest_values_only ⇒ request−applied 相減與 modified＝null **不能**判定同一筆命令是否被修改或限幅：本檔標「未能驗證」；
  - U_sol 是當輪規劃序列，實際逐輪重解 ⇒ 整段預測 vs 實測不是輸入對齊，50%／30% 不作模型實現率。

(ii) 一步殘差（第二次暫停、**中止前**：物理步 k 與 k＋1 皆在 [t_pause2, t_abort) 內）：
  手臂：r_arm＝x_{k+1} − [x_k ＋ α⊙(s_k − x_k) ＋ b_k]，s_k＝取證 arm_sp（同步）、x＝room q_arm_meas、b_k＝**k 時刻之前最近一輪求解記錄的 arm_bias**（α⊙d̂，非最終值）；
  底盤：Δ_base＝R(yaw_k)·(meas_vb_body − base_cmd_body)[:2]·Δt、Δθ＝(…)[2]·Δt（room 同一步欄位；base_cmd_body 為執行端該步套用的本體命令）；
  TCP 投影：J＝K.jacobian(q_k, link_tcp) 位置列；dp_arm＝J[:, arm]·r_arm、dp_base＝J[:, base]·(Δx, Δy, Δθ)；
  以「實測 TCP 對目標的垂直誤差方向」單位向量 n_⊥ 投影（正值＝使垂直誤差增大），每 0.25 s 窗累加，並列同窗實測垂直誤差變化。
(i) 時域敏感度（只取 23.8 s 同一快照）：以記錄的 z0（q_pred、s_pred）、u_prev、T_cyc、U_warm、arm_bias、park_hold.u_hold（cfg.base_hold）
  呼叫同一核心 solve_sp，N＝5 與 N＝10（U_warm 末列重複補長），其餘設定一致；比較共同前 0.25 s（k＝0..5）的預測垂直誤差與首筆命令 u0；
  另核 N＝5 重解與記錄 U_sol 的一致性。N＝10 改善與否都只是敏感度證據，不單獨證明主因。

    python3 evaluation/door_coopen_core_g_1_resid_check.py   # 寫 runs/door_coopen_core_g_1/analysis/door_coopen_core_g_1_resid_check.json（不覆寫）
"""
import bisect
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
from ammr_wholebody_mpc.wgmpc_core import task_error        # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_g_1_resid_check.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
TCP = 'link_tcp'


def perp_vec(K, qf, T, ax):
    p = K.fk(np.asarray(qf, float), TCP)[:3, 3] - T[:3, 3]
    along = float(p @ ax)
    v = p - along * ax
    return v, float(np.linalg.norm(v))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    K = WholeBodyKinematics.from_urdf_file(URDF)
    idn = json.load(open(IDENT))
    alpha = np.asarray(idn['alpha'], float)
    a_H = np.asarray(json.load(open(GB))['args']['a_H'], float)
    t = json.load(open(os.path.join(RUN, 'task.json')))
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    cols = {c: i for i, c in enumerate(r['steps_cols'])}
    pz = [e for e in t['policy_events'] if e.get('approach_gate_pause')]
    t0 = pz[1]['t']
    t_abort = next(e['t'] for e in t['policy_events'] if 'abort' in e)
    sp = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('arm_sp'), list):
            sp[e['step']] = np.asarray(e['arm_sp'], float)
    logs = sorted([x for x in sv['log'] if (x.get('solve_in') or {}).get('arm_bias') is not None],
                  key=lambda x: x['solve_in']['snap_sim_t_exact'])
    lt = [x['solve_in']['snap_sim_t_exact'] for x in logs]
    targ = [(x['solve_in']['snap_sim_t_exact'], np.asarray(x['solve_in']['T_cyc'], float).reshape(4, 4)) for x in logs]
    steps = sorted(r['steps'], key=lambda s: s[cols['step']])
    rows = []
    for a_, b_ in zip(steps, steps[1:]):
        ta, tb = a_[cols['sim_t']], b_[cols['sim_t']]
        if not (t0 <= ta and tb < t_abort) or b_[cols['step']] != a_[cols['step']] + 1 or a_[cols['step']] not in sp:
            continue
        i = bisect.bisect_right(lt, ta) - 1
        if i < 0:
            continue
        bias = np.asarray(logs[i]['solve_in']['arm_bias'], float)
        T = targ[i][1]
        xa, xb = np.asarray(a_[cols['q_arm_meas']], float), np.asarray(b_[cols['q_arm_meas']], float)
        r_arm = xb - (xa + alpha * (sp[a_[cols['step']]] - xa) + bias)
        ba = np.asarray(a_[cols['base_xyth']], float)
        dt = tb - ta
        vres = np.asarray(a_[cols['meas_vb_body']], float) - np.asarray(a_[cols['base_cmd_body']], float)
        c, s_ = math.cos(ba[2]), math.sin(ba[2])
        dbase = np.array([(c * vres[0] - s_ * vres[1]) * dt, (s_ * vres[0] + c * vres[1]) * dt, vres[2] * dt])
        qf = np.concatenate([ba, xa])
        J = np.asarray(K.jacobian(qf, TCP), float)[:3]
        H = KC._H_of(a_[cols['door_q_rad']])
        ax = H[:3, :3] @ a_H
        v, perp = perp_vec(K, qf, T, ax)
        n = v / perp if perp > 1e-9 else np.zeros(3)
        rows.append({'t': ta, 'perp_mm': perp * 1e3, 'res_arm_perp_um': float(n @ (J[:, 3:9] @ r_arm)) * 1e6,
                     'res_base_perp_um': float(n @ (J[:, 0:3] @ dbase)) * 1e6, 'r_arm_rad': r_arm.tolist(), 'vres_body': vres.tolist()})
    wins = []
    w0 = rows[0]['t'] if rows else None
    while rows and w0 < rows[-1]['t']:
        w = [x for x in rows if w0 <= x['t'] < w0 + 0.25]
        if len(w) >= 2:
            wins.append({'t0': w0, 'n': len(w), 'perp_change_meas_mm': w[-1]['perp_mm'] - w[0]['perp_mm'],
                         'res_arm_perp_sum_mm': sum(x['res_arm_perp_um'] for x in w) / 1e3, 'res_base_perp_sum_mm': sum(x['res_base_perp_um'] for x in w) / 1e3})
        w0 += 0.25
    out = {'schema': 'door_coopen_core_g_1_resid_check/1', 'interval_s': [t0, t_abort], 'n_steps': len(rows),
           'total': {'perp_change_meas_mm': (rows[-1]['perp_mm'] - rows[0]['perp_mm']) if rows else None,
                     'res_arm_perp_sum_mm': sum(x['res_arm_perp_um'] for x in rows) / 1e3,
                     'res_base_perp_sum_mm': sum(x['res_base_perp_um'] for x in rows) / 1e3,
                     'r_arm_mean_rad': np.mean([x['r_arm_rad'] for x in rows], axis=0).tolist() if rows else None,
                     'vres_body_mean': np.mean([x['vres_body'] for x in rows], axis=0).tolist() if rows else None},
           'windows_0p25s': wins, 'command_pairing': '未能驗證（stages 為 unpaired_latest_values_only；本檔不以 request−applied 判定修改／限幅）'}
    print(f"(ii) {len(rows)} 步 [{t0:.2f}, {t_abort:.2f})：實測垂直變化 {out['total']['perp_change_meas_mm']:.3f} mm；"
          f"手臂一步殘差投影累計 {out['total']['res_arm_perp_sum_mm']:+.3f} mm、底盤殘差投影累計 {out['total']['res_base_perp_sum_mm']:+.3f} mm；"
          f"底盤本體速度殘差平均 {np.round(out['total']['vres_body_mean'], 5).tolist()}", flush=True)
    for w in wins:
        print(f"   窗 {w['t0']:.2f}：實測 {w['perp_change_meas_mm']:+.3f}、手臂殘差 {w['res_arm_perp_sum_mm']:+.3f}、底盤殘差 {w['res_base_perp_sum_mm']:+.3f} mm", flush=True)
    # (i)
    args = sv['args']
    row = min(logs, key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - 23.8))
    si = row['solve_in']
    T = np.asarray(si['T_cyc'], float).reshape(4, 4)
    st = min(steps, key=lambda s: abs(s[cols['sim_t']] - si['snap_sim_t_exact']))
    ax = KC._H_of(st[cols['door_q_rad']])[:3, :3] @ a_H
    res_i = {'snap_sim_t': si['snap_sim_t_exact']}
    for N in (5, 10):
        am = S.ArmSetpointModel(alpha=idn['alpha'], bias=si['arm_bias'], phys_dt=idn['phys_dt_measured_s'])
        cfg = S.WGMPCConfigSP(N=N, dt=1.0 / float(args['rate']), tcp=TCP, arm_model=am, w_s=float(args['w_s']), w_a=float(args['w_a']),
                              w_s_arm=float(args['w_s_arm']))
        cfg.base_hold = tuple(float(x) for x in row['park_hold']['u_hold'])
        Uw = np.asarray(si['U_warm'], float)
        if N > len(Uw):
            Uw = np.vstack([Uw, np.repeat(Uw[-1:], N - len(Uw), axis=0)])
        z0 = S.make_z(si['q_pred'], si['s_pred'])
        rr = S.solve_sp(K, z0, np.asarray(si['u_prev'], float), T, cfg, U_warm=Uw[:N])
        if not rr.ok:
            res_i[f'N{N}'] = {'ok': False}
            continue
        Z = S.rollout_sp(z0, rr.U, cfg)
        perps = [perp_vec(K, Z[k][:9], T, ax)[1] * 1e3 for k in range(6)]
        e0 = task_error(K, Z[5][:9], T, TCP)
        res_i[f'N{N}'] = {'ok': True, 'perp_k0_5_mm': perps, 'u0': np.asarray(rr.U[0]).tolist(), 'rot_k5_rad': float(np.linalg.norm(e0[3:]))}
        if N == 5:
            res_i['N5_vs_logged_U_sol_absmax'] = float(np.max(np.abs(np.asarray(rr.U) - np.asarray(si['U_sol'], float))))
    if res_i.get('N5', {}).get('ok') and res_i.get('N10', {}).get('ok'):
        res_i['u0_diff_absmax'] = float(np.max(np.abs(np.asarray(res_i['N10']['u0']) - np.asarray(res_i['N5']['u0']))))
        print(f"(i) {res_i['snap_sim_t']:.2f} s：N5 重解 vs 記錄 U_sol 差 {res_i['N5_vs_logged_U_sol_absmax']:.2e}；"
              f"前 0.25 s 垂直 N5 {np.round(res_i['N5']['perp_k0_5_mm'], 3).tolist()}、N10 {np.round(res_i['N10']['perp_k0_5_mm'], 3).tolist()} mm；首筆命令差 {res_i['u0_diff_absmax']:.2e}", flush=True)
    else:
        print('(i) 求解失敗：', {k: v for k, v in res_i.items() if isinstance(v, dict)}, flush=True)
    out['horizon_sensitivity'] = res_i
    out['note'] = ('一步殘差用同步配對的實測設定點與當輪 α⊙d̂；N＝10 改善與否只是敏感度證據；殘差與 d̂ 同步變化不據以歸因時間常數；'
                   '底盤殘差欄位語意：base_cmd_body＝執行端該步套用之本體命令、meas_vb_body＝實測本體速度（room_run 定義）')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
