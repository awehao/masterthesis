#!/usr/bin/env python3
"""door_coopen_core_bench_3 末段求解紀錄核對（只讀；Codex r26：求解器本來就規劃容忍末段鉛直誤差，還是預測能修正、實際卻未實現？）。
快照：align_solver.json 已發布輪中最接近 90.0、100.0、104.9 s 者（時域終點須早於中止 105.32 s；排除中止後資料）。
每個快照：
  - 規劃（planning evidence，不是實際軌跡）：z0＝make_z(solve_in.q_pred, s_pred)（延遲補償後起點），手臂模型 α＝free4、bias＝solve_in.arm_bias（當輪 α⊙d̂），
    以記錄的 U_sol 經 wgmpc_core_sp.rollout_sp 展開；對當輪目標 T_cyc 算 TCP 位置誤差（實際−目標，世界），報鉛直分量 ez 與範數，k＝0…N。
  - 實測（配對同一物理區間）：預測步 k 對應 sim 時刻 snap＋d_state·dt＋k·dt；取 room_run 最近一步的 base_xyth＋q_arm_meas 算 TCP，對**同一 T_cyc** 的誤差。
    （門繞鉛直鉸鏈轉 ⇒ 目標鉛直座標不隨門角變，鉛直分量可跨週期比較；另報各快照 T_cyc 的 z。）
  - 起點一致性：預測 k＝0 與實測 k＝0 的鉛直差。
  - 規劃的鉛直修正量（k＝N − k＝0）與實測同期變化；偏移估計 d̂（offset_d_hat）。
界線：U_sol 是規劃序列，實際每輪重解；只回答「規劃是否已包含往 0 修正」與「同期實測是否跟上」，不判定原因。

    python3 evaluation/door_coopen_core_bench_3_solver_check.py   # 寫 runs/door_coopen_core_bench_3/analysis/door_coopen_core_bench_3_solver_check.json（不覆寫）
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
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_3')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_3_solver_check.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
SNAPS = (90.0, 100.0, 104.9)
T_ABORT = 105.31999764591455


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    steps = [s for s in r['steps'] if s[c['sim_t']] < T_ABORT]
    idn = json.load(open(IDENT))
    a = sv['args']
    K = WholeBodyKinematics.from_urdf_file(URDF)
    rows = [x for x in sv['log'] if x.get('published') and (x.get('solve_in') or {}).get('U_sol') is not None]
    out = {'schema': 'door_coopen_core_bench_3_solver_check/1', 'snapshots': []}
    for ts in SNAPS:
        row = min(rows, key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - ts))
        si = row['solve_in']
        cfg = S.WGMPCConfigSP(N=int(a['N']), dt=1.0 / float(a['rate']), tcp='link_tcp',
                              arm_model=S.ArmSetpointModel(alpha=idn['alpha'], bias=si['arm_bias'], phys_dt=idn['phys_dt_measured_s']))
        U = np.asarray(si['U_sol'], float)
        Z = S.rollout_sp(S.make_z(si['q_pred'], si['s_pred']), U, cfg)
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        tsnap, dst = float(si['snap_sim_t_exact']), float(row['delay_comp']['d_state_cycles'])
        if tsnap + (dst + len(U)) * cfg.dt >= T_ABORT:
            raise SystemExit(f'快照 {tsnap} 時域越過中止')
        pred, meas = [], []
        for k in range(len(U) + 1):
            p = K.fk(np.asarray(Z[k][:9], float), 'link_tcp')[:3, 3] - T[:3, 3]
            pred.append({'k': k, 'ez_mm': float(p[2]) * 1e3, 'e_norm_mm': float(np.linalg.norm(p)) * 1e3})
            tk = tsnap + (dst + k) * cfg.dt
            s = min(steps, key=lambda x: abs(x[c['sim_t']] - tk))
            q = np.asarray(list(s[c['base_xyth']]) + list(s[c['q_arm_meas']]), float)
            pm = K.fk(q, 'link_tcp')[:3, 3] - T[:3, 3]
            meas.append({'k': k, 't': s[c['sim_t']], 'ez_mm': float(pm[2]) * 1e3, 'e_norm_mm': float(np.linalg.norm(pm)) * 1e3})
        snap = {'snap_sim_t': tsnap, 'T_cyc_z_m': float(T[2, 3]), 'offset_d_hat': si.get('offset_d_hat'), 'pred': pred, 'meas': meas,
                'start_consistency_ez_mm': pred[0]['ez_mm'] - meas[0]['ez_mm'],
                'planned_ez_change_mm': pred[-1]['ez_mm'] - pred[0]['ez_mm'], 'measured_ez_change_mm': meas[-1]['ez_mm'] - meas[0]['ez_mm'],
                'planned_norm_change_mm': pred[-1]['e_norm_mm'] - pred[0]['e_norm_mm']}
        out['snapshots'].append(snap)
        print(f"快照 {tsnap:.2f} s（T_cyc z {T[2, 3]:.4f} m）：規劃 ez " + ' → '.join(f"{x['ez_mm']:.2f}" for x in pred)
              + " mm；實測 ez " + ' → '.join(f"{x['ez_mm']:.2f}" for x in meas)
              + f" mm；起點差 {snap['start_consistency_ez_mm']:+.2f}；規劃變化 {snap['planned_ez_change_mm']:+.2f}、實測變化 {snap['measured_ez_change_mm']:+.2f} mm；"
              f"d̂ j2 {None if si.get('offset_d_hat') is None else round(si['offset_d_hat'][1], 4)}", flush=True)
    out['note'] = 'U_sol 為規劃序列；實際逐輪重解。實測對同一 T_cyc 與延遲補償後的配對時刻。'
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
