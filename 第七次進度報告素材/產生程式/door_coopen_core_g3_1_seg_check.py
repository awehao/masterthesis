#!/usr/bin/env python3
"""door_coopen_core_g3_1 續進段垂直增長的紀錄核對（只讀；Codex r14 §3：只兩段 20.14–21.06、24.62–25.28 s，沿用一步殘差方法）。
逐物理步 k→k＋1（room_run 同一序列；位姿 k→k＋1 配命令 k）：
  e＝p_tcp − p_target（世界），垂直分量 e⊥＝P·e，P＝I − ax·axᵀ，ax＝H_R(門角_k)·a_H；n＝e⊥/|e⊥|（k 時）。對 |e⊥| 的一步貢獻（n·各分量，正值＝增大）：
  - target：−n·P·(p_T,k+1 − p_T,k)；目標取任務 trace 中 ≤ 該步時刻的最新一列 target（實際發布值）。
  - arm_model：n·J_arm·(α⊙(sp_k − x_k) ＋ b_k)；sp_k 取證據 arm_sp（同步號），b_k 取 k 之前最近一輪求解記錄的 arm_bias。
  - arm_resid：n·J_arm·(x_{k+1} − x_k − α⊙(sp_k − x_k) − b_k)。
  - base_cmd：n·J_base·(R(θ_k)·cmd_k·Δt)；cmd_k＝base_cmd_body（執行端該步套用）。
  - base_resid：n·J_base·(Δbase_k − R(θ_k)·cmd_k·Δt)（Δθ 已纏繞）。
  - lin_rem：n·(Δp_tcp − J·Δq)（線性化餘項）。
  合計與實測 Δ|e⊥| 對照（核對分解閉合）。另報同段沿軸誤差與目標沿軸推進量。
界線：只說明各分量在該段的帳目；不判定唯一機制、不提控制修正。

    python3 evaluation/door_coopen_core_g3_1_seg_check.py   # 寫 runs/door_coopen_core_g3_1/analysis/door_coopen_core_g3_1_seg_check.json（不覆寫）
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
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g3_1')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_g3_1_seg_check.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
TCP = 'link_tcp'
SEGS = ((20.14, 21.06), (24.62, 25.28))
KEYS = ('target', 'arm_model', 'arm_resid', 'base_cmd', 'base_resid', 'lin_rem')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    K = WholeBodyKinematics.from_urdf_file(URDF)
    alpha = np.asarray(json.load(open(IDENT))['alpha'], float)
    a_H = np.asarray(json.load(open(GB))['args']['a_H'], float)
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    steps = sorted(r['steps'], key=lambda s: s[c['step']])
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    logs = sorted([x for x in sv['log'] if (x.get('solve_in') or {}).get('arm_bias') is not None], key=lambda x: x['solve_in']['snap_sim_t_exact'])
    lt = [x['solve_in']['snap_sim_t_exact'] for x in logs]
    tr = sorted([json.loads(ln)['row'] for ln in open(os.path.join(RUN, 'task_trace.jsonl'))], key=lambda x: x['t'])
    trg = [(x['t'], np.asarray(x['target'], float).reshape(4, 4)) for x in tr if x.get('target') is not None]
    tt = [a for a, _ in trg]
    sp = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('arm_sp'), list):
            sp[e['step']] = np.asarray(e['arm_sp'], float)
    out = {'schema': 'door_coopen_core_g3_1_seg_check/1', 'segments': []}
    for lo, hi in SEGS:
        acc = {k: 0.0 for k in KEYS}
        n_used, n_skip, meas0, meas1, al0, al1, tgt_along = 0, {}, None, None, None, None, 0.0
        for a_, b_ in zip(steps, steps[1:]):
            ta = a_[c['sim_t']]
            if not (lo <= ta < hi):
                continue
            k = a_[c['step']]
            why = None
            if b_[c['step']] != k + 1:
                why = 'step_gap'
            elif k not in sp:
                why = 'no_arm_sp'
            i = bisect.bisect_right(lt, ta) - 1
            j0, j1 = bisect.bisect_right(tt, ta) - 1, bisect.bisect_right(tt, b_[c['sim_t']]) - 1
            if why is None and (i < 0 or j0 < 0):
                why = 'no_solver_or_target'
            if why:
                n_skip[why] = n_skip.get(why, 0) + 1
                continue
            qa = np.concatenate([np.asarray(a_[c['base_xyth']], float), np.asarray(a_[c['q_arm_meas']], float)])
            qb = np.concatenate([np.asarray(b_[c['base_xyth']], float), np.asarray(b_[c['q_arm_meas']], float)])
            ax = KC._H_of(a_[c['door_q_rad']])[:3, :3] @ a_H
            P = np.eye(3) - np.outer(ax, ax)
            TA, TB = trg[j0][1][:3, 3], trg[j1][1][:3, 3]
            pa, pb = K.fk(qa, TCP)[:3, 3], K.fk(qb, TCP)[:3, 3]
            ea, eb = P @ (pa - TA), P @ (pb - TB)
            na = float(np.linalg.norm(ea))
            n = ea / na if na > 1e-9 else np.zeros(3)
            if meas0 is None:
                meas0, al0 = na, float((pa - TA) @ ax)
            meas1, al1 = float(np.linalg.norm(eb)), float((pb - TB) @ ax)
            tgt_along += float((TB - TA) @ ax)
            J = np.asarray(K.jacobian(qa, TCP), float)[:3]
            bias = np.asarray(logs[i]['solve_in']['arm_bias'], float)
            xa, xb = qa[3:], qb[3:]
            dx_model = alpha * (sp[k] - xa) + bias
            dt = b_[c['sim_t']] - ta
            th = qa[2]
            R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
            dcmd = R @ np.asarray(a_[c['base_cmd_body']], float) * dt
            dbase = qb[:3] - qa[:3]
            dbase[2] = (dbase[2] + math.pi) % (2 * math.pi) - math.pi
            dq = np.concatenate([dbase, xb - xa])
            acc['target'] += float(-n @ (P @ (TB - TA)))
            acc['arm_model'] += float(n @ (J[:, 3:9] @ dx_model))
            acc['arm_resid'] += float(n @ (J[:, 3:9] @ (xb - xa - dx_model)))
            acc['base_cmd'] += float(n @ (J[:, 0:3] @ dcmd))
            acc['base_resid'] += float(n @ (J[:, 0:3] @ (dbase - dcmd)))
            acc['lin_rem'] += float(n @ ((pb - pa) - J @ dq))
            n_used += 1
        seg = {'interval_s': [lo, hi], 'n_steps': n_used, 'skipped': n_skip, 'perp_start_mm': None if meas0 is None else meas0 * 1e3,
               'perp_end_mm': None if meas1 is None else meas1 * 1e3, 'perp_change_mm': None if meas0 is None else (meas1 - meas0) * 1e3,
               'contrib_mm': {k: v * 1e3 for k, v in acc.items()}, 'contrib_sum_mm': sum(acc.values()) * 1e3,
               'along_start_mm': None if al0 is None else al0 * 1e3, 'along_end_mm': None if al1 is None else al1 * 1e3, 'target_along_advance_mm': tgt_along * 1e3}
        out['segments'].append(seg)
        print(f"段 {lo}–{hi} s（{n_used} 步，略 {n_skip}）：實測垂直 {seg['perp_start_mm']:.2f} → {seg['perp_end_mm']:.2f}（{seg['perp_change_mm']:+.3f} mm）；"
              f"分量合計 {seg['contrib_sum_mm']:+.3f}：" + '、'.join(f'{k} {v:+.3f}' for k, v in seg['contrib_mm'].items())
              + f"；沿軸 {seg['along_start_mm']:.2f} → {seg['along_end_mm']:.2f} mm、目標沿軸推進 {seg['target_along_advance_mm']:+.2f} mm", flush=True)
    out['note'] = ('n 取 k 時垂直方向；Δ|e⊥| ≈ Σ n·Δe⊥（一階）；目標以任務 trace 最新一列近似逐物理步發布值（trace 約 15 Hz，'
                   '目標分量集中在 trace 更新步）；只說明帳目，不判定唯一機制')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
