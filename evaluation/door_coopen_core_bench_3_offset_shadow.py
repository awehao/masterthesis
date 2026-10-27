#!/usr/bin/env python3
"""door_coopen_core_bench_3 開門段偏移估計的**因果影子重播**（只讀；Codex r27：唯一候選＝OPEN 期間恢復既有 interval 觀測器的合格區間更新）。
資料（只用當時可得）：OPEN 期間求解端已發布輪的快照時刻（solve_in.snap_sim_t_exact）→ 對應 room 物理步號；相鄰快照之間為一個區間 (k0, k1]：
  q0＝room q_arm_meas[k0]、q1＝room q_arm_meas[k1]、逐步設定點 s_j＝證據 arm_sp[j]（j＝k0…k1−1）。區間須：步號連續、皆 wholebody_applied、
  證據齊全、m ∈ [1, m_max]；否則該區間兩版本都略過（計數）。
兩個版本（同一起點：任務紀錄 offset_hold.d_frozen，即 4.72 s 凍結值；同 α、τ 0.5 s、限幅 0.05、m_max 10）：
  frozen：d̂ 全程固定＝d_frozen（重現實際行為）；shadow：既有 IntervalOffsetObserver.update_setpoints 逐區間更新（因果：預測一律用更新前 d̂）。
比較（樣本外）：每區間終點殘差 res＝q1 − q_pred(d̂_before)；逐軸 RMS；以 k0 構型的 TCP 位置雅可比把手臂殘差投影到世界鉛直（mm）。
  依門角 5° 分箱；另報 shadow 的 d̂ 軌跡（各軸範圍、是否觸及限幅、相鄰區間最大變化）與末段（門角 ≥ 40°）比較。
界線：直接驅動觀測器核心（非完整 IntervalFeed 的整合核對）；開迴路重播，**不代表閉迴路**（不能看到改用新估計後的命令與接觸力變化，即 Codex 擔心的
  「把接觸負載學成偏差、加重夾持內力」無法由本重播排除）。

    python3 evaluation/door_coopen_core_bench_3_offset_shadow.py   # 寫 runs/door_coopen_core_bench_3/analysis/door_coopen_core_bench_3_offset_shadow.json（不覆寫）
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
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import offset_interval as OINT                               # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_3')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_3_offset_shadow.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    by = {s[c['step']]: s for s in r['steps']}
    t2k = {round(s[c['sim_t']], 4): s[c['step']] for s in r['steps']}
    ev = t['policy_events']
    t_abort = next(e['t'] for e in ev if e.get('abort'))
    hold = sv['stats']['offset_hold']
    d0 = np.asarray(hold['d_frozen'], float)
    idn = json.load(open(IDENT))
    alpha = np.asarray(idn['alpha'], float) * np.ones(6)
    a = sv['args']
    tau, dmax = float(a.get('offset_tau_s', 0.5)), float(a.get('offset_max_rad', 0.05))
    sp = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('arm_sp'), list):
            sp[e['step']] = np.asarray(e['arm_sp'], float)
    snaps = sorted({round(x['solve_in']['snap_sim_t_exact'], 4) for x in sv['log'] if x.get('published') and x.get('solve_in')
                    and float(hold['sim_t']) <= x['solve_in']['snap_sim_t_exact'] < t_abort})
    ks = [t2k[x] for x in snaps if x in t2k]
    shadow = OINT.IntervalOffsetObserver(alpha, float(idn['phys_dt_measured_s']), tau, dmax, d0=d0)
    K = WholeBodyKinematics.from_urdf_file(URDF)
    rows, skip, dtraj = [], {}, []
    for k0, k1 in zip(ks, ks[1:]):
        m = k1 - k0
        steps = [by.get(j) for j in range(k0, k1 + 1)]
        why = None
        if m < 1 or m > shadow.m_max:
            why = 'interval_len'
        elif any(s is None for s in steps):
            why = 'step_gap'
        elif any(s[c['kind']] != 'wholebody_applied' for s in steps):
            why = 'not_wholebody'
        elif any(j not in sp for j in range(k0, k1)):
            why = 'evidence_missing'
        if why:
            skip[why] = skip.get(why, 0) + 1
            continue
        q0 = np.asarray(by[k0][c['q_arm_meas']], float)
        q1 = np.asarray(by[k1][c['q_arm_meas']], float)
        s_seq = [sp[j] for j in range(k0, k1)]
        qf = q0.copy()
        for sj in s_seq:
            qf = qf + alpha * (sj - qf) + alpha * d0
        res_f = q1 - qf
        rec = shadow.update_setpoints(q0, s_seq, q1)
        if rec['used'] != 'updated':
            skip['shadow_held:' + str(rec['why'])] = skip.get('shadow_held:' + str(rec['why']), 0) + 1
            continue
        res_s = np.asarray(rec['pred_residual'], float)
        qfull = np.asarray(list(by[k0][c['base_xyth']]) + list(q0), float)
        J = np.asarray(K.jacobian(qfull, 'link_tcp'), float)[:3, 3:9]
        rows.append({'t': by[k0][c['sim_t']], 'q_deg': math.degrees(by[k0][c['door_q_rad']]), 'm': m,
                     'res_f': res_f.tolist(), 'res_s': res_s.tolist(), 'vz_f_mm': float((J @ res_f)[2]) * 1e3, 'vz_s_mm': float((J @ res_s)[2]) * 1e3,
                     'd_after': rec['d_hat_after']})
        dtraj.append(rec['d_hat_after'])

    def rms(v):
        v = np.asarray(v, float)
        return np.sqrt((v ** 2).mean(axis=0)).tolist() if len(v) else None
    bins = []
    for lo in range(0, 45, 5):
        w = [x for x in rows if lo <= x['q_deg'] < lo + 5]
        if w:
            bins.append({'q_bin_deg': [lo, lo + 5], 'n': len(w), 'rms_res_frozen': rms([x['res_f'] for x in w]), 'rms_res_shadow': rms([x['res_s'] for x in w]),
                         'mean_vz_frozen_mm': float(np.mean([x['vz_f_mm'] for x in w])), 'mean_vz_shadow_mm': float(np.mean([x['vz_s_mm'] for x in w])),
                         'rms_vz_frozen_mm': float(np.sqrt(np.mean([x['vz_f_mm'] ** 2 for x in w]))), 'rms_vz_shadow_mm': float(np.sqrt(np.mean([x['vz_s_mm'] ** 2 for x in w])))})
    D = np.asarray(dtraj, float)
    late = [x for x in rows if x['q_deg'] >= 40.0]
    out = {'schema': 'door_coopen_core_bench_3_offset_shadow/1', 't_range': [float(hold['sim_t']), t_abort], 'd0_frozen': d0.tolist(),
           'n_intervals': len(rows), 'skipped': skip, 'bins': bins,
           'overall': {'rms_res_frozen': rms([x['res_f'] for x in rows]), 'rms_res_shadow': rms([x['res_s'] for x in rows]),
                       'rms_vz_frozen_mm': float(np.sqrt(np.mean([x['vz_f_mm'] ** 2 for x in rows]))),
                       'rms_vz_shadow_mm': float(np.sqrt(np.mean([x['vz_s_mm'] ** 2 for x in rows])))},
           'late_ge_40deg': {'n': len(late), 'mean_vz_frozen_mm': float(np.mean([x['vz_f_mm'] for x in late])) if late else None,
                             'mean_vz_shadow_mm': float(np.mean([x['vz_s_mm'] for x in late])) if late else None},
           'shadow_d_traj': {'min': D.min(axis=0).tolist(), 'max': D.max(axis=0).tolist(), 'end': D[-1].tolist(),
                             'hit_clip': bool(np.any(np.abs(D) >= dmax - 1e-12)),
                             'max_step_change': np.abs(np.diff(D, axis=0)).max(axis=0).tolist() if len(D) > 1 else None},
           'note': '開迴路因果重播；樣本外殘差（更新前 d̂）；不代表閉迴路，無法看見改用新估計後的接觸力與夾持內力變化'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('區間', len(rows), '略過', skip)
    print('整體 鉛直殘差 RMS（每區間 TCP 投影）凍結 %.4f mm → 影子 %.4f mm' % (out['overall']['rms_vz_frozen_mm'], out['overall']['rms_vz_shadow_mm']))
    for b in bins:
        print('  門角', b['q_bin_deg'], 'n', b['n'], '鉛直均值 凍結 %+.4f → 影子 %+.4f mm；RMS %.4f → %.4f' % (b['mean_vz_frozen_mm'], b['mean_vz_shadow_mm'], b['rms_vz_frozen_mm'], b['rms_vz_shadow_mm']))
    print('影子 d̂ 範圍 j2', round(D[:, 1].min(), 4), round(D[:, 1].max(), 4), '末', [round(v, 4) for v in D[-1]], '觸限', out['shadow_d_traj']['hit_clip'],
          '最大單區間變化', [round(v, 5) for v in out['shadow_d_traj']['max_step_change']])


if __name__ == '__main__':
    main()
