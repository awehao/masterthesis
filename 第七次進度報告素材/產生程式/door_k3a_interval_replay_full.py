#!/usr/bin/env python3
"""有限區間等效偏移觀測：**自求解啟動、零初值**的整段開迴路回放（只讀；依 Codex 20261022：尚未驗證從啟動零初值的整段行為）。

自第一個求解快照起到最後一個快照，d̂₀＝0，逐相鄰快照區間更新（m＝兩快照之間的物理步數；> m_max ⇒ 保持）。
離線以「快照設定點＋區間內各物理步套用速度積分」代替逐步設定點歷史（room 紀錄無逐步設定點；積分對帳先前驗證 ≤ 3e-8 rad，本檔逐區間再報）。
報：區間步數分布、保持次數與原因、首次更新時刻、各相位起點的 d̂ 與實錄 d̂ 比較、樣本外區間條件預測殘差分布（依相位）。
r2（Codex 20261022_111624 口徑修正）：首次可提交更新時刻＝首個有效區間的**終點** t1（不是起點）；逐列保存 t0／t1／d_before／d_after；
相位起點比較用**當時已可取得**的估計（t1 ≤ 相位起點的最後一筆 d_after），實錄 d̂ 取 ≤ 相位起點的最後一個快照。
開迴路：手臂軌跡不隨估計改變，不得稱閉迴路行為。

    python3 evaluation/door_k3a_interval_replay_full.py <run_dir> [...]   # 寫 <run_dir>/analysis/door_k3a_interval_replay_full_r2.json（已存在則拒絕覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_rot_locate as RL                      # noqa: E402
import offset_interval as OI                          # noqa: E402


def one(run):
    out = os.path.join(run, 'analysis', 'door_k3a_interval_replay_full_r2.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    alpha, dtp = np.asarray(ident['alpha'], float), float(ident['phys_dt_measured_s'])
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    st = r['steps']
    st_t = np.array([s[I['sim_t']] for s in st])

    def step_at(ts):
        return int(np.searchsorted(st_t, ts + 1e-9, side='right') - 1)

    def phase_at(ts):
        cur = None
        for e in ev:
            if e['t'] <= ts:
                cur = e['to']
        return cur
    snaps = [x['solve_in'] for x in a['log'] if x.get('solve_in')]
    obs = OI.IntervalOffsetObserver(alpha, dtp, a['args']['offset_tau_s'], a['args']['offset_max_rad'], d0=None)
    rows, held = [], {}
    for si, sn in zip(snaps, snaps[1:]):
        k0, k1 = step_at(si['snap_sim_t_exact']), step_at(sn['snap_sim_t_exact'])
        applied = [st[kk][I['applied_arm']] for kk in range(k0 + 1, k1 + 1)]
        _, s_int = RL.predict_next(np.array(si['q_meas'])[3:9], si['s_meas'], np.zeros(6), applied, alpha, dtp)
        rec = obs.update(np.array(si['q_meas'])[3:9], si['s_meas'], applied, np.array(sn['q_meas'])[3:9])
        if rec['used'] == 'held':
            key = (rec['why'] or '').split('（')[0]
            held[key] = held.get(key, 0) + 1
        rows.append({'t0': si['snap_sim_t_exact'], 't1': sn['snap_sim_t_exact'], 'd_before': rec['d_hat_before'],
                     'd_logged_t0': si.get('offset_d_hat'),
                     'phase': phase_at(si['snap_sim_t_exact']), 'm': len(applied), 'used': rec['used'],
                     'why': rec['why'], 's_int_mismatch': float(np.abs(s_int - np.array(sn['s_meas'])).max()),
                     'pred_residual': rec.get('pred_residual'), 'd_after': rec['d_hat_after'], 'd_logged_next': sn.get('offset_d_hat')})
    m_counts = {}
    for x in rows:
        m_counts[str(x['m'])] = m_counts.get(str(x['m']), 0) + 1
    first_upd = next(({'t0': x['t0'], 't1': x['t1']} for x in rows if x['used'] == 'updated'), None)
    at_phase = {}
    for e in ev:
        avail = [x for x in rows if x['t1'] <= e['t'] + 1e-9]
        logged = [x for x in rows if x['t0'] <= e['t'] + 1e-9]
        at_phase[e['to']] = {'phase_t': e['t'],
                             'd_interval_available': (avail[-1]['d_after'] if avail else [0.0] * 6),
                             'd_interval_available_since_t1': (avail[-1]['t1'] if avail else None),
                             'd_logged': (logged[-1]['d_logged_t0'] if logged else None),
                             'd_logged_snapshot_t': (logged[-1]['t0'] if logged else None)}
    by_phase = {}
    for ph in sorted({x['phase'] for x in rows if x['phase']}):
        w = [x for x in rows if x['phase'] == ph and x['used'] == 'updated']
        by_phase[ph] = {'n_updated': len(w), 'n_held': sum(1 for x in rows if x['phase'] == ph and x['used'] == 'held'),
                        'pred_residual_rad': RL.dist([x['pred_residual'] for x in w])}
    res = {'run': os.path.basename(os.path.normpath(run)), 'open_loop_only': True, 'd0': [0.0] * 6, 'm_max': obs.m_max,
           'n_intervals': len(rows), 'm_counts': m_counts, 'n_updated': obs.n_upd, 'n_held': obs.n_held, 'held_reasons': held,
           'first_update_t': first_upd, 'first_snapshot_t': snaps[0]['snap_sim_t_exact'],
           'max_s_integration_mismatch_rad': max(x['s_int_mismatch'] for x in rows),
           'at_phase_start': at_phase, 'by_phase': by_phase,
           'note': '開迴路、樣本外區間條件預測殘差；離線以套用速度積分代替逐步設定點歷史（線上規格要求改用 /coman/arm_setpoint 逐步值）。',
           'rows': rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    return res


def main():
    for run in sys.argv[1:]:
        res = one(run)
        print(json.dumps({k: res[k] for k in ('run', 'n_intervals', 'm_counts', 'n_updated', 'n_held', 'held_reasons', 'first_snapshot_t',
                                              'first_update_t', 'max_s_integration_mismatch_rad', 'at_phase_start')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
