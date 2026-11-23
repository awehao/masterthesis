#!/usr/bin/env python3
"""有限區間等效偏移觀測的**開迴路**比較（只讀；不代表閉迴路行為）。

窗口：HOLD 起 → 物理首次單指 < 0.5 N。以實錄求解快照（q_meas、s_meas）與兩快照間實際套用速度為輸入：
  logged：用實錄當輪 d̂ 預測下一快照（與 door_k3a_rot_locate r4 的相鄰殘差同式）；
  interval：IntervalOffsetObserver，初值＝HOLD 起點的實錄 d̂，逐區間以**更新前**估計預測（樣本外），區間結束再更新。
手臂軌跡不隨估計改變（開迴路），只比較同一輸入下的樣本外預測殘差，不得稱閉迴路改善或負載已辨識。

    python3 evaluation/door_k3a_interval_replay.py <run_dir> [...]   # 寫 <run_dir>/analysis/door_k3a_interval_replay.json（已存在則拒絕覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3a_rot_locate as RL                      # noqa: E402
import offset_interval as OI                          # noqa: E402

FINGERS = KC.FINGERS


def one(run):
    out = os.path.join(run, 'analysis', 'door_k3a_interval_replay.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    alpha, dtp = np.asarray(ident['alpha'], float), float(ident['phys_dt_measured_s'])
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    t_hold = next(e['t'] for e in ev if e['to'] == 'HOLD')
    t_open = next(e['t'] for e in ev if e['to'] == 'OPEN')
    G = np.array(t['G_T_H'])
    st = r['steps']
    st_t = np.array([s[I['sim_t']] for s in st])
    t_low = next(s[I['sim_t']] for s in st if s[I['sim_t']] >= t_open
                 and min((s[I['leaf_contacts']] or {}).get(f, {}).get('bar', 0.0) for f in FINGERS) < 0.5)

    def step_at(ts):
        return int(np.searchsorted(st_t, ts + 1e-9, side='right') - 1)
    snaps = [x['solve_in'] for x in a['log'] if x.get('solve_in') and t_hold - 1e-9 <= x['solve_in']['snap_sim_t_exact'] <= t_low + 1e-9]
    obs = OI.IntervalOffsetObserver(alpha, dtp, a['args']['offset_tau_s'], a['args']['offset_max_rad'], d0=snaps[0].get('offset_d_hat'))
    rows = []
    for si, sn in zip(snaps, snaps[1:]):
        k0, k1 = step_at(si['snap_sim_t_exact']), step_at(sn['snap_sim_t_exact'])
        applied = [st[kk][I['applied_arm']] for kk in range(k0 + 1, k1 + 1)]
        q0, q1 = np.array(si['q_meas']), np.array(sn['q_meas'])
        qp_log, _ = RL.predict_next(q0[3:9], si['s_meas'], si.get('offset_d_hat') or np.zeros(6), applied, alpha, dtp)
        rec = obs.update(q0[3:9], si['s_meas'], applied, q1[3:9])
        qp_int = q1[3:9] - np.array(rec['pred_residual']) if rec['used'] == 'updated' else None
        H1 = KC._H_of(st[k1][I['door_q_rad']])
        rm, _ = RL.grasp_err(H1, RL.fk(q1[:3], q1[3:9]), G)
        rl, _ = RL.grasp_err(H1, RL.fk(q1[:3], qp_log), G)
        row = {'t0': si['snap_sim_t_exact'], 'phase': 'HOLD' if si['snap_sim_t_exact'] < t_open else 'OPEN', 'm': len(applied),
               'res_logged_rad': (q1[3:9] - qp_log).tolist(), 'rot_logged_deg': (rm - rl).tolist(),
               'used': rec['used'], 'd_interval_before': rec['d_hat_before'], 'd_logged': si.get('offset_d_hat')}
        if qp_int is not None:
            ri, _ = RL.grasp_err(H1, RL.fk(q1[:3], qp_int), G)
            row['res_interval_rad'] = rec['pred_residual']
            row['rot_interval_deg'] = (rm - ri).tolist()
            row['d_obs'] = rec['d_obs']
        rows.append(row)
    summ = {}
    for ph in ('HOLD', 'OPEN'):
        w = [x for x in rows if x['phase'] == ph and 'res_interval_rad' in x]
        summ[ph] = {'n': len(w), 'm_counts': {str(m): sum(1 for x in w if x['m'] == m) for m in sorted({x['m'] for x in w})},
                    'logged': {'res_rad': RL.dist([x['res_logged_rad'] for x in w]), 'rot_deg': RL.dist([x['rot_logged_deg'] for x in w])},
                    'interval': {'res_rad': RL.dist([x['res_interval_rad'] for x in w]), 'rot_deg': RL.dist([x['rot_interval_deg'] for x in w])},
                    'mean_abs_res_rad': {'logged': np.mean(np.abs([x['res_logged_rad'] for x in w]), axis=0).tolist() if w else None,
                                         'interval': np.mean(np.abs([x['res_interval_rad'] for x in w]), axis=0).tolist() if w else None},
                    'd_obs': RL.dist([x['d_obs'] for x in w])}
    res = {'run': os.path.basename(os.path.normpath(run)), 'open_loop_only': True, 't_hold': t_hold, 't_open': t_open, 't_first_low_phys': t_low,
           'n_intervals': len(rows), 'n_held': obs.n_held, 'by_phase': summ, 'd_end': {'interval': obs.d.tolist(), 'logged': snaps[-1].get('offset_d_hat')},
           'note': '開迴路、樣本外（預測用更新前估計）。只比較同一輸入下的預測殘差；d_obs 是區間內近似恆定的等效偏移，不是真實負載；跨相位區間依起點歸類。',
           'rows': rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    return res


def main():
    for run in sys.argv[1:]:
        res = one(run)
        print(json.dumps({k: res[k] for k in ('run', 'n_intervals', 'n_held', 'by_phase', 'd_end')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
