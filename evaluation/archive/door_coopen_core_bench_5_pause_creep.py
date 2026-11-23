#!/usr/bin/env python3
"""door_coopen_core_bench_5 暫停後手臂緩移的有界對時（只讀；Codex r30 §3：只查暫停切換至中止這一段，不新增控制修正）。
窗口：執行端暫停切換步（room finger_effort.pause.t，103.02 s）至中止事件（task policy_events abort.t，108.04 s）。
逐物理步對時（同一步號）：
  1. 設定點：evidence arm_sp 六軸是否真正固定（相對切換步的最大偏差；首次變動時刻）；
  2. 實測：room q_arm_meas（相對切換步與相對 STOPPED 時刻的變化）、applied_arm（零速率命令）、applied_src（套用來源序號／鏈模式）；
  3. 手指：room q_finger（實測指位）、finger_contact_n（兩指—橫桿力）、驅動上限切換與定期讀回（room finger_effort）；
  4. 力矩：evidence dof_measured_efforts（joint1–6、兩指）、j6_reaction_wrench；room wrist_force_n；
  5. 估計：求解端 offset_hold（凍結值、時刻）。
  各量以 0.1 s 分箱報平均；另報 STOPPED（107.19 s）前後 j2 角度、j2 力矩、j2 設定點偏差（q − sp）與兩指合力的線性趨勢（斜率）。
  監督判定窗口（0.5 s 手臂、0.1 s 底盤／門）重算，與求解端 FAIL 時刻對照。
界線：只描述時序；不判定原因（零速率命令不等於沒有位置伺服力矩；不把緩移直接歸因於接觸力）。

    python3 evaluation/door_coopen_core_bench_5_pause_creep.py   # 寫 runs/door_coopen_core_bench_5/analysis/door_coopen_core_bench_5_pause_creep.json（不覆寫）
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_5')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_5_pause_creep.json')
J = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')


def slope(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return None if len(t) < 3 else float(np.polyfit(t - t[0], y, 1)[0])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    fe = r['finger_effort']['pause']
    t0 = float(fe['t'])
    ta = float(next(e['t'] for e in t['policy_events'] if e.get('abort')))
    sup = sv['stats']['k3m_pause']['supervisor_outcome']
    t_stop = float(sup['transitions']['STOPPED'])
    ev = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and t0 - 1e-6 <= e.get('t', -1) <= ta + 1e-6:
            ev[e['step']] = e
    S = [s for s in r['steps'] if t0 - 1e-6 <= s[c['sim_t']] <= ta + 1e-6 and s[c['step']] in ev]
    T = np.array([s[c['sim_t']] for s in S])
    Q = np.array([s[c['q_arm_meas']] for s in S], float)
    U = np.array([s[c['applied_arm']] for s in S], float)
    SP = np.array([ev[s[c['step']]]['arm_sp'] for s in S], float)
    F = np.array([s[c['finger_contact_n']] for s in S], float)
    QF = np.array([s[c['q_finger']] for s in S], float)
    TAU = np.array([[ev[s[c['step']]]['dof_measured_efforts'].get(j, np.nan) for j in J + ('finger_joint1', 'finger_joint2')] for s in S], float)
    W = np.array([ev[s[c['step']]]['j6_reaction_wrench'] for s in S], float)
    SRC = [s[c['applied_src']] for s in S]
    dsp = np.abs(SP - SP[0]).max(axis=1)
    sp_change = next((float(T[i]) for i in range(len(T)) if dsp[i] > 1e-12), None)
    i_stop = int(np.searchsorted(T, t_stop))
    bins = []
    for a in np.arange(t0, ta, 0.1):
        w = (T >= a) & (T < a + 0.1)
        if w.any():
            bins.append({'t': round(float(a), 2), 'q_j2_mrad_rel': float((Q[w, 1].mean() - Q[0, 1]) * 1e3), 'q_minus_sp_j2_mrad': float((Q[w, 1] - SP[w, 1]).mean() * 1e3),
                         'tau_j2': float(np.nanmean(TAU[w, 1])), 'tau_j3': float(np.nanmean(TAU[w, 2])), 'f_sum': float(F[w].sum(axis=1).mean()),
                         'f': F[w].mean(axis=0).tolist(), 'q_finger': QF[w].mean(axis=0).tolist(), 'tau_finger': np.nanmean(TAU[w, 6:8], axis=0).tolist(),
                         'wrench_fz': float(W[w, 2].mean()), 'u_arm_absmax': float(np.abs(U[w]).max())})
    pre, post = (T >= t_stop - 2.0) & (T < t_stop), (T >= t_stop) & (T <= ta)
    trend = {}
    for nm, y in (('q_j2_rad', Q[:, 1]), ('tau_j2_nm', TAU[:, 1]), ('q_minus_sp_j2_rad', Q[:, 1] - SP[:, 1]), ('f_sum_n', F.sum(axis=1)),
                  ('tau_finger1', TAU[:, 6]), ('tau_finger2', TAU[:, 7]), ('wrench_fz', W[:, 2])):
        trend[nm] = {'slope_2s_before_stopped_per_s': slope(T[pre], y[pre]), 'slope_after_stopped_per_s': slope(T[post], y[post])}
    sw = []
    for i in range(len(T)):
        k = int(np.searchsorted(T, T[i] - 0.5 - 1e-6))
        if T[i] >= t_stop and T[i] - T[k] > 0.4:
            v = np.abs(Q[i] - Q[k]) / (T[i] - T[k])
            sw.append((float(T[i]), float(v.max()), int(v.argmax()) + 1))
    first_over = next((x for x in sw if x[1] > sup['thresholds']['arm']), None)
    hold = sv['stats'].get('offset_hold')
    out = {'schema': 'door_coopen_core_bench_5_pause_creep/1', 'window': [t0, ta], 't_stopped': t_stop, 'supervisor': sup,
           'setpoint': {'max_dev_from_switch_rad': float(dsp.max()), 'first_change_t': sp_change,
                        'max_dev_after_stopped_rad': float(np.abs(SP[i_stop:] - SP[i_stop]).max()) if i_stop < len(SP) else None},
           'applied_arm_absmax_after_stopped': float(np.abs(U[i_stop:]).max()) if i_stop < len(U) else None,
           'applied_src_after_stopped': sorted({(None if x is None else (x[2], x[3])) for x in SRC[i_stop:]}, key=str),
           'q_change_after_stopped_mrad': ((Q[-1] - Q[i_stop]) * 1e3).tolist() if i_stop < len(Q) else None,
           'q_change_from_switch_mrad': ((Q[-1] - Q[0]) * 1e3).tolist(),
           'finger_effort': {'pause_switch': {k: fe.get(k) for k in ('step', 't', 'old', 'set', 'readback', 'ok')},
                             'periodic_in_window': [p for p in fe.get('periodic', []) if p['t'] <= ta + 1e-6]},
           'offset_hold': None if not hold else {k: hold.get(k) for k in ('sim_t', 'trigger', 'd_frozen')},
           'trend': trend, 'arm_window_first_over': first_over, 'arm_window_max_after_stopped': max(sw, key=lambda x: x[1]) if sw else None,
           'bins_0p1s': bins, 'note': '只描述時序；不判定原因'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('窗口', [round(t0, 2), round(ta, 2)], 'STOPPED', round(t_stop, 2))
    print('設定點自切換起最大偏差 %.3g rad、首次變動 %s；STOPPED 後最大偏差 %s' % (out['setpoint']['max_dev_from_switch_rad'], sp_change, out['setpoint']['max_dev_after_stopped_rad']))
    print('STOPPED 後 applied_arm |max| %s；套用來源 %s' % (out['applied_arm_absmax_after_stopped'], out['applied_src_after_stopped']))
    print('q 變化（mrad）自切換', [round(v, 3) for v in out['q_change_from_switch_mrad']], '自 STOPPED', [round(v, 3) for v in out['q_change_after_stopped_mrad']])
    for k, v in trend.items():
        print('  趨勢 %-18s STOPPED 前 2 s %s　之後 %s' % (k, None if v['slope_2s_before_stopped_per_s'] is None else '%+.4g' % v['slope_2s_before_stopped_per_s'],
                                                     None if v['slope_after_stopped_per_s'] is None else '%+.4g' % v['slope_after_stopped_per_s']))
    print('手臂 0.5 s 窗口首次超過', first_over, '最大', out['arm_window_max_after_stopped'], '監督 FAIL', sup['transitions']['FAIL'])
    for b in bins[::5]:
        print('  %.1f j2 %+.3f mrad（q−sp %+.3f）τj2 %.3f τj3 %.3f 合力 %.2f N 指 %s 指位 %s τ指 %s Fz %.2f' % (
            b['t'], b['q_j2_mrad_rel'], b['q_minus_sp_j2_mrad'], b['tau_j2'], b['tau_j3'], b['f_sum'], [round(v, 2) for v in b['f']],
            [round(v, 5) for v in b['q_finger']], [round(v, 2) for v in b['tau_finger']], b['wrench_fz']))


if __name__ == '__main__':
    main()
