#!/usr/bin/env python3
"""door_wide_w1c_30_1 暫停期幾何分析（Codex 20261025_005017：只讀本趟資料，聚焦 PAUSE→STOPPED→幾何中止）。

逐 0.1 s（每 10 物理步）以線上同一距離計算（door_k3m_geom_monitor.default_backend：FK 點雲 × 門角模型幾何）重算：
  S0 實際：qf(t)、q_door(t)、q_finger(t)；
  S1 門角固定在 STOPPED 時刻值，其餘實際（只看夾爪側運動）；
  S2 底盤＋手臂固定在 STOPPED 時刻值，門角實際（只看門角的影響）；
  S3 手指開度固定在 STOPPED 時刻值，其餘實際。
敏感性對照只用來比較伴隨關係，**不相加成因果百分比**。
另對齊：門角、底盤位移、手臂關節變化、手指開度、兩指接觸力、腕力、底盤套用命令與手臂套用值；
  夾爪相對**實測**把手（evidence T_WH）的位姿變化（mm／RPY 參數）；門角模型推得的把手位置與實測 T_WH 的差（mm）；
  線上幾何監看紀錄（trace k3m_geom：d、d_pred；接近率、預測扣除量＝d − d_pred）。
輸出：results/door/w1c_30_1_pause_geom.json、第七次素材／圖表／w1c_30_1_暫停期淨空.png。只描述，不推因果。

    python3 evaluation/door_w1c_pause_geom_analysis.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k3m_geom_monitor as GEO                    # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1c_30_1')
OUT = os.path.join(HERE, 'results', 'door', 'w1c_30_1_pause_geom.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1c_30_1_暫停期淨空.png')
FINGERS = ('uflite_finger1', 'uflite_finger2')


def rpy(R):
    return np.degrees([math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0]))), math.atan2(R[1, 0], R[0, 0])])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    s = json.load(open(os.path.join(RUN, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {}
    for line in open(os.path.join(RUN, 'evidence.jsonl')):
        d = json.loads(line)
        ev[d['step']] = d
    pe = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    t_pause, t_abort = pe['PAUSE'], next(e['t'] for e in t['policy_events'] if e.get('abort'))
    tr_st = (s['stats']['k3m_pause']['supervisor_outcome']['transitions'] or {})
    t_stop = tr_st.get('STOPPED')
    t_from = pe['SETTLE']
    steps = [x for x in r['steps'] if t_from - 1e-9 <= x[I['sim_t']] <= t_abort + 1e-9]
    ref = min(steps, key=lambda x: abs(x[I['sim_t']] - t_stop))
    qf_ref = np.array(list(ref[I['base_xyth']]) + list(ref[I['q_arm_meas']]), float)
    qd_ref = float(ref[I['door_q_rad']])
    qfg_ref = float(np.mean(ref[I['q_finger']]))
    T_ref_rel = None
    fn = GEO.default_backend()
    geo_log = {round(x['t'], 4): x['k3m_geom'] for x in t['trace'] if x.get('k3m_geom') and x['k3m_geom'].get('cls')}
    rows = []
    for x in steps[::10]:
        k, tt = int(x[I['step']]), float(x[I['sim_t']])
        qf = np.array(list(x[I['base_xyth']]) + list(x[I['q_arm_meas']]), float)
        qd, qfg = float(x[I['door_q_rad']]), float(np.mean(x[I['q_finger']]))
        S0 = fn(qf, qd, qfg)['nonfinger']
        S1 = fn(qf, qd_ref, qfg)['nonfinger']
        S2 = fn(qf_ref, qd, qfg)['nonfinger']
        S3 = fn(qf, qd, qfg_ref)['nonfinger']
        e = ev.get(k)
        rel_mm = rel_rpy = model_vs_meas_mm = None
        if e and e.get('T_WH'):
            TWH = np.array(e['T_WH'], float).reshape(4, 4)
            Trel = np.linalg.inv(TWH) @ KC._tcp(x[I['base_xyth']], x[I['q_arm_meas']])
            if T_ref_rel is None and tt >= t_stop - 1e-9:
                T_ref_rel = Trel.copy()
            if T_ref_rel is not None:
                rel_mm = ((Trel[:3, 3] - T_ref_rel[:3, 3]) * 1e3).tolist()
                rel_rpy = (rpy(Trel[:3, :3]) - rpy(T_ref_rel[:3, :3])).tolist()
            model_vs_meas_mm = ((KC._H_of(qd)[:3, 3] - TWH[:3, 3]) * 1e3).tolist()
        lc = x[I['leaf_contacts']] or {}
        g = geo_log.get(round(tt, 4))
        rows.append({'t': tt, 'step': k, 'q_deg': math.degrees(qd), 'dq_from_stop_deg': math.degrees(qd - qd_ref),
                     'base_d_mm': ((qf[:2] - qf_ref[:2]) * 1e3).tolist(), 'base_dyaw_mrad': (qf[2] - qf_ref[2]) * 1e3,
                     'arm_dmax_mrad': float(np.abs(qf[3:] - qf_ref[3:]).max() * 1e3), 'q_finger': qfg,
                     'fb': [float((lc.get(f) or {}).get('bar', 0.0)) for f in FINGERS], 'wrist_n': float(x[I['wrist_force_n']]),
                     'base_cmd': [float(v) for v in x[I['base_cmd_body']]], 'arm_applied_absmax': float(np.abs(x[I['applied_arm']]).max()),
                     'S0_mm': S0[0] * 1e3, 'S0_who': S0[1], 'S1_door_fixed_mm': S1[0] * 1e3, 'S2_robot_fixed_mm': S2[0] * 1e3,
                     'S3_finger_fixed_mm': S3[0] * 1e3, 'tcp_rel_handle_mm': rel_mm, 'tcp_rel_handle_rpy_deg': rel_rpy,
                     'model_minus_meas_handle_mm': model_vs_meas_mm,
                     'online': None if g is None else {'dec': g.get('dec'), 'd_mm': g['cls']['nonfinger'][0] * 1e3,
                                                         'd_pred_mm': g['cls']['nonfinger'][1] * 1e3,
                                                         'deduction_mm': (g['cls']['nonfinger'][0] - g['cls']['nonfinger'][1]) * 1e3}})
    a = [x for x in rows if x['t'] >= t_stop - 1e-9]
    first, last = a[0], a[-1]
    summ = {'t_settle': pe['SETTLE'], 't_pause': t_pause, 't_stopped': t_stop, 't_abort': t_abort,
            'who_set_after_stop': sorted({x['S0_who'] for x in a}),
            'delta_STOPPED_to_abort': {'S0_mm': last['S0_mm'] - first['S0_mm'], 'S1_door_fixed_mm': last['S1_door_fixed_mm'] - first['S1_door_fixed_mm'],
                                       'S2_robot_fixed_mm': last['S2_robot_fixed_mm'] - first['S2_robot_fixed_mm'],
                                       'S3_finger_fixed_mm': last['S3_finger_fixed_mm'] - first['S3_finger_fixed_mm'],
                                       'door_deg': last['q_deg'] - first['q_deg'], 'base_mm': last['base_d_mm'],
                                       'arm_dmax_mrad': last['arm_dmax_mrad'],
                                       'tcp_rel_handle_mm': last['tcp_rel_handle_mm'], 'model_minus_meas_handle_mm_first_last':
                                           [first['model_minus_meas_handle_mm'], last['model_minus_meas_handle_mm']]},
            'online_deduction_mm_range': [min(x['online']['deduction_mm'] for x in rows if x['online']),
                                          max(x['online']['deduction_mm'] for x in rows if x['online'])]}
    json.dump({'schema': 'w1c_pause_geom/1', 'note': '敏感性對照只比較伴隨關係，不相加成因果百分比；距離＝門角模型幾何×FK 點雲（同線上）。',
               'summary': summ, 'rows': rows}, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    T = [x['t'] for x in rows]
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(T, [x['S0_mm'] for x in rows], color='#2a6fdb', lw=1.8, label='S0 實際（重算）')
    ax[0].plot(T, [x['S1_door_fixed_mm'] for x in rows], color='#d9822b', lw=1.2, ls='--', label='S1 門角固定於 STOPPED')
    ax[0].plot(T, [x['S2_robot_fixed_mm'] for x in rows], color='#4a9a5a', lw=1.2, ls='--', label='S2 底盤＋手臂固定於 STOPPED')
    ax[0].plot(T, [x['S3_finger_fixed_mm'] for x in rows], color='#888', lw=1.0, ls=':', label='S3 手指開度固定')
    on = [x for x in rows if x['online']]
    ax[0].plot([x['t'] for x in on], [x['online']['d_pred_mm'] for x in on], color='#b33', lw=1.0, label='線上預測 d_pred')
    ax[0].axhline(5.0, color='#b33', lw=0.8, ls=':')
    ax[0].set_ylabel('夾爪殼—橫桿淨空（mm）')
    ax[0].legend(fontsize=7.5, frameon=False, loc='lower left')
    ax[1].plot(T, [x['dq_from_stop_deg'] for x in rows], color='#2a6fdb', lw=1.4, label='門角變化（°，相對 STOPPED）')
    ax2 = ax[1].twinx()
    ax2.plot(T, [x['arm_dmax_mrad'] for x in rows], color='#d9822b', lw=1.0, label='手臂最大關節變化（mrad）')
    ax2.plot(T, [math.hypot(*x['base_d_mm']) for x in rows], color='#4a9a5a', lw=1.0, label='底盤位移（mm）')
    ax[1].set_ylabel('門角變化（°）')
    ax2.set_ylabel('mrad／mm')
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc='upper left')
    rr = [x for x in rows if x['tcp_rel_handle_mm']]
    for j, lab, col in ((0, 'x 沿橫桿', '#888'), (1, 'y', '#4a9a5a'), (2, 'z 向上', '#2a6fdb')):
        ax[2].plot([x['t'] for x in rr], [x['tcp_rel_handle_mm'][j] for x in rr], color=col, lw=1.2, label=f'TCP 相對實測把手 Δ{lab}')
    mm = [x for x in rows if x['model_minus_meas_handle_mm']]
    ax[2].plot([x['t'] for x in mm], [np.linalg.norm(x['model_minus_meas_handle_mm']) - np.linalg.norm(mm[0]['model_minus_meas_handle_mm']) for x in mm],
               color='#b33', lw=1.0, ls='--', label='模型把手與實測把手差（變化量）')
    ax[2].set_ylabel('mm')
    ax[2].set_xlabel('模擬時間 s')
    ax[2].legend(fontsize=7.5, frameon=False, loc='upper left')
    for a_ in ax:
        for tt, c in ((pe['SETTLE'], '#444'), (t_pause, '#444'), (t_stop, '#2a6fdb'), (t_abort, '#000')):
            a_.axvline(tt, color=c, lw=0.7, ls='--')
        a_.grid(alpha=0.25)
    fig.suptitle('door_wide_w1c_30_1：SETTLE→PAUSE→STOPPED→幾何中止 的殼—桿淨空與對齊量（虛線：SETTLE、PAUSE、STOPPED、中止）', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)


if __name__ == '__main__':
    main()
