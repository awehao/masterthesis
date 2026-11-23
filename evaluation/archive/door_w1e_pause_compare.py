#!/usr/bin/env python3
"""door_wide_w1e_30_1（暫停期手指驅動輸出上限 5→20 N）與 door_wide_w1c_30_1（基準）暫停期對照（只讀；Codex 20261025_024339 分報要求）。

分段：PAUSE→STOPPED、STOPPED 後同長窗口（取 w1c STOPPED→中止的長度 5.32 s）、STOPPED→結束（w1c 中止 62.00 s；w1e DONE）。
各段：兩指法向承載（room leaf_contacts bar）、手指 DOF 量測力（evidence dof_measured_efforts，含約束、非驅動輸出）、手指開度、腕力、
門角變化（角度差）、TCP 相對實測把手位移及其分解（door_w1c_stop_relmotion_analysis.decompose；以段首為基準，段末為最後物理步）、
殼—桿淨空（trace k3m_geom nonfinger d 首／末／最小）、底盤位移與 0.1 s 滑動最大速度、手臂實測關節變化全程峰值、手臂套用命令最大值。
單趟對單趟，只描述；上限讀回只確認設定成立，不證明驅動飽和或法向力達某值。
輸出：results/door/w1e_30_1_pause_compare.json、第七次素材／圖表／w1e_30_1_暫停期對照.png。

    python3 evaluation/door_w1e_pause_compare.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_w1c_stop_relmotion_analysis as A           # noqa: E402

RUNS = {'w1c': 'door_wide_w1c_30_1', 'w1e': 'door_wide_w1e_30_1'}
OUT = os.path.join(HERE, 'results', 'door', 'w1e_30_1_pause_compare.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1e_30_1_暫停期對照.png')
FING = ('uflite_finger1', 'uflite_finger2')


def load(rid):
    run = os.path.join(HERE, 'runs', rid)
    r = json.load(open(os.path.join(run, 'room_run.json')))
    s = json.load(open(os.path.join(run, 'align_solver.json')))
    t = json.load(open(os.path.join(run, 'task.json')))
    pe = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    tr = s['stats']['k3m_pause']['supervisor_outcome']['transitions']
    t_end = next((e['t'] for e in t['policy_events'] if e.get('abort')), pe.get('DONE'))
    ev = {}
    for line in open(os.path.join(run, 'evidence.jsonl')):
        d = json.loads(line)
        if pe['PAUSE'] - 0.05 <= d['t'] <= t_end + 0.05:
            ev[d['step']] = d
    geo = {round(x['t'], 2): x['k3m_geom']['cls']['nonfinger'][0] for x in t['trace'] if (x.get('k3m_geom') or {}).get('cls')}
    return r, ev, pe, tr, t_end, geo


def seg(r, ev, geo, a, b):
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    st = [x for x in r['steps'] if a - 1e-9 <= x[I['sim_t']] <= b + 1e-9 and int(x[I['step']]) in ev]
    x0, x1 = st[0], st[-1]
    e0, e1 = ev[int(x0[I['step']])], ev[int(x1[I['step']])]
    H0, H1 = (np.array(e['T_WH'], float).reshape(4, 4) for e in (e0, e1))
    tot, Ao, _, split = A.decompose(H0, x0[I['base_xyth']], x0[I['q_arm_meas']], H1, x1[I['base_xyth']], x1[I['q_arm_meas']])
    fb = np.array([[float(((x[I['leaf_contacts']] or {}).get(f) or {}).get('bar', 0.0)) for f in FING] for x in st])
    eff = np.array([[ev[int(x[I['step']])]['dof_measured_efforts'].get(j, np.nan) for j in ('finger_joint1', 'finger_joint2')] for x in st])
    qf = np.array([x[I['q_finger']] for x in st])
    wr = np.array([float(x[I['wrist_force_n']] or np.nan) for x in st])
    q0 = np.array(x0[I['q_arm_meas']])
    dq = np.abs(np.array([np.array(x[I['q_arm_meas']]) - q0 for x in st])) * 1e3
    vmax = 0.0
    for j in range(len(st) - 10):
        p0, p1 = st[j][I['base_xyth']], st[j + 10][I['base_xyth']]
        vmax = max(vmax, math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (st[j + 10][I['sim_t']] - st[j][I['sim_t']]) * 1e3)
    cl = [geo[k] * 1e3 for k in sorted(geo) if a - 1e-9 <= k <= b + 1e-9]
    return {'t': [float(x0[I['sim_t']]), float(x1[I['sim_t']])], 'n_steps': len(st),
            'F_bar_median_n': np.median(fb, 0).tolist(), 'F_bar_p05_n': np.percentile(fb, 5, 0).tolist(),
            'finger_dof_effort_median_n': np.nanmedian(eff, 0).tolist(), 'finger_dof_effort_absmax_n': np.nanmax(np.abs(eff), 0).tolist(),
            'q_finger_median_mm': (np.median(qf, 0) * 1e3).tolist(), 'wrist_median_n': float(np.nanmedian(wr)), 'wrist_max_n': float(np.nanmax(wr)),
            'door_deg_change': math.degrees(float(x1[I['door_q_rad']]) - float(x0[I['door_q_rad']])),
            'tcp_rel_handle_total_mm': (tot * 1e3).tolist(), 'decomp_A_mm': {k: (v * 1e3).tolist() for k, v in Ao.items()},
            'dH_translation_mm': (split['dH_translation'] * 1e3).tolist(),
            'clearance_mm_first_last_min': None if not cl else [cl[0], cl[-1], min(cl)],
            'base_disp_mm': math.hypot(x1[I['base_xyth']][0] - x0[I['base_xyth']][0], x1[I['base_xyth']][1] - x0[I['base_xyth']][1]) * 1e3,
            'base_max_speed_0p1s_mm_s': vmax, 'arm_meas_change_peak_mrad': float(dq.max()),
            'applied_arm_absmax': max(float(np.abs(x[I['applied_arm']]).max()) for x in st)}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    L = None
    res = {'schema': 'w1e_pause_compare/1', 'runs': {}}
    data = {}
    for k in ('w1c', 'w1e'):
        r, ev, pe, tr, t_end, geo = load(RUNS[k])
        data[k] = (r, ev, pe, tr, t_end, geo)
    L = data['w1c'][4] - data['w1c'][3]['STOPPED']
    for k, (r, ev, pe, tr, t_end, geo) in data.items():
        ts = tr['STOPPED']
        res['runs'][k] = {'transitions': tr, 't_end': t_end, 'pause': pe['PAUSE'],
                          'PAUSE→STOPPED': seg(r, ev, geo, pe['PAUSE'], ts),
                          'STOPPED+同長': seg(r, ev, geo, ts, ts + L),
                          'STOPPED→結束': seg(r, ev, geo, ts, t_end)}
    res['same_len_s'] = L
    fe = data['w1e'][0].get('finger_effort') or {}
    res['w1e_finger_effort'] = {'init': (fe.get('init') or {}).get('readback'), 'pause_step': (fe.get('pause') or {}).get('step'),
                                'old': (fe.get('pause') or {}).get('old'), 'readback': (fe.get('pause') or {}).get('readback'),
                                'n_periodic': len((fe.get('pause') or {}).get('periodic') or []),
                                'periodic_all_ok': all(e.get('ok') is True for e in (fe.get('pause') or {}).get('periodic') or [])}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    col = {'w1c': '#888888', 'w1e': '#2a6fdb'}
    lab = {'w1c': 'w1c_30_1（上限 5 N）', 'w1e': 'w1e_30_1（暫停期 20 N）'}
    for k, (r, ev, pe, tr, t_end, geo) in data.items():
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        ts = tr['STOPPED']
        st = [x for x in r['steps'] if pe['PAUSE'] - 1e-9 <= x[I['sim_t']] <= t_end + 1e-9 and int(x[I['step']]) in ev]
        xs = st[::5]
        tt = [x[I['sim_t']] - pe['PAUSE'] for x in xs]
        ref = next(x for x in st if x[I['sim_t']] >= ts - 1e-9)
        Href = np.array(ev[int(ref[I['step']])]['T_WH'], float).reshape(4, 4)
        y = [((A.p_rel(np.array(ev[int(x[I['step']])]['T_WH'], float).reshape(4, 4), x[I['base_xyth']], x[I['q_arm_meas']])
               - A.p_rel(Href, ref[I['base_xyth']], ref[I['q_arm_meas']]))[1] * 1e3) for x in xs]
        ax[0].plot(tt, y, color=col[k], lw=1.3, label=lab[k])
        fb = [[float(((x[I['leaf_contacts']] or {}).get(f) or {}).get('bar', 0.0)) for f in FING] for x in xs]
        ax[1].plot(tt, [v[0] for v in fb], color=col[k], lw=1.0, label=f'{k} F1')
        ax[1].plot(tt, [v[1] for v in fb], color=col[k], lw=1.0, ls='--', label=f'{k} F2')
        cl = [(kk - pe['PAUSE'], geo[kk] * 1e3) for kk in sorted(geo) if pe['PAUSE'] <= kk <= t_end]
        ax[2].plot([u for u, _ in cl], [v for _, v in cl], color=col[k], lw=1.3, label=lab[k])
        q0 = float(st[0][I['door_q_rad']])
        ax[3].plot(tt, [math.degrees(float(x[I['door_q_rad']]) - q0) for x in xs], color=col[k], lw=1.3, label=lab[k])
        for a_ in ax:
            a_.axvline(ts - pe['PAUSE'], color=col[k], ls=':', lw=0.9)
            a_.axvline(t_end - pe['PAUSE'], color=col[k], ls='-.', lw=0.9)
    ax[2].axhline(5.0, color='#c0392b', ls='--', lw=0.8, label='幾何門檻 5 mm')
    ax[0].set_ylabel('TCP 相對把手 y（mm，相對 STOPPED）')
    ax[1].set_ylabel('兩指法向承載（N）')
    ax[2].set_ylabel('殼—桿淨空（mm）')
    ax[3].set_ylabel('門角變化（°，相對 PAUSE）')
    ax[3].set_xlabel('相對 PAUSE 時間（s）；點線＝STOPPED、點劃線＝結束（w1c 幾何中止、w1e DONE）')
    for a_ in ax:
        a_.grid(alpha=0.25)
        a_.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle('暫停期對照：w1c（手指上限 5 N）vs w1e（暫停期 20 N，執行器上限變更；單趟對單趟）', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    for k in res['runs']:
        for sname in ('PAUSE→STOPPED', 'STOPPED+同長', 'STOPPED→結束'):
            sg = res['runs'][k][sname]
            print(k, sname, json.dumps({kk: (np.round(vv, 3).tolist() if isinstance(vv, (list, float)) else vv) for kk, vv in sg.items()
                                        if kk in ('t', 'F_bar_median_n', 'finger_dof_effort_median_n', 'q_finger_median_mm', 'wrist_median_n',
                                                  'door_deg_change', 'tcp_rel_handle_total_mm', 'clearance_mm_first_last_min', 'base_disp_mm',
                                                  'base_max_speed_0p1s_mm_s', 'arm_meas_change_peak_mrad')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
