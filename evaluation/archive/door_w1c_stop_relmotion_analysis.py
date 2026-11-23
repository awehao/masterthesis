#!/usr/bin/env python3
"""door_wide_w1c_30_1：STOPPED→幾何中止期間 TCP 相對把手位移的運動學分解（只讀；Codex 20261025_022929）。

一、手臂：設定點（evidence arm_sp）、實測關節（room q_arm_meas）、套用命令（room applied_arm）是否持續變化；其 FK 對 TCP 的影響。
二、相對位移分解（共同座標、可閉合）：p(H, B, q)＝(H⁻¹ · T_tcp(B, q)) 的位置（把手座標，H＝evidence T_WH；B＝base_xyth；q＝q_arm_meas）。
    以 t0＝STOPPED 為基準，逐項替換（telescoping）：
      順序 A：Δq＝p(H0,B0,q1)−p(H0,B0,q0)；ΔB＝p(H0,B1,q1)−p(H0,B0,q1)；ΔH＝p(H1,B1,q1)−p(H0,B1,q1)；
      順序 B：ΔH 先、ΔB、Δq 後。兩者都與總變化 p(H1,B1,q1)−p(H0,B0,q0) 精確閉合（報告閉合殘差），順序差異另報。
    ΔH 再拆：把手原點平移項 R1ᵀ(o0−o1) 與把手座標旋轉項 (R1ᵀ−R0ᵀ)(x−o0)（x＝TCP 世界位置），兩者相加＝ΔH。
    另以門角變化 × 把手半徑 0.80 m 作量級對照（不是分解項）。
三、只回答「運動學上哪一項構成相對位移」，**不回答**「什麼受力造成門續轉」。
輸出：results/door/w1c_30_1_stop_relmotion.json、第七次素材／圖表／w1c_30_1_停止後相對位移分解.png。

    python3 evaluation/door_w1c_stop_relmotion_analysis.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1c_30_1')
OUT = os.path.join(HERE, 'results', 'door', 'w1c_30_1_stop_relmotion.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1c_30_1_停止後相對位移分解.png')
R_HANDLE = 0.80


def p_rel(H, B, q):
    T = KC._tcp(B, np.asarray(q, float))
    return (np.linalg.inv(H) @ T)[:3, 3]


def decompose(H0, B0, q0, H1, B1, q1):
    tot = p_rel(H1, B1, q1) - p_rel(H0, B0, q0)
    A = {'dq': p_rel(H0, B0, q1) - p_rel(H0, B0, q0), 'dB': p_rel(H0, B1, q1) - p_rel(H0, B0, q1),
         'dH': p_rel(H1, B1, q1) - p_rel(H0, B1, q1)}
    Bo = {'dH': p_rel(H1, B0, q0) - p_rel(H0, B0, q0), 'dB': p_rel(H1, B1, q0) - p_rel(H1, B0, q0),
          'dq': p_rel(H1, B1, q1) - p_rel(H1, B1, q0)}
    x = KC._tcp(B1, np.asarray(q1, float))[:3, 3]
    R0, o0, R1, o1 = H0[:3, :3], H0[:3, 3], H1[:3, :3], H1[:3, 3]
    split = {'dH_translation': R1.T @ (o0 - o1), 'dH_rotation': (R1.T - R0.T) @ (x - o0)}
    return tot, A, Bo, split


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    s = json.load(open(os.path.join(RUN, 'align_solver.json')))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    tr = s['stats']['k3m_pause']['supervisor_outcome']['transitions']
    t0, ta = tr['STOPPED'], next(e['t'] for e in t['policy_events'] if e.get('abort'))
    ev = {}
    for line in open(os.path.join(RUN, 'evidence.jsonl')):
        d = json.loads(line)
        if t0 - 0.05 <= d['t'] <= ta + 0.05:
            ev[d['step']] = d
    st = [x for x in r['steps'] if t0 - 1e-9 <= x[I['sim_t']] <= ta + 1e-9 and int(x[I['step']]) in ev]
    x0 = st[0]
    e0 = ev[int(x0[I['step']])]
    H0 = np.array(e0['T_WH'], float).reshape(4, 4)
    B0, q0, sp0 = x0[I['base_xyth']], x0[I['q_arm_meas']], np.array(e0['arm_sp'], float)
    qd0 = float(x0[I['door_q_rad']])
    rows = []
    for x in st[::5]:
        e = ev[int(x[I['step']])]
        H1 = np.array(e['T_WH'], float).reshape(4, 4)
        tot, A, Bo, split = decompose(H0, B0, q0, H1, x[I['base_xyth']], x[I['q_arm_meas']])
        rows.append({'t': float(x[I['sim_t']]), 'dt': float(x[I['sim_t']] - t0), 'total_mm': (tot * 1e3).tolist(),
                     'A_mm': {k: (v * 1e3).tolist() for k, v in A.items()}, 'B_mm': {k: (v * 1e3).tolist() for k, v in Bo.items()},
                     'dH_split_mm': {k: (v * 1e3).tolist() for k, v in split.items()},
                     'closure_A_mm': float(np.abs(tot - sum(A.values())).max() * 1e3),
                     'closure_B_mm': float(np.abs(tot - sum(Bo.values())).max() * 1e3),
                     'door_arc_mm': (float(x[I['door_q_rad']]) - qd0) * R_HANDLE * 1e3,
                     'base_disp_mm': [(x[I['base_xyth']][0] - B0[0]) * 1e3, (x[I['base_xyth']][1] - B0[1]) * 1e3],
                     'base_dyaw_mrad': (x[I['base_xyth']][2] - B0[2]) * 1e3,
                     'q_meas_minus_q0_mrad': ((np.array(x[I['q_arm_meas']]) - np.array(q0)) * 1e3).tolist(),
                     'sp_minus_sp0_mrad': ((np.array(e['arm_sp']) - sp0) * 1e3).tolist(),
                     'applied_arm_absmax': float(np.abs(x[I['applied_arm']]).max()),
                     'base_cmd_body_mm_s': [float(v) * 1e3 for v in x[I['base_cmd_body']][:2]]})
    last = rows[-1]
    allst = [x for x in r['steps'] if t0 - 1e-9 <= x[I['sim_t']] <= ta + 1e-9]
    summ = {'t_stopped': t0, 't_abort': ta, 'n_steps': len(allst), 'n_with_evidence': len(st),
            'total_mm': last['total_mm'], 'order_A_mm': last['A_mm'], 'order_B_mm': last['B_mm'], 'dH_split_mm': last['dH_split_mm'],
            'closure_max_mm': max(max(x['closure_A_mm'], x['closure_B_mm']) for x in rows),
            'door_arc_mm': last['door_arc_mm'], 'door_deg': math.degrees(float(allst[-1][I['door_q_rad']]) - qd0),
            'arm': {'applied_arm_absmax_all_steps': max(float(np.abs(x[I['applied_arm']]).max()) for x in allst),
                    'sp_change_absmax_mrad': max(max(abs(v) for v in x['sp_minus_sp0_mrad']) for x in rows),
                    'q_meas_change_absmax_mrad': max(max(abs(v) for v in x['q_meas_minus_q0_mrad']) for x in rows),
                    'sp_minus_meas_at_t0_mrad': ((sp0 - np.array(q0)) * 1e3).tolist()},
            'base': {'disp_mm_last': last['base_disp_mm'], 'dyaw_mrad_last': last['base_dyaw_mrad'],
                     'cmd_body_mm_s_mean': np.mean([x['base_cmd_body_mm_s'] for x in rows], 0).tolist()}}
    json.dump({'schema': 'w1c_stop_relmotion/1', 'summary': summ, 'rows': rows}, open(OUT, 'w'), ensure_ascii=False, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    tt = [x['dt'] for x in rows]
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
    ax[0].plot(tt, [x['total_mm'][1] for x in rows], color='#222', lw=1.8, label='總變化 y（把手座標）')
    ax[0].plot(tt, [x['A_mm']['dH'][1] for x in rows], color='#2a6fdb', lw=1.2, label='把手項 ΔH')
    ax[0].plot(tt, [x['A_mm']['dB'][1] for x in rows], color='#d9822b', lw=1.2, label='底盤項 ΔB')
    ax[0].plot(tt, [x['A_mm']['dq'][1] for x in rows], color='#4a9a5a', lw=1.2, label='手臂關節項 Δq')
    ax[0].plot(tt, [x['door_arc_mm'] for x in rows], color='#888', ls=':', lw=1.2, label='門角變化×0.80 m（量級對照）')
    ax[0].set_ylabel('TCP 相對把手 y 變化（mm）')
    ax[0].legend(fontsize=7.5, frameon=False, ncol=2)
    ax[1].plot(tt, [x['dH_split_mm']['dH_translation'][1] for x in rows], color='#2a6fdb', lw=1.2, label='ΔH 把手原點平移項 y')
    ax[1].plot(tt, [x['dH_split_mm']['dH_rotation'][1] for x in rows], color='#7b5ea7', lw=1.2, label='ΔH 把手座標旋轉項 y')
    ax[1].plot(tt, [math.hypot(*x['base_disp_mm']) for x in rows], color='#d9822b', lw=1.0, ls='--', label='底盤位移 |·|（mm）')
    ax[1].set_ylabel('mm')
    ax[1].legend(fontsize=7.5, frameon=False)
    for j in range(6):
        ax[2].plot(tt, [x['q_meas_minus_q0_mrad'][j] for x in rows], lw=1.0, label=f'j{j + 1} 實測')
    ax[2].plot(tt, [max(abs(v) for v in x['sp_minus_sp0_mrad']) for x in rows], color='#222', ls='--', lw=1.2, label='設定點變化 max|·|')
    ax[2].set_ylabel('手臂關節變化（mrad）')
    ax[2].set_xlabel('STOPPED 後時間（s）；至幾何中止')
    ax[2].legend(fontsize=7, frameon=False, ncol=4)
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.suptitle('door_wide_w1c_30_1：STOPPED→幾何中止的 TCP 相對把手位移分解（順序 A；閉合殘差 ≤ %.1e mm）' % summ['closure_max_mm'], fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
