#!/usr/bin/env python3
"""door_wide_w1d_30_1（保持錨點重設）與 door_wide_w1c_30_1（未重設）暫停期對照 **r2**（只讀；Codex 20261025_013926 更正）。
r2：post 窗口拆成三段，不把監督器 FAIL 後資料歸為重設效果——
  A 新命令實際套用前 [ZERO_PUB, T_apply)：T_apply＝room_run applied_src 的 source_seq 首次 ≥ first_published.source_seq 的物理步（實錄 55.93 s）；
  B 套用後至 FAIL [T_apply, T_FAIL]；C FAIL 後至任務中止 (T_FAIL, 中止]。w1c 取相對 ZERO_PUB 同長的 A／B 窗口作對照，另報至中止。
  速度為 0.1 s 後向差分，B 段（0.12 s）前段仍含套用前樣本，揭露。輸出 r2 檔（r1 保留）。
（以下為 r1 說明）

每物理步（0.01 s）取：底盤套用命令（本體）、底盤實測速度與門角速率（0.1 s 後向差分，同監督器窗口長度）、
夾爪相對**實測**把手（evidence T_WH）的位置變化（以 ZERO_PUB 時刻為基準，mm）、線上幾何監看殼—桿淨空（trace k3m_geom nonfinger d）。
窗口：pre＝[ZERO_PUB−1 s, ZERO_PUB)；post＝[ZERO_PUB, ZERO_PUB＋L]，L＝w1d 從 ZERO_PUB 到中止的長度（兩趟同長）；w1c 另報 [ZERO_PUB, 中止]。
另算 w1d 重設後底盤位移方向與「重設前位姿−舊錨點」及「重設前伺服命令（本體）」的夾角餘弦。單趟描述，不作統計或因果結論。
輸出：results/door/w1d_30_1_reanchor_compare.json、第七次素材／圖表／w1d_30_1_重設前後對照.png。

    python3 evaluation/door_w1d_reanchor_compare.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402

RUNS = {'w1c': os.path.join(HERE, 'runs', 'door_wide_w1c_30_1'), 'w1d': os.path.join(HERE, 'runs', 'door_wide_w1d_30_1')}
OUT = os.path.join(HERE, 'results', 'door', 'w1d_30_1_reanchor_compare_r2.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1d_30_1_重設前後對照_r2.png')


def load(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    s = json.load(open(os.path.join(run, 'align_solver.json')))
    ev = {}
    for line in open(os.path.join(run, 'evidence.jsonl')):
        d = json.loads(line)
        ev[d['step']] = d
    tr = s['stats']['k3m_pause']['supervisor_outcome']['transitions']
    t_abort = next(e['t'] for e in t['policy_events'] if e.get('abort'))
    geo = {round(x['t'], 2): x['k3m_geom']['cls']['nonfinger'][0] for x in t['trace'] if (x.get('k3m_geom') or {}).get('cls')}
    return t, r, s, ev, tr, t_abort, geo


def series(run, t0, t1):
    t, r, s, ev, tr, t_abort, geo = load(run)
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    st = r['steps']
    tz = tr['ZERO_PUB']
    k_of = {int(x[I['step']]): j for j, x in enumerate(st)}
    rel0 = None
    rows = []
    for j, x in enumerate(st):
        tt = float(x[I['sim_t']])
        if tt < t0 - 1e-9 or tt > t1 + 1e-9 or j < 10:
            continue
        p = st[j - 10]
        dt = tt - float(p[I['sim_t']])
        b, bp = x[I['base_xyth']], p[I['base_xyth']]
        v = math.hypot(b[0] - bp[0], b[1] - bp[1]) / dt
        qd = (float(x[I['door_q_rad']]) - float(p[I['door_q_rad']])) / dt
        e = ev.get(int(x[I['step']]))
        rel = None
        if e and e.get('T_WH'):
            Trel = np.linalg.inv(np.array(e['T_WH'], float).reshape(4, 4)) @ KC._tcp(b, x[I['q_arm_meas']])
            if rel0 is None and tt >= tz - 1e-9:
                rel0 = Trel[:3, 3].copy()
            if rel0 is not None:
                rel = ((Trel[:3, 3] - rel0) * 1e3).tolist()
        rows.append({'t': tt, 't_rel_zp': tt - tz, 'base_cmd_body_mm_s': [float(c) * 1e3 for c in x[I['base_cmd_body']][:2]],
                     'base_cmd_norm_mm_s': math.hypot(*x[I['base_cmd_body']][:2]) * 1e3, 'base_speed_mm_s': v * 1e3,
                     'door_rate_mrad_s': qd * 1e3, 'q_deg': math.degrees(float(x[I['door_q_rad']])), 'tcp_rel_handle_mm': rel,
                     'clear_mm': None if geo.get(round(tt, 2)) is None else geo[round(tt, 2)] * 1e3, 'base_xyth': list(b)})
    return rows, tr, t_abort, s


def win(rows, a, b):
    w = [x for x in rows if a - 1e-9 <= x['t'] < b - 1e-9] or [x for x in rows if a - 1e-9 <= x['t'] <= b + 1e-9]
    cl = [x['clear_mm'] for x in w if x['clear_mm'] is not None]
    rl = [x['tcp_rel_handle_mm'] for x in w if x['tcp_rel_handle_mm'] is not None]
    return {'t': [a, b], 'n': len(w),
            'base_cmd_norm_mm_s_median': float(np.median([x['base_cmd_norm_mm_s'] for x in w])),
            'base_speed_mm_s_median': float(np.median([x['base_speed_mm_s'] for x in w])),
            'base_speed_mm_s_max': float(max(x['base_speed_mm_s'] for x in w)),
            'door_rate_mrad_s_median': float(np.median([x['door_rate_mrad_s'] for x in w])),
            'door_deg_change': w[-1]['q_deg'] - w[0]['q_deg'],
            'clear_mm_first_last': [cl[0], cl[-1]] if cl else None,
            'tcp_rel_handle_mm_last': rl[-1] if rl else None}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    _, rd, sd, _, trd, tad, _ = load(RUNS['w1d'])
    Ir = {c: i for i, c in enumerate(rd['steps_cols'])}
    seq1 = sd['stats']['pause_reanchor']['record']['first_published']['source_seq']
    t_apply = next(float(x[Ir['sim_t']]) for x in rd['steps'] if x[Ir['applied_src']][2] >= seq1)
    t_fail = trd['FAIL']
    LA, LB = t_apply - trd['ZERO_PUB'], t_fail - trd['ZERO_PUB']
    L = tad - trd['ZERO_PUB']
    res = {'schema': 'w1d_reanchor_compare/2', 't_apply_w1d': t_apply, 't_fail_w1d': t_fail, 'note': '單趟描述；速度＝0.1 s 後向差分（同監督器窗口）；淨空＝線上幾何監看 nonfinger d；TCP 相對實測把手以 ZERO_PUB 為基準',
           'post_window_len_s': L, 'runs': {}}
    all_rows = {}
    for k, run in RUNS.items():
        _, _, _, _, tr, ta, _ = load(run)
        tz = tr['ZERO_PUB']
        rows, tr, ta, s = series(run, tz - 1.0, ta + 0.0)
        all_rows[k] = rows
        res['runs'][k] = {'transitions': tr, 't_abort': ta, 'abort': next(e['abort'] for e in json.load(open(os.path.join(run, 'task.json')))['policy_events'] if e.get('abort')),
                          'pre': win(rows, tz - 1.0, tz), 'A_before_apply': win(rows, tz, tz + LA),
                          'B_apply_to_fail': win(rows, tz + LA, tz + LB + 1e-6), 'C_after_fail': win(rows, tz + LB + 1e-6, tz + L),
                          'post_to_abort': win(rows, tz, ta)}
    rec = sd['stats']['pause_reanchor']['record']
    rows = all_rows['w1d']
    tz = res['runs']['w1d']['transitions']['ZERO_PUB']
    eff = rec['effective']['sim_t']
    a = min(rows, key=lambda x: abs(x['t'] - eff))['base_xyth']
    b = rows[-1]['base_xyth']
    d = np.array([b[0] - a[0], b[1] - a[1]])
    e = np.array([a[0] - rec['old_anchor'][0], a[1] - rec['old_anchor'][1]])
    c, s_ = math.cos(a[2]), math.sin(a[2])
    db = np.array([d[0] * c + d[1] * s_, -d[0] * s_ + d[1] * c])
    pre_cmd = np.array(next(x['base_cmd_body_mm_s'] for x in reversed(rows) if x['t'] < tz))
    cs = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
    res['w1d_direction'] = {'disp_after_effective_world_mm': (d * 1e3).tolist(), 'pose_minus_old_anchor_at_effective_mm': (e * 1e3).tolist(),
                            'cos_disp_vs_offset': cs(d, e), 'disp_body_mm': (db * 1e3).tolist(), 'pre_reset_cmd_body_mm_s': pre_cmd.tolist(),
                            'cos_disp_body_vs_pre_cmd': cs(db, pre_cmd)}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    col = {'w1c': '#888888', 'w1d': '#2a6fdb'}
    lab = {'w1c': 'w1c_30_1（未重設）', 'w1d': 'w1d_30_1（ZERO_PUB 重設）'}
    for k, rows in all_rows.items():
        tt = [x['t_rel_zp'] for x in rows if x['t_rel_zp'] <= 1.5]
        sel = rows[:len(tt)]
        ax[0].plot(tt, [x['base_cmd_norm_mm_s'] for x in sel], color=col[k], lw=1.2, label=lab[k])
        ax[1].plot(tt, [x['base_speed_mm_s'] for x in sel], color=col[k], lw=1.2)
        ax[2].plot(tt, [x['door_rate_mrad_s'] for x in sel], color=col[k], lw=1.2)
        cl = [(x['t_rel_zp'], x['clear_mm']) for x in sel if x['clear_mm'] is not None]
        ax[3].plot([u for u, _ in cl], [v for _, v in cl], color=col[k], lw=1.2, marker='.', ms=3)
        ta = res['runs'][k]['t_abort'] - res['runs'][k]['transitions']['ZERO_PUB']
        if ta <= 1.5:
            for a_ in ax:
                a_.axvline(ta, color=col[k], ls=':', lw=1)
    for a_ in ax:
        a_.axvline(LA, color='#4a9a5a', ls='--', lw=0.8)
        a_.axvline(LB, color='#c0392b', ls='--', lw=0.8)
    ax[1].axhline(1.0, color='#d9822b', ls='--', lw=0.8, label='監督器底盤門檻 1 mm/s')
    ax[2].axhline(2.0, color='#d9822b', ls='--', lw=0.8, label='門門檻 2 mrad/s')
    ax[0].set_ylabel('底盤套用命令 |v|（mm/s）')
    ax[1].set_ylabel('底盤實測速度（mm/s）')
    ax[2].set_ylabel('門角速率（mrad/s）')
    ax[3].set_ylabel('殼—桿淨空（mm）')
    ax[3].set_xlabel('相對 ZERO_PUB 時間（s）；綠虛線＝w1d 新命令實際套用、紅虛線＝w1d 監督器 FAIL、點線＝任務中止')
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].legend(fontsize=8, frameon=False)
    ax[2].legend(fontsize=8, frameon=False)
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.suptitle('保持錨點重設前後：命令、底盤速度、門角速率與淨空（單趟對照）', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(json.dumps({k: v for k, v in res.items() if k != 'runs'}, ensure_ascii=False, indent=1, default=float))
    for k in res['runs']:
        print(k, json.dumps({w: res['runs'][k][w] for w in ('pre', 'A_before_apply', 'B_apply_to_fail', 'C_after_fail', 'post_to_abort')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
