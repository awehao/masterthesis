#!/usr/bin/env python3
"""W1b 30° 單趟（door_wide_w1b_30_1）的接觸承載分析（Codex 20261025_001336：只讀既有資料、不擷取、不調參、不跑模擬）。

對齊：以物理步號配對 room_run.json（門角、底盤、實測手臂、手指開度、相位）與 evidence.jsonl（把手位姿 T_WH、逐點接觸 p／n／fn、
逐點摩擦 ft、joint6 反作用旋量）；求解紀錄以 **publish_sim_t_exact（已發布列）** 配對引導是否開啟（r2，Codex 20261025_002129 第 2 點；r1 誤用快照 sim_t）。
姿態變化稱「RPY 參數變化」。輸出 r2 檔（r1 保留）。
分段：HOLD、OPEN（至 finger1 首次低於 0.5 N 前）、首次低於門檻、持續低值至中止、中止後。
座標：接觸點與法向、摩擦切向力轉到**把手座標**（T_WH：x 沿橫桿、z 向上；原點＝橫桿中心）；
  每指：法向力幅值和 Σ|fn|、承載點（只取 |fn| > 0.05 N 的點，以 |fn| 加權；零承載點不定義承載方向）、
  承載加權法向、切向力和 Σft（把手座標）。
夾爪相對把手：TCP 在把手座標的位置與姿態，以 HOLD 末為基準的變化（mm、°）。
joint6 反作用旋量：規格登錄為「子關節座標、子關節原點」的入射反作用；**作用側未核清**，只報原座標的力大小與分量，
  轉到把手座標與合力／力矩平衡一律標 NA。
輸出：results/door/w1b_30_1_load_analysis.json＋圖（第七次素材／圖表）。只描述，不推因果。

    python3 evaluation/door_w1b_load_analysis.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1b_30_1')
OUT = os.path.join(HERE, 'results', 'door', 'w1b_30_1_load_analysis_r2.json')
FIG_DIR = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表')
FINGERS = ('uflite_finger1', 'uflite_finger2')
FN_MIN = 0.05
LOW_N = 0.5


def rot_to_rpy(R):
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[1, 0], R[0, 0])))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    task = json.load(open(os.path.join(RUN, 'task.json')))
    room = json.load(open(os.path.join(RUN, 'room_run.json')))
    solver = json.load(open(os.path.join(RUN, 'align_solver.json')))
    ev = {}
    for line in open(os.path.join(RUN, 'evidence.jsonl')):
        d = json.loads(line)
        ev[int(d['step'])] = d
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    pe = {e['to']: e['t'] for e in task['policy_events'] if 'to' in e}
    t_hold, t_open, t_abort = pe['HOLD'], pe['OPEN'], task.get('t_abort') or next(e['t'] for e in task['policy_events'] if e.get('abort'))
    glog = sorted((x['publish_sim_t_exact'], bool((x.get('coord') or {}).get('w_vref'))) for x in solver.get('log') or []
                  if x.get('published') and isinstance(x.get('coord'), dict) and x.get('publish_sim_t_exact') is not None)
    rows = []
    T_base = None
    for s in room['steps']:
        k, t = int(s[I['step']]), float(s[I['sim_t']])
        if t < t_hold - 1e-9:
            continue
        e = ev.get(k)
        if e is None or not e.get('T_WH'):
            continue
        TWH = np.array(e['T_WH'], float).reshape(4, 4)
        Rt, o = TWH[:3, :3].T, TWH[:3, 3]
        lc = s[I['leaf_contacts']] or {}
        fb_room = [float((lc.get(f) or {}).get('bar', 0.0)) for f in FINGERS]
        T_tcp = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        T_rel = np.linalg.inv(TWH) @ T_tcp
        row = {'step': k, 't': t, 'q_deg': math.degrees(float(s[I['door_q_rad']])), 'fb_room': fb_room,
               'q_finger': [float(v) for v in s[I['q_finger']]], 'wrist_n': float(s[I['wrist_force_n']]),
               'tcp_in_handle_mm': (T_rel[:3, 3] * 1e3).tolist(), 'tcp_rpy_in_handle_deg': rot_to_rpy(T_rel[:3, :3])}
        if T_base is None and t >= t_open - 0.011:
            T_base = T_rel.copy()
        for f in FINGERS:
            pts = (e.get('contacts') or {}).get(f) or []
            fr = (e.get('friction') or {}).get(f) or []
            fn = np.array([abs(float(p['fn'])) for p in pts]) if pts else np.zeros(0)
            P = np.array([Rt @ (np.array(p['p'], float) - o) for p in pts]) if pts else np.zeros((0, 3))
            N = np.array([Rt @ np.array(p['n'], float) for p in pts]) if pts else np.zeros((0, 3))
            m = fn > FN_MIN
            ft = np.sum([Rt @ np.array(q['ft'], float) for q in fr], axis=0) if fr else np.zeros(3)
            row[f] = {'fn_sum': float(fn.sum()), 'n_pts_loaded': int(m.sum()),
                      'load_point_mm': None if not m.any() else (np.average(P[m], axis=0, weights=fn[m]) * 1e3).tolist(),
                      'load_normal': None if not m.any() else (lambda v: (v / np.linalg.norm(v)).tolist())(np.average(N[m], axis=0, weights=fn[m])),
                      'ft_sum_handle': ft.tolist()}
        w = e.get('j6_reaction_wrench')
        row['j6_force_raw'] = None if not w else [float(v) for v in w[:3]]
        i = np.searchsorted([g[0] for g in glog], t, side='right') - 1
        row['guide_on'] = None if i < 0 else glog[i][1]
        rows.append(row)
    # 相對 OPEN 起點的夾爪姿態變化
    for r in rows:
        r['tcp_d_mm'] = (np.array(r['tcp_in_handle_mm']) - T_base[:3, 3] * 1e3).tolist()
        r['tcp_d_rpy_deg'] = [((a - b + 180.0) % 360.0) - 180.0 for a, b in zip(r['tcp_rpy_in_handle_deg'], rot_to_rpy(T_base[:3, :3]))]
    open_rows = [r for r in rows if t_open <= r['t'] < t_abort]
    t_first_low = next((r['t'] for r in open_rows if r['fb_room'][0] < LOW_N), None)
    t_sust = None                                            # 逐步值在門檻附近擺盪；中止前最後一段連續低於門檻的起點
    for r in reversed(open_rows):
        if r['fb_room'][0] >= LOW_N:
            break
        t_sust = r['t']
    segs = {'HOLD': (t_hold, t_open), 'OPEN_before_low': (t_open, t_first_low or t_abort),
            'first_low_to_abort': (t_first_low, t_abort) if t_first_low else None, 'post_abort': (t_abort, rows[-1]['t'])}

    def summ(a, b):
        xs = [r for r in rows if a <= r['t'] < b]
        if not xs:
            return None
        out = {'n': len(xs), 't': [a, b], 'q_deg': [xs[0]['q_deg'], xs[-1]['q_deg']], 'wrist_n': [min(r['wrist_n'] for r in xs), max(r['wrist_n'] for r in xs)]}
        for j, f in enumerate(FINGERS):
            fs = [r[f]['fn_sum'] for r in xs]
            lp = [r[f]['load_point_mm'] for r in xs if r[f]['load_point_mm'] is not None]
            out[f] = {'fn_sum_first_last': [fs[0], fs[-1]], 'fn_sum_min_max': [min(fs), max(fs)],
                      'load_point_mm_first_last': None if not lp else [lp[0], lp[-1]],
                      'ft_handle_last': xs[-1][f]['ft_sum_handle'], 'q_finger_first_last': [xs[0]['q_finger'][j], xs[-1]['q_finger'][j]]}
        out['finger1_frac_below_low'] = sum(1 for r in xs if r['fb_room'][0] < LOW_N) / len(xs)
        out['tcp_d_mm_last'] = xs[-1]['tcp_d_mm']
        out['tcp_d_rpy_deg_last'] = xs[-1]['tcp_d_rpy_deg']
        return out
    res = {'schema': 'w1b_load_analysis/2', 'run': 'door_wide_w1b_30_1', 'solver_pairing': 'publish_sim_t_exact（已發布列）', 'frames': {
        'handle': 'T_WH（evidence；x 沿橫桿、z 向上，原點＝把手 prim）', 'contacts': '世界座標 → 把手座標',
        'j6': '子關節座標、子關節原點（規格登錄）；作用側未核清 ⇒ 轉把手座標與力平衡標 NA'},
        'thresholds': {'fn_point_min_n': FN_MIN, 'finger_low_n': LOW_N}, 't_events': {'HOLD': t_hold, 'OPEN': t_open,
        'first_finger1_low': t_first_low, 'last_continuous_low_from': t_sust, 'abort': t_abort},
        'segments': {k: (None if v is None else summ(*v)) for k, v in segs.items()},
        'j6_handle_frame': 'NA', 'force_balance': 'NA',
        'rows_every_10': rows[::10]}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    figs(rows, t_open, t_first_low, t_sust, t_abort)
    print(json.dumps({k: v for k, v in res.items() if k not in ('rows_every_10',)}, ensure_ascii=False, indent=1, default=float)[:6000])


def figs(rows, t_open, t_low, t_sust, t_abort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    os.makedirs(FIG_DIR, exist_ok=True)
    xs = [r for r in rows if r['t'] >= t_open - 1.0 and r['t'] <= t_abort + 1.0]
    q = [r['q_deg'] for r in xs]
    t = [r['t'] for r in xs]
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t, [r['uflite_finger1']['fn_sum'] for r in xs], color='#2a6fdb', lw=1.6, label='finger1（上）法向力和')
    ax[0].plot(t, [r['uflite_finger2']['fn_sum'] for r in xs], color='#d9822b', lw=1.6, label='finger2（下）法向力和')
    ax[0].plot(t, [r['wrist_n'] for r in xs], color='#6b6b6b', lw=1.0, ls='--', label='腕力')
    ax[0].axhline(LOW_N, color='#b33', lw=0.8, ls=':')
    ax[0].set_ylabel('N')
    ax[0].legend(loc='upper left', fontsize=8, frameon=False)
    for k, lab in ((0, 'x 沿橫桿'), (1, 'y'), (2, 'z 向上')):
        ax[1].plot(t, [r['tcp_d_mm'][k] for r in xs], lw=1.4, label=f'TCP 相對把手 Δ{lab}')
    ax[1].set_ylabel('mm（相對 OPEN 起點）')
    ax[1].legend(loc='upper left', fontsize=8, frameon=False)
    for k, lab in ((0, 'roll（繞橫桿）'), (1, 'pitch'), (2, 'yaw')):
        ax[2].plot(t, [r['tcp_d_rpy_deg'][k] for r in xs], lw=1.4, label=f'RPY 參數變化 {lab}')
    ax[2].set_ylabel('°（相對 OPEN 起點）')
    ax[2].set_xlabel('模擬時間 s')
    ax[2].legend(loc='upper left', fontsize=8, frameon=False)
    for a in ax:
        for tt, c in ((t_open, '#444'), (t_low, '#b33'), (t_sust, '#b33'), (t_abort, '#000')):
            if tt:
                a.axvline(tt, color=c, lw=0.7, ls='--')
        a.grid(alpha=0.25)
    sec = ax[0].secondary_xaxis('top', functions=(lambda x: np.interp(x, t, q), lambda y: np.interp(y, q, t)))
    sec.set_xlabel('門角 °')
    fig.suptitle('door_wide_w1b_30_1：兩指承載與夾爪相對把手位姿（虛線：OPEN、finger1 首次／持續低於 0.5 N、中止）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'w1b_30_1_承載與相對位姿_r2.png'), dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for f, c in (('uflite_finger1', '#2a6fdb'), ('uflite_finger2', '#d9822b')):
        pts = [(r['q_deg'], r[f]['load_point_mm']) for r in xs if r[f]['load_point_mm'] is not None and r['t'] < t_abort]
        ax[0].plot([p[0] for p in pts], [p[1][0] for p in pts], color=c, lw=1.2, label=f'{f[7:]} 承載點 x（沿橫桿）')
        ax[1].plot([p[0] for p in pts], [p[1][2] for p in pts], color=c, lw=1.2, label=f'{f[7:]} 承載點 z')
    for a, yl in ((ax[0], 'mm（把手座標 x）'), (ax[1], 'mm（把手座標 z）')):
        a.set_xlabel('門角 °')
        a.set_ylabel(yl)
        a.legend(fontsize=8, frameon=False)
        a.grid(alpha=0.25)
    fig.suptitle('承載點位置（只取 |fn| > 0.05 N 的接觸點，以法向力加權）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'w1b_30_1_承載點位置_r2.png'), dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
