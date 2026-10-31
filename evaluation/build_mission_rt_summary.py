#!/usr/bin/env python3
"""第七次素材：完整往返（mission_rt_*）彙總與圖（只讀 evaluation/runs/ 紀錄；不跑模擬）。數字由紀錄現算，不手抄。

    python3 evaluation/build_mission_rt_summary.py

輸出：
  第七次進度報告素材/資料/mission_rt_summary.json   各趟結束碼、第一阻點（失敗趟）、成功趟的分段時刻與末態
  第七次進度報告素材/圖表/34_完整往返_各趟時間組成.png
  第七次進度報告素材/圖表/35_完整往返_各包局部縮時.png（局部改善／時長對照）
"""
from __future__ import annotations

import ast
import json
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402
import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_ws(d):
    """往上找含 evaluation/runs 的工作區根（本檔在 evaluation/ 或素材「產生程式/」副本皆可直接執行）。"""
    while True:
        if os.path.isdir(os.path.join(d, 'evaluation', 'runs')):
            return d
        if os.path.dirname(d) == d:
            raise SystemExit('找不到含 evaluation/runs 的工作區根')
        d = os.path.dirname(d)


WS = _find_ws(HERE)
RUNS = os.path.join(WS, 'evaluation', 'runs')
MAT = os.path.join(WS, '第七次進度報告素材')
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
CAT = ['#898781', '#2a78d6', '#eb6834', '#1baf7a', '#8a5cd0', '#d4a72c']   # 起跑（中性灰）＋五段，固定順序
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'font.size': 10, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 160, 'savefig.facecolor': SURF, 'axes.unicode_minus': False})
HOME = (-0.6, -2.8, math.pi / 2)


def rd(p):
    try:
        return open(p, errors='ignore').read()
    except OSError:
        return ''


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def phases(txt, pat=r'相位 → (\S+) @ sim ([\d.]+)'):
    out = {}
    for k, v in re.findall(pat, txt):
        out.setdefault(k, float(v))
    return out


def merged(R):
    m = jl(os.path.join(RUNS, R, 'room_run_mission_merged.json'))
    if m is None:
        return None
    I = {k: i for i, k in enumerate(m['steps_cols'])}
    T = m['steps']
    return (np.array([x[I['sim_t']] for x in T]), np.array([x[I['base_xyth']] for x in T], float),
            np.array([x[I['owner(0=nav,1=wb,2=glide)']] for x in T]), np.degrees([x[I['door_q_rad']] for x in T]),
            np.array([x[I['opening_m']] for x in T], float))


def ret_events(R, sg):
    d = jl(os.path.join(RUNS, R, f'task_return_{sg}.json'))
    if d is None:
        return None, None
    ev = {}
    for e in d['events']:
        k = e.get('to') or e.get('gate')
        if k:
            ev.setdefault(k, e['t'])
    return ev, d


def back_angle_s(t, b, a, c):
    """[a, c) 內移動方向與車頭夾角 >90° 且 v>0.01 的時間（0.1 s 位姿差分）。"""
    k = 10
    v = np.hypot(*(b[k:, :2] - b[:-k, :2]).T) / (t[k:] - t[:-k])
    mv = np.arctan2(b[k:, 1] - b[:-k, 1], b[k:, 0] - b[:-k, 0])
    ang = np.degrees(np.abs(np.arctan2(np.sin(mv - b[k:, 2]), np.cos(mv - b[k:, 2]))))
    tt = t[k:]
    sel = (tt >= a) & (tt < c) & (v > 0.01) & (ang > 90)
    return float(sel.sum() * np.median(np.diff(t)))


def first_blocker(D):
    """失敗趟第一阻點：返程段 ABORTED 的 why → 抽屜執行端停止世界 → 抽屜任務中止 → 末端對位未完成。"""
    for sg in ('A', 'B'):
        d = jl(os.path.join(D, f'task_return_{sg}.json'))
        if d and d.get('result') == 'ABORTED':
            return f'返程 {sg}：{d.get("why")}'
    m = re.findall(r'\[drawer\] \*\*(閉爪斜坡完成後[^*]+)\*\*（([^）]*)）', rd(os.path.join(D, 'sim.log')))
    if m:
        return f'抽屜：{m[0][0]}（{m[0][1]}）'
    m = re.findall(r'\*\*中止：(\S+)\*\*', rd(os.path.join(D, 'task_d.log')))
    if m:
        return f'抽屜：{m[0]}'
    m = re.findall(r'末端對位 \*\*未完成\*\* ([^\n]+)', rd(os.path.join(D, 'nav_align.log')))
    if m:
        return f'末端對位未完成：{m[0][:80]}'
    return None


def run_summary(R):
    D = os.path.join(RUNS, R)
    log = rd(os.path.join(D, 'run.log'))
    m = re.findall(r'=== 結束 rc=(\d+)[^\n]*', log)
    rc = int(m[-1]) if m else None
    s = {'run': R, 'rc': rc}
    if rc != 0:
        s['end_line'] = (re.findall(r'=== 結束 rc=\d+[^\n]*', log) or [''])[-1][:300]
        s['first_blocker'] = first_blocker(D)
        return s
    M = merged(R)
    if M is None:
        return s
    t, b, own, q, op = M
    door = phases(rd(os.path.join(D, 'task.log')))
    drw = phases(rd(os.path.join(D, 'task_d.log')), r'相位 \S+ → \*\*(\S+)\*\* @ sim ([\d.]+)')
    evA, dA = ret_events(R, 'A')
    evB, dB = ret_events(R, 'B')
    nv = jl(os.path.join(D, 'nav1.json')) or {}
    cruise = [x['t'] for x in nv.get('segments', []) if x['seg'] == 'CRUISE']
    ho = [x['t'] for x in nv.get('segments', []) if x['seg'] == 'HANDED_OVER']
    al = jl(os.path.join(D, 'nav_align.json')) or {}
    e = b[-1]
    s.update(t_end=float(t[-1]), t_nav_start=cruise[0] if cruise else None, door=door, drawer=drw, A=evA, B=evB,
             home_err_mm=float(math.hypot(e[0] - HOME[0], e[1] - HOME[1]) * 1e3), home_yaw_err_rad=float(wrap(e[2] - HOME[2])),
             door_final_deg=float(q[-1]), drawer_final_mm=float(op[-1] * 1e3),
             warn={'A': dA['monitor']['n_warn'], 'B': dB['monitor']['n_warn']}, stop={'A': dA['monitor']['n_stop'], 'B': dB['monitor']['n_stop']},
             nonfinger_min_mm={'A': dA['monitor']['pred_min']['nonfinger'] * 1e3, 'B': dB['monitor']['pred_min']['nonfinger'] * 1e3},
             nav_align=[al.get('t0'), al.get('t1')], nav1_handed_over=ho)
    g = (jl(os.path.join(D, 'room_run_drawer.json')) or {}).get('return_gate') or {}
    gr = {}
    for x in g.get('events', []):
        if x.get('to') in ('CLOSING', 'GRASPED'):
            gr.setdefault(x['op'], {})[x['to']] = x['t']
    s['grasp_s'] = {('A' if k == 1 else 'B'): v['GRASPED'] - v['CLOSING'] for k, v in gr.items() if 'GRASPED' in v}
    # 五段（＋起跑）
    t0 = s['t_nav_start'] or 0.0
    seg = [('起跑', 0.0, t0), ('門段', t0, door.get('DONE')), ('抽屜段', door.get('DONE'), drw.get('NAVIGATE_HOME')),
           ('返程 A', drw.get('NAVIGATE_HOME'), evA.get('TO_NAV')), ('返程 B', evA.get('TO_NAV'), evB.get('TO_NAV')), ('回家', evB.get('TO_NAV'), float(t[-1]))]
    s['segments'] = [(n, a, c) for n, a, c in seg]
    # 局部量
    loc = {}
    if 'RG4' in door:                                                     # R1 交導航（RG4 後首次控制者＝導航）前 3 s 到後 6 s
        kn = np.where((t > door['RG4']) & (own == 0))[0]
        if len(kn):
            loc['R1R2_back_s'] = back_angle_s(t, b, t[kn[0]] - 3.0, t[kn[0]] + 6.0)
    if len(ho) >= 2 and 'RG5' in door:
        loc['R3_s'] = door['RG5'] - ho[1]
    if 'RG7' in door and 'RG8' in door:
        loc['RG7_s'] = door['RG8'] - door['RG7']
    if evA and 'MOVING' in evA and 'OPENING' in evA:
        loc['A_S4_s'] = evA['OPENING'] - evA['MOVING']
    if al.get('t0') is not None and al.get('t1') is not None:
        loc['nav_align_s'] = al['t1'] - al['t0']
    loc['start_s'] = t0
    tj = jl(os.path.join(D, 'task.json')) or {}
    try:
        st = tj['k3m_pause']['last_status']
        o = ast.literal_eval(st) if isinstance(st, str) else st
        loc['obs_to_mode2_s'] = float(o['outcome']['checks']['2']['t_commit']) - float(o['outcome']['transitions']['OBS_PASSED'])
    except (KeyError, TypeError, ValueError, SyntaxError):
        pass
    rl = rd(os.path.join(D, 'relay_d.log'))
    a_ = re.findall(r'收到 subtask_done[^@]*@ sim ([\d.]+)', rl)
    c_ = re.findall(r'開放輸出 @ sim ([\d.]+)', rl)
    if a_ and c_:
        loc['door_to_drawer_handover_s'] = float(c_[0]) - float(a_[0])
    s['local'] = loc
    return s


def fig34(S):
    ok = [s for s in S if s.get('rc') == 0 and s.get('segments')]
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(ok) + 1.6))
    names = [n if n in ('起跑', '回家') else n + '區間' for n, _, _ in ok[0]['segments']]
    for i, s in enumerate(ok):
        y = len(ok) - 1 - i
        for j, (n, a, c) in enumerate(s['segments']):
            if a is None or c is None:
                continue
            ax.barh(y, c - a, left=a, height=0.62, color=CAT[j], edgecolor=SURF, linewidth=1.2)
            if s is ok[-1] and c - a > 25:
                ax.text(a + (c - a) / 2, y, f'{c - a:.0f}', ha='center', va='center', fontsize=7.5, color='white')
        ax.text(s['t_end'] + 6, y, f"{s['t_end']:.1f} s", va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels([s['run'].replace('mission_', '') for s in ok][::-1], fontsize=8)
    ax.set_xlabel('模擬時間（s）')
    ax.set_xlim(0, max(s['t_end'] for s in ok) * 1.09)
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Patch(color=CAT[j], label=n) for j, n in enumerate(names)], ncol=6, frameon=False, loc='lower left',
              bbox_to_anchor=(0.0, 1.0), fontsize=8.5)
    ax.set_title('完整往返各趟任務區間組成（rc 0 趟；配置逐包變動，n＝1／趟，不合併為可靠性）', loc='left', fontsize=10.5, pad=26)
    fig.text(0.01, -0.06, '各區間含導航、進場、交接與操作（不是純操作時間）。起跑＝模擬開始→導航起步；門段＝→門段 DONE；抽屜段＝→抽屜交回導航；返程 A／B＝→各段交回導航；回家＝→本趟結束（含末端對位，非 GMPC）。'
             '來源：evaluation/runs/mission_rt_*（task.log、task_d.log、task_return_*.json、room_run_mission_merged.json）。', fontsize=7.5, color=INK2, wrap=True)
    return fig


ITEMS = [('R1R2_back_s', 'R1→R2 移動方向與車頭夾角 >90° 時間', 'mission_rt_35', 'mission_rt_36'),
         ('R3_s', 'R3（R2 交回全身→RG5）', 'mission_rt_36', 'mission_rt_37'),
         ('start_s', '起跑（模擬開始→導航起步）', 'mission_rt_38', 'mission_rt_39'),
         ('obs_to_mode2_s', 'PAUSE：OBS_PASSED→mode 2 提交', 'mission_rt_39', 'mission_rt_40'),
         ('A_S4_s', '返程 A 推開（MOVING→OPENING）', 'mission_rt_40', 'mission_rt_41'),
         ('RG7_s', '去程穿門拉關（RG7→RG8）', 'mission_rt_41', 'mission_rt_42'),
         ('nav_align_s', '回家末端對位（非 GMPC）', 'mission_rt_42', 'mission_rt_43')]


def fig35(S):
    by = {s['run']: s for s in S}
    rows = []
    for k, lab, r0, r1 in ITEMS:
        a = (by.get(r0) or {}).get('local', {}).get(k)
        c = (by.get(r1) or {}).get('local', {}).get(k)
        rows.append((lab, r0, r1, a, c))
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(rows) + 1.4))
    for i, (lab, r0, r1, a, c) in enumerate(rows):
        y = len(rows) - 1 - i
        if a is None or c is None:
            ax.text(0, y, '（紀錄缺欄位，未畫）', va='center', fontsize=8, color=MUTED)
            continue
        ax.plot([c, a], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter([a], [y], s=46, color='#898781', zorder=2, edgecolor=SURF, linewidth=1.5)
        ax.scatter([c], [y], s=46, color='#2a78d6', zorder=3, edgecolor=SURF, linewidth=1.5)
        ax.text(max(a, c) + 1.2, y, f'{a:.2f}→{c:.2f} s（{r0[8:]}→{r1[8:]}）', va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlabel('局部時長（s，模擬時間）')
    ax.set_xlim(0, max(max(r[3] or 0, r[4] or 0) for r in rows) * 1.45)
    ax.grid(axis='y', visible=False)
    ax.scatter([], [], s=40, color='#898781', label='改動前一趟')
    ax.scatter([], [], s=40, color='#2a78d6', label='改動後一趟')
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    ax.set_title('各包局部改善／時長對照（相鄰兩趟實測，各 n＝1；局部量不等於整趟差；R1 列為反向時間，非提速）', loc='left', fontsize=10.5)
    return fig


def main():
    runs = sorted((d for d in os.listdir(RUNS) if re.fullmatch(r'mission_rt_\d+', d)), key=lambda x: int(x.split('_')[-1]))
    S = [run_summary(R) for R in runs if int(R.split('_')[-1]) >= 21]
    os.makedirs(os.path.join(MAT, '資料'), exist_ok=True)
    out = os.path.join(MAT, '資料', 'mission_rt_summary.json')
    json.dump({'source': 'evaluation/runs/mission_rt_*', 'generator': 'evaluation/build_mission_rt_summary.py', 'runs': S},
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫出', out)
    for name, f in (('34_完整往返_各趟時間組成.png', fig34), ('35_完整往返_各包局部縮時.png', fig35)):
        fig = f(S)
        p = os.path.join(MAT, '圖表', name)
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        print('寫出', p)
    for s in S:
        if s['rc'] == 0 and s.get('t_end'):
            print(s['run'], f"{s['t_end']:.1f} s", f"home {s['home_err_mm']:.2f} mm／{s['home_yaw_err_rad']:+.4f} rad", f"門 {s['door_final_deg']:.3f}°",
                  'warn', s['warn'], {k: round(v, 2) for k, v in s['local'].items()})
        else:
            print(s['run'], 'rc', s['rc'], s.get('first_blocker'))


if __name__ == '__main__':
    main()
