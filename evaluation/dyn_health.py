#!/usr/bin/env python3
"""導航動態批次逐趟健檢：dyn_health.py <批次名> [--json out.json]
把每一趟分成三類，決定要不要補跑（Howard 11/26「系統失敗要補齊趟次」）：
  INFRA  系統失敗＝和被測方法無關、跑法本身沒成立 ⇒ 該趟作廢、補跑：
         機器人整趟幾乎沒動（<1 m）、任一行人在區間內沒移動（位移 <0.3 m）、定位（地圖位姿 vs gz 真值里程）偏差 >0.5 m、
         區間內有節點非收尾性死亡、沒有 /plan。
  NAV    導航失敗＝方法本身沒到（Nav2 行為樹中止、卡住、繞圈逾時）⇒ 照實計入。
  OK     到達。
同時列出行為樹中止原因（"Goal failed" 前一則 planner／controller 警告），供揭露 Nav2 預設 BT 在 START_OCCUPIED 不做復原就中止。
需在 BIGARENA=1 SCENE=bigarena_ghost 環境下跑（dyn_summary 會自動設）。"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('BIGARENA', '1')
os.environ.setdefault('SCENE', 'bigarena_ghost')
import analyze as A                                                   # noqa: E402

BAGTAG = dict(gmpc_scan=('gmpc_cbf', 'scan'), mppi=('mppi', 'mppi'), dwb=('dwb', 'dwb'), rpp=('rpp', 'rpp'))
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def log_reasons(logf):
    """行為樹中止與中止前最後一則 planner/controller 警告；非收尾的節點死亡。"""
    if not os.path.exists(logf):
        return dict(bt_fail=None, reasons=[], early_death=[])
    lines = [ANSI.sub('', s) for s in open(logf, errors='replace')]
    fails, reasons, last_warn = 0, [], ''
    for s in lines:
        if ('[planner_server]' in s or '[controller_server]' in s) and ('WARN' in s or 'ERROR' in s) and 'Aborting handle' not in s:
            last_warn = re.sub(r'.*\]: ', '', s.strip())[:90]
        if 'Goal failed' in s:
            fails += 1
            reasons.append(last_warn)
    # 收尾：最後一次「process has died」之前若有大量死亡同時發生＝cleanup；只抓 exit code 非 1/-2/-15 且不在最後 40 行內的
    early = []
    for i, s in enumerate(lines):
        m = re.search(r'\[(\S+)\]: process has died \[pid \d+, exit code (-?\d+)', s)
        if m and m.group(2) not in ('1', '-2', '-15', '0') and i < len(lines) - 40:
            early.append(f'{m.group(1)}:{m.group(2)}')
    return dict(bt_fail=fails, reasons=reasons, early_death=early)


def one(bag, logf, goal):
    M = A.read_bag(bag)
    to, xyth = A.extract_odom(M)
    plans = M.get('/plan', [])
    out = dict(n_plan=len(plans))
    if not plans or len(to) < 10:
        out['infra'] = ['no_plan_or_odom']
        return out
    t0 = plans[0][0] / 1e9
    mt, mx = A.extract_robot_map_pose(M)[:2]
    near = np.where((mt >= t0) & (np.hypot(mx[:, 0] - goal[0], mx[:, 1] - goal[1]) <= 0.30))[0] if len(mt) else []
    t1 = mt[near[0]] if len(near) else to[-1]
    sel = (to >= t0) & (to <= t1)
    P = xyth[sel, :2]
    out['robot_path_m'] = round(float(np.sum(np.hypot(*np.diff(P, axis=0).T))), 2) if len(P) > 1 else 0.0
    # 定位：地圖位姿 vs 真值里程（gz odom 為世界座標、無漂移）
    if len(mt) > 2:
        ms = (mt >= t0) & (mt <= t1)
        if ms.sum() > 2:
            ox = np.interp(mt[ms], to, xyth[:, 0]); oy = np.interp(mt[ms], to, xyth[:, 1])
            out['loc_err_max_m'] = round(float(np.max(np.hypot(mx[ms, 0] - ox, mx[ms, 1] - oy))), 3)
    still = []
    for ts, xy, nm in A.extract_gt_obstacles(M):
        if not nm.startswith('dyn_obs'):
            continue
        s = (ts >= t0) & (ts <= t1)
        if s.sum() < 2 or float(np.ptp(xy[s, 0]) + np.ptp(xy[s, 1])) < 0.3:
            still.append(nm)
    out['still_movers'] = still
    inf = []
    if out['robot_path_m'] < 1.0:
        inf.append('robot_not_moving')
    if still:
        inf.append('mover_still:' + ','.join(still))
    if out.get('loc_err_max_m', 0) > 0.5:
        inf.append('loc_err>0.5')
    out.update(log_reasons(logf))
    if out['early_death']:
        inf.append('early_death:' + ','.join(out['early_death']))
    out['infra'] = inf
    return out


def main():
    batch = sys.argv[1]
    R = os.path.join(HERE, 'results', batch)
    G = {int(r['seed']): (float(r['goal_x']), float(r['goal_y'])) for r in csv.DictReader(open(os.path.join(HERE, 'results', 'bigarena_poses.csv')))}
    A.load_static_clearance()
    rows = []
    for f in sorted(glob.glob(os.path.join(R, '*_r*_*.csv'))):
        m = re.match(r'(\w+?)_(\w+?)_r(\d+)_(gmpc_scan|mppi|dwb|rpp)\.csv$', os.path.basename(f))
        if not m:
            continue
        sp, pt, rt, meth = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        rec = list(csv.DictReader(open(f)))
        if not rec:
            continue
        am, tg = BAGTAG[meth]
        bags = sorted(glob.glob(os.path.join(HERE, 'bags', f'{am}__{tg}_{batch}_{sp}_{pt}_r{rt}_t*_seed{rt}')))
        if not bags:
            continue
        b = bags[-1]
        logf = os.path.join(HERE, 'logs', os.path.basename(b) + '.log')
        r = dict(case=f'{sp}_{pt}_r{rt}', method=meth, bag=os.path.basename(b), success=rec[-1].get('success') == 'True')
        try:
            r.update(one(b, logf, G[rt]))
        except Exception as e:                                       # noqa: BLE001
            r['infra'] = ['analysis_error:' + repr(e)[:60]]
        r['cls'] = 'INFRA' if r['infra'] else ('OK' if r['success'] else 'NAV')
        rows.append(r)
        print('%-18s %-9s %-5s path %6s loc %6s plan %4s btfail %s %s %s' % (
            r['case'], meth, r['cls'], r.get('robot_path_m'), r.get('loc_err_max_m'), r.get('n_plan'), r.get('bt_fail'),
            ';'.join(r['infra']), ('| ' + ' / '.join(r.get('reasons') or [])) if r.get('reasons') else ''))
    n = {c: sum(1 for r in rows if r['cls'] == c) for c in ('OK', 'NAV', 'INFRA')}
    print('合計', n)
    if '--json' in sys.argv:
        json.dump(rows, open(sys.argv[sys.argv.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
