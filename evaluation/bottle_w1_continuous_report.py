#!/usr/bin/env python3
"""連續展臂名目趟 vs 基線 SHROW 名目的比較（只讀紀錄）。手臂停留＝展臂開始到朝瓶開始之間，六軸實測關節速度（0.10 s 差分）全部 < 0.01 rad/s 的累計時間；
另報展臂完成時刻（候選：位置進度到 1；基線：展臂開始＋t_unfold 3.4 s 的排程）、手腕量測與鎖定、閉爪、DONE、滑移、放置偏差。
    python3 evaluation/bottle_w1_continuous_report.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '產生程式'))
import bottle_mainline_table as M                       # noqa: E402

RUNS = (('基線 SHROW', 'bottle_w1_sr_nominal1'), ('連續展臂', 'bottle_w1_cont_nominal1'))
STILL = 0.01


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    t = c['t']
    k = np.flatnonzero(np.abs(t * 10 - np.round(t * 10)) < 1e-6)
    Q = np.column_stack([c[f'q{j}'] for j in range(1, 7)])[k]
    qd = np.abs(np.diff(Q, axis=0)).max(axis=1) / 0.1
    tm = t[k][1:]
    win = (tm > ev['unfold_start']) & (tm <= ev['blend_start'])
    still = float((qd[win] < STILL).sum() * 0.1)
    moving = tm[win][qd[win] >= STILL]
    ca = r.get('continuous_approach') or {}
    unf_end = next((x[0] for x in ca.get('progress_log', []) if x[2] >= 1.0), None) if ca else ev['unfold_start'] + 3.4
    e, _ = M.metrics(run)
    sp = np.convolve(np.hypot(np.gradient(c['base_x'], t), np.gradient(c['base_y'], t)), np.ones(5) / 5, mode='same')
    wb = (t > ev['unfold_start']) & (t <= ev['blend_start'])
    return dict(run=run, unfold_start=ev['unfold_start'], unfold_end=unf_end, blend_start=ev['blend_start'], arm_still_s=still, arm_last_motion_before_blend=float(moving.max()) if len(moving) else None,
                base_near_stop_in_unfold_window_s=float((sp[wb] < M.NEAR_STOP).sum() * 0.01), vision_freeze=ev.get('vision_freeze'), wrist_fix=ev.get('wrist_fix'), wrist_lock=ev.get('wrist_lock'),
                close=ev.get('close_request'), done=e['t_done'], ok=e['ok'], slip_mm=e['slip_mm'], near_grip_s=e['near_grip_s'], near_run_s=e['near_run_s'],
                place_offset_mm=(r['place']['metrics'] or {}).get('offset', float('nan')) * 1e3, w1_groups=r['w1'].get('n_groups'), w1_goal_err=r['w1'].get('goal_err_vs_truth_mm'),
                ms=e['ms'], shield=e['shield']['active'])


def main():
    out = {nm: one(run) for nm, run in RUNS}
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_continuous_physical.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    keys = [('完成／DONE', lambda o: f"{o['ok']}／{o['done']} s"), ('展臂開始→完成', lambda o: f"{o['unfold_start']:.2f}→{o['unfold_end']:.2f} s"), ('朝瓶開始', lambda o: f"{o['blend_start']:.2f} s"),
            ('展臂開始到朝瓶間手臂停留（六軸 < 0.01 rad/s）', lambda o: f"{o['arm_still_s']:.1f} s"), ('同段底盤近停', lambda o: f"{o['base_near_stop_in_unfold_window_s']:.2f} s"),
            ('底盤相機凍結／手腕量測／鎖定／閉爪', lambda o: f"{o['vision_freeze']}／{o['wrist_fix']}／{o['wrist_lock']}／{o['close']} s"), ('手腕量測組；目標誤差（評估）', lambda o: f"{o['w1_groups']}；{o['w1_goal_err']} mm"),
            ('抓取段近停；全程近停', lambda o: f"{o['near_grip_s']:.2f}；{o['near_run_s']:.2f} s"), ('滑移；放置偏差', lambda o: f"{o['slip_mm']:.2f}；{o['place_offset_mm']:.1f} mm"),
            ('求解 p95／max／>50 ms；Shield', lambda o: f"{o['ms']['p95']:.1f}／{o['ms']['max']:.1f}／{o['ms']['over50']}；{o['shield']}")]
    print('| 項目 | ' + ' | '.join(f"{nm} {o['run']}" for nm, o in out.items()) + ' |')
    print('|---|' + '---|' * len(out))
    for nm, f in keys:
        print(f'| {nm} | ' + ' | '.join(f(o) for o in out.values()) + ' |')


if __name__ == '__main__':
    sys.exit(main())
