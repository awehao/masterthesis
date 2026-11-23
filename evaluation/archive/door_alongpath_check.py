#!/usr/bin/env python3
"""主線 B：沿「原導航 Smac2D 的實際路徑」（探測：門前線 y≈−0.5、往 −x）核手臂追蹤可行性與 TCP 後退量排程。
底盤沿門前線（sb 由路徑點換算）自 a＝a_start 往 −a 走到抓持窗口起點 a_w；朝向＝任務朝向（該趟通過朝向，面門）。
手臂：由窗口起點的門旁分支預抓位解（r＝S_PRE）沿路徑往回延拓；每個 a 取「同分支、σ≥下限、限位、淨空（規劃門檻）」成立的最小後退量 r（候選 S_PRE…0.40），
並要求 r(a) 沿行進方向不增（往前走只會越靠近）。逐點另記相機是否看得到把手、Shield 殘差（命令＝沿路徑 −a 方向、速度依剖面）。
輸出＝可行的 r(a) 排程、盲段（看不到把手的 a 範圍）、σ／淨空最小值。運動學層離線，不是物理結果。
    python3 door_alongpath_check.py RUN_DIR PROBE_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
P = json.load(open(sys.argv[2]))
path = np.array([p[:2] for p in P[0]['plan']], float)
# 世界 → 門座標：rel = b − h；sb＝rel·n，a＝−rel·d
rel = path - TS.h_xy
A, SB = -(rel @ TS.d), rel @ TS.n_
on = (np.abs(SB - SB[-1]) < 0.03)                                   # 門前線段
sb_line = float(np.median(SB[on]))
print('door-line sb %.4f, a range on line %.3f..%.3f' % (sb_line, A[on].min(), A[on].max()), flush=True)
a_w = -0.09
R_GRID = np.round(np.arange(TS.S_PRE, 0.4001, 0.01), 3)
q_end = TS.end_config(a_w, sb_line)
assert q_end is not None, 'end config'
a_list = np.round(np.arange(a_w, min(0.55, float(A[on].max())) + 1e-9, 0.01), 3)
seed, r_prev, rows, fail = q_end.copy(), TS.S_PRE, [], None
for a in a_list:                                                    # 由窗口往回（a 增加）；往回走 r 只能不減
    b = TS.base(a, sb_line)
    got = None
    for r in R_GRID[R_GRID >= r_prev - 1e-9]:
        q, sm = TS.solve(b, float(r), seed)
        if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
            continue
        lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
        if lim < 0:
            continue
        cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
        if any(cl[k] < TS.THRP[k] for k in ('env', 'base', 'self')):
            continue
        got = (float(r), q, sm, lim, cl)
        break
    if got is None:
        fail = dict(a=float(a), why='no_r_on_branch', r_from=r_prev); break
    r, q, sm, lim, cl = got
    vb, _ = TS.world_vel(-0.05, 0.0)
    shm, _ = TS.SH.margin(b, vb, 0.0)
    rows.append(dict(a=float(a), r=r, sigma=round(float(sm), 4), lim=round(lim, 3), self=round(float(cl['self']), 4), base=round(float(cl['base']), 4), vis=bool(TS.visible(np.r_[b, q])), shield=round(shm, 4), q=[round(float(x), 4) for x in q]))
    seed, r_prev = q, r
rows = rows[::-1]                                                   # 依行進順序（a 由大到小）
vis_a = [x['a'] for x in rows if x['vis']]
blind = [x['a'] for x in rows if not x['vis']]
Q = np.array([x['q'] for x in rows])
out = dict(note='運動學層離線；底盤沿原導航實際路徑的門前線；把手＝該趟近距鎖定參考', run=sys.argv[1], probe=sys.argv[2], sb_line=round(sb_line, 4), a_w=a_w, fail=fail, n=len(rows),
           r_schedule=[(x['a'], x['r']) for x in rows], sigma_min=min(x['sigma'] for x in rows) if rows else None, self_min=min(x['self'] for x in rows) if rows else None,
           shield_min=min(x['shield'] for x in rows) if rows else None, visible_a=[max(vis_a), min(vis_a)] if vis_a else None, blind_a=[max(blind), min(blind)] if blind else None,
           joint_travel_max=round(float(np.abs(np.diff(Q, axis=0)).sum(axis=0).max()), 3) if len(Q) > 1 else None, rows=rows)
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1)
print({k: out[k] for k in ('sb_line', 'fail', 'n', 'sigma_min', 'self_min', 'shield_min', 'visible_a', 'blind_a', 'joint_travel_max')})
print('r(a):', [(x['a'], x['r']) for x in rows[::5]])
