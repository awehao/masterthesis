#!/usr/bin/env python3
"""主線 B：沿導航自然門前線（sb≈0.46，往 −a）接近時，底盤朝向排程 δ(a)（相對面門朝向，逆時針為正＝轉向行進方向 −x）讓手臂全程在門旁分支上追把手。
由接手點（a_w、δ＝0、r＝S_PRE）沿門前線往回延拓（a 增加）：每點在 δ≥前一點、r≥前一點中取最小可行（同分支 Δq＜0.3、σ≥下限、限位、淨空規劃門檻），
往回 δ 與 r 只能不減（往前走只會轉回面門、靠近把手）；記相機可見、δ 的變化率（以 V 換算 deg/s）。運動學層離線。
    python3 door_yaw_schedule.py RUN_DIR OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
SB, A_W = 0.46, -0.10
D_GRID = [math.radians(x) for x in range(0, 91, 3)]
R_GRID = np.round(np.arange(TS.S_PRE, 0.451, 0.02), 3)


def base(a, dy):
    b = TS.base(a, SB); b[2] += dy; return b


res = {}
for v_app in (0.05, 0.10):
    seed, d_prev, r_prev, rows, fail = TS.end_config(A_W, SB), 0.0, TS.S_PRE, [], None
    for a in np.round(np.arange(A_W, 0.80 + 1e-9, 0.01), 3):
        got = None
        for dy in [x for x in D_GRID if x >= d_prev - 1e-9]:
            for r in R_GRID[R_GRID >= r_prev - 1e-9]:
                b = base(a, dy)
                q, sm = TS.solve(b, float(r), seed)
                if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
                    continue
                lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
                cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
                if lim < 0 or any(cl[k] < TS.THRP[k] for k in ('env', 'base', 'self')):
                    continue
                got = (dy, float(r), q, float(sm), lim, cl, bool(TS.visible(np.r_[b, q])))
                break
            if got is not None:
                break
        if got is None:
            fail = dict(a=float(a), d_from_deg=round(math.degrees(d_prev), 1), r_from=r_prev); break
        dy, r, q, sm, lim, cl, vis = got
        rows.append(dict(a=float(a), dyaw_deg=round(math.degrees(dy), 1), r=r, sigma=round(sm, 4), lim=round(lim, 3), self=round(float(cl['self']), 4), base=round(float(cl['base']), 4), vis=vis, q=[round(float(x), 4) for x in q]))
        seed, d_prev, r_prev = q, dy, r
    if rows:
        D = np.array([x['dyaw_deg'] for x in rows]); Q = np.array([x['q'] for x in rows])
        rate = float(np.abs(np.diff(D)).max() / (0.01 / v_app)) if len(D) > 1 else 0.0
        vis_a = [x['a'] for x in rows if x['vis']]
        res[str(v_app)] = dict(v=v_app, fail=fail, reached_a=rows[-1]['a'], dyaw_at=[(x['a'], x['dyaw_deg'], x['r'], x['vis']) for x in rows[::5]], dyaw_rate_max_deg_s=round(rate, 1),
                               sigma_min=min(x['sigma'] for x in rows), self_min=min(x['self'] for x in rows), visible_a=[min(vis_a), max(vis_a)] if vis_a else None,
                               joint_travel_max=round(float(np.abs(np.diff(Q, axis=0)).sum(axis=0).max()), 3), joint_rate_max=round(float(np.abs(np.diff(Q, axis=0)).max() / (0.01 / v_app)), 3), rows=rows)
        print('v', v_app, {k: res[str(v_app)][k] for k in ('fail', 'reached_a', 'dyaw_rate_max_deg_s', 'sigma_min', 'self_min', 'visible_a', 'joint_travel_max', 'joint_rate_max')}, flush=True)
        print('   (a, dyaw, r, vis):', res[str(v_app)]['dyaw_at'], flush=True)
    break                                                              # 排程與速度無關（只影響變化率換算）；只算一次
json.dump(dict(note='運動學層離線；沿門前線的底盤朝向排程＋同分支追蹤', sb=SB, a_w=A_W, res=res), open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
