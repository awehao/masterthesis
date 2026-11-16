#!/usr/bin/env python3
"""主線 B：沿導航自然門前線（sb≈0.46，往 −a）接近時，手臂由導航姿態（K2_READY 實測）以關節空間最小急動在 T 秒內移到接手點的預抓位（門旁分支、r＝S_PRE），
期間基座以 V 沿門前線由 a_w＋V·T 走到 a_w（越過把手正前方的奇異帶時手臂不做 IK 追蹤，只走關節空間）。
逐點核：限位、淨空（規劃門檻 env／base／self）、關節速率／加速度上限；末段 0.5 s 記相機可見；接手點之後的抓持窗口由 door_window_map 另核。運動學層離線。
    python3 door_line_transition_check.py RUN_DIR OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])
V_ARM, A_ARM = float(TS.W.V_ARM), float(TS.W.A_ARM)
res = []
for sb in (0.46,):
    for a_w in (-0.08, -0.10, -0.12):
        q_end = TS.end_config(a_w, sb)
        for (T, V) in ((3.0, 0.05), (4.0, 0.05), (3.0, 0.03)):
            N, bad, rows = 90, None, []
            for k in range(N + 1):
                u = k / N
                s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                q = Q_NAV + s * (q_end - Q_NAV)
                a = a_w + V * T * (1 - u)
                b = TS.base(a, sb)
                lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
                cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
                bad_k = [k_ for k_ in ('env', 'base', 'self') if cl[k_] < TS.THRP[k_]]
                vis = bool(TS.visible(np.r_[b, q])) if u >= 1 - 0.5 / T else None
                rows.append(dict(u=round(u, 3), a=round(a, 3), self=float(cl['self']), env=float(cl['env']), lim=lim, vis=vis))
                if lim < -0.05 or bad_k:
                    bad = dict(u=round(u, 3), a=round(a, 3), clear_bad=bad_k, who=cl['who'], lim=round(lim, 3)); break
            dq = np.abs(q_end - Q_NAV)
            rec = dict(sb=sb, a_w=a_w, T=T, V=V, a_start=round(a_w + V * T, 3), ok=bad is None, fail=bad, joint_rate_max=round(float(dq.max() * 1.875 / T), 3), joint_acc_max=round(float(dq.max() * 5.7735 / T ** 2), 3),
                       rate_ok=bool(dq.max() * 1.875 / T <= V_ARM), self_min=round(min(x['self'] for x in rows), 4), env_min=round(min(x['env'] for x in rows), 4), vis_end=[x['vis'] for x in rows if x['vis'] is not None][-3:],
                       q_end=[round(float(x), 4) for x in q_end], joint_delta=[round(float(x), 3) for x in (q_end - Q_NAV)])
            res.append(rec)
            print({k: rec[k] for k in ('a_w', 'T', 'V', 'a_start', 'ok', 'fail', 'joint_rate_max', 'rate_ok', 'self_min', 'env_min', 'vis_end')}, flush=True)
json.dump(dict(note='運動學層離線；沿門前線關節空間轉換（導航姿態→接手點預抓位）', q_nav=Q_NAV.tolist(), results=res), open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
