#!/usr/bin/env python3
"""主線 B（Codex 20261117：連續接近須從導航中的實際手臂構型開始核）：導航姿態 K2_READY（重力靜態後的實測值同遠距乾跑起點）→ 走廊入口的追蹤分支構型。
做法：底盤沿「導航來向」直線以 v_nav 走到走廊入口（最後 T_s 秒內），手臂同時由導航姿態以關節空間最小急動內插到入口追蹤構型（door_turn_search 最佳路徑的起點 q_start）；
逐點核：限位、淨空（規劃門檻；env／base／self）、關節速率／加速度上限（V_ARM／A_ARM）、腕部相機在內插後段（入口前 0.5 s 起）看得到把手（只記錄）。運動學層離線，不是執行動作。
    python3 evaluation/door_nav2track_check.py RUN_DIR TURN_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
T = json.load(open(sys.argv[2]))
best = T['best'][0]
q_track = np.array(best['q_start'], float)
b_entry = np.array(best['base_start'], float)
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])   # 遠距乾跑起點（K2_READY＋重力靜態偏置的實測值）
V_ARM, A_ARM = float(TS.W.V_ARM), float(TS.W.A_ARM)
res = []
for (T_s, v_nav, came_from_deg) in ((2.0, 0.15, 0.0), (3.0, 0.15, 0.0), (2.0, 0.15, -30.0), (3.0, 0.10, 30.0)):
    th = math.radians(came_from_deg)                                 # 導航來向：入口切向（朝門 −sb，偏 th）往回
    dirw = -(np.cos(th) * (-TS.n_) + np.sin(th) * (-TS.d))           # 世界座標：由入口往回的方向
    N = 60
    out, bad = [], None
    for k in range(N + 1):
        u = k / N
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5                   # 最小急動
        q = Q_NAV + s * (q_track - Q_NAV)
        bxy = b_entry[:2] + dirw * v_nav * T_s * (1 - u)
        b = np.r_[bxy, b_entry[2]]
        lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
        cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
        r_ = [k_ for k_ in ('env', 'base', 'self') if cl[k_] < TS.THRP[k_]]
        vis = TS.visible(np.r_[b, q]) if u >= 1 - 0.5 / T_s else None
        out.append(dict(u=round(u, 3), lim=lim, env=cl['env'], base=cl['base'], self=cl['self'], vis=vis))
        if lim < -0.05 or r_:
            bad = dict(u=round(u, 3), lim=round(lim, 3), clear_bad=r_, who=cl['who'])
            break
    sdot_max, sddot_max = 1.875 / T_s, 5.7735 / T_s ** 2                # 最小急動的最大速度／加速度係數
    dq = np.abs(q_track - Q_NAV)
    res.append(dict(T_s=T_s, v_nav=v_nav, came_from_deg=came_from_deg, ok=bad is None, fail=bad, joint_rate_max=round(float(dq.max() * sdot_max), 3), joint_acc_max=round(float(dq.max() * sddot_max), 3),
                    rate_ok=bool(dq.max() * sdot_max <= V_ARM), acc_ok=bool(dq.max() * sddot_max <= A_ARM), self_min=round(min(x['self'] for x in out), 4), env_min=round(min(x['env'] for x in out), 4),
                    lim_min=round(min(x['lim'] for x in out), 3), visible_last_0p5s=[x['vis'] for x in out if x['vis'] is not None][-3:]))
    print(res[-1], flush=True)
json.dump(dict(note='運動學層離線；導航姿態→走廊入口追蹤構型的關節空間最小急動內插，底盤同時沿導航來向直線進入口', q_nav=Q_NAV.tolist(), q_track=q_track.tolist(), base_entry=b_entry.tolist(), joint_delta=[round(float(x), 3) for x in (q_track - Q_NAV)], results=res), open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
