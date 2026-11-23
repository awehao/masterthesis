#!/usr/bin/env python3
"""主線 B1（Codex 20261117）：沿原導航 Smac2D 的**實際路徑**（兩段：起點→入口 E、E→終點 G）評估移動交權點。
對第 2 段上每個候選接手點 P_t（間隔 2 cm，且離 G 的弧長 ≥ 0.45 m：GMPC 到達容差 0.30 m 前不會停）：
  1 窗口：方向 d_t＝路徑在 P_t 的切向（固定方向，同 MovingGraspCtrl 模型），以 V＝0.05 m/s 走 2.2 s；TCP 後退由 S_PRE 插到 0；逐點同分支、σ、限位、淨空（規劃門檻）、Shield（命令＝d_t·V）。
  2 往回延拓：由 P_t 沿實際路徑往回（1 cm 取樣），同分支、後退量 r 單調（往回不減、≤0.40）、σ、限位、淨空；r 首次到 0.33（追蹤起點）處記弧長 s_trk；
     途中逐點記相機可見與 Shield（命令＝切向×速度剖面：v_nav 至 P_t 前 1.0 m 線性降到 V）。追蹤段內不換分支、不偏轉姿態（B2 不採）。
  3 路徑轉角：P_t 前 0.3 m 內切向變化（接手前不得仍在轉向）。
輸出可行接手點與成本（追蹤段長、關節行程、σ／自碰撞／Shield 最小值、可見到哪個 r、P_t 前轉角）。運動學層離線，路徑為實際 Smac 規劃（use_start 探測）。
    python3 door_b1_eval.py RUN_DIR PROBE_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
V, V_NAV, T_WIN = 0.05, 0.15, 2.2
YAW = TS.yaw


def to_door(p):
    rel = np.asarray(p, float) - TS.h_xy
    return float(-(rel @ TS.d)), float(rel @ TS.n_)


def resample(P, ds=0.01):
    P = np.asarray(P, float)
    keep = [P[0]]
    for p in P[1:]:
        if np.linalg.norm(p - keep[-1]) > 1e-6:
            keep.append(p)
    P = np.array(keep)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    ss = np.arange(0.0, s[-1] + 1e-9, ds)
    return np.c_[np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], ss


def tangent(P, i, w=3):
    a, b = max(i - w, 0), min(i + w, len(P) - 1)
    t = P[b] - P[a]
    return t / max(np.linalg.norm(t), 1e-9)


def bvel(tw, v):
    c, s = math.cos(YAW), math.sin(YAW)
    vw = tw * v
    return np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], 0.0])


def window(P_t, d_t, q0):
    seed, out = q0.copy(), []
    for k in range(23):
        p = P_t + d_t * V * 0.1 * k
        r_ = TS.S_PRE * max(0.0, 1.0 - k / 12)
        b = np.r_[p, YAW]
        q, sm = TS.solve(b, r_, seed)
        if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
            return None, dict(k=k, why='ik_or_sigma')
        lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
        cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
        if lim < 0 or any(cl[k_] < TS.THRP[k_] for k_ in ('env', 'base', 'self')):
            return None, dict(k=k, why='lim_or_clear')
        shm, _ = TS.SH.margin(b, bvel(d_t, V), 0.0)
        if shm < 0:
            return None, dict(k=k, why='shield', m=round(shm, 4))
        out.append((sm, lim, cl['self'], shm))
        seed = q
    return dict(sigma=min(x[0] for x in out), lim=min(x[1] for x in out), self=min(x[2] for x in out), shield=min(x[3] for x in out)), None


def backward(P, ss, i_t, q_t):
    seed, r_prev, rows = q_t.copy(), TS.S_PRE, []
    s_slow = ss[i_t] - 1.0
    R = np.round(np.arange(TS.S_PRE, 0.4001, 0.01), 3)
    for i in range(i_t, -1, -1):
        b = np.r_[P[i], YAW]
        got = None
        for r in R[R >= r_prev - 1e-9]:
            q, sm = TS.solve(b, float(r), seed)
            if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
                continue
            lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
            cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
            if lim < 0 or any(cl[k_] < TS.THRP[k_] for k_ in ('env', 'base', 'self')):
                continue
            got = (float(r), q, sm, lim, cl)
            break
        if got is None:
            return rows, dict(i=i, s=round(float(ss[i]), 3), why='no_r', r_from=r_prev, a_sb=[round(x, 3) for x in to_door(P[i])])
        r, q, sm, lim, cl = got
        v = V if ss[i] >= s_slow + 1.0 else (V_NAV + (V - V_NAV) * max(0.0, ss[i] - s_slow))
        shm, _ = TS.SH.margin(b, bvel(tangent(P, i), v), 0.0)
        rows.append(dict(i=i, s=float(ss[i]), r=r, sigma=float(sm), lim=lim, self=float(cl['self']), shield=shm, vis=bool(TS.visible(np.r_[b, q])), q=q))
        seed, r_prev = q, r
        if r >= 0.33 - 1e-9:
            return rows, None
    return rows, dict(why='path_start')


P_ = json.load(open(sys.argv[2]))
legs1 = {tuple(np.round(l['goal'][:2], 4)): l['plan'] for l in P_ if abs(l['start'][0] - 1.5) < 1e-6}
res = []
for l in P_:
    if abs(l['start'][0] - 1.5) < 1e-6:
        continue
    E = tuple(np.round(l['start'][:2], 4))
    if E not in legs1:
        continue
    path, ss = resample(legs1[E] + l['plan'])
    s_E = float(resample(legs1[E])[1][-1])
    G = l['goal'][:2]
    cands = [i for i in range(len(path)) if ss[i] >= s_E and ss[-1] - ss[i] >= 0.45 and i % 2 == 0]
    for i_t in cands:
        P_t, d_t = path[i_t], tangent(path, i_t)
        a_t, sb_t = to_door(P_t)
        q_t = TS.end_config(a_t, sb_t)
        rec = dict(E=list(E), G=G, i_t=i_t, s_t=round(float(ss[i_t]), 3), P_t=[round(float(x), 4) for x in P_t], a_sb=[round(a_t, 3), round(sb_t, 3)], d_t=[round(float(x), 3) for x in d_t])
        if q_t is None:
            rec.update(ok=False, why='end_ik'); res.append(rec); continue
        w, wf = window(P_t, d_t, q_t)
        if w is None:
            rec.update(ok=False, why='window', detail=wf); res.append(rec); continue
        rows, bf = backward(path, ss, i_t, q_t)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(tangent(path, max(i_t - 30, 0)) @ d_t)))))
        if bf is not None:
            rec.update(ok=False, why='backward', detail=bf, tracked_m=round(float(ss[i_t] - rows[-1]['s']), 3) if rows else 0.0, window={k: round(v, 4) for k, v in w.items()}); res.append(rec); continue
        Q = np.array([x['q'] for x in rows])
        vis_r = [x['r'] for x in rows if x['vis']]
        rec.update(ok=True, window={k: round(v, 4) for k, v in w.items()}, track_len_m=round(float(ss[i_t] - rows[-1]['s']), 3), sigma_min=round(min(x['sigma'] for x in rows), 4),
                   self_min=round(min(x['self'] for x in rows), 4), shield_min=round(min(x['shield'] for x in rows), 4), visible_down_to_r=round(min(vis_r), 3) if vis_r else None,
                   joint_travel_max=round(float(np.abs(np.diff(Q, axis=0)).sum(axis=0).max()), 3) if len(Q) > 1 else 0.0, turn_last_0p3m_deg=round(turn, 1),
                   track_start_xy=[round(float(x), 3) for x in path[rows[-1]['i']]], q_track_start=[round(float(x), 4) for x in rows[-1]['q']])
        res.append(rec)
        print('OK', rec['E'], rec['G'], 'P_t', rec['P_t'], rec['a_sb'], 'd', rec['d_t'], 'track', rec['track_len_m'], 'sig', rec['sigma_min'], 'self', rec['self_min'], 'vis_r', rec['visible_down_to_r'], 'turn', rec['turn_last_0p3m_deg'], flush=True)
ok = [r for r in res if r['ok']]
fails = {}
for r in res:
    if not r['ok']:
        k = r['why'] + ':' + str((r.get('detail') or {}).get('why'))
        fails[k] = fails.get(k, 0) + 1
json.dump(dict(note='運動學層離線；路徑＝實際 Smac2D（ComputePathToPose use_start 探測）；把手＝far_door_9 近距鎖定參考', n=len(res), n_ok=len(ok), fails=fails, ok=ok, all=res), open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
print('n', len(res), 'ok', len(ok), fails)
