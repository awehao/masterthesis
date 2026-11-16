#!/usr/bin/env python3
"""主線 B（Codex 20261117_022552：換抓法與最佳底盤位置一起選）：把手抓取繞橫桿軸旋轉 ψ 的抓法族——
T_HG(ψ)＝Rx(ψ)·T_HG、a_H(ψ)＝Rx(ψ)·a_H（把手座標 x＝橫桿軸；手指仍垂直橫桿閉合）；T_WE＝T_WH·Trans(s·a_H(ψ))·T_HG(ψ)。
對每個 ψ（底盤沿原導航 Smac 門前線 sb≈0.46、朝向＝面門）：
  1 窗口：接手點 a_w 沿門 −a、V＝0.05、2.2 s（插入 S_PRE→0）：同分支、σ、限位、淨空（規劃門檻，含夾爪殼—橫桿）、Shield。
  2 由接手點沿門前線往回（a 增加）延拓同分支、後退量 r 單調不減（≤0.45）：能連續追到多遠（是否穿過原奇異帶 a≈0～0.02）、
    哪一段相機看得到把手（中心與兩端都在影像內），最後可見時的 r。
  3 盲段估計（目前假設下）：由「往前走時最後可見點」到 READY 的時間（底盤 V、抓持排程 2.05 s；不重複計入插入）。
運動學層離線；把手＝far_door_9 近距鎖定參考；分支由 ψ＝0 的門旁種子逐 5° 連續轉到 ψ。
    python3 door_grasp_variants.py RUN_DIR OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
SB, V, T_READY = 0.46, 0.05, 2.05
T_HG0, A_H0 = np.asarray(TS.T_HG, float), np.asarray(TS.a_H, float)


def Rx(psi):
    c, s = math.cos(psi), math.sin(psi)
    R = np.eye(4); R[1:3, 1:3] = [[c, -s], [s, c]]
    return R


def target(psi, r):
    a = Rx(psi)[:3, :3] @ A_H0
    Tt = np.eye(4); Tt[:3, 3] = r * a
    return TS.T_H @ Tt @ Rx(psi) @ T_HG0


def solve(b, psi, r, seed):
    q, res, sm = TS.DFK.ik_closure_fc(target(psi, r), b, seed)
    if not (res < 1e-6 and np.isfinite(q).all()):
        return None, None
    return q, sm


def ok_cfg(b, q):
    lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
    cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
    return lim >= 0 and all(cl[k] >= TS.THRP[k] for k in ('env', 'base', 'self', 'shell_bar')), lim, cl


def pregrasp_branch(psi, a, sb):
    sd = TS.end_config(-0.115, 0.45)
    if sd is None:
        return None
    n = max(1, int(abs(math.degrees(psi)) / 5))
    for k in range(1, n + 1):
        sd, _ = solve(TS.base(-0.115, 0.45), psi * k / n, TS.S_PRE, sd)
        if sd is None:
            return None
    for k in range(1, 41):
        sd, _ = solve(TS.base(-0.115 + (a + 0.115) * k / 40, 0.45), psi, TS.S_PRE, sd)
        if sd is None:
            return None
    for k in range(1, 11):
        sd, _ = solve(TS.base(a, 0.45 + (sb - 0.45) * k / 10), psi, TS.S_PRE, sd)
        if sd is None:
            return None
    return sd


def window(psi, a_w, q0):
    seed, mins = q0.copy(), dict(sig=9.0, self=9.0, shield=9.0)
    for k in range(23):
        a_ = a_w - V * 0.1 * k
        r_ = TS.S_PRE * max(0.0, 1.0 - k / 12)
        b = TS.base(a_, SB)
        q, sm = solve(b, psi, r_, seed)
        if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
            return None, dict(k=k, why='ik_or_sigma')
        okc, lim, cl = ok_cfg(b, q)                                  # 插入段也照核（env 不含工具—橫桿；手指—橫桿接觸由抓持控制器的豁免規則處理）
        if not okc:
            return None, dict(k=k, why='clear_or_limit')
        shm, _ = TS.SH.margin(b, TS.world_vel(-V, 0.0)[0], 0.0)
        mins = dict(sig=min(mins['sig'], sm), self=min(mins['self'], float(cl['self'])), shield=min(mins['shield'], shm))
        seed = q
    return mins, None


def backward(psi, a_w, q0):
    seed, r_prev, rows = q0.copy(), TS.S_PRE, []
    R = np.round(np.arange(TS.S_PRE, 0.4501, 0.02), 3)
    for a in np.round(np.arange(a_w, 0.60 + 1e-9, 0.01), 3):
        b = TS.base(float(a), SB)
        got, first_ok = None, None
        for r in R[R >= r_prev - 1e-9]:
            q, sm = solve(b, psi, float(r), seed)
            if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
                continue
            okc, lim, cl = ok_cfg(b, q)
            if not okc:
                continue
            if first_ok is None:
                first_ok = (float(r), q, sm)
            if os.environ.get('PREFER_VIS', '1') != '1' or TS.visible(np.r_[b, q]):
                got = (float(r), q, sm)
                break
        if got is None:
            got = first_ok
        if got is None:
            return rows, dict(a=float(a), r_from=r_prev)
        rows.append(dict(a=float(a), r=got[0], sigma=got[2], vis=bool(TS.visible(np.r_[b, got[1]])), q=got[1]))
        seed, r_prev = got[1], got[0]
    return rows, None


res = []
for deg in [int(x) for x in os.environ.get('PSI_LIST', '0,-15,-30,-45,15,30,45').split(',')]:
    psi = math.radians(deg)
    rec = dict(psi_deg=deg)
    best = None
    for a_w in (-0.08, -0.10, -0.12):
        q0 = pregrasp_branch(psi, a_w, SB)
        if q0 is None:
            continue
        w, wf = window(psi, a_w, q0)
        if w is None:
            continue
        rows, bf = backward(psi, a_w, q0)
        reach = rows[-1]['a'] if rows else a_w
        vis = [x for x in rows if x['vis']]
        # 往前走（a 由大到小）時最後一個可見點：可見點中 a 最小者
        last_vis_a = min(x['a'] for x in vis) if vis else None
        blind = None if last_vis_a is None else (last_vis_a - a_w) / V + T_READY
        Q = np.array([x['q'] for x in rows]) if rows else np.zeros((0, 6))
        cand = dict(a_w=a_w, window={k: round(v, 4) for k, v in w.items()}, track_reach_a=reach, track_fail=bf, visible_a=[round(min(x['a'] for x in vis), 3), round(max(x['a'] for x in vis), 3)] if vis else None,
                    last_visible_r=None if not vis else [x['r'] for x in vis if x['a'] == last_vis_a][0], blind_est_s=None if blind is None else round(blind, 2),
                    sigma_min=round(min(x['sigma'] for x in rows), 4) if rows else None, joint_travel_max=round(float(np.abs(np.diff(Q, axis=0)).sum(axis=0).max()), 3) if len(Q) > 1 else 0.0,
                    crosses_band=bool(reach >= 0.05))
        if best is None or (cand['crosses_band'], -(cand['blind_est_s'] or 99)) > (best['crosses_band'], -(best['blind_est_s'] or 99)):
            best = cand
    rec['best'] = best
    res.append(rec)
    print('psi %+3d' % deg, None if best is None else {k: best[k] for k in ('a_w', 'track_reach_a', 'crosses_band', 'visible_a', 'last_visible_r', 'blind_est_s', 'sigma_min', 'joint_travel_max')}, flush=True)
json.dump(dict(note='繞橫桿軸旋轉抓取的抓法族；運動學層離線；盲段＝目前假設（底盤 V 0.05、抓持排程 2.05 s）下的估計，不是下限', sb=SB, res=res), open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=str)
