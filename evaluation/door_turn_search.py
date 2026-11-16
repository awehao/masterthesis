#!/usr/bin/env python3
"""主線 B（離線第 1、2 項；Codex 20261117：任務導向底盤位置選擇＋接近→沿門的連續轉向，取代候選影片的瞬間換方向）。
底盤路徑＝三次 Bézier（門座標：沿門 a、離門 sb；朝向固定＝該趟通過朝向，全向底盤只平移）：
  起點 P0＝(a0, sb0)，起始切向＝朝門（−n）偏 θ0；終點 P3＝(a_w, sb_w)＝可抓窗口起點，終點切向＝抓持時的沿門移動方向（沿門 −a，速度 v_w＝0.05 m/s）。
  控制點長度 L0、L3 決定轉向的「圓滑度」；速度剖面＝由 v0 線性降到 v_w（以弧長參數化）。
手臂：由終點（門旁分支的預抓位解）沿路徑往回 IK 延拓（同一分支、每步 Δq＜0.3 rad），TCP 後退量 r 由 r0 隨路徑進度降到 S_PRE。
逐點核：IK、限位餘裕 0.05 rad、σ_min ≥ SIG_FLOOR、淨空（規劃門檻 THR_PLAN：底盤中心到障礙 ≥0.42 m）、原 Shield 速度—距離條件（shield_geom，命令＝路徑切向×速度，門關閉）、
  r ≥0.31 時腕部相機看得到把手中心與兩端；另算路徑曲率、底盤加速度（含轉向的向心）、關節速率（依速度剖面）。
任務導向位置選擇＝在 (sb_w, a_w) 小範圍內比較「進入狀態之後的抓持窗口」綜合餘裕（σ_min、限位、淨空、Shield 殘差），取最大者；不掃滿網格。
把手估計＝指定 Isaac 趟的近距鎖定參考（實際影像，同第 1 段）；不讀真值。運動學層離線，不是物理結果。
    python3 evaluation/door_turn_search.py RUN_DIR OUT.json"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                   # noqa: E402
import door_far_link as FL, door_motm_grasp_ctrl as G, door_wbopt as W, door_clear_door_fast as _F, door_fast_kin as DFK, wrist_v0_capture as W0   # noqa: E401,E402
import shield_geom as SHG                                            # noqa: E402

W.clear_door = _F.clear_door_fast
K = G.K
RUN = sys.argv[1]
J = json.load(open(os.path.join(RUN, 'ros_merge.json')))
T_H = np.array(J['terminal']['lock']['T'], float)
Kc = np.array(json.load(open(os.path.join(RUN, 'frames_meta.json')))['K'], float)
Rg, T_HG, a_H = G.PG.baseline_reference()
FL.set_approach_axis(a_H)
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
S_PRE = float(G.PG.S_PRE)
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
THRP, SIG = W.THR_PLAN, W.SIG_FLOOR
pg = J['plan']['pg']
d = np.array(pg['d'], float); n_ = np.array(pg['n'], float); yaw = float(pg['yaw'])
h_xy = T_H[:2, 3]
a_n = T_H[:3, :3] @ np.asarray(a_H, float)
ENDS = [T_H[:3, 3], T_H[:3, 3] + T_H[:3, 0] * 0.06, T_H[:3, 3] - T_H[:3, 0] * 0.06]
DS_SEED = np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157])
SH = SHG.ShieldGeom()
V_W = 0.05                                                           # 抓持窗口沿門速度（同 MovingGraspCtrl V）
A_BODY = np.asarray(W.A_BODY, float)


def base(a, sb):
    return np.r_[h_xy + sb * n_ - a * d, yaw]


def target(r):
    T = np.asarray(T_at(T_H, 0.0), float).copy()
    T[:3, 3] += r * a_n
    return T


def visible(q9):
    Ti = np.linalg.inv(K.fk(q9, W0.OPT))
    for p in ENDS:
        pc = (Ti @ np.r_[p, 1.0])[:3]
        if pc[2] < 0.08:
            return False
        u, v = (Kc @ (pc / pc[2]))[:2]
        if not (25 <= u <= 615 and 25 <= v <= 455):
            return False
    return True


def solve(b, r, seed):
    q, res, sm = DFK.ik_closure_fc(target(r), b, seed)
    if not (res < 1e-6 and np.isfinite(q).all()):
        return None, None
    return q, sm


def world_vel(da, dsb):
    """門座標速度 (ȧ, ṡb) → 世界 → 本體（朝向 yaw）。"""
    vw = dsb * n_ - da * d
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], 0.0]), vw


def bezier(P0, P1, P2, P3, u):
    m = 1 - u
    p = m ** 3 * P0 + 3 * m * m * u * P1 + 3 * m * u * u * P2 + u ** 3 * P3
    dp = 3 * m * m * (P1 - P0) + 6 * m * u * (P2 - P1) + 3 * u * u * (P3 - P2)
    ddp = 6 * m * (P2 - 2 * P1 + P0) + 6 * u * (P3 - 2 * P2 + P1)
    return p, dp, ddp


# 門旁分支的預抓位解（同第 1 段：a＝−0.115、sb 0.45 的抓取位種子 → 後退 S_PRE → 沿線延拓）
def end_config(a_w, sb_w):
    sd, _ = solve(base(-0.115, 0.45), 0.0, DS_SEED)
    sd, _ = solve(base(-0.115, 0.45), S_PRE, sd)
    a_cur, sb_cur = -0.115, 0.45
    for k in range(1, 41):                                           # 先沿 a、再沿 sb 漸進到 (a_w, sb_w)，同一分支
        a_ = -0.115 + (a_w + 0.115) * k / 40
        sd, _ = solve(base(a_, 0.45), S_PRE, sd)
        if sd is None:
            return None
    for k in range(1, 21):
        sb_ = 0.45 + (sb_w - 0.45) * k / 20
        sd, _ = solve(base(a_w, sb_), S_PRE, sd)
        if sd is None:
            return None
    return sd


def window_margins(a_w, sb_w, q_end):
    """抓持窗口（沿門 −a 方向以 V_W 走 2.2 s＝0.11 m、TCP 由 S_PRE 插入到 0）的綜合餘裕。"""
    seed, out = q_end.copy(), []
    for k in range(23):
        a_ = a_w - V_W * 0.1 * k
        r_ = S_PRE * max(0.0, 1.0 - k / 12)
        q, sm = solve(base(a_, sb_w), r_, seed)
        if q is None or np.abs(q - seed).max() > 0.3:
            return None
        cl = W.clear_door(np.r_[base(a_, sb_w), q, 0.0], T_H)
        vb, _ = world_vel(-V_W, 0.0)
        shm, _ = SH.margin(base(a_, sb_w), vb, 0.0)
        out.append(dict(sig=sm, lim=float(min((q - LO).min(), (HI - q).min())), base=cl['base'] - THRP['base'], self=cl['self'] - THRP['self'], env=cl['env'] - THRP['env'], shield=shm))
        seed = q
    return {k: min(x[k] for x in out) for k in out[0]}


def path_check(a0, sb0, th0_deg, L0, L3, a_w, sb_w, q_end, v0, r0=0.33, N=80):
    th0 = math.radians(th0_deg)
    P0, P3 = np.array([a0, sb0]), np.array([a_w, sb_w])
    t0 = np.array([-math.sin(th0), -math.cos(th0)])                  # 起始切向：朝門（−sb）偏 θ0（往 −a 為正）
    t3 = np.array([-1.0, 0.0])                                       # 終點切向：沿門 −a
    P1, P2 = P0 + L0 * t0, P3 - L3 * t3
    us = np.linspace(0.0, 1.0, N + 1)
    pts = [bezier(P0, P1, P2, P3, u) for u in us]
    seg = np.array([np.linalg.norm(pts[i + 1][0] - pts[i][0]) for i in range(N)])
    s_cum = np.r_[0.0, np.cumsum(seg)]
    Ltot = float(s_cum[-1])
    v_of = lambda s: v0 + (V_W - v0) * s / max(Ltot, 1e-9)           # noqa: E731
    Q, info, seed = [None] * (N + 1), [None] * (N + 1), q_end.copy()
    for i in range(N, -1, -1):                                       # 由終點往回延拓
        p, dp, ddp = pts[i]
        r = r0 + (S_PRE - r0) * (s_cum[i] / Ltot)
        b = base(p[0], p[1])
        q, sm = solve(b, r, seed)
        if q is None:
            return dict(ok=False, fail=dict(i=i, why='ik'))
        if np.abs(q - seed).max() > 0.3:
            return dict(ok=False, fail=dict(i=i, why='branch_jump'))
        lim = float(min((q - LO).min(), (HI - q).min()))
        if lim < 0:
            return dict(ok=False, fail=dict(i=i, why='limit', a=round(p[0], 3), sb=round(p[1], 3)))
        if sm < SIG:
            return dict(ok=False, fail=dict(i=i, why='manip', a=round(p[0], 3), sb=round(p[1], 3), r=round(r, 3)))
        cl = W.clear_door(np.r_[b, q, 0.0], T_H)
        bad = [k for k in ('env', 'base', 'self') if cl[k] < THRP[k]]
        if bad:
            return dict(ok=False, fail=dict(i=i, why='clear:' + ','.join(bad), a=round(p[0], 3), sb=round(p[1], 3)))
        tang = dp / max(np.linalg.norm(dp), 1e-12)
        v = v_of(s_cum[i])
        vb, _ = world_vel(tang[0] * v, tang[1] * v)
        shm, _ = SH.margin(b, vb, 0.0)
        if shm < 0:
            return dict(ok=False, fail=dict(i=i, why='shield', margin=round(shm, 4), a=round(p[0], 3), sb=round(p[1], 3)))
        vis = visible(np.r_[b, q])
        if r >= 0.31 and not vis:
            return dict(ok=False, fail=dict(i=i, why='not_visible', r=round(r, 3)))
        kappa = float(abs(dp[0] * ddp[1] - dp[1] * ddp[0]) / max(np.linalg.norm(dp), 1e-12) ** 3)
        Q[i], info[i] = q, dict(sig=sm, lim=lim, shield=shm, base=cl['base'], self=cl['self'], vis=vis, kappa=kappa, v=v, r=r)
        seed = q
    Qa = np.array(Q)
    dt = np.array([seg[i] / max(0.5 * (v_of(s_cum[i]) + v_of(s_cum[i + 1])), 1e-9) for i in range(N)])
    qd = np.abs(np.diff(Qa, axis=0)) / dt[:, None]
    acc_c = max(x['kappa'] * x['v'] ** 2 for x in info)              # 向心加速度上界
    acc_t = abs(V_W ** 2 - v0 ** 2) / (2 * Ltot)                     # 切向（等減速）
    return dict(ok=True, L=round(Ltot, 3), T_s=round(float(dt.sum()), 2), sigma_min=round(min(x['sig'] for x in info), 4), lim_min=round(min(x['lim'] for x in info), 3),
                shield_min=round(min(x['shield'] for x in info), 4), base_clear_min=round(min(x['base'] for x in info), 4), self_min=round(min(x['self'] for x in info), 4),
                visible_until_r=round(min([x['r'] for x in info if x['vis']] or [np.nan]), 3), kappa_max=round(max(x['kappa'] for x in info), 2), acc_centripetal_max=round(acc_c, 3), acc_tangential=round(acc_t, 3),
                joint_rate_max=round(float(qd.max()), 3), joint_travel_max=round(float(np.abs(np.diff(Qa, axis=0)).sum(axis=0).max()), 3), q_start=[round(float(x), 4) for x in Qa[0]],
                base_start=[round(float(x), 4) for x in base(a0, sb0)], base_end=[round(float(x), 4) for x in base(a_w, sb_w)], path=[[round(float(p[0][0]), 4), round(float(p[0][1]), 4)] for p in pts[::10]])


if __name__ == '__main__':
    t0 = time.perf_counter()
    print('door frame: d', d.round(4).tolist(), 'n', n_.round(4).tolist(), 'yaw', round(yaw, 4), 'handle xy', h_xy.round(4).tolist(), flush=True)
    # 1 任務導向進入狀態：(sb_w, a_w) 小範圍比較抓持窗口綜合餘裕
    ents = []
    for sb_w in (0.45, 0.47, 0.50):
        for a_w in (-0.06, -0.075, -0.09):
            qe = end_config(a_w, sb_w)
            if qe is None:
                ents.append(dict(sb_w=sb_w, a_w=a_w, ok=False, why='end_ik')); continue
            wm = window_margins(a_w, sb_w, qe)
            if wm is None:
                ents.append(dict(sb_w=sb_w, a_w=a_w, ok=False, why='window_ik')); continue
            score = min(wm['sig'] - SIG, wm['lim'] * 0.05, wm['base'], wm['self'] * 2, wm['shield'] * 0.1)
            ents.append(dict(sb_w=sb_w, a_w=a_w, ok=bool(wm['sig'] >= SIG and wm['lim'] >= 0 and wm['base'] >= 0 and wm['self'] >= 0 and wm['shield'] >= 0), window={k: round(v, 4) for k, v in wm.items()}, score=round(score, 5), q_end=qe))
            print('entry', sb_w, a_w, ents[-1]['ok'], ents[-1]['window'], flush=True)
    okE = sorted([e for e in ents if e['ok']], key=lambda e: -e['score'])
    # 2 連續轉向路徑：對最佳 2 個進入狀態試起點與切向
    paths = []
    for e in okE[:2]:
        for sb0 in (0.75, 0.90):
            for a0 in (e['a_w'] + 0.05, e['a_w'] + 0.10, e['a_w'] + 0.20):
                for th0 in (0.0, 20.0, 40.0):
                    for (L0, L3) in ((0.15, 0.15), (0.25, 0.20)):
                        for v0 in (0.10, 0.15):
                            c = path_check(a0, sb0, th0, L0, L3, e['a_w'], e['sb_w'], e['q_end'], v0)
                            c.update(entry=dict(sb_w=e['sb_w'], a_w=e['a_w']), a0=round(a0, 3), sb0=sb0, th0=th0, L0=L0, L3=L3, v0=v0)
                            paths.append(c)
    okP = [p for p in paths if p['ok']]
    okP.sort(key=lambda p: (-min(p['sigma_min'] - SIG, p['shield_min'] * 0.1, p['base_clear_min'] - THRP['base'], p['self_min'] - THRP['self']), p['joint_travel_max']))
    fails = {}
    for p in paths:
        if not p['ok']:
            k = p['fail']['why'].split(':')[0]
            fails[k] = fails.get(k, 0) + 1
    out = dict(source=RUN, note='運動學層離線；把手＝該趟近距鎖定參考（實際影像）；Shield 條件以路徑切向×速度剖面近似命令、門關閉；不是物理結果', S_PRE=S_PRE, V_W=V_W, thr_plan={k: float(v) for k, v in THRP.items()}, sig_floor=SIG,
               entries=[{k: v for k, v in e.items() if k != 'q_end'} for e in ents], best_entry=None if not okE else {k: v for k, v in okE[0].items() if k != 'q_end'}, n_paths=len(paths), n_ok=len(okP), fail_reasons=fails,
               best=okP[:5], wall_s=round(time.perf_counter() - t0, 1))
    if okE:
        out['best_entry_q_end'] = [round(float(x), 5) for x in okE[0]['q_end']]
    json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=str)
    print('paths', len(paths), 'ok', len(okP), fails, flush=True)
    for p in okP[:5]:
        print({k: p[k] for k in ('entry', 'a0', 'sb0', 'th0', 'L0', 'L3', 'v0', 'L', 'T_s', 'sigma_min', 'lim_min', 'shield_min', 'base_clear_min', 'self_min', 'visible_until_r', 'kappa_max', 'acc_centripetal_max', 'joint_rate_max', 'joint_travel_max')})
    for p in [x for x in paths if not x['ok']][:6]:
        print('FAIL', p['entry'], p['a0'], p['sb0'], p['th0'], p['fail'])
