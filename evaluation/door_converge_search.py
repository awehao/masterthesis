#!/usr/bin/env python3
"""Howard 11/17 主線第 1 段（Codex：局部候選搜尋，不掃滿網格）：觀察 → 抓持的連續接近動作。
底盤沿「收斂直線」由較遠處（離門面 sb0、沿門位置 a0）走到預抓位（sb1＝0.45、a1）；TCP 後退量 r 隨路徑進度由 0.33 線性降到 S_PRE（0.03）。
手臂由終點（門旁抓持／開門分支的預抓位解，已驗過可接開門首段）沿路徑往回做 IK 延拓，要求同一分支連續（每步 Δq＜0.3 rad）。
逐點核：IK 殘差、限位餘裕 0.05 rad、可操作度 ≥ 下限、淨空（env／base／self）、TCP 後退量 ≥0.31 m 時腕部相機看得到把手中心與兩端。
把手估計＝指定 Isaac 趟的近距鎖定參考（實際影像），通過線方向同該趟；不讀真值。
    python3 evaluation/door_converge_search.py RUN_DIR OUT.json"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import door_far_link as FL, door_motm_grasp_ctrl as G, door_wbopt as W, door_clear_door_fast as _F, door_fast_kin as DFK, wrist_v0_capture as W0

W.clear_door = _F.clear_door_fast
K = G.K
RUN = sys.argv[1]
J = json.load(open(os.path.join(RUN, 'ros_merge.json')))
lk = J['terminal']['lock']
T_H = np.array(lk['T'], float)                                      # 近距鎖定參考（該趟實際影像）
Kc = np.array(json.load(open(os.path.join(RUN, 'frames_meta.json')))['K'], float)
Rg, T_HG, a_H = G.PG.baseline_reference()
FL.set_approach_axis(a_H)
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
S_PRE = float(G.PG.S_PRE)
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
THR, SIG = W.THR, W.SIG_FLOOR
pg = J['plan']['pg']
d = np.array(pg['d'], float); n_ = np.array(pg['n'], float); yaw = float(pg['yaw'])
h_xy = T_H[:2, 3]
a_n = T_H[:3, :3] @ np.asarray(a_H, float)
ENDS = [T_H[:3, 3], T_H[:3, 3] + T_H[:3, 0] * 0.06, T_H[:3, 3] - T_H[:3, 0] * 0.06]
DS_SEED = np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157])


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
        return None, 'ik', None
    return q, None, sm


def check(b, q, sm, r):
    m_lim = float(min((q - LO).min(), (HI - q).min()))
    if m_lim < 0:
        return 'limit', None
    if sm < SIG:
        return 'manip', None
    cl = W.clear_door(np.r_[b, q, 0.0], T_H)
    bad = [k for k in ('env', 'base', 'self') if cl[k] < THR[k]]
    if bad:
        return 'clear:' + ','.join(bad), None
    vis = visible(np.r_[b, q])
    if r >= 0.31 and not vis:
        return 'not_visible', None
    return None, dict(sigma=float(sm), lim=m_lim, clear={k: float(cl[k]) for k in ('env', 'base', 'self')}, vis=bool(vis))


# 終點：門旁分支的預抓位解（由門旁種子在 a＝−0.115、sb 0.45 的抓取位 → 後退 S_PRE → 沿線到 a1）
q_end = {}
sd, _, _ = solve(base(-0.115, 0.45), 0.0, DS_SEED)
sd, _, _ = solve(base(-0.115, 0.45), S_PRE, sd)
for a1 in np.arange(-0.115, -0.0249, 0.005):
    sd, why, _ = solve(base(float(a1), 0.45), S_PRE, sd)
    if sd is None:
        break
    q_end[round(float(a1), 3)] = sd.copy()
print('end configs at a1:', sorted(q_end)[:3], '...', sorted(q_end)[-3:])

N = 60
t0 = time.perf_counter()
cands = []
for a1 in (-0.075, -0.10):
    if round(a1, 3) not in q_end:
        continue
    for sb0 in (0.60, 0.70, 0.80, 0.90):
        for a0 in (-0.05, 0.05, 0.15, 0.25, 0.35, 0.45):
            if a0 <= a1 + 0.02:
                continue
            Q, info, why, seed = [], [], None, q_end[round(a1, 3)].copy()
            for k in range(N, -1, -1):                              # 由終點往回延拓（同一分支）
                u = k / N
                a, sb, r = a0 + u * (a1 - a0), sb0 + u * (0.45 - sb0), 0.33 + u * (S_PRE - 0.33)
                b = base(a, sb)
                q, w, sm = solve(b, r, seed)
                if q is None:
                    why = dict(at=round(u, 3), why=w); break
                if np.abs(q - seed).max() > 0.3:
                    why = dict(at=round(u, 3), why='branch_jump', dq=round(float(np.abs(q - seed).max()), 3)); break
                w2, inf = check(b, q, sm, r)
                if w2 is not None:
                    why = dict(at=round(u, 3), why=w2, r=round(r, 3), a=round(a, 3), sb=round(sb, 3)); break
                Q.append(q); info.append(inf); seed = q
            c = dict(a0=a0, sb0=sb0, a1=a1, ok=why is None, fail=why)
            if why is None:
                Q = np.array(Q[::-1]); info = info[::-1]
                L_base = float(math.hypot(a0 - a1, sb0 - 0.45))
                dq = np.abs(np.diff(Q, axis=0))
                c.update(base_path_m=round(L_base, 3), joint_travel_rad=round(float(dq.sum(axis=0).max()), 3), dq_step_max=round(float(dq.max()), 4),
                         sigma_min=round(min(x['sigma'] for x in info), 4), lim_margin_min=round(min(x['lim'] for x in info), 4),
                         clear_min={k: round(min(x['clear'][k] for x in info), 4) for k in ('env', 'base', 'self')},
                         visible_until_r=round(min(0.33 + (k / N) * (S_PRE - 0.33) for k in range(N + 1) if info[k]['vis']), 3) if any(x['vis'] for x in info) else None,
                         q_start=[round(float(x), 3) for x in Q[0]], q_end=[round(float(x), 3) for x in Q[-1]],
                         time_s_at_0p1=round(L_base / 0.1, 2), arm_rate_max_at_0p1=round(float(dq.max()) / ((L_base / 0.1) / N), 3))
            cands.append(c)
ok = [c for c in cands if c['ok']]
ok.sort(key=lambda c: (-min(c['sigma_min'] - SIG, c['lim_margin_min'], c['clear_min']['self'] - THR['self']), c['joint_travel_rad']))
fails = {}
for c in cands:
    if not c['ok']:
        k = c['fail']['why'].split(':')[0]
        fails[k] = fails.get(k, 0) + 1
out = dict(source=RUN, note='運動學層離線局部搜尋；把手＝該趟近距鎖定參考（實際影像）；不是物理結果', S_PRE=S_PRE, sig_floor=SIG, thr={k: float(v) for k, v in THR.items()}, n_cands=len(cands), n_ok=len(ok), fail_reasons=fails, best=ok[:6], all=cands, wall_s=round(time.perf_counter() - t0, 1))
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=str)
print('cands', len(cands), 'ok', len(ok), fails)
for c in ok[:6]:
    print({k: c[k] for k in ('a0', 'sb0', 'a1', 'base_path_m', 'joint_travel_rad', 'sigma_min', 'lim_margin_min', 'clear_min', 'visible_until_r', 'time_s_at_0p1', 'arm_rate_max_at_0p1')})
for c in [x for x in cands if not x['ok']][:8]:
    print('FAIL', c['a0'], c['sb0'], c['a1'], c['fail'])
