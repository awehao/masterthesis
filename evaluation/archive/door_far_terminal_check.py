#!/usr/bin/env python3
"""K3（Codex 20261116）：觀察位（TCP 沿接近軸後退 S_OBS＝0.33 m）→ 預抓位（後退 S_PRE）的終端接近，底盤沿通過線等速移動中。
運動學層的完整軌跡核對（離線；不是物理結果）：
  1. 可行圖：沿線位置 a（距抓取點）× TCP 後退量 s 的格點，由觀察分支連續解 IK；每格記可操作度、限位、淨空（env／base／self）。
  2. 排程族：底盤 a(t)=a_start−v_pass·t；TCP 後退量由 S_OBS 以最小急動在時間 T 內降到 S_PRE，起點 a_start、時間 T 掃描。
     每條排程逐控制拍（0.05 s）連續解 IK，核：殘差、分支連續（每拍關節變化）、限位、可操作度下限（不降）、淨空、關節速率與加速度上限。
  3. 回報可行排程、各自到達預抓位時的沿線位置（須留給移動抓持控制器的窗口）、最小可操作度、最大關節速率。
把手估計取既有 Isaac 趟的固定參考（途中實際影像估計）；通過線與觀察分支用同一套選擇程式。
    python3 evaluation/door_far_terminal_check.py RUN_DIR OUT.json"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import door_far_link as FL, door_far_route as FR, door_motm_grasp_ctrl as G, door_wbopt as W, door_clear_door_fast as _F, door_fast_kin as DFK

W.clear_door = _F.clear_door_fast
K = G.K
RUN = sys.argv[1]
J = json.load(open(os.path.join(RUN, 'ros_merge.json')))
T_est = np.array(J['anchor']['T_odom_handle'], float)
Rg, T_HG, a_H = G.PG.baseline_reference()
FL.set_approach_axis(a_H)
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
S_PRE, S_OBS, V = float(G.PG.S_PRE), float(FR.S_OBS), 0.05
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
clear_fn = lambda x: W.clear_door(np.r_[x, 0.0], T_est)            # noqa: E731
THR, SIG = W.THR, W.SIG_FLOOR
pgj = J['plan']['pg']
pg = dict(p0=np.array(pgj['p0']), d=np.array(pgj['d']), n=np.array(pgj['n']), yaw=float(pgj['yaw']))
q0 = np.array(J['plan']['br']['q_start_op'], float)
a_n = T_est[:3, :3] @ np.asarray(a_H, float)


def target(s):
    T = np.asarray(T_at(T_est, 0.0), float).copy()
    T[:3, 3] += s * a_n
    return T


def base(a):
    return np.r_[pg['p0'] - pg['d'] * a, pg['yaw']]


def cell(a, s, seed):
    q, r, sm = DFK.ik_closure_fc(target(s), base(a), seed)
    if not (r < 1e-6 and np.isfinite(q).all()):
        return None, 'ik'
    if np.abs(q - seed).max() > 0.6:
        return None, 'branch_jump'
    if np.any(q < LO) or np.any(q > HI):
        return q, 'limit'
    if sm < SIG:
        return q, 'manip:%.4f' % sm
    cl = clear_fn(np.r_[base(a), q])
    bad = [k for k in ('env', 'base', 'self') if cl[k] < THR[k]]
    if bad:
        return q, 'clear:' + ','.join(bad)
    return q, dict(sigma=float(sm), clear={k: float(cl[k]) for k in ('env', 'base', 'self')})


t0 = time.perf_counter()
# ---- 1. 可行圖
A = np.round(np.arange(0.225, -0.0501, -0.0125), 4)
S = np.round(np.r_[np.arange(S_OBS, S_PRE, -0.02), S_PRE, 0.015, 0.0], 4)      # 末兩欄＝插入段（後退 0.015、0＝抓取位）；插入段的兩指—橫桿豁免由抓持控制器處理，這裡的淨空是未豁免的值，原因照列
grid, seed_a = [], q0.copy()
for a in A:
    qa, ra = cell(a, S_OBS, seed_a)
    row, seed = [], (qa if qa is not None else seed_a)
    if qa is not None:
        seed_a = qa
    for s in S:
        q, r = cell(a, s, seed)
        ok = isinstance(r, dict)
        row.append(dict(ok=ok, why=None if ok else r, sigma=r['sigma'] if ok else None, self=r['clear']['self'] if ok else None, q=None if q is None else [round(float(x), 3) for x in q]))
        if q is None:
            break
        seed = q
    while len(row) < len(S):
        row.append(dict(ok=False, why='not_reached', sigma=None, self=None, q=None))
    grid.append(row)
fmap = [''.join('#' if c['ok'] else ('m' if str(c['why']).startswith('manip') else ('c' if str(c['why']).startswith('clear') else ('l' if c['why'] == 'limit' else '.'))) for c in row) for row in grid]


# ---- 2. 排程族
def minjerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def trap(u, f=0.2):
    """梯形速度（加減速各佔 f）：峰值速度＝平均的 1/(1−f) 倍（最小急動是 1.875 倍）。"""
    vp = 1.0 / (1.0 - f)
    if u < f:
        return 0.5 * vp * u * u / f
    if u > 1 - f:
        return 1.0 - 0.5 * vp * (1 - u) ** 2 / f
    return vp * (u - 0.5 * f)


PROFILE = dict(minjerk=minjerk, trap=trap)


def schedule(a_start, T, prof='minjerk'):
    n = int(round(T / 0.05))
    q, Q, sig, mc, why = None, [], np.inf, dict(env=np.inf, base=np.inf, self=np.inf), None
    seed = q0.copy()
    for a_pre in np.arange(0.225, a_start - 1e-9, -0.0025):             # 先沿觀察分支走到 a_start
        seed, r = cell(a_pre, S_OBS, seed)
        if not isinstance(r, dict):
            return dict(ok=False, why='observe_branch:' + str(r), a_fail=round(float(a_pre), 4))
    for k in range(n + 1):
        a = a_start - V * 0.05 * k
        s = S_OBS + (S_PRE - S_OBS) * PROFILE[prof](k / n)
        q, r = cell(a, s, seed)
        if not isinstance(r, dict):
            why = dict(ok=False, why=str(r), k=k, a=round(float(a), 4), s=round(float(s), 4), frac=round(k / n, 3))
            break
        Q.append(q); seed = q
        sig = min(sig, r['sigma']); mc = {k_: min(mc[k_], r['clear'][k_]) for k_ in mc}
    if why is not None:
        return why
    Q = np.array(Q)
    rate = np.abs(np.diff(Q, axis=0)) / 0.05
    acc = np.abs(np.diff(Q, n=2, axis=0)) / 0.05 ** 2
    a_end = a_start - V * T
    out = dict(ok=bool(rate.max() <= FR.V_ARM and acc.max() <= FR.A_ARM), a_start=a_start, T=T, a_at_pregrasp=round(float(a_end), 4), sigma_min=round(float(sig), 4), rate_max=round(float(rate.max()), 3), acc_max=round(float(acc.max()), 2),
               min_clear={k_: round(float(v_), 4) for k_, v_ in mc.items()}, tcp_speed_max=round(float(1.875 * (S_OBS - S_PRE) / T), 3), q_end=[round(float(x), 4) for x in Q[-1]])
    if not out['ok']:
        out['why'] = 'rate_or_acc'
    return out


scheds = []
for a_start in (0.225, 0.2125):
    for T in (1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0):
        if a_start - V * T < 0.0:
            continue
        for prof in ('minjerk', 'trap'):
            r = schedule(a_start, T, prof)
            r.setdefault('a_start', a_start); r.setdefault('T', T); r['profile'] = prof
            scheds.append(r)
ok = [r for r in scheds if r['ok']]
ok.sort(key=lambda r: (-r['a_at_pregrasp'], -r['sigma_min']))
fail = {}
for r in scheds:
    if not r['ok']:
        k_ = str(r.get('why')).split(':')[0]
        fail[k_] = fail.get(k_, 0) + 1
gs = getattr(G, 'SCHED', None)
res = dict(source=RUN, note='運動學層離線核對；不是物理結果。把手估計＝該趟固定參考（途中實際影像）', S_OBS=S_OBS, S_PRE=S_PRE, v_pass=V, sig_floor=SIG, thr={k: float(v) for k, v in THR.items()}, limits_margin_rad=0.05, v_arm=FR.V_ARM, a_arm=FR.A_ARM,
           map=dict(alongs=A.tolist(), retreats=S.tolist(), legend='#＝可行；m＝可操作度不足；c＝淨空不足；l＝限位；.＝IK 無解或分支不連續／未到達。每列一個沿線位置（由 0.225 到 −0.05），每欄一個後退量（由 0.33 到 S_PRE）', rows=fmap,
                    sigma_at_pregrasp=[None if not row[-3]['ok'] else round(row[-3]['sigma'], 4) for row in grid], last3_cols=[[c_['why'] if not c_['ok'] else 'ok(sig %.3f, self %.4f)' % (c_['sigma'], c_['self']) for c_ in row[-3:]] for row in grid], q_at_pregrasp=[row[-3]['q'] for row in grid], first_block_reason=[next((c['why'] for c in row if not c['ok']), None) for row in grid]),
           schedules=dict(n=len(scheds), n_ok=len(ok), fail_reasons=fail, ok=ok[:12], all=scheds), grasp_ctrl_sched=gs, grasp_window_m=list(FL.GRASP_WINDOW_M), wall_s=round(time.perf_counter() - t0, 1))
json.dump(res, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=str)
print('S_PRE', S_PRE, 'SIG', SIG)
for a, row in zip(A, fmap):
    print('%+.4f %s' % (a, row))
print('schedules ok', len(ok), '/', len(scheds), fail)
for r in ok[:8]:
    print({k: r[k] for k in ('profile', 'a_start', 'T', 'a_at_pregrasp', 'sigma_min', 'rate_max', 'acc_max', 'min_clear')})
for r in [x for x in scheds if not x['ok']][:6]:
    print('FAIL', {k: r.get(k) for k in ('a_start', 'T', 'why', 'k', 'a', 's', 'frac', 'rate_max', 'acc_max')})
