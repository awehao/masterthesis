#!/usr/bin/env python3
"""A 包（Codex r41）：同一時間軸串接「外側把手開門末態 → 放手退臂 → 移到門洞入口 → 接近並正面抓內側把手 → 通過中拉門關 → 解除、收臂、清空」（離線全身運動學）。
門規格 door_local_v3（雙側把手；內側把手同型、同高、對稱位置為**本輪建模假設**）；幾何 door_geom_v3。Isaac 門資產尚未加入內側把手。
起點：release_bench6_v4 的保存構型（bench_6 STOPPED 後 110.0 s 實錄起點 → 張爪、沿工具 −z 退、越過自由端、收臂；已重新時間化 12.62 s）。
門角：放手到內側閉爪前依壓力情境（static／k3b_stress／reverse_0p5；release_plan_v1 定義，k3b_stress 依定義線性延續）持續演變，不在退臂終點重設；
  內側閉爪後門角由持握決定（剛性跟隨），只減不增。
階段與指定接觸（手指類逐物件分別核，不合併身分）：
  M（移動）：手臂 STOW、手指張開；底盤沿格點 A*（0.0125 m；底盤圓盤 ≥ 20 mm、v3 形狀）到 B1＝(0.60, −0.10, 270°)，偏航沿途內插；全身淨空同下。
  A（接近）：底盤鎖定 B1；預抓取 T_pre＝H_in(q)·T_HG·平移(工具 −z 0.06 m) 以多種子 IK 取最佳分支，自 STOW **關節空間內插**過去逐點核
    （直角座標內插版第 6 軸逼近限位；全身解版底盤漂向門閂側牆）；兩支橫桿對手指與非手指都核（≥ 5 mm）。
  I（插入）：TCP 沿工具 z 前進 0.06 m 到 T_grasp＝H_in(q)·T_HG（手指張開）；手指—內側橫桿 ≥ 0（不得穿入），其餘 ≥ 5 mm。
  G（閉爪）：手指張→合 3 點；只允許手指—內側橫桿接觸（距離 ≥ −1 mm 視為接觸），外側橫桿與其他形狀 ≥ 5 mm。
  P（通過中拉關）：每步門角 Δ ∈ {−0.5°, 0}、底盤期望位移 2 cm 朝 (0.62, y＋)；TCP＝H_in(q)·T_HG；全身解；手指只豁免內側橫桿；到門角 Q_END（2.0°）止。
  R（解除）：張爪 3 點；TCP 沿工具 −z 退 0.06 m（手指離開內側橫桿後不得重新穿入）；之後手臂關節內插到 STOW（底盤固定）。
判準（每點）：關節餘裕（LITE6_SAFE）≥ 0.05 rad、非手指（含兩支橫桿）≥ 5 mm、自碰 ≥ 5 mm、底盤 ≥ 20 mm、手指對非指定物件 ≥ 5 mm；單步關節 ≤ 0.10 rad、底盤 ≤ 0.03 m。
報：各階段是否成立、第一個阻點；全程重新時間化（求解器設定：底盤 0.035255 m/s、0.1999 rad/s、0.5／2.0；手臂 0.9999 rad/s、19.984 rad/s²）後的時間；
  「通過中關門」重疊：底盤圓盤尚未越過 y 0.44 期間的門角減少量與時間；整機清空（全部連桿點 y ≥ 0.14＝牆遠側面＋0.02）時刻與門角。
界線：離線運動學；取樣間不連續；無接觸力、摩擦、門動力學；持握剛性跟隨；不含 W-GMPC 閉迴路（下一步）。

    python3 -u evaluation/door_coopen_pass_close_inner_v1.py [static|k3b_stress|reverse_0p5]   # 寫 results/door/door_coopen_pass_close_inner_v1_<情境>.json（不覆寫）
"""
import heapq
import json
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import door_geom_v3 as G3                              # noqa: E402
import door_coopen_pass_close_v3 as PC3                # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_release_bench6_v3 as RB3            # noqa: E402
import door_coopen_release_plan_v1 as RP               # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
from ammr_wholebody_mpc import arm_limits as AL        # noqa: E402

C = RV.C
D3 = yaml.safe_load(open(os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
H = float(D3['handle']['height'])
THG = RV.T_HG()
REL = os.path.join(HERE, 'results', 'door', 'door_coopen_release_bench6_v4.json')
B1 = np.array([0.60, -0.10, math.radians(270.0)])
Q_END, Y_CLEAR, WALL_FAR = 2.0, 0.44, 0.14
OPEN, CLOSED = RP.OPEN, RP.Q_CLOSED


def tz(T, d):
    """沿工具 z 平移 d。"""
    M = np.eye(4)
    M[2, 3] = d
    return T @ M


def dists(x, q_deg, qfi):
    """回傳 {nonfinger, self, base, fingers:{f:{物件:距離}}}；非手指含兩支橫桿；手指逐物件（含 bar_out／bar_in）。"""
    q = math.radians(q_deg)
    qf = [float(v) for v in x]
    sh = G3.shapes(D3, q, H)
    pts = {}
    for n in C.ARM_LINKS:
        T = C.K.fk(qf, n)
        P = C.CL[n] + (C.FINGER_AXIS[n] * qfi if n in C.FINGER_AXIS else 0.0)
        pts[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
    nonf, fing = (math.inf, None), {}
    for n, P in pts.items():
        dd = {name: float(np.min(G3.box_dist(P, c, R, hs))) for name, c, R, hs in sh}
        dd['bar_out'] = float(np.min(G3._bar_outer(D3, P, q, H)))
        dd['bar_in'] = float(np.min(G3.bar_capsule_dist_inner(D3, P, q)))
        if n in C.FINGERS:
            fing[n] = dd
        else:
            k = min(dd, key=dd.get)
            if dd[k] < nonf[0]:
                nonf = (dd[k], f'{n}:{k}')
    base = min(((float(G3.disc_clear(qf[0], qf[1], c[2], C.RB, c, R, hs)), name) for name, c, R, hs in sh), key=lambda z: z[0])
    self_ = TJ.GEO(qf, q, qfi)['self']
    ymin = min(float(P[:, 1].min()) for P in pts.values())
    return {'nonfinger': nonf, 'self': self_, 'base': base, 'fingers': fing, 'arm_ymin': ymin}


def check(x, q_deg, qfi, stage, prev=None):
    """階段指定接觸。回傳 (ok, why, info)。"""
    qa = np.asarray(x[3:])
    jm = float(min(min(qa - AL.LITE6_SAFE.lower), min(AL.LITE6_SAFE.upper - qa)))
    if jm < 0.05:
        return False, f'joint_margin:{jm:.3f}', None
    d = dists(x, q_deg, qfi)
    if d['nonfinger'][0] < 0.005:
        return False, f"nonfinger:{d['nonfinger'][1]}:{d['nonfinger'][0] * 1e3:.1f}mm", d
    if d['self'][0] < 0.005:
        return False, f"self:{d['self'][1]}", d
    if d['base'][0] < 0.02:
        return False, f"base:{d['base'][1]}", d
    for f, dd in d['fingers'].items():
        for k, v in dd.items():
            if k == 'bar_out' and stage == 'release_outer_c':        # 指定初始接觸：此處不判，由呼叫端依 v4 規則（離開後不得重新穿入）另核
                continue
            if k == 'bar_out' and stage == 'release_outer':          # 沿用 release_bench6_v4：外側橫桿為指定配對，只禁止穿入
                if v < 0.0:
                    return False, f'finger:{f}:bar_out 穿入 {v * 1e3:.1f}mm', d
                continue
            if k == 'bar_in' and stage in ('I', 'R_slide'):
                if v < 0.0:
                    return False, f'finger:{f}:bar_in 穿入 {v * 1e3:.1f}mm', d
                continue
            if k == 'bar_in' and stage in ('G', 'P', 'R_open'):
                if v < -0.001:
                    return False, f'finger:{f}:bar_in 穿入 {v * 1e3:.1f}mm', d
                continue
            if v < 0.005:
                return False, f'finger:{f}:{k}:{v * 1e3:.1f}mm', d
    return True, None, dict(d, jm=jm)


def step_ok(x0, x1):
    return float(np.max(np.abs(x1[3:] - x0[3:]))) <= 0.10 and float(np.linalg.norm(x1[:2] - x0[:2])) <= 0.03


def astar_base(b0, b1, q_deg):
    """底盤圓盤格點 A*（0.0125 m）；手臂 STOW；回傳路徑點（不含偏航）。"""
    q = math.radians(q_deg)
    sh = G3.shapes(D3, q, H)
    res = 0.0125                                   # 瓶頸處 25 mm 格點會錯過（47.9° 最大可保留淨空 27.2 mm）

    def ok(p):
        return min(float(G3.disc_clear(p[0], p[1], c[2], C.RB, c, R, hs)) for _, c, R, hs in sh) >= 0.02
    s = (round(b0[0] / res), round(b0[1] / res))
    g = (round(b1[0] / res), round(b1[1] / res))
    dist, prev, pq = {s: 0.0}, {}, [(0.0, s)]
    nb = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    closed = set()
    while pq:
        _, u = heapq.heappop(pq)                     # 佇列存 f＝g＋h；實際成本取 dist[u]
        if u in closed:
            continue
        closed.add(u)
        if u == g:
            break
        d = dist[u]
        for a, b in nb:
            v = (u[0] + a, u[1] + b)
            if not ok((v[0] * res, v[1] * res)):
                continue
            nd = d + res * math.hypot(a, b)
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd + res * math.hypot(g[0] - v[0], g[1] - v[1]), v))
    if g not in dist:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return [np.array([p[0] * res, p[1] * res]) for p in path[::-1]]


def interp_T(Ta, Tb, a):
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    w = PC3.logR(Ra.T @ Rb)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        R = Ra
    else:
        k = w / th
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = Ra @ (np.eye(3) + math.sin(a * th) * Kx + (1 - math.cos(a * th)) * Kx @ Kx)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (1 - a) * Ta[:3, 3] + a * Tb[:3, 3]
    return T


M_ARM = np.diag([1e-6, 1e-6, 1e-6] + [1.0 / 25.0] * 6)         # 底盤鎖定（A／I：底盤停在已確認可抓的 B1）


def solve_w(x, T, Minv):
    xn = x.copy()
    for _ in range(4):
        e = PC3.err(xn, T)
        J = np.asarray(C.K.jacobian(xn, 'link_tcp'), float)
        xn = xn + Minv @ J.T @ np.linalg.solve(J @ Minv @ J.T + 1e-12 * np.eye(6), e)
    e = PC3.err(xn, T)
    return xn if (np.linalg.norm(e[:3]) <= 1e-3 and np.linalg.norm(e[3:]) <= 0.01) else None


def track(x, Ts, q_of, qfi, stage, rows, t_tag, base_lock=False):
    """逐點追蹤 TCP 目標序列（全身解；base_lock 時底盤鎖定）；失敗回傳 (x, why)。"""
    for k, T in enumerate(Ts):
        xn = solve_w(x, T, M_ARM) if base_lock else PC3.solve_step(x, T, np.zeros(9))
        if xn is None:
            return x, f'{stage}:IK 失敗（第 {k} 點）'
        if not step_ok(x, xn):
            return x, f'{stage}:單步過大（第 {k} 點；關節 {float(np.max(np.abs(xn[3:] - x[3:]))):.3f} rad、底盤 {float(np.linalg.norm(xn[:2] - x[:2])) * 1e3:.1f} mm）'
        ok, why, info = check(xn, q_of(k), qfi, stage)
        if not ok:
            return x, f'{stage}:{why}（第 {k} 點）'
        x = xn
        rows.append({'stage': stage, 'x': x.tolist(), 'q': q_of(k), 'qf': qfi, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}, 'tag': t_tag})
    return x, None


def main():
    scen = sys.argv[1] if len(sys.argv) > 1 else 'static'
    out = os.path.join(HERE, 'results', 'door', f'door_coopen_pass_close_inner_v1_{scen}.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t0w = time.time()
    rel = json.load(open(REL))
    rr = rel['results'][scen]
    q0 = rel['start']['q_deg']
    dq = RP.SCEN[scen]
    t_rel = rr['retimed']['t_end_s']
    last = rr['rows'][-1]
    x = np.array(list(last['base']) + list(last['qa']), float)
    rows, block = [], None
    # 退臂段沿用（每點門角＝重新時間化結果）
    for r_, rt in zip(rr['rows'], rr['retimed']['rows']):
        rows.append({'stage': 'release_outer', 'x': list(r_['base']) + list(r_['qa']), 'q': rt['q_door'], 'qf': r_['q_finger'], 'tag': 'v4'})
    # 以後的門角：在內側閉爪前依情境時間演變；這裡先以「步數×名目時間」估計，重新時間化後再依新時間重算並重核（見文末）
    t_est = [t_rel]

    def q_free():
        return q0 + dq(t_est[0])
    # ---- M：移動到門洞入口 ----
    qd_now = q_free()
    path = astar_base(x[:2], B1[:2], qd_now)
    if path is None:
        block = {'stage': 'M', 'why': '底盤 A* 無路徑'}
    else:
        yaw0 = float(x[2])
        dyaw = math.atan2(math.sin(B1[2] - yaw0), math.cos(B1[2] - yaw0))
        for k, p in enumerate(path[1:], start=1):
            xn = x.copy()
            xn[0], xn[1] = p
            xn[2] = yaw0 + dyaw * k / (len(path) - 1)
            t_est[0] += max(0.0125 * 1.42 / 0.035, abs(dyaw / (len(path) - 1)) / 0.2)
            ok, why, info = check(xn, q_free(), OPEN, 'M')
            if not ok:
                block = {'stage': 'M', 'why': why, 'at': xn[:3].tolist()}
                break
            x = xn
            rows.append({'stage': 'M', 'x': x.tolist(), 'q': q_free(), 'qf': OPEN, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}})
    # ---- A／I：接近與插入（門角在閉爪前仍自由演變；以預估時間取值） ----
    if block is None:
        qg = q_free()
        Tg = G3.handle_pose_inner(D3, math.radians(qg)) @ THG
        Tpre = tz(Tg, -0.06)
        # 預抓取構型：底盤 B1 固定，多種子 IK，取關節餘裕＋淨空最好的分支；再自 STOW 關節空間內插過去（逐點核）
        best = None
        for seed in [None] + [np.array(sd) for sd in C.SEEDS]:
            q9 = C.ik(np.asarray(x[:3]), Tpre, seed, only_prev=seed is not None)
            if q9 is None:
                continue
            xp = np.concatenate([x[:3], np.asarray(q9)[C.IDX]])
            ok, why_, info = check(xp, qg, OPEN, 'A')
            if ok:
                scv = min(info['jm'], 2.0) + 5.0 * min(info['nonfinger'][0], info['self'][0])
                if best is None or scv > best[0]:
                    best = (scv, xp)
        if best is None:
            block = {'stage': 'A', 'why': '預抓取位姿在 B1 無合格 IK 分支', 'q': qg}
        else:
            qa0, qa1 = x[3:].copy(), best[1][3:]
            n_a = max(10, int(math.ceil(float(np.max(np.abs(qa1 - qa0))) / 0.08)))
            for k in range(1, n_a + 1):
                xn = x.copy()
                xn[3:] = qa0 + (qa1 - qa0) * k / n_a
                ok, why_, info = check(xn, qg, OPEN, 'A')
                if not ok:
                    block = {'stage': 'A', 'why': f'{why_}（關節內插第 {k}/{n_a} 點）', 'q': qg}
                    break
                rows.append({'stage': 'A', 'x': xn.tolist(), 'q': qg, 'qf': OPEN, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}})
            x = np.array(rows[-1]['x'])
    if block is None:
        x, why = track(x, [tz(Tg, -0.06 + 0.06 * (k + 1) / 12) for k in range(12)], lambda k: qg, OPEN, 'I', rows, None, base_lock=True)
        if why:
            block = {'stage': 'I', 'why': why, 'q': qg}
    if block is None:
        for k in range(3):
            qfi = OPEN + (CLOSED - OPEN) * (k + 1) / 3
            ok, why, info = check(x, qg, qfi, 'G')
            if not ok:
                block = {'stage': 'G', 'why': why}
                break
            rows.append({'stage': 'G', 'x': x.tolist(), 'q': qg, 'qf': qfi, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}})
    # ---- P：通過中拉關 ----
    q = qg if block is None else None
    if block is None:
        it = 0
        while q > Q_END + 1e-9 and it < 600:
            it += 1
            TJ.cool()
            best = None
            for dqd in (-0.5, 0.0):
                qn = max(Q_END, q + dqd)
                T = G3.handle_pose_inner(D3, math.radians(qn)) @ THG
                for ang in (90.0, 70.0, 110.0, 50.0, 130.0):
                    a = math.radians(ang)
                    tgt = np.array([0.62 - x[0], 0.0])
                    dx_des = np.zeros(9)
                    dx_des[0] = 0.02 * math.cos(a) + 0.2 * tgt[0]
                    dx_des[1] = 0.02 * math.sin(a)
                    xn = PC3.solve_step(x, T, dx_des)
                    if xn is None or not step_ok(x, xn):
                        continue
                    ok, why, info = check(xn, qn, CLOSED, 'P')
                    if not ok:
                        continue
                    sc = (q - qn) * 0.02 + float(xn[1] - x[1]) + 0.05 * min(info['nonfinger'][0], info['self'][0])
                    if best is None or sc > best[0]:
                        best = (sc, xn, qn, info)
            if best is None:
                block = {'stage': 'P', 'why': '所有候選動作不成立', 'q': q, 'x': x.tolist()}
                break
            _, x, q, info = best
            rows.append({'stage': 'P', 'x': x.tolist(), 'q': q, 'qf': CLOSED, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}})
            p40 = [r_ for r_ in rows if r_['stage'] == 'P'][-40:]
            if len(p40) == 40 and p40[0]['q'] == q and abs(p40[0]['x'][1] - x[1]) < 1e-3:
                block = {'stage': 'P', 'why': '連續 40 步門角與底盤 y 都無進展', 'q': q, 'x': x.tolist()}
                break
    # ---- R：解除、退出、收臂 ----
    if block is None:
        for k in range(3):
            qfi = CLOSED + (OPEN - CLOSED) * (k + 1) / 3
            ok, why, info = check(x, q, qfi, 'R_open')
            if not ok:
                block = {'stage': 'R_open', 'why': why}
                break
            rows.append({'stage': 'R_open', 'x': x.tolist(), 'q': q, 'qf': qfi, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}})
    if block is None:
        Tg2 = C.K.fk(x, 'link_tcp')
        x, why = track(x, [tz(Tg2, -0.06 * (k + 1) / 12) for k in range(12)], lambda k: q, OPEN, 'R_slide', rows, None)
        if why:
            block = {'stage': 'R_slide', 'why': why}
    if block is None:
        qa0 = x[3:].copy()
        for k in range(1, 41):
            xn = x.copy()
            xn[3:] = (1 - k / 40) * qa0 + (k / 40) * TJ.STOW
            ok, why, info = check(xn, q, OPEN, 'R_stow')
            if not ok:
                block = {'stage': 'R_stow', 'why': why}
                break
            rows.append({'stage': 'R_stow', 'x': xn.tolist(), 'q': q, 'qf': OPEN, 'clear': {kk: info[kk][0] for kk in ('nonfinger', 'self', 'base')}})
        x = np.array(rows[-1]['x'])
    # ---- 重新時間化（全程）＋閉爪前門角依新時間重算並重核 ----
    cfgs = [(r_['x'][:3], r_['x'][3:], 'open' if r_['stage'] in ('G', 'R_open') else r_['stage']) for r_ in rows]
    ts = RB3.retime(cfgs)
    n_rel = sum(1 for r_ in rows if r_['stage'] == 'release_outer')
    i_grasp = next((i for i, r_ in enumerate(rows) if r_['stage'] == 'G'), None)
    recheck_bad = []
    for i, r_ in enumerate(rows):
        r_['t'] = ts[i]
        if i >= n_rel and (i_grasp is None or i < i_grasp):
            qn = q0 + dq(ts[i])
            if abs(qn - r_['q']) > 1e-9:
                ok, why, _ = check(np.array(r_['x']), qn, r_['qf'], r_['stage'])
                r_['q_retimed'] = qn
                if not ok:
                    recheck_bad.append({'i': i, 't': ts[i], 'stage': r_['stage'], 'q': qn, 'why': why})
    if i_grasp is not None and scen != 'static':
        qn = q0 + dq(ts[i_grasp])
        if abs(qn - rows[i_grasp]['q']) > 0.25:
            recheck_bad.append({'i': i_grasp, 'stage': 'G', 'why': f'閉爪時門角依新時間為 {qn:.2f}°，與規劃時的 {rows[i_grasp]["q"]:.2f}° 差 > 0.25°（抓取位姿需重算）'})
    # 整機 y 下界：全部連桿點雲（含底盤 base_link、手指依當時開度）與底盤圓盤 y − RB 的最小值
    for r_ in rows:
        W = C.world([float(v) for v in r_['x']], list(C.CL), r_['qf'])
        r_['ymin_body'] = float(min(min(float(P[:, 1].min()) for P in W.values()), r_['x'][1] - C.RB))
    pr = [r_ for r_ in rows if r_['stage'] == 'P']
    after_p = [r_ for r_ in rows if r_['stage'] in ('P', 'R_open', 'R_slide', 'R_stow')]
    base_clear_t = next((r_['t'] for r_ in after_p if r_['x'][1] >= Y_CLEAR), None)
    whole = next((r_ for r_ in after_p if r_['ymin_body'] >= WALL_FAR), None)
    whole_clear = {'t_s': whole['t'] if whole else None, 'stage': whole['stage'] if whole else None, 'q_deg': whole['q'] if whole else None,
                   'final_ymin_body_m': rows[-1]['ymin_body'], 'final_ok': bool(rows[-1]['ymin_body'] >= WALL_FAR)}
    overlap = None
    if pr:
        pre = [r_ for r_ in pr if r_['x'][1] < Y_CLEAR]
        overlap = {'q_at_P_start': pr[0]['q'], 'q_at_base_clear': next((r_['q'] for r_ in after_p if r_['x'][1] >= Y_CLEAR), None),
                   'stage_at_base_clear': next((r_['stage'] for r_ in after_p if r_['x'][1] >= Y_CLEAR), None),
                   'closing_deg_before_base_clear': (pr[0]['q'] - min(r_['q'] for r_ in pre)) if pre else 0.0,
                   'duration_overlap_s': (pre[-1]['t'] - pre[0]['t']) if len(pre) > 1 else 0.0,
                   'base_y_at_P_start': pr[0]['x'][1], 'base_y_at_P_end': pr[-1]['x'][1]}
    res = {'schema': 'door_coopen_pass_close_inner/1', 'scenario': scen, 'door_spec': 'door_local_v3（內側把手同型、同高、對稱＝建模假設）', 'block': block,
           'completed': block is None, 'n_points': len(rows), 't_end_s': ts[-1] if ts else None, 'recheck_after_retime': recheck_bad,
           'stage_counts': {s_: sum(1 for r_ in rows if r_['stage'] == s_) for s_ in ('release_outer', 'M', 'A', 'I', 'G', 'P', 'R_open', 'R_slide', 'R_stow')},
           'stage_t_start': {s_: next((r_['t'] for r_ in rows if r_['stage'] == s_), None) for s_ in ('M', 'A', 'I', 'G', 'P', 'R_open', 'R_slide', 'R_stow')},
           'overlap_pass_close': overlap, 'base_clear_t_s': base_clear_t, 'whole_robot_clear': whole_clear,
           'min_clear_m': {k: min((r_['clear'][k] for r_ in rows if 'clear' in r_), default=None) for k in ('nonfinger', 'self', 'base')},
           'final': {'x': rows[-1]['x'], 'q_deg': rows[-1]['q']}, 'rows': rows, 'wall_s': time.time() - t0w}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, ensure_ascii=False, default=float)[:2500], flush=True)


if __name__ == '__main__':
    main()
