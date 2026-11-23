#!/usr/bin/env python3
"""mode 4 重設計 R1–R3 離線參考（Codex 20261029_145418 核可完整 R1–R3；20261029_145654 核可圓形包絡 0.39 一致化）。
Howard 原話：「移動時能朝前，但是做動作時能平滑的轉動到達最舒適的位置方便手臂作用」。
起點＝pb1f mode 3（安全退離）末態：底盤 (0.0712, −0.9370, 76.0°)、手臂放手構型、手指張開；門角 47.89°（v3 雙把手）。終點＝B1 (0.65, −0.05, −90°)、手臂 STOW，接既有 mode 5。
路徑：底盤圓盤格點 A*（0.0125 m、底盤圓盤 ≥ 20 mm）；步成本＝長度×(1＋K·max(0, D_PREF−d)/0.3)，d＝底盤中心到靜態形狀的距離 ⇒ 能走寬處就走寬處。
分段（同一條路徑）：
  R1（W-GMPC）：手臂 放手構型→STOW 關節內插，同時偏航轉向 R2 起點切線；底盤沿路徑前進量隨「偏航與路徑切線夾角」縮放（cos⁺），
      即朝向未對上前幾乎不平移、對上後才前進（不先停車收臂、不原地轉完才走）。結束條件＝手臂到 STOW 且路徑點 d ≥ D_NAV。
  R2（導航 GMPC＋CBF＋護盾）：路徑上 d ≥ D_NAV 的連續段；車頭＝路徑切線（朝前）。D_NAV＝0.55＝GMPC 靜態 CBF（margin 0.47＋牆點半徑 0.05＝0.52）＋0.03。
      時間只作名目（0.25 m/s），實跑由導航閉迴路決定。
  R3（W-GMPC）：R2 末點→B1。偏航以動態規劃求：每 2 cm 路徑點 × 偏航格點 5°，整機核對（PI.check 'M'：關節餘裕、非手指 ≥ 5 mm、自碰、底盤 ≥ 20 mm、手指 ≥ 5 mm）
      可行者才可走；轉移 |Δψ| ≤ 10°／點、時間＝max(Δs/v, |Δψ|/ω)；成本＝時間＋W_FWD·w(s)·(ψ−切線)²＋W_SMOOTH·Δψ²，w(s) 在末 L_BLEND 線性降到 0；終點偏航＝−90°。
      ⇒ 前段朝前，接近 B1 時連續轉入操作朝向（轉多快由幾何與限速決定，不另設停車原地轉）。
時間化：R1、R3 用 door_coopen_pass_close_inner_retime_v3 的 plan／sample（求解器上限 0.995 倍；**本程式在自身行程內把 V_FREE 換成 TIGHT**，不改原檔），20 Hz 取樣後逐點再核。
報：各段時間、交棒點（位置、偏航、d、包絡 ρ）、整機最小淨空、偏航最大速率、與舊 mode 4（140.4 s）對照。
界線：離線運動學；R2 為名目；門角固定（放手後門靜止的情境）；未含 W-GMPC／導航閉迴路、5 Hz 振盪未解。
    python3 -u evaluation/door_regrasp_m4r_v1.py   # 寫 results/door/door_regrasp_m4r_v1.json（不覆寫）
"""
import heapq
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_retime_v3 as RT    # noqa: E402
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import door_nav_envelope_monitor as NE                 # noqa: E402

REF = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1f.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_m4r_v1.json')
RES = 0.0125
D_PREF, K_PEN = 0.60, 4.0
D_NAV = 0.55
V_NAV = 0.25
DS = 0.02
YAW_STEP = 5.0
MAX_DY = 2
W_FWD, W_SMOOTH = 4.0, 0.5
L_BLEND = 0.40
V3, W3 = 0.035255 * RT.TIGHT, 0.1999 * RT.TIGHT


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    ref = json.load(open(REF))
    s3 = ref['segments']['3']
    S = np.array(s3['base'][-1], float)
    A0 = np.array(s3['arm'][-1], float)
    QF = float(s3['qf'][-1])
    QD = float(ref['q_ref_deg'])
    STOW = np.array(ref['stow'], float)
    B1 = np.array(ref['B1'], float)
    sh = G3.shapes(PI.D3, math.radians(QD), PI.H)

    def dctr(x, y):
        best = 9.0
        for _, c, R, hs in sh:
            l = R[:2, :2].T @ np.array([x - c[0], y - c[1]])
            best = min(best, float(np.hypot(*np.maximum(np.abs(l) - hs[:2], 0.0))))
        return best

    # ---- A*（底盤圓盤 ≥ 20 mm ⇔ d ≥ RB＋0.02；與 PI.astar_base 同格點）
    s = (round(S[0] / RES), round(S[1] / RES))
    g = (round(B1[0] / RES), round(B1[1] / RES))
    dmemo = {}

    def dd(u):
        if u not in dmemo:
            dmemo[u] = dctr(u[0] * RES, u[1] * RES)
        return dmemo[u]
    nb = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    dist, prev, pq, closed = {s: 0.0}, {}, [(0.0, s)], set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == g:
            break
        for a, b in nb:
            v = (u[0] + a, u[1] + b)
            if dd(v) < PI.C.RB + 0.02:
                continue
            c = RES * math.hypot(a, b) * (1.0 + K_PEN * max(0.0, D_PREF - dd(v)) / 0.3)
            if dist[u] + c < dist.get(v, math.inf):
                dist[v], prev[v] = dist[u] + c, u
                heapq.heappush(pq, (dist[v] + RES * math.hypot(v[0] - g[0], v[1] - g[1]), v))
    if g not in prev:
        raise SystemExit('A* 無路徑')
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    P = np.array([[u[0] * RES, u[1] * RES] for u in path[::-1]])
    P[0], P[-1] = S[:2], B1[:2]
    # 重新取樣（弧長 DS）＋輕度平滑（端點固定；平滑後重核 d）
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ss = np.linspace(0.0, seg[-1], int(math.ceil(seg[-1] / DS)) + 1)
    Pr = np.c_[np.interp(ss, seg, P[:, 0]), np.interp(ss, seg, P[:, 1])]
    for _ in range(30):
        Q = Pr.copy()
        Q[1:-1] = 0.25 * Pr[:-2] + 0.5 * Pr[1:-1] + 0.25 * Pr[2:]
        if min(dctr(*p) for p in Q) < PI.C.RB + 0.02:
            break
        Pr = Q
    dP = np.array([dctr(*p) for p in Pr])
    tang = np.array([math.atan2(*(Pr[min(k + 1, len(Pr) - 1)] - Pr[max(k - 1, 0)])[::-1]) for k in range(len(Pr))])
    L = float(np.sum(np.linalg.norm(np.diff(Pr, axis=0), axis=1)))
    print(f'路徑 {L:.3f} m、{len(Pr)} 點；d 範圍 {dP.min():.3f}–{dP.max():.3f} m', flush=True)

    # ---- 分段：R2＝R1 之後第一段 d ≥ D_NAV 的最長連續區間
    runs, k = [], 0
    while k < len(Pr):
        if dP[k] >= D_NAV:
            j = k
            while j + 1 < len(Pr) and dP[j + 1] >= D_NAV:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    if not runs:
        raise SystemExit(f'路徑上無 d ≥ {D_NAV} 的點，導航無法接手')
    i_h, i_r = max(runs, key=lambda r: r[1] - r[0])
    print(f'導航段 i {i_h}–{i_r}（{(i_r - i_h) * DS:.2f} m）；起 {np.round(Pr[i_h], 3)} d {dP[i_h]:.3f}；止 {np.round(Pr[i_r], 3)} d {dP[i_r]:.3f}', flush=True)

    # ---- R1：收臂＋轉向＋依朝向前進
    psi_h = tang[i_h]
    n1 = max(i_h, 1)
    yaw0 = S[2]
    dyaw = wrap(psi_h - yaw0)
    W1 = []
    # 偏航與手臂先行：前 K0 點不平移（偏航誤差大）；之後按 cos⁺(誤差) 分配路徑弧長
    n_turn = max(int(math.ceil(abs(dyaw) / math.radians(YAW_STEP))), 1)
    m = max(n_turn, 20)
    yaws = yaw0 + dyaw * np.linspace(0.0, 1.0, m + 1)
    wts = np.array([max(0.0, math.cos(wrap(tang[0] - y))) for y in yaws[1:]])
    alloc = np.r_[0.0, np.cumsum(wts)]
    alloc = alloc / alloc[-1] * n1 if alloc[-1] > 0 else np.linspace(0, n1, m + 1)
    for j in range(m + 1):
        f = j / m
        sidx = alloc[j]
        a, b = int(math.floor(sidx)), min(int(math.floor(sidx)) + 1, len(Pr) - 1)
        xy = Pr[a] + (Pr[b] - Pr[a]) * (sidx - a)
        if j == 0:
            xy = S[:2]
        arm = A0 + (STOW - A0) * (0.5 - 0.5 * math.cos(math.pi * f))
        W1.append(np.r_[xy, yaws[j], arm])
    W1 = np.array(W1)

    # ---- R3：偏航動態規劃
    ks = list(range(i_r, len(Pr)))
    ygrid = np.radians(np.arange(-180.0, 180.0, YAW_STEP))
    NY = len(ygrid)
    feas = np.zeros((len(ks), NY), bool)
    nchk = 0
    for a, k in enumerate(ks):
        for j, y in enumerate(ygrid):
            if a == len(ks) - 1 and abs(wrap(y - B1[2])) > 1e-6:
                continue
            ok, _, _ = PI.check(np.r_[Pr[k], y, STOW], QD, QF, 'M')
            nchk += 1
            feas[a, j] = ok
    jb1 = int(np.argmin([abs(wrap(y - B1[2])) for y in ygrid]))
    if not feas[-1, jb1]:
        raise SystemExit('B1（STOW、−90°）整機核對不成立')
    s_r = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pr[ks], axis=0), axis=1))]
    s_tot = s_r[-1]
    INF = math.inf
    cost = np.full((len(ks), NY), INF)
    back = np.full((len(ks), NY), -1, int)
    j0 = int(np.argmin([abs(wrap(y - tang[i_r])) for y in ygrid]))
    if not feas[0, j0]:
        raise SystemExit(f'R3 起點以切線偏航 {math.degrees(tang[i_r]):.1f}° 整機核對不成立')
    cost[0, j0] = 0.0
    for a in range(1, len(ks)):
        dsa = s_r[a] - s_r[a - 1]
        wf = W_FWD * min(1.0, max(0.0, (s_tot - s_r[a]) / L_BLEND))
        for j in np.nonzero(feas[a])[0]:
            e = wrap(ygrid[j] - tang[ks[a]])
            for dj in range(-MAX_DY, MAX_DY + 1):
                jp = (j - dj) % NY
                if cost[a - 1, jp] == INF:
                    continue
                dy = math.radians(YAW_STEP) * abs(dj)
                c = cost[a - 1, jp] + max(dsa / V3, dy / W3) + wf * e * e + W_SMOOTH * dj * dj
                if c < cost[a, j]:
                    cost[a, j], back[a, j] = c, jp
    if cost[-1, jb1] == INF:
        raise SystemExit('R3 偏航動態規劃無可行解（通道內無法連續轉入 −90°）')
    jj = [jb1]
    for a in range(len(ks) - 1, 0, -1):
        jj.append(back[a, jj[-1]])
    jj = jj[::-1]
    yaw3 = np.unwrap(ygrid[jj])
    W3p = np.array([np.r_[Pr[k], yaw3[a], STOW] for a, k in enumerate(ks)])
    # 起點偏航精確對上切線、終點精確 −90°（差值 < 2.5°，屬格點量化）
    W3p[0, 2] = yaw3[0] + wrap(tang[i_r] - yaw3[0])
    W3p[-1, 2] = yaw3[-1] + wrap(B1[2] - yaw3[-1])

    # ---- 時間化（R1、R3）
    RT.V_FREE = RT.TIGHT

    def timed(W, tag):
        n = len(W)
        Pn = np.c_[W, np.zeros((n, 0))]
        Qn = np.full(n, QD)
        Fn = np.full(n, QF)
        stg = [tag] * n
        dt, tau, v, vq, it = RT.plan(list(Pn), list(Qn), list(Fn), stg)
        rows, tw = RT.sample(list(Pn), list(Qn), list(Fn), dt, tau, v, vq, stg)
        return rows

    R1 = timed(W1, 'R1')
    R3 = timed(W3p, 'R3')

    # ---- 逐點再核
    env = NE.EnvelopeMonitor()

    def recheck(rows, name):
        mins = {'nonfinger': math.inf, 'self': math.inf, 'base': math.inf}
        first_bad = None
        for r_ in rows:
            ok, why, info = PI.check(np.asarray(r_['x']), QD, QF, 'M')
            if not ok and first_bad is None:
                first_bad = {'t': r_['t'], 'why': why, 'x': np.asarray(r_['x']).tolist()}
            if info:
                for kk in mins:
                    mins[kk] = min(mins[kk], info[kk][0])
        X = np.array([r_['x'] for r_ in rows])
        tt = np.array([r_['t'] for r_ in rows])
        wz = np.abs(np.diff(X[:, 2]) / np.diff(tt))
        vb = np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1) / np.diff(tt)
        return {'name': name, 'T_s': float(tt[-1]), 'n': len(rows), 'first_bad': first_bad, 'min_clear_m': mins,
                'wz_max': float(wz.max()), 'v_max': float(vb.max())}

    c1, c3 = recheck(R1, 'R1'), recheck(R3, 'R3')
    rho_h = env.admit(STOW, QF)
    L2 = float(np.sum(np.linalg.norm(np.diff(Pr[i_h:i_r + 1], axis=0), axis=1)))
    T2 = L2 / V_NAV
    tot = c1['T_s'] + T2 + c3['T_s']
    yaw_fwd_err = [abs(math.degrees(wrap(W3p[a, 2] - tang[k]))) for a, k in enumerate(ks)]
    out = {'schema': 'door_regrasp_m4r/1', 'ref': os.path.relpath(REF, HERE), 'q_door_deg': QD, 'start': S.tolist(), 'B1': B1.tolist(),
           'params': {'D_PREF': D_PREF, 'K_PEN': K_PEN, 'D_NAV': D_NAV, 'V_NAV_nominal': V_NAV, 'DS': DS, 'YAW_STEP_deg': YAW_STEP,
                      'MAX_DY': MAX_DY, 'W_FWD': W_FWD, 'W_SMOOTH': W_SMOOTH, 'L_BLEND': L_BLEND, 'V3': V3, 'W3': W3},
           'path': {'xy': Pr.tolist(), 'd_m': dP.tolist(), 'tangent': tang.tolist(), 'length_m': L},
           'handover_nav': {'i': i_h, 'xy': Pr[i_h].tolist(), 'yaw': float(psi_h), 'd_m': float(dP[i_h]), 'rho_admit': rho_h[1]},
           'handover_wb': {'i': i_r, 'xy': Pr[i_r].tolist(), 'yaw': float(tang[i_r]), 'd_m': float(dP[i_r])},
           'R1': {'check': c1, 'rows': [{'t': r_['t'], 'x': np.asarray(r_['x']).tolist()} for r_ in R1]},
           'R2': {'length_m': L2, 'T_nominal_s': T2, 'd_min_m': float(dP[i_h:i_r + 1].min()), 'heading': 'tangent'},
           'R3': {'check': c3, 'n_dp_checks': nchk, 'yaw_fwd_err_deg': yaw_fwd_err,
                  'rows': [{'t': r_['t'], 'x': np.asarray(r_['x']).tolist()} for r_ in R3]},
           'total_s': tot, 'old_mode4_s': 140.4, 'wall_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for c in (c1, c3):
        print(f"{c['name']}：{c['T_s']:.2f} s、第一個阻點 {c['first_bad']}、最小淨空 非手指 {c['min_clear_m']['nonfinger'] * 1e3:.1f} mm／"
              f"自碰 {c['min_clear_m']['self'] * 1e3:.1f}／底盤 {c['min_clear_m']['base'] * 1e3:.1f}；ω max {c['wz_max']:.3f}、v max {c['v_max']:.4f}")
    print(f'R2 名目 {L2:.2f} m／{T2:.1f} s（d min {out["R2"]["d_min_m"]:.3f}）；交導航 ρ {rho_h[1]["rho_m"]:.4f}')
    print(f'R3 偏航：起 {math.degrees(W3p[0, 2]):.1f}° → 終 {math.degrees(W3p[-1, 2]):.1f}°；朝前誤差（前段 50%）max {max(yaw_fwd_err[:len(ks) // 2]):.1f}°')
    print(f'合計 {tot:.1f} s（舊 mode 4 140.4 s）；寫 {OUT}；{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
