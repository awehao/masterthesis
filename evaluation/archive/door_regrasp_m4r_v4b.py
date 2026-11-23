#!/usr/bin/env python3
"""v4b（Codex 20261029_174949 採 A、20261029_175625 同意 A′；door_m4r_1 R3 起算偏 0.185 m）：R2 最後一段與 R3 前置共用**同一直線、同一朝向**。
  - 導航第二段折線＝原 R2 路徑保留到 J（index J_IDX）→ 半徑 FILLET_R 的圓弧 → 直線進場到 P_r（方向＝P_r 處切線 ψr）；直線長由幾何決定（記錄；A′ 名目 0.6 m）。
    保留段、圓弧、直線每 1 cm 核 d ≥ D_NAV，並以行進方向＋STOW 每 5 cm 整機核對。
  - R3＝直線最後 N_PRE_M（0.40 m）沿 ψr、車頭固定 ψr、STOW、起速 V_HAND → 接 v3 的 R3（P_r → B1，同一偏航動態規劃）；整機核對、時間化、端點核對同 v3。
  - 記 R3_pre：直線起訖、ψr、通過 P_r 的參考時刻 tau_pre_end；任務端只在 [0, tau_pre_end] 依直線進度投影選接點。
  （v4＝沿原彎道前置，door_m4r_1 位姿投影偏 0.08 m／34° 仍會被拒，結果保留、不使用。）
v3（Codex 20261029_154350：v2 的非零端速放進拋物線混合 ⇒ 端點位置偏、未取到精確終止時刻）：只改 sample_bv（兩端三次 Hermite、含 T_end、每列解析速度）＋端點核對；其餘同 v2。
mode 4 重設計 R1–R3 離線參考 v2（Codex 20261029_151139：選 (a) 為第一版接線基線，並修正 ①收臂不拉長等偏航 ②偏航上限用已登錄的 0.0738 rad/s ③接點速度連續）。
v1（door_regrasp_m4r_v1.py，結果保留）：偏航用 QP 上限 0.1999、收臂與偏航同步拉長、R1／R3 端點規劃成靜止 —— 三處依 151139 改正，路徑與分段規則不變。
  ① R1 手臂以自身速度（0.5×0.9999 rad/s，同舊自由段）在前 m_arm 段收完，偏航獨立進行；
  ② 偏航速率上限 YAW_REF＝0.0738 rad/s（door_regrasp_retime_v7：受控體偏航增益 0.82×coord 引導 w_cap 0.10×0.9；名目模型餘裕，非實機保證），時間化與 R3 動態規劃都用它；
  ③ 端點速度：R1 末＝沿 R2 起點切線 V_HAND（交導航時仍在前進），R3 起＝沿切線 V_HAND（導航 GLIDE 減速到此再交回），B1＝靜止；
     LSPB 規劃改用本檔 plan_bv／sample_bv（同 retime_v3 的算法，只把兩端零速改成指定速度）；R2 名目改梯形（V_HAND→0.25→V_HAND，a 0.3 m/s²＝GLIDE 減速上限）。
原 v1 說明：
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
    python3 -u evaluation/door_regrasp_m4r_v4b.py   # 寫 results/door/door_regrasp_m4r_v4b.json（不覆寫）
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
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_m4r_v4b.json')
N_PRE_M, J_IDX, FILLET_R = 0.40, 44, 0.15
RES = 0.0125
D_PREF, K_PEN = 0.60, 4.0
D_NAV = 0.55
V_NAV = 0.25
DS = 0.02
YAW_STEP = 5.0
MAX_DY = 2
W_FWD, W_SMOOTH = 4.0, 0.5
L_BLEND = 0.40
V3, W3 = 0.035255 * RT.TIGHT, 0.0738
V_HAND = 0.02          # 交棒時沿切線的底盤速度（≤ v_box_lin 0.035255）
A_NAV = 0.3
V_ARM_FRAC = 0.5
VMAXB = np.array([V3, V3, W3] + [RT.VA * RT.TIGHT] * 6)
AMAXB = np.array([RT.AB, RT.AB, RT.AW] + [RT.AA] * 6) * RT.TIGHT


def plan_bv(P, F, v0, v1):
    """同 retime_v3.plan，但兩端速度＝v0／v1（世界座標 9 維），速度上限 VMAXB（偏航 0.0738）。"""
    n = len(P)
    dt = []
    for k in range(n - 1):
        d = P[k + 1] - P[k]
        bx, by = RT.body_xy_max(d, float(P[k][2]), float(P[k + 1][2]))
        db = np.abs(d)
        db[0], db[1] = bx, by
        dt.append(max(float(np.max(db / VMAXB)), 1e-3))
    dt = np.array(dt)
    for it in range(5000):
        v = [(P[k + 1] - P[k]) / dt[k] for k in range(n - 1)]
        vext = [np.asarray(v0, float)] + v + [np.asarray(v1, float)]
        tau = np.zeros(n)
        for j in range(n):
            dv = RT.body(vext[j + 1] - vext[j], float(P[j][2]))
            tau[j] = float(np.max(np.abs(dv) / AMAXB))
        bad = [k for k in range(n - 1) if 0.5 * (tau[k] + tau[k + 1]) > dt[k] * (1 - 1e-9)]
        if not bad:
            return dt, tau, v, vext
        for k in bad:
            dt[k] *= 1.1
    raise RuntimeError('時間化未收斂')


def _herm(t, T, x0, v0, x1, v1):
    """三次 Hermite：t∈[0,T]，兩端位置與速度精確。回傳 (x, v)。"""
    u = t / T
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    d00, d10, d01, d11 = (6 * u ** 2 - 6 * u) / T, 3 * u ** 2 - 4 * u + 1, (-6 * u ** 2 + 6 * u) / T, 3 * u ** 2 - 2 * u
    return h00 * x0 + h10 * T * v0 + h01 * x1 + h11 * T * v1, d00 * x0 + d10 * v0 + d01 * x1 + d11 * v1


def sample_bv(P, dt, tau, v, vext):
    """v3：內部接點同 retime_v3 的拋物線混合；**兩端**改三次 Hermite，使起點＝(P0, v0)、終點＝(P_end, v1) 精確
    （v2 把非零端速放進拋物線公式，端點位置偏 v·τ/2，且不一定取到終止時刻）。取樣含精確 T_end；每列附解析速度 'v'。"""
    P = [np.asarray(p, float) for p in P]
    n = len(P)
    tw = np.concatenate([[tau[0] / 2], tau[0] / 2 + np.cumsum(dt)])
    T_end = tw[-1] + tau[-1] / 2
    v0, v1 = np.asarray(vext[0], float), np.asarray(vext[-1], float)

    def lin(k, t):
        return P[k] + v[k] * (t - tw[k]), np.asarray(v[k], float)

    def state(t):
        if tau[0] > 0 and t <= tau[0]:
            xa, va = lin(0, tau[0])
            return _herm(t, tau[0], P[0], v0, xa, va)
        if tau[-1] > 0 and t >= T_end - tau[-1]:
            ta = T_end - tau[-1]
            xa, va = lin(n - 2, ta)
            return _herm(t - ta, tau[-1], xa, va, P[-1], v1)
        j = int(np.argmin(np.abs(tw - t)))
        if 0 < j < n - 1 and abs(t - tw[j]) <= tau[j] / 2 and tau[j] > 0:
            s_ = t - tw[j] + tau[j] / 2
            return (P[j] + vext[j] * (t - tw[j]) + (vext[j + 1] - vext[j]) * s_ * s_ / (2 * tau[j]),
                    vext[j] + (vext[j + 1] - vext[j]) * s_ / tau[j])
        k = j if t > tw[j] else j - 1
        k = min(max(k, 0), n - 2)
        return lin(k, t)
    ts = list(np.arange(0.0, T_end, RT.DT_CTRL))
    if T_end - ts[-1] > 1e-9:
        ts.append(T_end)
    out = []
    for t in ts:
        x, vv = state(float(t))
        out.append({'t': float(t), 'x': x, 'v': vv})
    return out


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
    T_arm = float(np.max(np.abs(STOW - A0))) / (RT.VA * V_ARM_FRAC)
    T_yaw = abs(dyaw) / W3
    m_arm = min(m, max(3, int(math.ceil(m * T_arm / max(T_yaw, 1e-6)))))
    yaws = yaw0 + dyaw * np.linspace(0.0, 1.0, m + 1)
    wts = np.array([max(0.0, math.cos(wrap(tang[0] - y))) for y in yaws[1:]])
    alloc = np.r_[0.0, np.cumsum(wts)]
    alloc = alloc / alloc[-1] * n1 if alloc[-1] > 0 else np.linspace(0, n1, m + 1)
    for j in range(m + 1):
        f = min(1.0, j / m_arm)
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

    # ---- v4b：導航折線（保留段→圓弧→直線）與 R3 前置直線
    psr = float(tang[i_r]); ur = np.array([math.cos(psr), math.sin(psr)])
    J = Pr[J_IDX]; thj = float(tang[J_IDX]); ej = np.array([math.cos(thj), math.sin(thj)])
    aa, ss_ = np.linalg.solve(np.array([ej, ur]).T, Pr[i_r] - J)
    X = J + aa * ej
    dl = wrap(psr - thj)
    Tl = FILLET_R * math.tan(abs(dl) / 2)
    A1, A2 = X - Tl * ej, X + Tl * ur
    if aa < Tl or np.dot(Pr[i_r] - A2, ur) < N_PRE_M:
        raise SystemExit(f'圓弧不合：保留段可用 {aa:.3f} < {Tl:.3f} 或直線 {np.dot(Pr[i_r] - A2, ur):.3f} < {N_PRE_M}')
    nrm = np.array([-ej[1], ej[0]]) * (1 if dl > 0 else -1)
    Cc = A1 + FILLET_R * nrm
    a0 = math.atan2(A1[1] - Cc[1], A1[0] - Cc[0])
    n_arc = max(4, int(math.ceil(FILLET_R * abs(dl) / 0.05)))
    arc = [Cc + FILLET_R * np.array([math.cos(a0 + dl * k / n_arc), math.sin(a0 + dl * k / n_arc)]) for k in range(n_arc + 1)]
    keep = [i_h]
    for i in range(i_h + 1, J_IDX + 1):
        if np.linalg.norm(Pr[i] - Pr[keep[-1]]) >= 0.05 - 1e-9:
            keep.append(i)
    r2v = [Pr[i] for i in keep] + [A1] + arc[1:] + [Pr[i_r]]
    r2v = np.array([q for k_, q in enumerate(r2v) if k_ == 0 or np.linalg.norm(q - r2v[k_ - 1]) > 1e-6])
    r2_check = {'d_min_m': math.inf, 'fullbody_bad': None, 'n': 0}
    for a_, b_ in zip(r2v[:-1], r2v[1:]):
        hdg = math.atan2(b_[1] - a_[1], b_[0] - a_[0])
        nseg = max(2, int(np.linalg.norm(b_ - a_) / 0.01) + 1)
        for k_ in range(nseg):
            q = a_ + (b_ - a_) * k_ / (nseg - 1)
            r2_check['d_min_m'] = min(r2_check['d_min_m'], dctr(*q))
            r2_check['n'] += 1
            if k_ % 5 == 0 and r2_check['fullbody_bad'] is None:
                okq, whyq, _ = PI.check(np.r_[q, hdg, STOW], QD, QF, 'M')
                if not okq:
                    r2_check['fullbody_bad'] = {'xy': q.tolist(), 'hdg': hdg, 'why': whyq}
    if r2_check['d_min_m'] < D_NAV - 1e-9 or r2_check['fullbody_bad'] is not None:
        raise SystemExit(f'v4b 導航折線核對不過：{r2_check}')
    L_straight = float(np.dot(Pr[i_r] - A2, ur))
    print(f'v4b 導航折線 {len(r2v)} 點；J {np.round(J, 3)}、圓弧 r {FILLET_R}（轉 {math.degrees(dl):.1f}°）、直線 {L_straight:.3f} m；淨空 min {r2_check["d_min_m"]:.3f}、整機全成立', flush=True)
    n_pre = int(round(N_PRE_M / DS))
    pre = [np.r_[Pr[i_r] - N_PRE_M * ur * (1 - k_ / n_pre), psr, STOW] for k_ in range(n_pre)]
    W3p = np.vstack([np.array(pre), W3p])
    W3p[n_pre, 2] = W3p[n_pre - 1, 2] + wrap(W3p[n_pre, 2] - W3p[n_pre - 1, 2])
    for k_ in range(n_pre + 1, len(W3p)):
        W3p[k_, 2] = W3p[k_ - 1, 2] + wrap(W3p[k_, 2] - W3p[k_ - 1, 2])

    # ---- 時間化（R1、R3；端點速度連續）
    def vt(psi):
        return np.r_[V_HAND * math.cos(psi), V_HAND * math.sin(psi), 0.0, np.zeros(6)]

    def timed(W, v0, v1):
        """兩端 Hermite 的加速度峰值高於拋物線混合 ⇒ 端點混合時間逐次 ×1.5 直到解析速度差分的車體加速度 ≤ 上限（必要時加長首／末段時距）。"""
        dt, tau, v, vext = plan_bv(list(W), None, v0, v1)
        tau = np.array(tau, float)
        for it in range(40):
            rows = sample_bv(list(W), dt, tau, v, vext)
            X = np.array([r_['x'] for r_ in rows]); V = np.array([r_['v'] for r_ in rows]); tt = np.array([r_['t'] for r_ in rows])
            rat = [float(np.max(np.abs(RT.body((V[i + 1] - V[i]) / (tt[i + 1] - tt[i]), float(X[i, 2]))) / AMAXB)) for i in range(len(tt) - 1)]
            bad0 = max(r_ for i, r_ in enumerate(rat) if tt[i] <= tau[0] + 1e-9) > 1.0 + 1e-9 if tau[0] > 0 else False
            bad1 = max(r_ for i, r_ in enumerate(rat) if tt[i + 1] >= tt[-1] - tau[-1] - 1e-9) > 1.0 + 1e-9 if tau[-1] > 0 else False
            if not (bad0 or bad1):
                return rows
            if bad0:
                tau[0] *= 1.5
                dt[0] = max(dt[0], 0.5 * (tau[0] + tau[1]))
            if bad1:
                tau[-1] *= 1.5
                dt[-1] = max(dt[-1], 0.5 * (tau[-1] + tau[-2]))
            v = [(np.asarray(W[k + 1]) - np.asarray(W[k])) / dt[k] for k in range(len(W) - 1)]
            vext = [np.asarray(v0, float)] + v + [np.asarray(v1, float)]
        raise RuntimeError('端點混合加長 40 次仍超加速度上限')

    R1 = timed(W1, np.zeros(9), vt(psi_h))
    R3 = timed(W3p, vt(psr), np.zeros(9))

    # ---- 端點核對（位置、速度、精確時刻；解析速度與取樣同一條軌跡）＋加速度（解析速度差分，車體座標）
    def endpoint_check(rows, W, v0, v1, name):
        X = np.array([r_['x'] for r_ in rows]); V = np.array([r_['v'] for r_ in rows]); tt = np.array([r_['t'] for r_ in rows])
        acc = np.array([RT.body((V[i + 1] - V[i]) / (tt[i + 1] - tt[i]), float(X[i, 2])) for i in range(len(tt) - 1)])
        ratio = np.max(np.abs(acc) / AMAXB, axis=0)
        vb = np.array([RT.body(V[i], float(X[i, 2])) for i in range(len(tt))])
        return {'name': name, 'T_end_s': float(tt[-1]), 'last_dt_s': float(tt[-1] - tt[-2]),
                'start_pos_err': float(np.max(np.abs(X[0] - W[0]))), 'start_vel_err': float(np.max(np.abs(V[0] - v0))),
                'end_pos_err': float(np.max(np.abs(X[-1] - W[-1]))), 'end_vel_err': float(np.max(np.abs(V[-1] - v1))),
                'v_start_base_mps': float(np.hypot(*V[0, :2])), 'v_end_base_mps': float(np.hypot(*V[-1, :2])),
                'acc_over_limit_ratio_max': float(ratio.max()), 'acc_ratio_axis': ratio.tolist(),
                'vel_over_limit_ratio_max': float(np.max(np.abs(vb) / VMAXB))}
    e1 = endpoint_check(R1, W1, np.zeros(9), vt(psi_h), 'R1')
    e3 = endpoint_check(R3, W3p, vt(psr), np.zeros(9), 'R3')
    X3r = np.array([r_['x'] for r_ in R3])
    k_pre = int(np.argmin(np.linalg.norm(X3r[:, :2] - Pr[i_r], axis=1)))
    r3pre = {'n_pre_m': N_PRE_M, 'start_xy': (Pr[i_r] - N_PRE_M * ur).tolist(), 'end_xy': Pr[i_r].tolist(), 'yaw': psr,
             'tau_pre_end': float(R3[k_pre]['t']), 'pre_end_dist_m': float(np.linalg.norm(X3r[k_pre, :2] - Pr[i_r])),
             'r2_vertices': r2v.tolist(), 'r2_check': r2_check, 'straight_m': L_straight, 'J_idx': J_IDX, 'fillet_r': FILLET_R}
    print(f"R3 前置直線：τ_pre_end {r3pre['tau_pre_end']:.2f} s（取樣距 P_r {r3pre['pre_end_dist_m'] * 1e3:.1f} mm）", flush=True)
    for e in (e1, e3):
        print(f"端點 {e['name']}：起 位置差 {e['start_pos_err']:.2e}、速度差 {e['start_vel_err']:.2e}（{e['v_start_base_mps']:.4f} m/s）；"
              f"終 位置差 {e['end_pos_err']:.2e}、速度差 {e['end_vel_err']:.2e}（{e['v_end_base_mps']:.4f} m/s）、T_end {e['T_end_s']:.3f}（末間隔 {e['last_dt_s']:.3f}）；"
              f"加速度／上限 max {e['acc_over_limit_ratio_max']:.3f}、速度／上限 max {e['vel_over_limit_ratio_max']:.3f}", flush=True)

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
    rho_h = env.admit(STOW, STOW, (QF, QF))
    L2 = float(np.sum(np.linalg.norm(np.diff(Pr[i_h:i_r + 1], axis=0), axis=1)))
    d_acc = (V_NAV ** 2 - V_HAND ** 2) / (2 * A_NAV)
    T2 = (L2 - 2 * d_acc) / V_NAV + 2 * (V_NAV - V_HAND) / A_NAV if L2 > 2 * d_acc else 2 * math.sqrt(L2 / A_NAV)
    X1 = np.array([r_['x'] for r_ in R1]); t1 = np.array([r_['t'] for r_ in R1])
    sp1 = np.linalg.norm(np.diff(X1[:, :2], axis=0), axis=1) / np.diff(t1)
    i_arm = next(i for i in range(len(X1)) if float(np.max(np.abs(X1[i, 3:] - STOW))) < 0.05)
    r1_metric = {'yaw_deg': math.degrees(float(X1[-1, 2] - X1[0, 2])), 'disp_m': float(np.linalg.norm(X1[-1, :2] - X1[0, :2])),
                 'frac_time_v_lt_5mm_s': float(np.mean(sp1 < 0.005)), 't_arm_within_0p05_s': float(t1[i_arm]),
                 'v_end_mps': float(sp1[-1])}
    tot = c1['T_s'] + T2 + c3['T_s']
    yaw_fwd_err = [abs(math.degrees(wrap(W3p[n_pre + a, 2] - tang[k]))) for a, k in enumerate(ks)]   # v4b：前置列 n_pre 之後才對應路徑點 ks
    out = {'schema': 'door_regrasp_m4r/4b', 'ref': os.path.relpath(REF, HERE), 'q_door_deg': QD, 'start': S.tolist(), 'B1': B1.tolist(),
           'params': {'D_PREF': D_PREF, 'K_PEN': K_PEN, 'D_NAV': D_NAV, 'V_NAV_nominal': V_NAV, 'DS': DS, 'YAW_STEP_deg': YAW_STEP,
                      'MAX_DY': MAX_DY, 'W_FWD': W_FWD, 'W_SMOOTH': W_SMOOTH, 'L_BLEND': L_BLEND, 'V3': V3, 'W3_yaw_ref': W3,
                      'V_HAND': V_HAND, 'A_NAV': A_NAV, 'V_ARM_FRAC': V_ARM_FRAC},
           'not_claimed': '名目時間不含導航閉迴路、交棒與 5 Hz 振盪；不可宣稱比實際舊流程快',
           'path': {'xy': Pr.tolist(), 'd_m': dP.tolist(), 'tangent': tang.tolist(), 'length_m': L},
           'handover_nav': {'i': i_h, 'xy': Pr[i_h].tolist(), 'yaw': float(psi_h), 'd_m': float(dP[i_h]), 'rho_admit': rho_h[1]},
           'R3_pre': r3pre,
           'handover_wb': {'i': i_r, 'xy': Pr[i_r].tolist(), 'yaw': float(tang[i_r]), 'd_m': float(dP[i_r])},
           'endpoint_check': {'R1': e1, 'R3': e3},
           'R1': {'check': c1, 'near_in_place': r1_metric, 'm_arm': m_arm, 'rows': [{'t': r_['t'], 'x': np.asarray(r_['x']).tolist(), 'v': np.asarray(r_['v']).tolist()} for r_ in R1]},
           'R2': {'length_m': L2, 'T_nominal_s': T2, 'd_min_m': float(dP[i_h:i_r + 1].min()), 'heading': 'tangent'},
           'R3': {'check': c3, 'n_dp_checks': nchk, 'yaw_fwd_err_deg': yaw_fwd_err,
                  'rows': [{'t': r_['t'], 'x': np.asarray(r_['x']).tolist(), 'v': np.asarray(r_['v']).tolist()} for r_ in R3]},
           'total_s': tot, 'old_mode4_s': 140.4, 'wall_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for c in (c1, c3):
        print(f"{c['name']}：{c['T_s']:.2f} s、第一個阻點 {c['first_bad']}、最小淨空 非手指 {c['min_clear_m']['nonfinger'] * 1e3:.1f} mm／"
              f"自碰 {c['min_clear_m']['self'] * 1e3:.1f}／底盤 {c['min_clear_m']['base'] * 1e3:.1f}；ω max {c['wz_max']:.3f}、v max {c['v_max']:.4f}")
    print(f'R2 名目 {L2:.2f} m／{T2:.1f} s（d min {out["R2"]["d_min_m"]:.3f}）；交導航 ρ {rho_h[1]["rho_m"]:.4f}')
    print(f'R1 近原地：轉 {r1_metric["yaw_deg"]:.1f}°、位移 {r1_metric["disp_m"]:.3f} m、速度 < 5 mm/s 時間比 {r1_metric["frac_time_v_lt_5mm_s"]:.0%}、'
          f'手臂到 STOW±0.05 於 {r1_metric["t_arm_within_0p05_s"]:.1f} s、末段平均速 {r1_metric["v_end_mps"]:.4f} m/s（規劃末速 V_HAND {V_HAND}，混合 τ 短於取樣間隔）')
    print(f'R3 偏航：起 {math.degrees(W3p[0, 2]):.1f}° → 終 {math.degrees(W3p[-1, 2]):.1f}°；朝前誤差（前段 50%）max {max(yaw_fwd_err[:len(ks) // 2]):.1f}°')
    print(f'合計 {tot:.1f} s（舊 mode 4 140.4 s）；寫 {OUT}；{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
