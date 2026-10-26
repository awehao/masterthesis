#!/usr/bin/env python3
"""連續協同通行：持握末段＋放手位置＋後續繞行的聯合搜尋 v1（Codex 20261026_213833 §6-2；離線靜態，無 Isaac；單一程序＋溫度暫停）。

目標（Howard 原要求，Codex 固定措辭）：開門期間底盤就沿通往出口的路徑繞行；解除持握前後不另安排停止、退離再重新起步；不要求握到整機通過。
出口進展 D(x, y)：門角 Q_REF＝60° 時、底盤淨空 ≥ 30 mm 的格點上，到 y ≥ 0.45 的測地距離（8 連通 Dijkstra）；
  不可通行格點取「到最近可通行格點的距離 ＋ 該格 D ＋ 1.0 m」（會被門扇掃過的位置自然被避開）。開門中 D 下降＝朝出口進展（允許後退，不要求分量同號）。
明列有限搜尋範圍：
  夾法 φ ∈ {45, 60, 75}°、s ∈ {0.80, 0.85}、roll 0（6 種；φ 60、s 0.85 為既有主夾法）；
  起點（門 0°）：x 0.70–1.20、y −0.70 – −0.32（每 0.06 m）、偏航＝「底盤→抓取點」方位 ±{0, 30, 60}°，粗篩同 edgegrasp_v1，
    IK（多種子）可行且底盤 ≥ 30 mm、非手指／手指／自碰 ≥ 5 mm；每種夾法取 D 最小的 3 個（彼此 ≥ 0.10 m）；
  正向 beam：門角每 1° 到 65°，底盤 Δx、Δy ∈ {−0.04…0.04}（0.02 步）、Δ偏航 ±4°，寬 6、每親 8；評分＝−D；判準同 traj_v1＋底盤 ≥ 30 mm。
放手評估（q ≥ 44°，beam 每層全部狀態）：
  1. 底盤在門扇背面（side ≥ 0.05 m）；
  2. 門靜止於 q：以 door_coopen_pass_smooth_v1.dijkstra 在門角包絡 [q − 2, q + 5]° 的淨空場求自底盤到出口的路徑，
     夾爪仍握持（TCP 固定）、底盤沿該路徑前進 1 cm 一步至多 10 cm（IK 以前一點為種子、相鄰 ≤ 0.10 rad、底盤 ≥ 30 mm）；前進 ≥ 3 cm 才算成立；
  3. 放手點＝前進終點：包絡場淨空 ≥ 30 mm 且到出口的最寬路徑瓶頸 ≥ 30 mm；
  4. 連續性：開門末 3 步方向 v_open、握持前進方向 v_fwd、通過起頭方向 v_pass 兩兩轉角；開門末 3 步 D 下降量。
  候選分數＝max(轉角)（小者佳），並列報；不在本檔做完整軌跡核驗（另檔，用同一條完整軌跡）。
**界線**：格點與取樣（非距離下界，候選須另做下界核驗）；剛性夾持、靜態；門餘轉以包絡代替；只用本範圍內的夾法與起點。

    python3 evaluation/door_coopen_joint_v1.py   # 寫 results/door/door_coopen_joint_v1.json（不覆寫）
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_pass_smooth_v1 as PS                # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_post_v1 as PO                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_coopen_widest_v1 as WD                     # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v1.json')
C = RV.C
XS, YS = PO.DG.XS, PO.DG.YS
PLAN, Q_REF, Q_END = 0.030, 60, 65
GRASPS = [(phi, s, 0) for phi in (45, 60, 75) for s in (0.80, 0.85)]
STEPS = (-0.04, -0.02, 0.0, 0.02, 0.04)


def d_map():
    F = PO.grid(float(Q_REF))
    ok = F >= PLAN
    D = np.full(F.shape, np.inf)
    pq = []
    for i in range(len(XS)):
        for j in range(len(YS)):
            if YS[j] >= 0.45 and ok[i, j]:
                D[i, j] = 0.0
                heapq.heappush(pq, (0.0, i, j))
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > D[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS) and ok[ni, nj]:
                    nd = d + math.hypot(XS[ni] - XS[i], YS[nj] - YS[j])
                    if nd < D[ni, nj]:
                        D[ni, nj] = nd
                        heapq.heappush(pq, (nd, ni, nj))
    fin = np.isfinite(D)
    dist, (ii, jj) = ndimage.distance_transform_edt(~fin, return_indices=True, sampling=(XS[1] - XS[0], YS[1] - YS[0]))
    Dfull = np.where(fin, D, D[ii, jj] + dist + 1.0)
    return Dfull


def D_at(Dm, b):
    i = int(np.clip(round((b[0] - XS[0]) / (XS[1] - XS[0])), 0, len(XS) - 1))
    j = int(np.clip(round((b[1] - YS[0]) / (YS[1] - YS[0])), 0, len(YS) - 1))
    return float(Dm[i, j])


def side_m(x, y, q):
    return (x - 0.475 * math.cos(q)) * math.sin(q) + (y + 0.475 * math.sin(q)) * math.cos(q)


def check(b, q, T, seed, full=False):
    r = TJ.check(b, q, T, seed, full=full)
    return r if (r is not None and r['clear']['base'] >= PLAN) else None


def starts(g, Dm):
    T0 = EG.grasp_T(0.0, *g)
    pg, az = T0[:3, 3], T0[:3, 2]
    out = []
    for x in np.round(np.arange(0.70, 1.201, 0.06), 2):
        for y in np.round(np.arange(-0.70, -0.319, 0.06), 2):
            brg = math.atan2(pg[1] - y, pg[0] - x)
            for dy in (-60, -30, 0, 30, 60):
                yaw = brg + math.radians(dy)
                m = np.array([x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw), 0.574])
                v = pg - m
                if not (0.12 <= math.hypot(v[0], v[1]) <= 0.50) or float(az @ v) / (np.linalg.norm(v) + 1e-12) < math.cos(math.radians(80)):
                    continue
                TJ.cool()
                b = np.array([x, y, yaw])
                r = check(b, 0.0, T0, None, full=True)
                if r is not None:
                    out.append({'base': b, 'qa': r['qa'], 'jm': r['jm'], 'clear': r['clear'], 'D': D_at(Dm, b)})
    out.sort(key=lambda z: z['D'])
    pick = []
    for o in out:
        if all(np.hypot(*(o['base'][:2] - p['base'][:2])) >= 0.10 for p in pick):
            pick.append(o)
        if len(pick) >= 3:
            break
    return pick, len(out)


def release_eval(st, prev3, q_deg, g):
    q = math.radians(q_deg)
    b = st['base']
    if side_m(b[0], b[1], q) < 0.05:
        return None
    F = RS.field(float(q_deg))
    gp = PS.dijkstra(F, b[:2])
    if gp is None:
        return None
    P, s = PS.smooth(F, gp, b[:2])
    T = EG.grasp_T(q, *g)
    seed, fwd = st['qa'], []
    for k in range(1, 11):
        kk = min(k, len(P) - 1)
        bb = np.array([P[kk][0], P[kk][1], b[2]])
        r = check(bb, q, T, seed)
        if r is None:
            break
        seed = r['qa']
        fwd.append({'base': bb.tolist(), 'qa': r['qa'].tolist(), 'clear': r['clear'], 'jm': r['jm']})
    if len(fwd) < 3:
        return None
    rb = np.array(fwd[-1]['base'])
    fr = PS.F_at(F, rb[:2])
    k_rel = len(fwd)
    W = WD.widest_field(F, rb[:2])
    if fr < PLAN or W is None or W < PLAN:
        return None
    v_open = np.array(b[:2]) - np.array(prev3[:2])
    v_fwd = rb[:2] - np.array(b[:2])
    v_pass = P[min(k_rel + 10, len(P) - 1)] - P[k_rel]

    def ang(a, c):
        if np.linalg.norm(a) < 1e-3 or np.linalg.norm(c) < 1e-3:
            return None
        return math.degrees(math.acos(float(np.clip(a @ c / np.linalg.norm(a) / np.linalg.norm(c), -1, 1))))
    t1, t2 = ang(v_open, v_fwd), ang(v_fwd, v_pass)
    return {'q_deg': q_deg, 'base_hold_end': b.tolist(), 'fwd_m': float(np.linalg.norm(v_fwd)), 'release_base': rb.tolist(),
            'release_F_m': fr, 'bottleneck_m': W, 'turn_open_fwd_deg': t1, 'turn_fwd_pass_deg': t2,
            'open_move_m': float(np.linalg.norm(v_open)), 'path': P.tolist(), 'fwd': fwd}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    Dm = d_map()
    res = []
    for g in GRASPS:
        st_list, n_ok = starts(g, Dm)
        print(f"夾法 φ{g[0]} s{g[1]:.2f}：0° 可行起點 {n_ok}，取 {[tuple(np.round(o['base'][:2], 2)) for o in st_list]}；{time.time() - t0:.0f} s", flush=True)
        for s0 in st_list:
            T_of = lambda q: EG.grasp_T(q, *g)        # noqa: E731
            layer = [dict(s0, path=[dict(s0, q=0)])]
            cands, why, qend = [], 'end', Q_END
            for qd in range(1, Q_END + 1):
                TJ.cool()
                q = math.radians(qd)
                nxt = []
                for st in layer:
                    nb = []
                    for dx in STEPS:
                        for dy in STEPS:
                            for dyaw in (-4, 0, 4):
                                b = np.array([st['base'][0] + dx, st['base'][1] + dy, st['base'][2] + math.radians(dyaw)])
                                nb.append((-D_at(Dm, b), b))
                    nb.sort(key=lambda z: -z[0])
                    k = 0
                    for sc, b in nb:
                        if k >= 8:
                            break
                        r = check(b, q, T_of(q), st['qa'])
                        if r is None:
                            continue
                        k += 1
                        ns = {'base': b, 'qa': r['qa'], 'jm': r['jm'], 'clear': r['clear'], 'D': -sc}
                        ns['path'] = st['path'] + [dict(ns, q=qd)]
                        nxt.append(ns)
                if not nxt:
                    why, qend = 'no_feasible_child', qd - 1
                    break
                nxt.sort(key=lambda z: z['D'])
                seen, layer = set(), []
                for c_ in nxt:
                    key = (round(c_['base'][0], 2), round(c_['base'][1], 2), round(math.degrees(c_['base'][2]) / 3))
                    if key not in seen:
                        seen.add(key)
                        layer.append(c_)
                    if len(layer) >= 6:
                        break
                if qd >= 44:
                    for st in layer:
                        TJ.cool()
                        prev = st['path'][max(0, len(st['path']) - 4)]['base']
                        e = release_eval(st, prev, qd, g)
                        if e is not None:
                            Ds = [p['D'] for p in st['path'] if 'D' in p]
                            e['D_drop_last3_m'] = float(st['path'][max(0, len(st['path']) - 4)].get('D', math.nan) - st['D'])
                            e['D_first_last'] = [float(Ds[0]) if Ds else None, float(st['D'])]
                            e['hold_path'] = [{'q_deg': p['q'], 'base': [float(v) for v in p['base']], 'qa': [float(v) for v in p['qa']],
                                               'clear': p['clear'], 'jm': p['jm']} for p in st['path']]
                            e['grasp'] = list(g)
                            e['start'] = s0['base'].tolist()
                            cands.append(e)
            best = min(cands, key=lambda c: max(c['turn_open_fwd_deg'] or 180, c['turn_fwd_pass_deg'] or 180), default=None)
            rec = {'grasp': list(g), 'start': s0['base'].tolist(), 'start_D': s0['D'], 'reached_q_deg': qend, 'why': why,
                   'n_release_cands': len(cands), 'best': best,
                   'others': [{k: c[k] for k in ('q_deg', 'release_base', 'release_F_m', 'bottleneck_m', 'turn_open_fwd_deg', 'turn_fwd_pass_deg', 'fwd_m', 'D_drop_last3_m')}
                              for c in sorted(cands, key=lambda c: max(c['turn_open_fwd_deg'] or 180, c['turn_fwd_pass_deg'] or 180))[:10]]}
            res.append(rec)
            bs = '' if best is None else (f"；最佳放手 {best['q_deg']}° 於 {np.round(best['release_base'][:2], 2)}，握持前進 {best['fwd_m']:.2f} m，"
                                          f"轉角 開→前 {best['turn_open_fwd_deg']}／前→過 {best['turn_fwd_pass_deg']}，F {best['release_F_m'] * 1000:.0f}、瓶頸 {best['bottleneck_m'] * 1000:.0f} mm、"
                                          f"末 3 步 D 降 {best['D_drop_last3_m']:.3f} m")
            print(f"  起點 {np.round(s0['base'][:2], 2)}：持握到 {qend}°（{why}），放手候選 {len(cands)}{bs}；{time.time() - t0:.0f} s", flush=True)
    json.dump({'schema': 'door_coopen_joint/1', 'q_ref_deg': Q_REF, 'plan_m': PLAN, 'grasps': GRASPS, 'q_end_deg': Q_END, 'per_start': res,
               'search_range': '夾法 6 種 × 每種至多 3 起點；beam 寬 6、每親 8；放手評估 q ≥ 44°',
               'not_modeled': ['格點與取樣（非距離下界）', '剛性夾持、靜態', '門餘轉以包絡代替', '本範圍外的夾法與起點']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
