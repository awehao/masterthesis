#!/usr/bin/env python3
"""連續協同通行：聯合搜尋 v2——自門扇背面放手狀態反向，要求開門末段已朝出口推進（離線靜態，無 Isaac；單一程序＋溫度暫停）。

v1（door_coopen_joint_v1，正向 beam、評分 −D）診斷：φ45 s0.80 第一個起點 44–65° 的 132 次放手評估全部是 front_side——
貪婪正向搜尋被把手帶向鉸鏈側（20° (0.46, −0.56) → 50° (0.12, −1.00)），不會先「繞過自由端」（那段暫時離出口更遠）；
既有的鉤形解（edgehold_v1）是由背面種子反向找到的。v1 只跑了 1 個起點即停止，部分紀錄在 door_coopen_joint_v1_partial.log。
本版（明列有限範圍）：
  夾法 φ ∈ {45, 60, 75}°、s ∈ {0.80, 0.85}、roll 0；放手門角 q_rel ∈ {48, 50, 52, 54, 56, 58, 60}°。
  關 1（背面放手狀態）：底盤格點每 0.04 m，門扇背面（side ≥ 0.05 m）、門角包絡 [q_rel − 2, q_rel + 5]° 淨空場 ≥ 30 mm 且最寬路徑瓶頸 ≥ 30 mm；
    偏航＝「底盤→抓取點」方位 ±{0, 15, 30, 45, 60}°，粗篩同 edgegrasp_v1；IK（多種子）、判準同 traj_v1＋底盤 ≥ 30 mm；
    再核門靜止時仍握持沿出口路徑（pass_smooth_v1.dijkstra）前進 ≥ 3 cm（至多 10 cm），前進終點同樣滿足包絡淨空與瓶頸 ≥ 30 mm。
  關 2（末段推進）：自持握末端反向 N＝10°（beam 寬 5、每親 8），評分＝反向時 D_q_rel 增加（＝正向 D 下降），D_q_rel＝包絡場上到出口的測地距離
    （引導評分對不可通行格點加「到最近可通行格點距離＋1.0 m」）；**推進量只用真實測地距離計算**：正向末 10° 兩端都在可通行區才計，
    否則記 blend_start_not_on_exit_route（首跑誤用含懲罰值，φ60 s0.85 q50 報出 1.34 m 假推進——底盤是在讓出門扇將掃過的位置，不是朝出口；已停止並重跑）；
    成立條件：真實 D 下降 ≥ 0.05 m，且開門末 3 步方向與握持前進方向轉角 ≤ 60°。
  關 3（完整持握）：關 2 成立者依 D 下降排序取前 3，自其第 10° 狀態續反向到 0°（評分往門前側 (0.5, −1.0)，同 edgehold_v2）。
每關報淘汰計數；找不到就如實報告卡在哪一關。
**界線**：格點與取樣（非距離下界）；剛性夾持、靜態；門餘轉以包絡代替；只在本範圍。

    python3 evaluation/door_coopen_joint_v2.py   # 寫 results/door/door_coopen_joint_v2.json（不覆寫）
    python3 evaluation/door_coopen_joint_v2.py 60 0.85 48   # 只跑單組，寫 door_coopen_joint_v2_phi60_s0.85_q48.json
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
import door_coopen_edgehold_v1 as EH                   # noqa: E402
import door_coopen_pass_smooth_v1 as PS                # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_post_v1 as PO                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_coopen_widest_v1 as WD                     # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v2.json')
XS, YS = PO.DG.XS, PO.DG.YS
PLAN, N_BLEND = 0.030, 10
GRASPS = [(phi, s, 0) for phi in (45, 60, 75) for s in (0.80, 0.85)]
Q_RELS = (48, 50, 52, 54, 56, 58, 60)
STEPS = EH.STEPS


def d_map(F):
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
    return np.where(fin, D, D[ii, jj] + dist + 1.0), D


def D_at(Dm, b):
    i = int(np.clip(round((b[0] - XS[0]) / (XS[1] - XS[0])), 0, len(XS) - 1))
    j = int(np.clip(round((b[1] - YS[0]) / (YS[1] - YS[0])), 0, len(YS) - 1))
    return float(Dm[i, j])


def side_m(x, y, q):
    return (x - 0.475 * math.cos(q)) * math.sin(q) + (y + 0.475 * math.sin(q)) * math.cos(q)


def check(b, q, T, seed, full=False):
    r = TJ.check(b, q, T, seed, full=full)
    return r if (r is not None and r['clear']['base'] >= PLAN) else None


def stage1(g, q_rel, F, rej):
    q = math.radians(q_rel)
    T = EG.grasp_T(q, *g)
    pg, az = T[:3, 3], T[:3, 2]
    W = RS.to_goal_bottleneck(F)
    out = []
    for x in np.round(np.arange(0.40, 1.401, 0.04), 2):
        for y in np.round(np.arange(-1.40, -0.199, 0.04), 2):
            if side_m(x, y, q) < 0.05:
                continue
            fx = PS.F_at(F, (x, y))
            i = int(np.argmin([abs(v - x) for v in XS]))
            j = int(np.argmin([abs(v - y) for v in YS]))
            if fx < PLAN or W[i, j] < PLAN:
                rej['budget'] = rej.get('budget', 0) + 1
                continue
            brg = math.atan2(pg[1] - y, pg[0] - x)
            got = False
            for dy in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
                yaw = brg + math.radians(dy)
                m = np.array([x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw), 0.574])
                v = pg - m
                if not (0.12 <= math.hypot(v[0], v[1]) <= 0.50) or float(az @ v) / (np.linalg.norm(v) + 1e-12) < math.cos(math.radians(80)):
                    continue
                TJ.cool()
                b = np.array([x, y, yaw])
                r = check(b, q, T, None, full=True)
                if r is None:
                    continue
                got = True
                gp = PS.dijkstra(F, b[:2])
                if gp is None:
                    rej['no_exit_path'] = rej.get('no_exit_path', 0) + 1
                    continue
                P, _ = PS.smooth(F, gp, b[:2])
                seed, fwd = r['qa'], []
                for k in range(1, 11):
                    kk = min(k, len(P) - 1)
                    bb = np.array([P[kk][0], P[kk][1], yaw])
                    rr = check(bb, q, T, seed)
                    if rr is None:
                        break
                    seed = rr['qa']
                    fwd.append(bb)
                if len(fwd) < 3:
                    rej['fwd_lt_3cm'] = rej.get('fwd_lt_3cm', 0) + 1
                    continue
                rb = fwd[-1]
                ii = int(np.argmin([abs(v - rb[0]) for v in XS]))
                jj = int(np.argmin([abs(v - rb[1]) for v in YS]))
                if PS.F_at(F, rb[:2]) < PLAN or W[ii, jj] < PLAN:
                    rej['release_budget'] = rej.get('release_budget', 0) + 1
                    continue
                out.append({'base': b, 'qa': r['qa'], 'jm': r['jm'], 'clear': r['clear'], 'fwd_m': float(np.linalg.norm(rb[:2] - b[:2])),
                            'v_fwd': (rb[:2] - b[:2]).tolist(), 'release_base': rb.tolist(), 'path': P.tolist()})
            if not got:
                rej['ik'] = rej.get('ik', 0) + 1
    return out


def blend(g, q_rel, s1, Dm, Dtrue):
    T_of = lambda q: EG.grasp_T(q, *g)                 # noqa: E731
    st0 = {'base': s1['base'], 'qa': s1['qa'], 'jm': s1['jm'], 'clear': s1['clear'], 'D': D_at(Dm, s1['base'])}
    layer = [dict(st0, path=[dict(st0, q=q_rel)])]
    for qd in range(q_rel - 1, q_rel - N_BLEND - 1, -1):
        TJ.cool()
        q = math.radians(qd)
        nxt = []
        for st in layer:
            nb = []
            for dx in STEPS:
                for dy in STEPS:
                    for dyaw in (-4, 0, 4):
                        b = np.array([st['base'][0] + dx, st['base'][1] + dy, st['base'][2] + math.radians(dyaw)])
                        nb.append((D_at(Dm, b), b))
            nb.sort(key=lambda z: -z[0])
            k = 0
            for sc, b in nb:
                if k >= 8:
                    break
                r = check(b, q, T_of(q), st['qa'])
                if r is None:
                    continue
                k += 1
                ns = {'base': b, 'qa': r['qa'], 'jm': r['jm'], 'clear': r['clear'], 'D': sc}
                ns['path'] = st['path'] + [dict(ns, q=qd)]
                nxt.append(ns)
        if not nxt:
            return None
        nxt.sort(key=lambda z: -z['D'])
        seen, layer = set(), []
        for c_ in nxt:
            key = (round(c_['base'][0], 2), round(c_['base'][1], 2), round(math.degrees(c_['base'][2]) / 3))
            if key not in seen:
                seen.add(key)
                layer.append(c_)
            if len(layer) >= 5:
                break
    best = layer[0]
    fw = best['path'][::-1]
    d0, d1 = D_at(Dtrue, fw[0]['base']), D_at(Dtrue, fw[-1]['base'])
    drop = (d0 - d1) if (math.isfinite(d0) and math.isfinite(d1)) else None
    v_open = np.array(fw[-1]['base'][:2]) - np.array(fw[-4]['base'][:2])
    vf = np.array(s1['v_fwd'])
    turn = None
    if np.linalg.norm(v_open) > 1e-3 and np.linalg.norm(vf) > 1e-3:
        turn = math.degrees(math.acos(float(np.clip(v_open @ vf / np.linalg.norm(v_open) / np.linalg.norm(vf), -1, 1))))
    return {'D_drop_m': None if drop is None else float(drop), 'D_guided_drop_m': float(fw[0]['D'] - fw[-1]['D']),
            'turn_open_fwd_deg': turn, 'state10': best, 'fw': fw}


def main():
    global OUT, GRASPS, Q_RELS
    if len(sys.argv) > 1:                              # 單組：phi s q_rel（與全掃描同一程式，輸出另檔）
        phi, s_, qr = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
        GRASPS, Q_RELS = [(phi, s_, 0)], (qr,)
        OUT = OUT.replace('.json', f'_phi{phi:g}_s{s_:g}_q{qr}.json')
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    fields = {q: RS.field(float(q)) for q in Q_RELS}
    dmaps = {q: d_map(fields[q]) for q in Q_RELS}     # (引導用含懲罰, 真實測地距離)
    survivors, summary = [], []
    for g in GRASPS:
        for q_rel in Q_RELS:
            rej = {}
            s1 = stage1(g, q_rel, fields[q_rel], rej)
            n2, ok2 = 0, []
            for c in s1:
                b = blend(g, q_rel, c, *dmaps[q_rel])
                n2 += 1
                if b is None:
                    rej['blend_infeasible'] = rej.get('blend_infeasible', 0) + 1
                    continue
                if b['D_drop_m'] is None:
                    rej['blend_start_not_on_exit_route'] = rej.get('blend_start_not_on_exit_route', 0) + 1
                    continue
                if b['D_drop_m'] < 0.05:
                    rej['blend_D_drop_lt5cm'] = rej.get('blend_D_drop_lt5cm', 0) + 1
                    continue
                if b['turn_open_fwd_deg'] is None or b['turn_open_fwd_deg'] > 60:
                    rej['blend_turn_gt60'] = rej.get('blend_turn_gt60', 0) + 1
                    continue
                ok2.append((b, c))
            best_drop = max((bb['D_drop_m'] for bb, _ in ok2), default=None)
            summary.append({'grasp': list(g), 'q_rel_deg': q_rel, 'n_stage1': len(s1), 'n_stage2_ok': len(ok2), 'reject': rej, 'best_D_drop_m': best_drop})
            for b, c in ok2:
                survivors.append({'grasp': list(g), 'q_rel_deg': q_rel, 'stage1': c, 'blend': b})
            print(f"φ{g[0]} s{g[1]:.2f} q{q_rel}：關1 {len(s1)}、關2 通過 {len(ok2)}（最佳 D 降 {best_drop}）；淘汰 {rej}；{time.time() - t0:.0f} s", flush=True)
    survivors.sort(key=lambda z: -z['blend']['D_drop_m'])
    full = []
    for sv in survivors[:3]:
        g, q_rel = tuple(sv['grasp']), sv['q_rel_deg']
        s10 = sv['blend']['state10']
        seed = {'x': float(s10['base'][0]), 'y': float(s10['base'][1]), 'yaw_deg': math.degrees(s10['base'][2]), 'phi': g[0], 's': g[1], 'roll': g[2],
                'qa': [float(v) for v in s10['qa']], 'jm': s10['jm'], 'clear': s10['clear']}
        orig = TJ.check
        TJ.check = lambda b, q, T, sd, push=False, full=False: (lambda r: r if (r is not None and r['clear']['base'] >= PLAN) else None)(orig(b, q, T, sd, push=push, full=full))
        try:
            layer, qend, why = EH.run(seed, q_rel - N_BLEND, 0, EH.FRONT)
        finally:
            TJ.check = orig
        P = layer[0]['path'][::-1] + [dict(p, q=p['q']) for p in sv['blend']['fw'][1:]]
        rec = {'grasp': list(g), 'q_rel_deg': q_rel, 'reached_q_deg': qend, 'why': why, 'D_drop_last10_m': sv['blend']['D_drop_m'],
               'turn_open_fwd_deg': sv['blend']['turn_open_fwd_deg'], 'fwd_m': sv['stage1']['fwd_m'], 'release_base': sv['stage1']['release_base'],
               'exit_path': sv['stage1']['path'],
               'trajectory': [{'q_deg': float(p['q']), 'base': [float(v) for v in p['base']], 'qa': [float(v) for v in p['qa']],
                               'clear': p['clear'], 'jm': p['jm']} for p in P] if qend == 0 else None}
        full.append(rec)
        print(f"完整持握 φ{g[0]} s{g[1]:.2f} q{q_rel}：反向到 {qend}°（{why}）；末 10° D 降 {rec['D_drop_last10_m']:.3f} m、轉角 {rec['turn_open_fwd_deg']}；{time.time() - t0:.0f} s", flush=True)
    json.dump({'schema': 'door_coopen_joint/2', 'grasps': GRASPS, 'q_rels': Q_RELS, 'plan_m': PLAN, 'n_blend_deg': N_BLEND,
               'summary': summary, 'n_survivors': len(survivors),
               'survivors_brief': [{'grasp': s['grasp'], 'q_rel_deg': s['q_rel_deg'], 'D_drop_m': s['blend']['D_drop_m'],
                                    'turn_deg': s['blend']['turn_open_fwd_deg'], 'base': s['stage1']['base'].tolist()} for s in survivors],
               'full': full, 'not_modeled': ['格點與取樣（非距離下界）', '剛性夾持、靜態', '門餘轉以包絡代替', '本範圍外的夾法與放手門角']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
