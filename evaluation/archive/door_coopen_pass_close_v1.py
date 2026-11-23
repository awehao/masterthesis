#!/usr/bin/env python3
"""A 包第 3–5 步：「後退開門 → 通過 → 通過途中關門」同一條候選的幾何／全身運動學核對（Howard 10/27 最終方向；Codex r38 修正後共同幾何）。
離線運動學，無接觸動力學、無控制；只回答「同一條路徑、同一時間軸上能否接起來」，不成立就報第一個阻點。

起點：bench_6 OPEN 末（門 47.7°）的實測底盤位姿與手臂角（指定初態開門的實跑結果）；抓取＝繞緣夾法 φ60、s 0.85、roll 0（door_coopen_edgegrasp_v1.grasp_T；
     與 joint_v2 q48 候選同）。門形狀／牆＝door_geom_v2.shapes（門扇單側 40 mm 厚、兩側牆），底盤圓盤淨空＝door_coopen_reach_v1.disc_ok（登錄 base 0.02 m）。
第一段（底盤×門角搜尋）：三維格點 (x, y, q)：x 0.05 m、y 0.05 m、門角 2.5° 階；門角只能減（關門由持握拉動）。
  可行狀態：disc_ok(x, y, q)；持握期間（q ≥ Q_REL）把手到手臂安裝點的水平距離 ≤ 0.50 m（安裝點在底盤朝把手方向 0.11 m；近似，第二段以全身 IK 核實）。
  移動：底盤 8 鄰格（門角不變）；或底盤不動、門角降一階（只在持握期間）。成本＝底盤移動距離×(1＋q/47.5)（門開著移動較貴 ⇒ 偏好能關就先關）＋關門一階 0.001。
  目標：底盤中心 y ≥ 0.44（底盤圓盤越過牆厚 0.12＋半徑 0.30＋0.02）。另報：底盤越過牆前（y＜0.44）門角已減少多少（「通過途中關門」）。
第二段（全身連續 IK）：沿第一段路徑（含門角），每步以前一步手臂解為種子（door_offline_check_v1.ik、only_prev），偏航在前一步 ±0.3 rad（7 個）中選淨空最大者；
  核：IK、關節餘裕 ≥ 0.05 rad、單步關節變化 ≤ 0.10 rad、非手指／手指／自碰 ≥ 5 mm、底盤 ≥ 20 mm（door_coopen_traj_v1.check）。第一個不成立的步 ⇒ 阻點（位置、門角、部位）。
界線：取樣間不連續；無接觸力、摩擦、門動力學；持握期間假設把手隨門角（剛性跟隨）；放手、退出縫隙、末段關門接觸另報（本版不含）。

    python3 evaluation/door_coopen_pass_close_v1.py   # 寫 results/door/door_coopen_pass_close_v1.json（不覆寫）
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
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_v1.json')
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
GRASP = (60.0, 0.85, 0.0)
DX, DQ = 0.05, 2.5
XS = np.round(np.arange(0.10, 1.501, DX), 3)
YS = np.round(np.arange(-1.30, 0.701, DX), 3)
QS = np.round(np.arange(0.0, 47.51, DQ), 2)
Q_REL = 0.0                     # 本版：持握約束適用到門角 0（退出縫隙另核）
Y_CLEAR = 0.44
REACH, MOUNT = 0.50, 0.11


def handle_xy(q_deg):
    return KC._H_of(math.radians(q_deg))[:2, 3]


def reach_ok(x, y, q_deg):
    h = handle_xy(q_deg)
    d = float(math.hypot(h[0] - x, h[1] - y))
    return d - MOUNT <= REACH


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    r6 = json.load(open(os.path.join(RUN, 'room_run.json')))
    t6 = json.load(open(os.path.join(RUN, 'task.json')))
    c = {k: i for i, k in enumerate(r6['steps_cols'])}
    t_set = next(e['t'] for e in t6['policy_events'] if e.get('to') == 'SETTLE')
    s_end = [s for s in r6['steps'] if s[c['sim_t']] < t_set][-1]
    b0, qa0, q0 = list(s_end[c['base_xyth']]), np.array(s_end[c['q_arm_meas']], float), math.degrees(s_end[c['door_q_rad']])
    # ---- 第一段 ----
    valid = {}
    for iq, qd in enumerate(QS):
        TJ.cool()
        qr = math.radians(qd)
        for ix, x in enumerate(XS):
            for iy, y in enumerate(YS):
                valid[(ix, iy, iq)] = bool(RV.disc_ok(float(x), float(y), qr)) and reach_ok(x, y, qd)
    ix0 = int(np.argmin(np.abs(XS - b0[0])))
    iy0 = int(np.argmin(np.abs(YS - b0[1])))
    iq0 = int(np.argmin(np.abs(QS - min(q0, QS[-1]))))
    start = (ix0, iy0, iq0)
    if not valid.get(start):
        print('起點格不可行（取最近可行格）', flush=True)
        cand = sorted((k for k, v in valid.items() if v and k[2] == iq0), key=lambda k: (k[0] - ix0) ** 2 + (k[1] - iy0) ** 2)
        start = cand[0]
    dist, prev, pq = {start: 0.0}, {}, [(0.0, start)]
    goal = None
    nb = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while pq:
        d, s = heapq.heappop(pq)
        if d > dist[s] + 1e-12:
            continue
        if YS[s[1]] >= Y_CLEAR:
            goal = s
            break
        ix, iy, iq = s
        moves = [((ix + a, iy + b, iq), DX * math.hypot(a, b) * (1.0 + QS[iq] / 47.5)) for a, b in nb]
        if iq > 0:
            moves.append(((ix, iy, iq - 1), 0.001))
        for n, w in moves:
            if not valid.get(n):
                continue
            nd = d + w
            if nd < dist.get(n, 1e18):
                dist[n], prev[n] = nd, s
                heapq.heappush(pq, (nd, n))
    res = {'schema': 'door_coopen_pass_close/1', 'start': {'base': b0, 'qa': qa0.tolist(), 'q_deg': q0, 'grid_start': [float(XS[start[0]]), float(YS[start[1]]), float(QS[start[2]])]},
           'grid': {'dx': DX, 'dq_deg': DQ, 'reach_proxy_m': REACH, 'y_clear': Y_CLEAR}, 'stage1': None, 'stage2': None}
    if goal is None:
        res['stage1'] = {'found': False, 'why': '格點內無「門角只減、底盤淨空、持握近似可達」到 y ≥ 0.44 的路徑'}
        json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
        print(res['stage1'])
        return
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path = path[::-1]
    P = [{'x': float(XS[a]), 'y': float(YS[b]), 'q_deg': float(QS[k])} for a, b, k in path]
    q_at_clear_start = next((p['q_deg'] for p in P if p['y'] >= 0.0), None)
    closing_before_clear = [p for p in P if p['y'] < Y_CLEAR]
    res['stage1'] = {'found': True, 'n': len(P), 'path': P, 'q_final_deg': P[-1]['q_deg'],
                     'q_min_before_clear_deg': min(p['q_deg'] for p in closing_before_clear),
                     'q_when_base_crosses_y0_deg': q_at_clear_start,
                     'closing_progress_before_clear_deg': P[0]['q_deg'] - min(p['q_deg'] for p in closing_before_clear),
                     'first_close_base': next(({'x': p['x'], 'y': p['y']} for p, pn in zip(P, P[1:]) if pn['q_deg'] < p['q_deg']), None)}
    print('第一段：', {k: v for k, v in res['stage1'].items() if k != 'path'}, flush=True)
    for p in P[::max(1, len(P) // 15)] + [P[-1]]:
        print('   ', p, flush=True)
    # ---- 第二段 ----
    seed, yaw, rows, block = qa0.copy(), float(b0[2]), [], None
    for i, p in enumerate(P):
        TJ.cool()
        qr = math.radians(p['q_deg'])
        T = EG.grasp_T(qr, *GRASP)
        best = None
        for dy in np.linspace(-0.3, 0.3, 7):
            yw = yaw + float(dy)
            r = TJ.check([p['x'], p['y'], yw], qr, T, seed)
            if r is not None and (best is None or min(r['clear'].values()) > min(best[1]['clear'].values())):
                best = (yw, r)
        if best is None:
            dbg = {}
            for dy in (0.0,):
                yw = yaw
                base = [p['x'], p['y'], yw]
                qf9 = RV.C.ik(np.asarray(base, float), T, seed, only_prev=True)
                if qf9 is None:
                    dbg['ik'] = 'IK 失敗（以前一步解為種子）'
                else:
                    qa = np.asarray(qf9, float)[RV.C.IDX]
                    d = TJ.GEO([float(v) for v in base + list(qa)], qr, RV.Q_FINGER)
                    dbg = {'step_rad': float(np.max(np.abs(qa - seed))), 'clear': {k: [float(d[k][0]), d[k][1]] for k in ('nonfinger', 'finger', 'base', 'self')},
                           'disc_ok': bool(RV.disc_ok(p['x'], p['y'], qr))}
            block = {'i': i, 'at': p, 'yaw_prev': yaw, 'detail': dbg}
            break
        yaw, r = best
        seed = r['qa']
        rows.append({'i': i, **p, 'yaw': yaw, 'qa': [float(v) for v in r['qa']], 'jm': r['jm'], 'clear': r['clear']})
    cl = {k: min(rw['clear'][k] for rw in rows) for k in ('nonfinger', 'finger', 'base', 'self')} if rows else None
    res['stage2'] = {'n_ok': len(rows), 'n_total': len(P), 'block': block, 'min_clear_m': cl, 'min_joint_margin_rad': min(rw['jm'] for rw in rows) if rows else None,
                     'rows': rows}
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o))
    print('第二段：通過', len(rows), '/', len(P), '最小淨空', cl, '阻點', json.dumps(block, ensure_ascii=False, default=float)[:900], f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
