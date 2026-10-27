#!/usr/bin/env python3
"""A 包第 3–5 步 v2：全身感知的束搜尋（door_coopen_pass_close_v1 第二段在第 1 步即因 link2 碰門扇而中斷——其底盤路徑只看底盤幾何選出）。
同一條候選、同一時間軸：起點＝bench_6 OPEN 末實測（底盤、手臂、門 47.7°）；持握繞緣夾法 φ60 s0.85 roll 0；門角只減（由持握拉動）。
每步動作：底盤平移 (dx, dy) ∈ {−0.025, 0, 0.025}²、偏航 Δ ∈ {−0.08, 0, 0.08} rad、門角 Δ ∈ {0, −1.0°}；子狀態以 door_coopen_traj_v1.check 核
  （底盤淨空 ≥ 20 mm、IK 以父步手臂解為種子、單步關節 ≤ 0.10 rad、關節餘裕 ≥ 0.05、非手指／手指／自碰 ≥ 5 mm）。
評分（越小越好）：未到門洞前 ＝ 到門洞入口 W(0.62, −0.15) 的距離＋0.6；到門洞後（y ≥ −0.15）＝0.6 − (y＋0.15)；另加 0.002·門角（偏好通過中關門）。
束寬 12、每父 8 子、最多 160 步；到 y ≥ 0.44（整機越過牆厚）即停；最佳評分連續 15 步未改善即提前停（重開機後重跑加入）。走不通 ⇒ 報最深前緣狀態與其子狀態失敗原因分布（具體阻點）。
界線：同 v1（離線運動學；取樣間不連續；無接觸力；持握剛性跟隨；放手、退縫、末段關門接觸另報）。

    python3 evaluation/door_coopen_pass_close_v2.py   # 寫 results/door/door_coopen_pass_close_v2.json（不覆寫）
"""
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

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_v2.json')
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
GRASP = (60.0, 0.85, 0.0)
W = np.array([0.62, -0.15])
Y_CLEAR, WIDTH, PER, MAXS = 0.44, 12, 8, 160
ACT = [(dx, dy, dw, dq) for dx in (-0.025, 0.0, 0.025) for dy in (-0.025, 0.0, 0.025) for dw in (-0.08, 0.0, 0.08) for dq in (0.0, -1.0)
       if not (dx == 0 and dy == 0 and dw == 0 and dq == 0)]


def score(x, y, q):
    if y < W[1]:
        h = float(math.hypot(x - W[0], y - W[1])) + 0.6
    else:
        h = 0.6 - (y - W[1])
    return h + 0.002 * q


def why_fail(base, qr, T, seed):
    if not RV.disc_ok(float(base[0]), float(base[1]), qr):
        return 'base_disc'
    qf9 = RV.C.ik(np.asarray(base, float), T, seed, only_prev=True)
    if qf9 is None:
        return 'ik'
    qa = np.asarray(qf9, float)[RV.C.IDX]
    if float(np.max(np.abs(qa - seed))) > TJ.LIM['joint_step_rad']:
        return 'joint_step'
    jm = float(min(min(qa - TJ.V2.ELO + 0.05), min(TJ.V2.EHI + 0.05 - qa)))
    if jm < TJ.LIM['joint_margin_rad']:
        return 'joint_margin'
    d = TJ.GEO([float(v) for v in list(base) + list(qa)], qr, RV.Q_FINGER)
    for k, lim in (('nonfinger', 'nonfinger_m'), ('base', 'base_m'), ('self', 'self_m'), ('finger', 'finger_m')):
        if d[k][0] < TJ.LIM[lim]:
            return f'{k}:{d[k][1]}'
    return 'ok'


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    r6 = json.load(open(os.path.join(RUN, 'room_run.json')))
    t6 = json.load(open(os.path.join(RUN, 'task.json')))
    c = {k: i for i, k in enumerate(r6['steps_cols'])}
    t_set = next(e['t'] for e in t6['policy_events'] if e.get('to') == 'SETTLE')
    s_end = [s for s in r6['steps'] if s[c['sim_t']] < t_set][-1]
    b0, qa0, q0 = [float(v) for v in s_end[c['base_xyth']]], np.array(s_end[c['q_arm_meas']], float), math.degrees(s_end[c['door_q_rad']])
    r0 = TJ.check(b0, math.radians(q0), EG.grasp_T(math.radians(q0), *GRASP), qa0)
    start = {'b': b0, 'qa': qa0 if r0 is None else r0['qa'], 'q': q0, 'path': [], 'clear': None if r0 is None else r0['clear']}
    beam, best_depth, it = [start], start, 0
    done = None
    while beam and it < MAXS:
        it += 1
        TJ.cool()
        kids = []
        for st in beam:
            cand = []
            for dx, dy, dw, dq in ACT:
                qn = st['q'] + dq
                if qn < 0.0:
                    continue
                b = [st['b'][0] + dx, st['b'][1] + dy, st['b'][2] + dw]
                qr = math.radians(qn)
                r = TJ.check(b, qr, EG.grasp_T(qr, *GRASP), st['qa'])
                if r is None:
                    continue
                cand.append((score(b[0], b[1], qn), {'b': b, 'qa': r['qa'], 'q': qn, 'clear': r['clear'], 'jm': r['jm'],
                                                     'path': st['path'] + [{'b': [float(v) for v in b], 'q': qn, 'qa': [float(v) for v in r['qa']], 'clear': r['clear']}]}))
            cand.sort(key=lambda z: z[0])
            kids.extend(cand[:PER])
        if not kids:
            break
        kids.sort(key=lambda z: z[0])
        uniq, seen = [], set()
        for sc, k in kids:
            key = (round(k['b'][0], 3), round(k['b'][1], 3), round(k['b'][2], 2), round(k['q'], 1))
            if key not in seen:
                seen.add(key)
                uniq.append(k)
        beam = uniq[:WIDTH]
        if score(beam[0]['b'][0], beam[0]['b'][1], beam[0]['q']) < score(best_depth['b'][0], best_depth['b'][1], best_depth['q']):
            best_depth = beam[0]
        hit = [k for k in beam if k['b'][1] >= Y_CLEAR]
        if hit:
            done = min(hit, key=lambda k: k['q'])
            break
        stall = 0 if score(beam[0]['b'][0], beam[0]['b'][1], beam[0]['q']) <= score(best_depth['b'][0], best_depth['b'][1], best_depth['q']) + 1e-12 and \
            best_depth is beam[0] else locals().get('stall', 0) + 1
        if stall >= 15:
            print(f'  步 {it}：最佳評分連續 15 步未改善 ⇒ 提前停', flush=True)
            break
        if it % 10 == 0:
            print(f"  步 {it}：最佳 底盤 ({beam[0]['b'][0]:.3f}, {beam[0]['b'][1]:.3f}, {beam[0]['b'][2]:.2f}) 門 {beam[0]['q']:.1f}°", flush=True)
    res = {'schema': 'door_coopen_pass_close/2', 'start': {'base': b0, 'q_deg': q0, 'start_check_ok': r0 is not None},
           'params': {'grasp': GRASP, 'W': W.tolist(), 'width': WIDTH, 'per_parent': PER, 'max_steps': MAXS, 'n_actions': len(ACT)}}
    if done is not None:
        P = done['path']
        ys = [p['b'][1] for p in P]
        res['found'] = True
        res['path'] = P
        res['summary'] = {'n': len(P), 'q_final_deg': done['q'], 'q_when_y_ge_0': next((p['q'] for p in P if p['b'][1] >= 0.0), None),
                          'closing_before_clear_deg': q0 - min(p['q'] for p in P if p['b'][1] < Y_CLEAR),
                          'min_clear_m': {k: min(p['clear'][k] for p in P) for k in ('nonfinger', 'finger', 'base', 'self')}, 'y_range': [min(ys), max(ys)]}
    else:
        bd = best_depth
        fails = {}
        for dx, dy, dw, dq in ACT:
            qn = bd['q'] + dq
            if qn < 0:
                continue
            b = [bd['b'][0] + dx, bd['b'][1] + dy, bd['b'][2] + dw]
            w = why_fail(b, math.radians(qn), EG.grasp_T(math.radians(qn), *GRASP), np.asarray(bd['qa'], float))
            fails[w] = fails.get(w, 0) + 1
        res['found'] = False
        res['frontier'] = {'base': bd['b'], 'q_deg': bd['q'], 'qa': [float(v) for v in bd['qa']], 'clear': bd['clear'], 'n_steps': len(bd['path']),
                           'child_fail_reasons': fails, 'path': bd['path']}
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o))
    if done is not None:
        print('找到：', res['summary'], f"{res['wall_s']:.0f} s", flush=True)
    else:
        f = res['frontier']
        print('未找到：最深前緣 底盤', np.round(f['base'], 3), '門', round(f['q_deg'], 1), '步數', f['n_steps'], '子狀態失敗原因', f['child_fail_reasons'], f"{res['wall_s']:.0f} s", flush=True)


if __name__ == '__main__':
    main()
