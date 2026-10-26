#!/usr/bin/env python3
"""連續協同通行：候選軌跡搜尋 v1（Codex 20261026_141054 §3；無 Isaac、無接觸力；單一程序、IK 以上一步為種子）。

候選接觸模式「拉 → 換側 → 推著通過」：
  A 拉門（抓持把手）：採用既有 W-GMPC 閉迴路運動學重播（door_wide_offline_sweep_90：剛性夾持、底盤可動、門跟隨夾爪）的軌跡，
    取每 1° 最接近的一列，逐列以同一組判準重核：底盤圓盤 ≥ 20 mm、非手指 ≥ 5 mm、手指（不含設計接觸）≥ 5 mm、自碰 ≥ 5 mm、
    關節餘裕 ≥ 0.05 rad；並報相鄰列手臂關節變化。（首版以貪婪束搜尋自造 A 相，15° 即無可行子節點而停；改用本方法實際產生的軌跡。）
  T 換側（門靜止於 q_A）：放開把手；在門閂側格點找可推門扇背面的起始位姿（door_coopen_push_v1.push_pose，接觸點距鉸鏈 r_c）；
    底盤由抓持終點到推門起點的直線（或經前方中途點）在 q_A 下全程 ≥ 20 mm。手臂換位路徑未核（只核兩端）。
  B 推門前進：門角每 1° 一步（由推觸帶動，假設門跟著推觸點；無力學），底盤 Δ ∈ {−0.04…+0.04}²（0.02 步）、Δ偏航 ±4°，
    接觸點可沿門扇滑動（r_c ±0.03，界於 0.75–0.93）；目標：底盤進入通行帶 x ∈ [0.35, 0.68] 並往 +y 前進。
    直到底盤前緣過牆面（y ≥ −0.30）且門角 ≥ 該 x 的靜態可通過門角（door_coopen_pass_v2）。
  C 直行通過（門靜止）：手臂收到攜行構型（K2_READY），底盤沿 +y 到 y＝0.60，全程底盤與手臂各類淨空照核。
輸出整條軌跡（相、門角、底盤、手臂）、各相首末、開門—通行重疊區間（B 相中門角增加且底盤 y 增加的門角範圍）、
最小淨空、關節餘裕、相鄰步最大關節變化與換算速度（門 1°/s）。
**界線**：無接觸力與摩擦、門被推動的動力學、取樣間碰撞、底盤外凸物、手臂換位與收臂路徑；只是幾何候選，不代表 Isaac 可行。

    python3 evaluation/door_coopen_traj_v1.py [q_A ...]   # 寫 results/door/door_coopen_traj_v1_sweepA.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_push_v1 as PU                       # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_traj_v1_sweepA.json')   # 首版（貪婪 A 相）輸出 door_coopen_traj_v1.json 保留
C, V2, GM = RV.C, RV.V2, RV.GM
LIM = dict(RV.LIM, finger_push_min_m=0.001, joint_step_rad=0.10)
BAND = (0.35, 0.68)
STOW = np.array([0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0])
GEO = GM.default_backend()
THG = RV.T_HG()
PASS = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_pass_v2.json')))
Q_STATIC = {r['x_c']: r['q_static_deg'] for r in PASS['per_xc']}


def cool():
    """單程序仍守溫度：k10temp ≥ 85 °C 時暫停到 ≤ 75 °C。"""
    for h in os.listdir('/sys/class/hwmon'):
        try:
            if open(f'/sys/class/hwmon/{h}/name').read().strip() == 'k10temp':
                while int(open(f'/sys/class/hwmon/{h}/temp1_input').read()) >= 85000:
                    time.sleep(5)
                    if int(open(f'/sys/class/hwmon/{h}/temp1_input').read()) <= 75000:
                        break
        except OSError:
            pass


def q_static_at(x):
    ks = sorted(Q_STATIC)
    if x < ks[0] or x > ks[-1]:
        return None
    k = min(ks, key=lambda v: abs(v - x))
    return Q_STATIC[k]


def check(base, q, T, seed, push=False, full=False):
    if not RV.disc_ok(float(base[0]), float(base[1]), q):
        return None
    qf9 = C.ik(np.asarray(base, float), T, seed, only_prev=not full) if seed is not None or full else None
    if qf9 is None:
        return None
    qa = np.asarray(qf9, float)[C.IDX]
    if seed is not None and not full and float(np.max(np.abs(qa - seed))) > LIM['joint_step_rad']:
        return None
    jm = float(min(min(qa - V2.ELO + 0.05), min(V2.EHI + 0.05 - qa)))
    if jm < LIM['joint_margin_rad']:
        return None
    qf = [float(v) for v in list(base) + list(qa)]
    d = GEO(qf, q, RV.Q_FINGER)
    fmin = LIM['finger_push_min_m'] if push else LIM['finger_m']
    if d['nonfinger'][0] < LIM['nonfinger_m'] or d['base'][0] < LIM['base_m'] or d['self'][0] < LIM['self_m'] or d['finger'][0] < fmin:
        return None
    return {'qa': qa, 'jm': jm, 'clear': {k: float(d[k][0]) for k in ('nonfinger', 'finger', 'base', 'self')}}


def beam(start, q0, q1, T_of, score, deltas, push=False, width=6, per_parent=8, stop=None):
    """start＝{'base', 'qa', 'extra'}；回傳每步最佳鏈（list of states），與是否中途停止。"""
    layer = [dict(start, path=[dict(start, q=q0)])]
    for qd in range(q0 + 1, q1 + 1):
        cool()
        q = math.radians(qd)
        cand = []
        for st in layer:
            nb = []
            for dx, dy, dyaw, ext in deltas(st):
                b = np.array([st['base'][0] + dx, st['base'][1] + dy, st['base'][2] + math.radians(dyaw)])
                nb.append((score(b, ext, qd), b, ext))
            nb.sort(key=lambda z: -z[0])
            k = 0
            for sc, b, ext in nb:
                if k >= per_parent:
                    break
                r = check(b, q, T_of(q, ext), st['qa'], push=push)
                if r is None:
                    continue
                k += 1
                ns = {'base': b, 'qa': r['qa'], 'extra': ext, 'score': sc, 'jm': r['jm'], 'clear': r['clear']}
                ns['path'] = st['path'] + [dict(ns, q=qd)]
                cand.append(ns)
        if not cand:
            return layer, qd - 1, 'no_feasible_child'
        cand.sort(key=lambda z: -z['score'])
        seen, layer = set(), []
        for c_ in cand:
            key = (round(c_['base'][0], 2), round(c_['base'][1], 2), round(math.degrees(c_['base'][2]) / 3))
            if key in seen:
                continue
            seen.add(key)
            layer.append(c_)
            if len(layer) >= width:
                break
        if stop is not None and stop(layer[0], qd):
            return layer, qd, 'stop_condition'
    return layer, q1, 'end'


def ser(path, phase):
    return [{'phase': phase, 'q_deg': p['q'], 'base': [float(v) for v in p['base']], 'qa': [float(v) for v in p['qa']],
             'r_c': (p.get('extra') or {}).get('r_c'), 'jm': p.get('jm'), 'clear': p.get('clear')} for p in path]


def transit_clear(b0, b1, q):
    for via in (None, np.array([0.5 * (b0[0] + b1[0]), min(b0[1], b1[1]) - 0.25, b1[2]])):
        pts = [b0, b1] if via is None else [b0, via, b1]
        ok = True
        for a, b in zip(pts, pts[1:]):
            for s in np.linspace(0.0, 1.0, 26):
                p = a + s * (b - a)
                if not RV.disc_ok(float(p[0]), float(p[1]), q):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True, None if via is None else [float(v) for v in via]
    return False, None


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    sw = json.load(open(os.path.join(HERE, 'results', 'door', 'door_wide_offline_sweep_90.json')))
    q0 = np.array(sw['trajectory'][0]['q'], float)
    start = {'base': q0[:3].copy(), 'qa': q0[3:].copy(), 'extra': {}}
    r0 = check(start['base'], 0.0, KC._H_of(0.0) @ THG, start['qa'], full=False)
    assert r0 is not None, '起點握持不可行'
    start.update(qa=r0['qa'], jm=r0['jm'], clear=r0['clear'])
    gT = lambda q, ext: KC._H_of(q) @ THG                                        # noqa: E731
    dA = lambda st: [(dx, dy, dyaw, {}) for dx in (-0.03, 0.0, 0.03) for dy in (-0.03, 0.0, 0.03) for dyaw in (-3, 0, 3)]   # noqa: E731
    sA = lambda b, ext, qd: float(b[0] + 0.5 * b[1])                            # noqa: E731
    qa_list = [int(a) for a in sys.argv[1:]] or [20, 25, 30, 35, 40]
    best_A, qA_reached, whyA = [], 0, 'end'
    tr = sw['trajectory']
    for qd in range(0, max(qa_list) + 1):
        row = min(tr, key=lambda r_: abs(r_['q_door_deg'] - qd))
        qq = np.array(row['q'], float)
        q_ = math.radians(row['q_door_deg'])
        r_ = check(qq[:3], q_, KC._H_of(q_) @ THG, qq[3:], full=True)
        if r_ is None:
            whyA = f'sweep_row_infeasible_at_{qd}'
            break
        best_A.append({'base': qq[:3], 'qa': r_['qa'], 'extra': {}, 'jm': r_['jm'], 'clear': r_['clear'], 'q': qd, 'q_door_row_deg': row['q_door_deg']})
        qA_reached = qd
    print(f'A：W-GMPC 重播軌跡逐 1° 重核可行至 q {qA_reached}°（{whyA}），{time.time() - t0:.0f} s', flush=True)
    cands = []
    for qa_ in qa_list:
        if qa_ > qA_reached:
            continue
        pA = [p for p in best_A if p['q'] <= qa_]
        endA = pA[-1]
        q = math.radians(qa_)
        # T：門閂側推門起點
        found = []
        for rc in (0.90, 0.85, 0.93):
            Tp = PU.push_pose(q, rc)
            pc = Tp[:3, 3]
            for x in np.arange(0.85, 1.46, 0.05):
                for y in np.arange(-1.0, -0.29, 0.05):
                    for a in range(0, 360, 15):
                        yaw = math.radians(a)
                        brg = math.atan2(pc[1] - y, pc[0] - x)
                        mx, my = x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw)
                        if abs(math.degrees(math.atan2(math.sin(yaw - brg), math.cos(yaw - brg)))) > 60 or not (0.12 <= math.hypot(mx - pc[0], my - pc[1]) <= 0.48):
                            continue
                        if not RV.disc_ok(float(x), float(y), q):
                            continue
                        cool()
                        r = check(np.array([x, y, yaw]), q, Tp, None, push=True, full=True)
                        if r is None:
                            continue
                        ok_t, via = transit_clear(np.asarray(endA['base']), np.array([x, y, yaw]), q)
                        if ok_t:
                            found.append((float(y) - 2.0 * max(0.0, abs(float(x) - 0.55) - 0.13), np.array([x, y, yaw]), r, rc, via))
                if len(found) >= 12:
                    break
            if len(found) >= 12:
                break
        if not found:
            cands.append({'q_A': qa_, 'phase_A': ser(pA, 'A'), 'transition': None, 'why': 'no_push_start'})
            print(f'q_A {qa_}°：找不到可達的推門起點', flush=True)
            continue
        found.sort(key=lambda z: -z[0])
        sc, bT, rT, rcT, via = found[0]
        startB = {'base': bT, 'qa': rT['qa'], 'extra': {'r_c': rcT}, 'jm': rT['jm'], 'clear': rT['clear']}

        def pT(q, ext):
            return PU.push_pose(q, ext['r_c'])

        def dB(st):
            rc0 = st['extra']['r_c']
            out = []
            for drc in (0.0, -0.03, 0.03):
                rc = round(rc0 + drc, 3)
                if not (0.75 <= rc <= 0.93):
                    continue
                for dx in (-0.04, -0.02, 0.0, 0.02, 0.04):
                    for dy in (-0.02, 0.0, 0.02, 0.04):
                        for dyaw in (-4, 0, 4):
                            out.append((dx, dy, dyaw, {'r_c': rc}))
            return out

        def sB(b, ext, qd):
            return float(b[1] - 2.0 * max(0.0, abs(float(b[0]) - 0.515) - 0.165))

        def stopB(st, qd):
            qs = q_static_at(float(st['base'][0]))
            return st['base'][1] >= -0.30 and qs is not None and qd >= qs

        layerB, qB, whyB = beam(startB, qa_, 100, pT, sB, dB, push=True, width=5, per_parent=8, stop=stopB)
        pB = layerB[0]['path']
        endB = pB[-1]
        # C：門靜止、手臂收到攜行構型、直行通過
        qC = math.radians(endB['q'])
        okC, minC = True, {'nonfinger': math.inf, 'base': math.inf, 'self': math.inf}
        for y in np.arange(float(endB['base'][1]), 0.61, 0.02):
            b = [float(endB['base'][0]), float(y), math.pi / 2]
            d = GEO(b + [float(v) for v in STOW], qC, RV.Q_FINGER)
            for k in minC:
                minC[k] = min(minC[k], float(d[k][0]))
        okC = minC['nonfinger'] >= LIM['nonfinger_m'] and minC['base'] >= LIM['base_m'] and minC['self'] >= LIM['self_m']
        ys = [p['base'][1] for p in pB]
        qs_ = [p['q'] for p in pB]
        inc = [qs_[i] for i in range(1, len(pB)) if ys[i] > ys[i - 1] + 1e-9]
        rec = {'q_A': qa_, 'phase_A': ser(pA, 'A'), 'transition': {'push_start_base': [float(v) for v in bT], 'r_c': rcT, 'via': via, 'n_found': len(found)},
               'phase_B': ser(pB, 'B'), 'phase_B_end': {'q_deg': endB['q'], 'why': whyB, 'base': [float(v) for v in endB['base']]},
               'phase_C': {'ok': bool(okC), 'min_clear': minC, 'q_static_deg': q_static_at(float(endB['base'][0]))},
               'overlap_q_deg': [min(inc), max(inc)] if inc else None, 'n_forward_steps_B': len(inc)}
        allp = rec['phase_A'] + rec['phase_B']
        for ph in ('A', 'B'):
            P = [p for p in allp if p['phase'] == ph]
            js = [float(np.max(np.abs(np.array(b['qa']) - np.array(a['qa'])))) for a, b in zip(P, P[1:])]
            bs = [math.hypot(b['base'][0] - a['base'][0], b['base'][1] - a['base'][1]) for a, b in zip(P, P[1:])]
            rec[f'stats_{ph}'] = {'joint_step_max_rad_per_deg': max(js) if js else None, 'base_step_max_m_per_deg': max(bs) if bs else None,
                                  'joint_margin_min': min(p['jm'] for p in P if p['jm'] is not None),
                                  'clear_min': {k: min(p['clear'][k] for p in P if p['clear']) for k in ('nonfinger', 'finger', 'base', 'self')}}
        cands.append(rec)
        print(f"q_A {qa_}°：推門起點 {np.round(bT, 3)} r_c {rcT}；B 至 {endB['q']}°（{whyB}）底盤 {np.round(endB['base'], 3)}；"
              f"重疊 {rec['overlap_q_deg']}；C {'可' if okC else '不可'}通過 {({k: round(v, 3) for k, v in minC.items()})}；{time.time() - t0:.0f} s", flush=True)
    json.dump({'schema': 'door_coopen_traj/1', 'limits': LIM, 'band_x': BAND, 'phase_A_reached_deg': qA_reached, 'phase_A_why': whyA,
               'candidates': cands, 'door_rate_assumed_deg_s': 1.0,
               'not_modeled': ['接觸力、摩擦、門被推動的動力學（假設門跟著推觸點）', '取樣間碰撞', '底盤外凸物', '手臂換位（抓持→推觸）與收臂路徑',
                               '控制器追蹤與延遲']}, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
