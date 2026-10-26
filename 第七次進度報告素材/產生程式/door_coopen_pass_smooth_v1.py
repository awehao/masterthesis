#!/usr/bin/env python3
"""連續協同通行：放手後通過路徑重建＋平滑＋收臂掃掠＋時間化 v1（Codex 20261026_183611 §3-3；離線靜態，無 Isaac）。

輸入：door_coopen_edgehold_v2 中握回 0° 的候選（取底盤餘裕最大者；可用參數指定放手門角）。
1. 淨空場：放手後門角在 [q_rel − 2°, q_rel + 5°] 的逐點最小值（同 release_set_v1）；底盤規劃門檻 30 mm（20＋誤差 10）。
2. 路徑：自放手底盤格點 Dijkstra 到 y ≥ 0.45，只走 F ≥ 30 mm 的格點，成本＝步長 ×（1 ＋ 3·max(0, 0.10 − F)/0.10）（偏好離障礙 ≥ 10 cm）；
   再做視線捷徑（每 1 cm 取樣仍須 F ≥ 30 mm）與 Chaikin 圓角（3 次，任一取樣違反即保留原角），以 1 cm 重取樣。
   **此即同一條路徑**：本檔報的最小淨空、轉角與速度都在這條路徑上算（回應 8° 與 28 mm 不同路徑的指正）。
3. 收臂（底盤同時沿路徑前進，偏航維持放手值）：
   a. 張爪：0.0016 → 0.0089（finger_joint），底盤沿路徑前 2 cm；
   b. 沿工具 −z 退 60 mm，底盤再前進 6 cm（TCP 目標以門角 q_rel 的抓取位姿為基準，IK 以前一點為種子）；
   c. 關節線性內插到 K2_READY（STOW），底盤再前進 30 cm；
   每個取樣點以門角 q_rel − 2、q_rel、q_rel + 5 三者核非手指／手指（張爪開度）／自碰 ≥ 5 mm 與底盤 ≥ 30 mm，報最小值與最近物件。
4. 時間化（持握段＋通過段同一時間軸）：
   底盤合速度 ≤ 0.05 m/s 且每軸 ≤ 0.035 m/s（E2 低速介面界限／求解器方框）、偏航 ≤ 0.20 rad/s、底盤加速度 ≤ 0.5 m/s²（E2 預設）、
   手臂 ≤ 1.0 rad/s（另報 ≤ 0.10 rad/s 的保守版時長）、門 ≤ 1 °/s；
   持握段逐 1° 步取「各界限所需時間」最大者；通過段以梯形速度（a 0.5、v 0.035）沿路徑；放手交界處速度向量連續性以加速度預算核對。
   輪速以 wb_wheel_limit.wheel_matrix（本體座標）換算，核 ≤ 0.2775 m/s。
**界線**：靜態幾何、圓盤底盤、剛性夾持；門餘轉以區間包絡代替；時間化為運動學下界，不含控制器追蹤；張爪後門的實際受力與滑移不在內。

    python3 evaluation/door_coopen_pass_smooth_v1.py [q_rel]   # 寫 results/door/door_coopen_pass_smooth_v1.json（不覆寫）
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import wb_wheel_limit as WL                            # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v1.json')
C = RV.C
XS, YS = RS.XS, RS.YS
PLAN = 0.030
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005}
V_AXIS, V_LIN, W_ANG, A_LIN = 0.035, 0.05, 0.20, 0.5
ARM_RATE, ARM_RATE_CONS, DOOR_RATE = 1.0, 0.10, 1.0
Q_CLOSED, OPEN = RV.Q_FINGER, C.FINGER_OPEN


def F_at(F, p):
    """雙線性內插淨空。"""
    fx = (p[0] - XS[0]) / (XS[1] - XS[0])
    fy = (p[1] - YS[0]) / (YS[1] - YS[0])
    i, j = int(np.clip(math.floor(fx), 0, len(XS) - 2)), int(np.clip(math.floor(fy), 0, len(YS) - 2))
    a, b = fx - i, fy - j
    return float((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i + 1, j] + (1 - a) * b * F[i, j + 1] + a * b * F[i + 1, j + 1])


def seg_ok(F, p, q):
    n = max(2, int(math.ceil(np.linalg.norm(np.subtract(q, p)) / 0.01)) + 1)
    return all(F_at(F, np.add(p, t * np.subtract(q, p))) >= PLAN for t in np.linspace(0, 1, n))


def dijkstra(F, b):
    i0 = int(np.argmin([abs(x - b[0]) for x in XS]))
    j0 = int(np.argmin([abs(y - b[1]) for y in YS]))
    dist, prev = {(i0, j0): 0.0}, {(i0, j0): None}
    pq = [(0.0, i0, j0)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[(i, j)]:
            continue
        if YS[j] >= 0.45:
            path, c = [], (i, j)
            while c is not None:
                path.append((XS[c[0]], YS[c[1]]))
                c = prev[c]
            return path[::-1]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if not ((di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS)) or F[ni, nj] < PLAN:
                    continue
                L = math.hypot(XS[ni] - XS[i], YS[nj] - YS[j])
                nd = d + L * (1 + 3 * max(0.0, 0.10 - F[ni, nj]) / 0.10)
                if nd < dist.get((ni, nj), math.inf):
                    dist[(ni, nj)], prev[(ni, nj)] = nd, (i, j)
                    heapq.heappush(pq, (nd, ni, nj))
    return None


def smooth(F, pts, start):
    pts = [tuple(start)] + [tuple(p) for p in pts[1:]]
    out, k = [pts[0]], 0
    while k < len(pts) - 1:
        m = len(pts) - 1
        while m > k + 1 and not seg_ok(F, pts[k], pts[m]):
            m -= 1
        out.append(pts[m])
        k = m
    for _ in range(3):
        new = [out[0]]
        for a, b in zip(out, out[1:]):
            q1 = (0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1])
            q2 = (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])
            new += [q1, q2]
        new.append(out[-1])
        new[1] = new[1] if seg_ok(F, new[0], new[1]) else out[1]
        if all(seg_ok(F, a, b) for a, b in zip(new, new[1:])):
            out = new
    L = np.cumsum([0.0] + [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(out, out[1:])])
    s = np.arange(0.0, L[-1] + 1e-9, 0.01)
    xs = np.interp(s, L, [p[0] for p in out])
    ys = np.interp(s, L, [p[1] for p in out])
    return np.stack([xs, ys], 1), s


def heading(P, k0, k1):
    v = P[k1] - P[k0]
    return v / (np.linalg.norm(v) + 1e-12)


def geo3(qf, q_rel, qfi):
    best = {}
    for dq in (-2.0, 0.0, 5.0):
        d = TJ.GEO([float(v) for v in qf], math.radians(q_rel + dq), qfi)
        for k in ('nonfinger', 'finger', 'self', 'base'):
            if k not in best or d[k][0] < best[k][0]:
                best[k] = (float(d[k][0]), f'{d[k][1]}@{q_rel + dq:.0f}°')
    return best


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    src = json.load(open(SRC))
    done = [r for r in src['per_end'] if r['reached_q_deg'] == 0]
    if len(sys.argv) > 1:
        cand = next(r for r in done if r['q_rel_deg'] == int(sys.argv[1]))
    else:
        cand = max(done, key=lambda r: min(r['margin_min_mm']['base'] - 10.0, r['margin_min_mm']['self']))
    q_rel = float(cand['q_rel_deg'])
    H = cand['trajectory']
    rel = H[-1]
    F = RS.field(q_rel)
    gp = dijkstra(F, rel['base'][:2])
    assert gp is not None, '放手後無路'
    P, s = smooth(F, gp, rel['base'][:2])
    clear_path = [F_at(F, p) for p in P]
    kmin = int(np.argmin(clear_path))
    # 交界方向：持握末 3 步 vs 通過前 5 cm
    v_in = np.array(H[-1]['base'][:2]) - np.array(H[-4]['base'][:2])
    v_out = heading(P, 0, min(5, len(P) - 1))
    turn = math.degrees(math.acos(float(np.clip(v_in @ v_out / (np.linalg.norm(v_in) + 1e-12), -1, 1))))
    print(f"候選放手 {q_rel:.0f}°；通過路徑 {s[-1]:.2f} m，最小淨空 {min(clear_path) * 1000:.1f} mm（於 {np.round(P[kmin], 2)}），交界轉角 {turn:.0f}°", flush=True)
    # 收臂
    yaw = rel['base'][2]
    Tg = EG.grasp_T(math.radians(q_rel), *(src['grasp'][k] for k in ('phi', 's', 'roll')))
    arm, seed = [], np.array(rel['qa'])
    kk = 0
    for k in range(0, 3):                                 # a. 張爪（底盤前進 2 cm）
        kk = min(k, len(P) - 1)
        qfi = Q_CLOSED + (OPEN - Q_CLOSED) * k / 2
        qf = [P[kk][0], P[kk][1], yaw] + list(seed)
        arm.append({'phase': 'open', 's_m': float(s[kk]), 'q_finger': qfi, 'qa': list(map(float, seed)), 'geo': geo3(qf, q_rel, qfi)})
    for k in range(1, 7):                                 # b. 沿 −z 退 60 mm（底盤再前進 6 cm）
        kk = min(2 + k, len(P) - 1)
        Td = Tg @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.01 * k], [0, 0, 0, 1.0]])
        base = np.array([P[kk][0], P[kk][1], yaw])
        q9 = C.ik(base, Td, seed, only_prev=True)
        if q9 is None:
            arm.append({'phase': 'retract', 's_m': float(s[kk]), 'ik': False})
            break
        seed = np.asarray(q9)[C.IDX]
        arm.append({'phase': 'retract', 's_m': float(s[kk]), 'q_finger': OPEN, 'qa': list(map(float, seed)),
                    'geo': geo3(list(base) + list(seed), q_rel, OPEN)})
    k0 = kk
    for k in range(1, 31):                                # c. 內插到 STOW（底盤再前進 30 cm）
        TJ.cool()
        kk = min(k0 + k, len(P) - 1)
        qa = (1 - k / 30) * seed + (k / 30) * TJ.STOW
        arm.append({'phase': 'stow', 's_m': float(s[kk]), 'q_finger': OPEN, 'qa': list(map(float, qa)),
                    'geo': geo3([P[kk][0], P[kk][1], yaw] + list(qa), q_rel, OPEN)})
    ok_arm = [a for a in arm if 'geo' in a]
    arm_min = {c: min(ok_arm, key=lambda a: a['geo'][c][0])['geo'][c] for c in ('nonfinger', 'finger', 'self', 'base')}
    arm_ik_fail = any(a.get('ik') is False for a in arm)
    print(f"收臂：IK 失敗 {arm_ik_fail}；最小 {({c: (round(v[0] * 1000, 1), v[1]) for c, v in arm_min.items()})}", flush=True)
    # 時間化
    hold_dt, hold_dt_cons, vb = [], [], []
    for a, b in zip(H, H[1:]):
        d = np.subtract(b['base'][:2], a['base'][:2])
        dyw = abs((b['base'][2] - a['base'][2] + math.pi) % (2 * math.pi) - math.pi)
        dqa = float(np.max(np.abs(np.subtract(b['qa'], a['qa']))))
        base_t = max(abs(d[0]) / V_AXIS, abs(d[1]) / V_AXIS, np.linalg.norm(d) / V_LIN)
        t = max(abs(b['q_deg'] - a['q_deg']) / DOOR_RATE, base_t, dyw / W_ANG, dqa / ARM_RATE)
        hold_dt.append(t)
        hold_dt_cons.append(max(t, dqa / ARM_RATE_CONS))
        vb.append(np.linalg.norm(d) / t)
    v_end = (np.array(H[-1]['base'][:2]) - np.array(H[-2]['base'][:2])) / hold_dt[-1]
    v_pass = V_AXIS
    dv = np.linalg.norm(v_pass * v_out - v_end)
    t_junction = dv / A_LIN
    L = s[-1]
    t_acc = v_pass / A_LIN
    pass_t = (L / v_pass + t_acc) if L > v_pass * t_acc else 2 * math.sqrt(L / A_LIN)
    Wm = WL.wheel_matrix(WL.WheelLimitConfig().wheel_base_L)

    def wheel(vw, yaw_, wz=0.0):
        c, s_ = math.cos(yaw_), math.sin(yaw_)
        vb_ = np.array([c * vw[0] + s_ * vw[1], -s_ * vw[0] + c * vw[1], wz])
        return float(np.max(np.abs(Wm @ vb_)))
    w_hold = max(wheel(np.subtract(b['base'][:2], a['base'][:2]) / t, a['base'][2],
                       ((b['base'][2] - a['base'][2] + math.pi) % (2 * math.pi) - math.pi) / t) for a, b, t in zip(H, H[1:], hold_dt))
    w_pass = max(wheel(v_pass * heading(P, k, min(k + 1, len(P) - 1)), yaw) for k in range(len(P) - 1))
    timing = {'hold_s': float(sum(hold_dt)), 'hold_s_arm_0p1': float(sum(hold_dt_cons)), 'hold_base_speed_max_m_s': float(max(vb)),
              'door_rate_mean_deg_s': q_rel / float(sum(hold_dt)), 'junction': {'v_end_m_s': v_end.tolist(), 'v_pass_dir': v_out.tolist(),
              'dv_m_s': float(dv), 't_min_s': float(t_junction)}, 'pass_s': float(pass_t), 'total_s': float(sum(hold_dt) + t_junction + pass_t),
              'wheel_speed_max_m_s': {'hold': w_hold, 'pass': w_pass, 'limit': WL.WheelLimitConfig().w_lim}}
    print(f"時間化：持握 {timing['hold_s']:.0f} s（手臂 0.1 rad/s 版 {timing['hold_s_arm_0p1']:.0f} s；門平均 {timing['door_rate_mean_deg_s']:.2f} °/s）、"
          f"交界 Δv {dv:.3f} m/s（{t_junction:.2f} s）、通過 {pass_t:.0f} s、合計 {timing['total_s']:.0f} s；輪速最大 持握 {w_hold:.3f}／通過 {w_pass:.3f}（上限 {timing['wheel_speed_max_m_s']['limit']:.4f}）", flush=True)
    json.dump({'schema': 'door_coopen_pass_smooth/1', 'source': os.path.relpath(SRC, HERE), 'q_rel_deg': q_rel, 'plan_base_m': PLAN,
               'door_envelope_deg': [q_rel - 2, q_rel + 5], 'path': P.tolist(), 'path_len_m': float(L),
               'path_clear_min_m': float(min(clear_path)), 'path_clear_min_at': P[kmin].tolist(), 'junction_turn_deg': turn,
               'arm_retract': arm, 'arm_min': {c: [v[0], v[1]] for c, v in arm_min.items()}, 'arm_ik_fail': arm_ik_fail,
               'limits': {'v_axis': V_AXIS, 'v_lin': V_LIN, 'w_ang': W_ANG, 'a_lin': A_LIN, 'arm_rate': ARM_RATE, 'door_rate': DOOR_RATE},
               'timing': timing, 'hold_candidate': {k: cand[k] for k in ('q_rel_deg', 'end', 'start_base_q0', 'margin_min_mm')},
               'not_modeled': ['控制器追蹤與延遲', '門餘轉動力學（區間包絡）', '張爪時的接觸力與滑移', '圓盤底盤', '取樣間碰撞']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
