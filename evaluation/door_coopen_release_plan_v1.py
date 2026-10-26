#!/usr/bin/env python3
"""連續協同通行：放手收臂規劃＋時間化門餘轉壓力情境 v1（離線靜態幾何、逐時門角；無 Isaac）。

背景：door_coopen_verify_v1（q48 候選）沿工具 −z 退時 link2 逼近門扇（δ=0 已降到 0.5 mm），繞回自由端 IK 失敗；
固定退法（−z 60 mm → 沿 u → 沿 −n、偏航 +90°）是為 55° 預算版調的，不可沿用。
本檔：
  門餘轉壓力情境（door_wide_k3b_30_1 實測，NOTE r2 分段表；正面夾法、30°，**只作壓力情境，不作本抓法上界**）：
    張爪後 0–0.16 s 以 1.0 °/s、其後 0.24 °/s 往開門方向續轉（3 s 共約 +0.75°）；另核 0 與反向（−0.5°）兩情境。
  收臂（張爪後即開始；TCP 速度 ≤ 0.05 m/s、手臂 ≤ 1.0 rad/s、偏航 ≤ 0.20 rad/s、底盤沿出口路徑 ≤ 0.035 m/s）：
    路徑點（門角 q_rel 的門扇座標）：W1＝TCP＋0.08 n（離開門面）；W2＝1.07 u＋0.08 n（越過自由端、門面側）；
    W3＝1.07 u − 0.15 n（門扇背面）；之後關節內插到 STOW。
    每步 1 cm：TCP 往當前路徑點移動，候選＝{夾爪姿態不變、繞世界 z ±10°、繞工具 x ±10°} × {底盤偏航 0、+2°、−2°}，
    底盤沿出口路徑前進 0.5 cm；IK 以前一點為種子（相鄰 ≤ 0.10 rad）；
    在**該步時刻的門角**（各情境）下核非手指／手指（張爪開度）／自碰 ≥ 5 mm、底盤 ≥ 30 mm，取最小距離最大的候選；
    張爪本身耗時 0.3 s（3 點，TCP 不動）。
報：每情境是否全程成立、首個失敗時刻與部位、最小距離、收臂時長、底盤前進量、偏航總變化。

    python3 evaluation/door_coopen_release_plan_v1.py <joint_v2_單組.json>   # 寫 results/door/door_coopen_release_plan_v1_<組>.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_coopen_verify_v1 as VF                     # noqa: E402

C = RV.C
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005}
BASE_PLAN = 0.030
V_TCP, ARM_RATE, W_ANG, V_BASE = 0.05, 1.0, 0.20, 0.035
Q_CLOSED, OPEN = RV.Q_FINGER, C.FINGER_OPEN
SCEN = {'k3b_stress': lambda t: (1.0 * min(t, 0.16) + 0.24 * max(0.0, t - 0.16)),
        'static': lambda t: 0.0,
        'reverse_0p5': lambda t: -min(0.5, 1.0 * t)}


def Rax(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def dmin(qf, q_deg, qfi):
    g = TJ.GEO([float(v) for v in qf], math.radians(q_deg), qfi)
    m = min((g[c][0] - TH[c], c, g[c][1]) for c in TH)
    return m, g


def plan(cand, scen_name):
    return plan_state(tuple(cand['grasp']), float(cand['q_rel_deg']), cand['exit_path'], cand['trajectory'][-1], float(cand['fwd_m']), scen_name)


def plan_state(g, q_rel, exit_path, end, fwd_m, scen_name):
    """end＝{'base', 'qa'}（持握末端，門角 q_rel）；底盤先沿出口路徑握持前進 fwd_m 後張爪。"""
    dq = SCEN[scen_name]
    q = math.radians(q_rel)
    path = VF.Path(exit_path)
    Tg = EG.grasp_T(q, *g)
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    s = float(fwd_m)
    b = path.at(s)
    yaw = end['base'][2]
    seed = np.asarray(C.ik(np.array([b[0], b[1], yaw]), Tg, np.array(end['qa']), only_prev=True)[C.IDX]) \
        if C.ik(np.array([b[0], b[1], yaw]), Tg, np.array(end['qa']), only_prev=True) is not None else np.array(end['qa'])
    t = 0.0
    rows = []
    for k in range(3):
        t += 0.1
        qfi = Q_CLOSED + (OPEN - Q_CLOSED) * (k + 1) / 3
        m, gg = dmin([b[0], b[1], yaw] + list(seed), q_rel + dq(t), qfi)
        rows.append({'t': t, 'phase': 'open', 'q_door': q_rel + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0]})
        if m[0] < 0 or gg['base'][0] < BASE_PLAN:
            return rows, {'t': t, 'phase': 'open', 'who': f'{m[1]}:{m[2]}'}, s, yaw - end['base'][2]
    p = Tg[:3, 3].copy()
    R = Tg[:3, :3].copy()
    W = [p + 0.08 * n, 1.07 * u * np.array([1, 1, 0]) + np.array([0, 0, p[2]]) + 0.08 * n,
         1.07 * u * np.array([1, 1, 0]) + np.array([0, 0, p[2]]) - 0.15 * n]
    for wi, w in enumerate(W):
        it = 0
        while np.linalg.norm(w - p) > 0.005 and it < 80:
            it += 1
            TJ.cool()
            step = (w - p) * min(1.0, 0.01 / np.linalg.norm(w - p))
            p_new = p + step
            best = None
            for rot in (np.eye(3), Rax([0, 0, 1], 0.17), Rax([0, 0, 1], -0.17), Rax(R[:, 0], 0.17), Rax(R[:, 0], -0.17)):
                R2 = rot @ R
                for dyw in (0.0, math.radians(2), -math.radians(2)):
                    s2 = s + 0.005
                    bb = path.at(s2)
                    base = np.array([bb[0], bb[1], yaw + dyw])
                    T = np.eye(4)
                    T[:3, :3] = R2
                    T[:3, 3] = p_new
                    q9 = C.ik(base, T, seed, only_prev=True)
                    if q9 is None:
                        continue
                    qa = np.asarray(q9)[C.IDX]
                    if float(np.max(np.abs(qa - seed))) > TJ.LIM['joint_step_rad']:
                        continue
                    dt = max(0.01 / V_TCP, float(np.max(np.abs(qa - seed))) / ARM_RATE, abs(dyw) / W_ANG, 0.005 / V_BASE)
                    m, gg = dmin(list(base) + list(qa), q_rel + dq(t + dt), OPEN)
                    if gg['base'][0] < BASE_PLAN:
                        continue
                    if best is None or m[0] > best[0][0]:
                        best = (m, qa, R2, base, dt, s2, gg)
            if best is None:
                return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': 'ik_or_base'}, s, yaw - end['base'][2]
            m, seed, R, base, dt, s, gg = best
            yaw = base[2]
            t += dt
            p = p_new
            rows.append({'t': t, 'phase': f'W{wi + 1}', 'q_door': q_rel + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0],
                         'tcp': p.tolist(), 'qa': seed.tolist(), 'base': base.tolist()})
            if m[0] < 0:
                return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': f'{m[1]}:{m[2]}'}, s, yaw - end['base'][2]
        if np.linalg.norm(w - p) > 0.005:
            return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': 'not_reached'}, s, yaw - end['base'][2]
    qa0 = seed.copy()
    for k in range(1, 41):
        TJ.cool()
        qa = (1 - k / 40) * qa0 + (k / 40) * TJ.STOW
        dt = max(float(np.max(np.abs(qa - ((1 - (k - 1) / 40) * qa0 + ((k - 1) / 40) * TJ.STOW)))) / ARM_RATE, 0.0075 / V_BASE)
        s += 0.0075
        t += dt
        bb = path.at(s)
        base = np.array([bb[0], bb[1], yaw])
        m, gg = dmin(list(base) + list(qa), q_rel + dq(t), OPEN)
        rows.append({'t': t, 'phase': 'stow', 'q_door': q_rel + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0]})
        if m[0] < 0 or gg['base'][0] < BASE_PLAN:
            return rows, {'t': t, 'phase': 'stow', 'who': f'{m[1]}:{m[2]}'}, s, yaw - end['base'][2]
    return rows, None, s, yaw - end['base'][2]


def main():
    src = sys.argv[1]
    tag = os.path.basename(src).replace('door_coopen_joint_v2_', '').replace('.json', '')
    out = os.path.join(HERE, 'results', 'door', f'door_coopen_release_plan_v1_{tag}.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    cand = json.load(open(src))['full'][0]
    res = {}
    for sc in ('static', 'k3b_stress', 'reverse_0p5'):
        rows, fail, s_end, dyaw = plan(cand, sc)
        mm = min(r['min_margin_m'] for r in rows) if rows else None
        res[sc] = {'ok': fail is None, 'fail': fail, 'n': len(rows), 't_end_s': rows[-1]['t'] if rows else 0.0, 'min_margin_mm': None if mm is None else mm * 1000,
                   'base_advance_m': s_end - float(cand['fwd_m']), 'dyaw_deg': math.degrees(dyaw), 'q_door_end': rows[-1]['q_door'] if rows else None, 'rows': rows}
        print(f"{tag} 情境 {sc}：{'成立' if fail is None else f'失敗於 {fail}'}；{len(rows)} 點、{res[sc]['t_end_s']:.1f} s、門角終值 {res[sc]['q_door_end']}；"
              f"最小餘裕 {res[sc]['min_margin_mm']:.1f} mm；底盤前進 {res[sc]['base_advance_m']:.2f} m、偏航 {res[sc]['dyaw_deg']:+.0f}°", flush=True)
    json.dump({'schema': 'door_coopen_release_plan/1', 'source': os.path.relpath(src, HERE), 'scenarios': list(SCEN), 'scenario_source': 'runs/door_wide_k3b_30_1/analysis/NOTE_door_wide_k3b_30_1_r2.md 分段表（壓力情境，非上界）',
               'results': res, 'not_modeled': ['張爪瞬間接觸力釋放的實際動力學', '本抓法的實際餘轉（以壓力情境代替）', '取樣間碰撞']},
              open(out, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
