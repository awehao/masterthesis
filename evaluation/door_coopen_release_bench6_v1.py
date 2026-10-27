#!/usr/bin/env python3
"""A 包（Codex r39 §下一候選 1）：bench_6 實錄開門末態能否「張爪 → 解除 → 退開」（door_coopen_release_plan_v1 的版本化改寫；只改起點與底盤處置）。
起點：bench_6 STOPPED 後、門已靜止的一個物理步（預設 T_PICK 110.0 s；門角約 47.9°）的實測底盤位姿、手臂角與門角；
  起點 TCP＝實測關節正運動學（不重設抓取關係、不瞬間對準名目抓取位姿）；門扇座標 u、n 依該步實測門角。
底盤：位置固定（不沿出口路徑前進），偏航候選 {0, ±2°}；其餘（張爪 3 點 0.3 s、路徑點 W1＝TCP＋0.08 n、W2＝1.07 u＋0.08 n、W3＝1.07 u − 0.15 n、
  每步 1 cm、姿態候選 {不變、繞世界 z ±10°、繞工具 x ±10°}、IK 前一點種子且單步 ≤ 0.10 rad、門角依壓力情境、非手指／手指（張爪開度）／自碰 ≥ 5 mm、
  底盤 ≥ 30 mm、之後關節內插到 STOW 40 點）同原版。壓力情境：static、k3b_stress、reverse_0p5（原版定義；只作壓力，不作本抓法上界）。
報：每情境是否全程成立、首個失敗的時刻／階段／部位、最小餘裕。不成立 ⇒ 即 A 包本候選的第一個阻點。

    python3 evaluation/door_coopen_release_bench6_v1.py [T_PICK]   # 寫 results/door/door_coopen_release_bench6_v1.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_release_plan_v1 as RP               # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_release_bench6_v1.json')
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
C = RP.C


def plan(b0, qa0, q_deg, scen):
    dq = RP.SCEN[scen]
    q = math.radians(q_deg)
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    base0 = np.array(b0, float)
    seed = np.array(qa0, float)
    Tg = C.K.fk(np.concatenate([base0, seed]), 'link_tcp')
    yaw = float(base0[2])
    t, rows = 0.0, []
    for k in range(3):
        t += 0.1
        qfi = RP.Q_CLOSED + (RP.OPEN - RP.Q_CLOSED) * (k + 1) / 3
        m, gg = RP.dmin(list(base0) + list(seed), q_deg + dq(t), qfi)
        rows.append({'t': t, 'phase': 'open', 'q_door': q_deg + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0]})
        if m[0] < 0 or gg['base'][0] < RP.BASE_PLAN:
            return rows, {'t': t, 'phase': 'open', 'who': f'{m[1]}:{m[2]}', 'margin_mm': m[0] * 1e3}
    p, R = Tg[:3, 3].copy(), Tg[:3, :3].copy()
    W = [p + 0.08 * n, 1.07 * u * np.array([1, 1, 0]) + np.array([0, 0, p[2]]) + 0.08 * n,
         1.07 * u * np.array([1, 1, 0]) + np.array([0, 0, p[2]]) - 0.15 * n]
    for wi, w in enumerate(W):
        it = 0
        while np.linalg.norm(w - p) > 0.005 and it < 80:
            it += 1
            TJ.cool()
            p_new = p + (w - p) * min(1.0, 0.01 / np.linalg.norm(w - p))
            best, why_all = None, {}
            for rot in (np.eye(3), RP.Rax([0, 0, 1], 0.17), RP.Rax([0, 0, 1], -0.17), RP.Rax(R[:, 0], 0.17), RP.Rax(R[:, 0], -0.17)):
                R2 = rot @ R
                for dyw in (0.0, math.radians(2), -math.radians(2)):
                    base = np.array([base0[0], base0[1], yaw + dyw])
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = R2, p_new
                    q9 = C.ik(base, T, seed, only_prev=True)
                    if q9 is None:
                        why_all['ik'] = why_all.get('ik', 0) + 1
                        continue
                    qa = np.asarray(q9)[C.IDX]
                    if float(np.max(np.abs(qa - seed))) > TJ.LIM['joint_step_rad']:
                        why_all['step'] = why_all.get('step', 0) + 1
                        continue
                    dt = max(0.01 / RP.V_TCP, float(np.max(np.abs(qa - seed))) / RP.ARM_RATE, abs(dyw) / RP.W_ANG)
                    m, gg = RP.dmin(list(base) + list(qa), q_deg + dq(t + dt), RP.OPEN)
                    if gg['base'][0] < RP.BASE_PLAN:
                        why_all['base'] = why_all.get('base', 0) + 1
                        continue
                    if best is None or m[0] > best[0][0]:
                        best = (m, qa, R2, base, dt, gg)
            if best is None:
                return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': 'ik_or_step_or_base', 'detail': why_all, 'tcp': p.tolist()}
            m, seed, R, base, dt, gg = best
            yaw = float(base[2])
            t += dt
            p = p_new
            rows.append({'t': t, 'phase': f'W{wi + 1}', 'q_door': q_deg + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0],
                         'tcp': p.tolist(), 'qa': seed.tolist(), 'base': base.tolist()})
            if m[0] < 0:
                return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': f'{m[1]}:{m[2]}', 'margin_mm': m[0] * 1e3, 'tcp': p.tolist()}
        if np.linalg.norm(w - p) > 0.005:
            return rows, {'t': t, 'phase': f'W{wi + 1}', 'who': 'not_reached'}
    qa0s = seed.copy()
    for k in range(1, 41):
        TJ.cool()
        qa = (1 - k / 40) * qa0s + (k / 40) * TJ.STOW
        t += 0.05
        base = np.array([base0[0], base0[1], yaw])
        m, gg = RP.dmin(list(base) + list(qa), q_deg + dq(t), RP.OPEN)
        rows.append({'t': t, 'phase': 'stow', 'q_door': q_deg + dq(t), 'min_margin_m': m[0], 'who': f'{m[1]}:{m[2]}', 'base_m': gg['base'][0]})
        if m[0] < 0 or gg['base'][0] < RP.BASE_PLAN:
            return rows, {'t': t, 'phase': 'stow', 'who': f'{m[1]}:{m[2]}', 'margin_mm': m[0] * 1e3}
    return rows, None


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t_pick = float(sys.argv[1]) if len(sys.argv) > 1 else 110.0
    r6 = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r6['steps_cols'])}
    s = min(r6['steps'], key=lambda z: abs(z[c['sim_t']] - t_pick))
    b0, qa0, q_deg = list(s[c['base_xyth']]), list(s[c['q_arm_meas']]), math.degrees(s[c['door_q_rad']])
    res = {'schema': 'door_coopen_release_bench6/1', 'start': {'sim_t': s[c['sim_t']], 'step': s[c['step']], 'base': b0, 'qa': qa0, 'q_deg': q_deg,
                                                                'finger_contact_n': s[c['finger_contact_n']]}, 'results': {}}
    for sc in ('static', 'k3b_stress', 'reverse_0p5'):
        rows, fail = plan(b0, qa0, q_deg, sc)
        mm = min(r['min_margin_m'] for r in rows) if rows else None
        res['results'][sc] = {'ok': fail is None, 'fail': fail, 'n': len(rows), 'min_margin_mm': None if mm is None else mm * 1e3, 'rows': rows}
        print(f"情境 {sc}：{'成立' if fail is None else '失敗於 ' + json.dumps(fail, ensure_ascii=False, default=float)[:300]}；{len(rows)} 點；最小餘裕 {res['results'][sc]['min_margin_mm']}", flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
