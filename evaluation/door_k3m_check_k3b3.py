#!/usr/bin/env python3
"""K3-B 整體判定器 r3（Codex 20261026_012939 最後必修；door_k3m_check_k3b2.py〔r2，凍結 k3b r2〕不改，本檔重用其全部分項）。
核心 door_k3b_release4。唯一差異：K13 終點保持改為 K13'——
  以固定 H(q_rel) 座標中、room 逐物理步 FK 的實測 TCP **三維位置**：
  - 每個 0.5 s 窗口（窗口終點涵蓋 RETREAT_DONE 前 0.5 s 到 COMPLETE）最大點對位移 ≤ 1 mm；
  - s_meas ≥ 0.095（RETREAT_DONE～COMPLETE）、時長 ≥ 3 s、步連續；
  - 姿態偏移界沿用 K5'，不另新增門檻。
其餘（舊鏈只取代兩項、M1–M2、K0–K12、X1–X3、錨點）同 r2。

    python3 evaluation/door_k3m_check_k3b3.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_k3b3.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_release4 as KR                         # noqa: E402
import door_k3m_check_k3b as J1                        # noqa: E402
import door_k3m_check_k3b2 as J2                       # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402

R = KR.REG
K3B_REPLACED = J2.K3B_REPLACED


def check_end_hold_3d(room, task, g, why, st):
    import door_k2_check as KC
    tt = ((((task or {}).get('k3b') or {}).get('monitor') or {}).get('t') or {})
    a, b = tt.get('RETREAT_DONE'), tt.get('COMPLETE')
    if g is None or not (J1._fin(a) and J1._fin(b)):
        why.append('K13_end_hold_missing')
        return
    S, I = J1._room(room)
    rows = [s for s in S if a - R['converge_win_s'] - 1e-9 <= s[I['sim_t']] <= b + 1e-9]
    if not rows or rows[-1][I['sim_t']] - a < R['complete_hold_s'] - 0.011 or rows[0][I['sim_t']] > a - R['converge_win_s'] + 0.011 \
            or any(y[I['step']] - x[I['step']] != 1 for x, y in zip(rows, rows[1:])):
        why.append('K13_end_hold_coverage')
        return
    Hinv = np.linalg.inv(g['Hr'])
    pts = []
    for s in rows:
        E = Hinv @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        sm = float((E[:3, 3] - g['E_rel'][:3, 3]) @ KR.A_H)
        pts.append((float(s[I['sim_t']]), sm, E[:3, 3].copy()))
    if any(sm < R['s_done_m'] - 1e-9 for t, sm, _ in pts if t >= a - 1e-9):
        why.append('K13_end_hold_left_zone')
    worst, j = 0.0, 0
    for i in range(len(pts)):
        while pts[i][0] - pts[j][0] > R['converge_win_s'] + 1e-9:
            j += 1
        if pts[i][0] < a - 1e-9:
            continue                                                          # 窗口終點自 RETREAT_DONE 起（窗口本身涵蓋其前 0.5 s）
        P = np.array([p[2] for p in pts[j:i + 1]])
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)).max()
        worst = max(worst, float(d))
    if worst > R['converge_max_m'] + 1e-9:
        why.append('K13_end_hold_moved_3d')
    st['k3b3_end_hold'] = {'t': [a, b], 'n': len(pts), 'worst_window_move_3d_mm': worst * 1e3}


def check_k3b3(task, room, gate, solver, st):
    why = []
    rw = (room or {}).get('w2') or {}
    cs = J1.check_multi(rw, why)
    if cs is None:
        return why
    J2.check_room_coverage(room, cs, why, st)
    t_conf = J1.check_exec_release(room, rw, cs, why, st)
    J2.check_effort2(room, rw, task, cs, t_conf, why, st)
    g = J2.check_geom_steps(room, cs, why, st)
    J2.check_exit_recompute(room, task, g, why, st)
    J2.check_tracking(task, g, why, st)
    J2.check_general_clearance(room, cs, why, st)
    check_end_hold_3d(room, task, g, why, st)
    J1.check_base_contacts(room, gate, cs, why, st)
    J1.check_task(task, why, st)
    J1.check_solver(solver, why, st)
    J2.check_reg_id_time(task, room, solver, cs, why, st)
    J2.anchors(room, gate, solver, cs, st)
    return why


def judge(task, room, gate, solver, observer):
    res = F2.judge(task, room, gate, solver, observer)
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if w in K3B_REPLACED]
    why_old = [w for w in why_all if w not in K3B_REPLACED]
    st = dict(res.get('stats') or {})
    why_k = check_k3b3(task, room, gate, solver, st)
    why = why_old + why_k
    return dict(res, why=why, stats=st, judge='k3b3', replaced_old_reasons=replaced, k3b_replaced_names=list(K3B_REPLACED),
                old_pass_excluding_replaced=not why_old, k3b_ok=not why_k, **{'pass': not why},
                semantics='DONE＝解除、退出近區、恢復 5 N、退離終點並三維收斂保持 ≥ 3 s；不是收臂')


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_k3b3.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3b_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
