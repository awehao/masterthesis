#!/usr/bin/env python3
"""K3-M＋受控暫停判定器 p2（door_k3m_check_p.py r3 的版本化擴充；Codex 20261024_101421 (a)）。

一、door_k3m_check_p.judge 全部照用（K3-M 判準、PAUSE→END_HOLD 對應、暫停 P1–P4）；task schema door_k3m_task_p2/1 在呼叫前
    轉為 door_k3m_task_p/1（其他欄位不動）。原始 tensor 門速的安全核對（全程逐物理步 |q̇| ≤ 0.05）照舊。
二、唯一替換：K3-A 判定器的 `settle_condition_not_held_1s`（PAUSE 前 1 s 逐物理步：角度、raw |q̇| ≤ 0.002、漂移、兩指）
    改由本檔 S1 重算；父判定器若給出該理由則移除（stats.parent_settle_reason_replaced 記錄），其餘理由全部保留。
  S1 完成條件（room 逐物理步重算，t ∈ [t_pause − 1.0, t_pause)，首步距 t_pause − 1.0 ≤ 0.02 s）：
     每步 |q − 8°| ≤ 0.5°、抓取漂移 ≤ 5 mm、兩指 ≥ 0.5 N；
     同步流節奏（偶數步）上每步 0.1 s 後向差分 |(q_k − q_{k−10})/(t_k − t_{k−10})| ≤ 0.002（兩端都在紀錄中、步號連續）。
  S2 線上訊號一致：trace 每列 SETTLE 的 settle_sig 須存在；valid 列的 step1＝該列步號、0.1 ≤ t1−t0 ≤ 0.13、
     (q0, q1, t0, t1) 與 room 同步號 ≤ 1e-7（時間 ≤ 1e-5 s）、rate 與 room 重算 ≤ 1e-6 rad/s；
     PAUSE 請求那一輪（policy_events SETTLE→END_HOLD 事件）的 settle_sig 為 valid 且 |rate| ≤ 0.002；
     轉換前 1 s 內的 SETTLE 列都 valid、rate ≤ 0.002（trace 列層級，與策略連續計時同一依據）。
  未進入 PAUSE ⇒ S1／S2 記 settle_p2_unverifiable（不算通過）。

    python3 evaluation/door_k3m_check_p2.py <run_dir>    # 寫 <run_dir>/analysis/door_k3m_check_p2.json；結束碼 0＝通過
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                          # noqa: E402
import door_k3a_check as KC3                           # noqa: E402
import door_k3m_settle_sig as SS                       # noqa: E402

PARENT_SETTLE_REASON = 'settle_condition_not_held_1s'
WIN_STEPS = 10                                         # 0.1 s＝10 物理步（同步流 5 筆）


def _t_pause(task):
    for e in task.get('policy_events') or []:
        if e.get('to') == 'PAUSE':
            return float(e['t']), e
    return None, None


def check_settle_p2(task, room, why, st):
    t_p, _ = _t_pause(task)
    eh = next((e for e in task.get('policy_events') or [] if e.get('to') == 'END_HOLD' and e.get('from') == 'SETTLE'), None)
    if t_p is None or eh is None:
        why.append('settle_p2_unverifiable')
        return False
    cols = room.get('steps_cols') or []
    need = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'leaf_contacts']
    if any(c not in cols for c in need):
        why.append('settle_p2_room_cols_missing')
        return False
    ix = {c: cols.index(c) for c in need}
    S = {int(s[ix['step']]): s for s in room['steps']}
    GTH = KC3.KC._rigid(task.get('G_T_H'))
    if GTH is None:
        why.append('settle_p2_grasp_matrix_missing')
        return False
    c = KC3.CFG
    ok = True
    # S1：逐物理步
    rows = sorted((s for s in room['steps'] if t_p - c.settle_hold_s - 1e-9 <= s[ix['sim_t']] < t_p - 1e-9), key=lambda s: s[ix['step']])
    st['settle_p2_n_steps'] = len(rows)
    if not rows or rows[0][ix['sim_t']] - (t_p - c.settle_hold_s) > 0.02 + 1e-9:
        why.append('settle_p2_window_not_covered')
        return False
    max_rate, n_rate = 0.0, 0
    for s in rows:
        q, b, qa, lc = s[ix['door_q_rad']], s[ix['base_xyth']], s[ix['q_arm_meas']], s[ix['leaf_contacts']]
        if not KC3.KC._fin(q) or not KC3.KC._fin(b) or not KC3.KC._fin(qa) or lc is None:
            why.append('settle_p2_nonfinite')
            return False
        dp, _ = KC3.KP.grasp_drift(KC3.KC._tcp(b, qa), GTH, KC3.KC._H_of(q))
        fb = [lc.get(f, {}).get('bar', 0.0) for f in KC3.FINGERS]
        if abs(q - KC3.THETA) > math.radians(c.settle_tol_deg) or dp > KC3.DRIFT_M or min(fb) < KC3.GRIP_MIN_N:
            ok = False
            st.setdefault('settle_p2_first_bad', {'t': s[ix['sim_t']], 'q_deg': math.degrees(q), 'drift': dp, 'fb': fb})
        k = int(s[ix['step']])
        if k % 2 == 0:
            s0 = S.get(k - WIN_STEPS)
            if s0 is None or any((k - j) not in S for j in range(WIN_STEPS)):
                why.append('settle_p2_rate_window_missing')
                return False
            r = (q - s0[ix['door_q_rad']]) / (s[ix['sim_t']] - s0[ix['sim_t']])
            n_rate += 1
            max_rate = max(max_rate, abs(r))
    st['settle_p2_max_rate'] = max_rate
    st['settle_p2_n_rate'] = n_rate
    if not ok or max_rate > c.settle_rate or n_rate == 0:
        why.append('settle_p2_condition_not_held_1s')
        ok = False
    # S2：線上訊號與 room 一致
    steps_q = {k: (float(v[ix['sim_t']]), float(v[ix['door_q_rad']])) for k, v in S.items()}
    tr = [x for x in task.get('trace') or [] if x.get('phase') == 'SETTLE']
    n_bad, n_valid = 0, 0
    for x in tr:
        sig = x.get('settle_sig')
        if not isinstance(sig, dict):
            n_bad += 1
            continue
        if not sig.get('valid'):
            continue
        n_valid += 1
        s0, s1 = sig.get('step0'), sig.get('step1')
        r = SS.recompute_rate(steps_q, s0, s1)
        if (s1 != int(x['step']) or r is None or not (0.1 - 1e-9 <= sig['t1'] - sig['t0'] <= 0.13 + 1e-9)
                or abs(steps_q[s0][1] - sig['q0']) > 1e-7 or abs(steps_q[s1][1] - sig['q1']) > 1e-7
                or abs(steps_q[s0][0] - sig['t0']) > 1e-5 or abs(steps_q[s1][0] - sig['t1']) > 1e-5 or abs(r - sig['rate']) > 1e-6):
            n_bad += 1
    st['settle_p2_trace'] = {'n_rows': len(tr), 'n_valid': n_valid, 'n_bad': n_bad}
    if not tr or n_bad:
        why.append('settle_p2_trace_signal_mismatch')
        ok = False
    sig = eh.get('settle_sig')
    if not isinstance(sig, dict) or not sig.get('valid') or abs(sig.get('rate', math.inf)) > c.settle_rate:
        why.append('settle_p2_transition_signal_invalid')
        ok = False
    last = [x for x in tr if x['t'] >= t_p - c.settle_hold_s - 1e-9]
    if not last or any(not (isinstance(x.get('settle_sig'), dict) and x['settle_sig'].get('valid')
                            and abs(x['settle_sig']['rate']) <= c.settle_rate) for x in last):
        why.append('settle_p2_trace_not_held_1s')
        ok = False
    return ok


def judge(task, room, gate, solver):
    t1 = copy.deepcopy(task)
    t1['schema'] = 'door_k3m_task_p/1' if task.get('schema') == 'door_k3m_task_p2/1' else 'mismatch'
    # 與 door_k3m_check_p.judge 相同的兩段（K3-M 判準、暫停 P1–P4），只把父判定器的 SETTLE 1 s 理由換成 S1／S2
    res = CP.KM.judge(CP.to_k3m_task(t1), room, gate, solver)
    st = dict(res.get('stats') or {})
    why_k = list(res['why'])
    if t1.get('schema') != 'door_k3m_task_p/1':
        why_k.append('task_schema_k3m_p2')
    st['parent_settle_reason_replaced'] = PARENT_SETTLE_REASON in why_k
    why_k = [w for w in why_k if w != PARENT_SETTLE_REASON]
    settle_ok = check_settle_p2(task, room, why_k, st)
    why_p = []
    cmd_ok, phys_ok = CP.check_pause(t1, room, solver, why_p, st)
    return {'pass': not why_k and not why_p, 'k3m_pass': not why_k, 'settle_p2_ok': settle_ok,
            'pause_command_stop': cmd_ok, 'pause_physical_stop': phys_ok,
            'why': why_k + why_p, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    solver = json.load(open(os.path.join(run, 'align_solver.json')))
    res = judge(task, room, gate, solver)
    p = os.path.join(run, 'analysis', 'door_k3m_check_p2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
