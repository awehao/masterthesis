#!/usr/bin/env python3
"""K2 判定器（缺資料、非有限、時間不連續一律不通過）。只讀：<run>/task.json、room_run.json、park_gate.json。

以**執行端逐物理步紀錄**（room_run.json）為準核對（K2 規格 draft-2 §G＋draft-3）；任務節點 trace 只用來取相位時刻。
監看窗＝UNFOLD 開始（任務事件）到結束。逐步：
  門角 |q| ≤ 1°；HOLD 期間兩指對橫桿接觸各 ≥ 0.5 N；非預期接觸（任何剛體對門板／立柱；非手指剛體對橫桿）≤ 0.1 N；
  手指對橫桿只在 ENGAGE～RELEASE 允許（其餘相位 > 0.1 N ⇒ 不通過）；牆接觸 ≤ 0.1 N；腕力 < 30 N；
  底盤相對停車錨點（park_gate.json）≤ 20 mm、偏航 ≤ 2°。
事件：相位序列含 ENGAGE → HOLD → RELEASE → RETREAT → DONE、無中止、HOLD 時長 ≥ 3.0 s。

    python3 evaluation/door_k2_check.py <run_dir>    # 寫 <run_dir>/analysis/door_k2_check.json；結束碼 0＝通過
"""
import json
import math
import os
import sys

import numpy as np

DOOR_MAX_DEG, CONTACT_N, GRIP_MIN_N, WRIST_N = 1.0, 0.1, 0.5, 30.0
BASE_M, BASE_YAW = 0.020, math.radians(2.0)
SEQ = ['ENGAGE', 'HOLD', 'RELEASE', 'RETREAT', 'DONE']
FINGERS = ('uflite_finger1', 'uflite_finger2')


def _fin(x):
    try:
        return bool(np.isfinite(np.asarray(x, float)).all())
    except (TypeError, ValueError):
        return False


def phase_windows(task):
    """由任務節點相位事件得 {phase: [(t0, t1), ...]}（模擬時間）。"""
    ev = [e for e in task.get('policy_events', []) if 'to' in e]
    out, tr = {}, task.get('trace') or []
    t_end = tr[-1]['t'] if tr else None
    for i, e in enumerate(ev):
        t1 = ev[i + 1]['t'] if i + 1 < len(ev) else t_end
        out.setdefault(e['to'], []).append((e['t'], t1))
    return out


def judge(task, room, gate):
    why = []
    if task.get('schema') != 'door_k2_task/1':
        why.append('task_schema')
    seq = [e['to'] for e in task.get('policy_events', []) if 'to' in e]
    if task.get('abort'):
        why.append(f"aborted:{task['abort']}")
    if not all(p in seq for p in SEQ) or [p for p in seq if p in SEQ] != SEQ:
        why.append(f'phase_sequence:{seq}')
    W = phase_windows(task)
    hold = W.get('HOLD', [])
    if not hold or hold[0][1] is None or hold[0][1] - hold[0][0] < 3.0 - 1e-6:
        why.append('hold_too_short_or_missing')
    t_unfold = W.get('UNFOLD', [(None, None)])[0][0]
    if t_unfold is None:
        why.append('no_unfold_event')
        return {'pass': False, 'why': why}
    cols = room.get('steps_cols') or []
    need = ['step', 'sim_t', 'base_xyth', 'door_q_rad', 'leaf_contacts', 'wall_contacts', 'wrist_force_n']
    if any(c not in cols for c in need):
        why.append('room_cols_missing')
        return {'pass': False, 'why': why}
    ix = {c: cols.index(c) for c in need}
    steps = [s for s in room.get('steps', []) if s[ix['sim_t']] >= t_unfold - 1e-9]
    if not steps:
        why.append('no_steps_in_window')
        return {'pass': False, 'why': why}
    sid = [s[ix['step']] for s in steps]
    if sid != list(range(sid[0], sid[0] + len(sid))):
        why.append('step_discontinuity')
    anchor = (gate or {}).get('gate', {}).get('anchor') if gate else None
    if anchor is None:
        why.append('park_anchor_missing')

    def ph_at(t):
        for ph, ws in W.items():
            for t0, t1 in ws:
                if t1 is not None and t0 <= t < t1:
                    return ph
        return None
    stats = {'max_abs_q_deg': 0.0, 'max_unexpected_n': 0.0, 'max_wall_n': 0.0, 'max_wrist_n': 0.0, 'max_base_m': 0.0,
             'max_base_yaw_deg': 0.0, 'min_hold_finger_bar_n': [math.inf, math.inf], 'max_finger_bar_outside_n': 0.0}
    bad = set()
    for s in steps:
        t = s[ix['sim_t']]
        q, lc, wc, wf, b = s[ix['door_q_rad']], s[ix['leaf_contacts']], s[ix['wall_contacts']], s[ix['wrist_force_n']], s[ix['base_xyth']]
        if not _fin(q) or lc is None or wc is None or wf is None or not _fin(wf) or not _fin(b):
            bad.add('missing_or_nonfinite')
            continue
        stats['max_abs_q_deg'] = max(stats['max_abs_q_deg'], abs(math.degrees(q)))
        ph = ph_at(t)
        fb = [lc.get(f, {}).get('bar', 0.0) for f in FINGERS]
        unexp = max([v.get('panel', 0.0) + (0.0 if k in FINGERS else v.get('bar', 0.0)) for k, v in lc.items()] or [0.0])
        stats['max_unexpected_n'] = max(stats['max_unexpected_n'], unexp)
        stats['max_wall_n'] = max(stats['max_wall_n'], max(wc.values()) if wc else 0.0)
        stats['max_wrist_n'] = max(stats['max_wrist_n'], wf)
        if ph == 'HOLD':
            stats['min_hold_finger_bar_n'] = [min(stats['min_hold_finger_bar_n'][k], fb[k]) for k in range(2)]
        elif ph not in ('ENGAGE', 'RELEASE'):
            stats['max_finger_bar_outside_n'] = max(stats['max_finger_bar_outside_n'], max(fb))
        if anchor is not None:
            stats['max_base_m'] = max(stats['max_base_m'], math.hypot(b[0] - anchor[0], b[1] - anchor[1]))
            stats['max_base_yaw_deg'] = max(stats['max_base_yaw_deg'],
                                            abs(math.degrees(math.atan2(math.sin(b[2] - anchor[2]), math.cos(b[2] - anchor[2])))))
    why += sorted(bad)
    if stats['max_abs_q_deg'] > DOOR_MAX_DEG:
        why.append('door_angle_over_1deg')
    if stats['max_unexpected_n'] > CONTACT_N:
        why.append('unexpected_contact')
    if stats['max_wall_n'] > CONTACT_N:
        why.append('wall_contact')
    if stats['max_wrist_n'] >= WRIST_N:
        why.append('wrist_force')
    if min(stats['min_hold_finger_bar_n']) < GRIP_MIN_N:
        why.append('hold_finger_contact_below_min')
    if stats['max_finger_bar_outside_n'] > CONTACT_N:
        why.append('finger_bar_outside_allowed_phase')
    if stats['max_base_m'] > BASE_M or stats['max_base_yaw_deg'] > math.degrees(BASE_YAW):
        why.append('base_drift')
    return {'pass': not why, 'why': why, 'stats': stats, 'phase_sequence': seq, 'n_steps_window': len(steps)}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    res = judge(task, room, gate)
    p = os.path.join(run, 'analysis', 'door_k2_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'why', 'stats')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
