#!/usr/bin/env python3
"""K3-M 受控暫停 p4b 判定器 r2（door_k3m_check_p4b.py 的版本化擴充；Claude 依 Howard 授權裁定）。

唯一修正：PAUSE→END_HOLD 對應的**零長度事件**。
  策略（door_k3m_policy_p／p2）在完整 SETTLE 成立的同一輪先記 SETTLE→END_HOLD、再記 END_HOLD→PAUSE（同一時刻），
  PAUSE 取代 END_HOLD。door_k3m_check_p.to_k3m_task 把 PAUSE 對應成 END_HOLD 後，相位序列出現兩個 END_HOLD；
  K3-A 判準只看第一個 END_HOLD 窗口（零長度），因此 door_k3m_pause_p4b_1（首次到 DONE）被判
  `phase_sequence` 與 `end_hold_too_short_or_missing`，但真正的 END_HOLD（＝PAUSE）窗口是 31.46→43.86 s。
  r2 在送入 K3-M 判準前，**只刪除**「to＝END_HOLD 且下一事件為同一時刻（|Δt| ≤ 1e-9）的 END_HOLD→PAUSE」這一筆事件；
  刪除筆數記入 stats.zero_length_end_hold_removed。其餘判準（含 END_HOLD ≥ 3 s、序列、幾何、監看、SETTLE p2、
  主動保持停止 P1–P4、手臂 0.5 s 窗口）全部沿用 door_k3m_check_p4b。

    python3 evaluation/door_k3m_check_p4b2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_p4b2.json；不覆寫
"""
from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                          # noqa: E402
import door_k3m_check_p2 as C2                         # noqa: E402
import door_k3m_check_p4b as C4B                       # noqa: E402


def drop_zero_length_end_hold(task):
    """回傳 (task 複本, 刪除筆數)。只刪「END_HOLD 事件緊接同時刻 END_HOLD→PAUSE」。"""
    t2 = copy.deepcopy(task)
    ev = t2.get('policy_events') or []
    keep, n = [], 0
    for i, e in enumerate(ev):
        nxt = next((x for x in ev[i + 1:] if 'to' in x), None)
        if (e.get('to') == 'END_HOLD' and nxt is not None and nxt.get('from') == 'END_HOLD' and nxt.get('to') == 'PAUSE'
                and abs(float(nxt['t']) - float(e['t'])) <= 1e-9):
            n += 1
            continue
        keep.append(e)
    t2['policy_events'] = keep
    return t2, n


def judge(task, room, gate, solver):
    t0, n_drop = drop_zero_length_end_hold(task)
    t1 = copy.deepcopy(t0)
    t1['schema'] = 'door_k3m_task_p/1' if task.get('schema') == 'door_k3m_task_p2/1' else 'mismatch'
    res = CP.KM.judge(CP.to_k3m_task(t1), room, gate, solver)
    st = dict(res.get('stats') or {})
    st['zero_length_end_hold_removed'] = n_drop
    why_k = list(res['why'])
    if t1.get('schema') != 'door_k3m_task_p/1':
        why_k.append('task_schema_k3m_p2')
    st['parent_settle_reason_replaced'] = C2.PARENT_SETTLE_REASON in why_k
    why_k = [w for w in why_k if w != C2.PARENT_SETTLE_REASON]
    settle_ok = C2.check_settle_p2(task, room, why_k, st)                  # SETTLE p2 用原始事件（轉換事件帶 settle_sig）
    why_p = []
    cmd_ok, phys_ok = C4B.check_pause_p4(copy.deepcopy(task) | {'schema': 'door_k3m_task_p/1'}, room, solver, why_p, st)
    return {'pass': not why_k and not why_p, 'k3m_pass': not why_k, 'settle_p2_ok': settle_ok,
            'pause_command_stop': cmd_ok, 'pause_physical_stop': phys_ok, 'stop_definition': 'active_hold_arm_win_0.5s',
            'why': why_k + why_p, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    solver = json.load(open(os.path.join(run, 'align_solver.json')))
    res = judge(task, room, gate, solver)
    p = os.path.join(run, 'analysis', 'door_k3m_check_p4b2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'why')},
                     ensure_ascii=False, default=float))
    print('zero_length_end_hold_removed', res['stats']['zero_length_end_hold_removed'])
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
