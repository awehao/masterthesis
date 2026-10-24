#!/usr/bin/env python3
"""W1e 判定器 w1e2（**看過 door_wide_w1e_30_1 結果後**才發現並修正的判定器缺陷，須揭露；比照 p4b2 先例；door_k3m_check_w1e.py 凍結不改）。

缺陷：w1e 以 1e-6 s 比對執行端紀錄時間與 room_run 步時間；room_run 的 sim_t 以兩位小數保存（54.21），執行端紀錄為 float32 累積的
world.current_time（54.2099987883），實測差約 1.2e-6 s ⇒ 切換與 12 筆定期讀回全被誤判「不在真實步」。
修正：唯一改動＝時間對位容差 1e-6 → 5e-6 s（與 W1d r2 判定器相同理由；物理步 0.01 s，不會配錯步）。步號必須相同的要求、
其餘所有判準（F1–F3、w1c／w1b 全部判準、ok 明確 True、嚴格前進、間隔 ≥ 1 s、非手指不變）一律不變。

    python3 evaluation/door_k3m_check_w1e2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1e2.json；不覆寫
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w1b as W1B                       # noqa: E402
import door_k3m_check_w1e as E                         # noqa: E402

T_TOL = 5e-6


def _at_room_r2(st_t, step, t):
    return isinstance(step, int) and W1B._num(t) and step in st_t and abs(st_t[step] - float(t)) <= T_TOL


def judge(task, room, gate, solver, observer):
    orig = E._at_room
    E._at_room = _at_room_r2
    try:
        res = E.judge(task, room, gate, solver, observer)
    finally:
        E._at_room = orig
    return dict(res, judge='w1e2', time_tol_s=T_TOL)


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_w1e2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop',
                                             'w1b_registration_ok', 'w1b_guidance_ok', 'offset_hold_ok', 'finger_effort_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
