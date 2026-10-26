#!/usr/bin/env python3
"""核心段判定器 r3（door_coopen_check_core2.py 的版本化擴充；Codex 20261027_003024 接近進度閘控；r1、r2 保留不改）。
r2 全部照用（chain_pass＝w1f2＋J1–J3；核心證據 core 與舊鏈分開），另加：
  J4（chain_pass 條件）閘控登錄：task.json coopen_gate.config＝登錄檔 approach_gate（逐鍵相等）；缺 ⇒ coopen_gate_missing／mismatch；
  核心證據報告另記 approach_gate_stats 與 pause／resume 事件（只報告）。

    python3 evaluation/door_coopen_check_core3.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core3.json；不覆寫
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core2 as J2C                  # noqa: E402


def check_gate(task, why):
    co = (task or {}).get('coopen') or {}
    p = co.get('params_path')
    if not (isinstance(p, str) and os.path.exists(os.path.join(WS, p))):
        why.append('coopen_gate_params_missing')
        return None
    reg = json.load(open(os.path.join(WS, p))).get('approach_gate')
    got = ((task or {}).get('coopen_gate') or {}).get('config')
    if not isinstance(reg, dict) or not isinstance(got, dict):
        why.append('coopen_gate_missing')
    elif set(reg) != set(got) or any(abs(float(reg[k]) - float(got[k])) > 1e-12 for k in reg):
        why.append('coopen_gate_mismatch')
    return reg


def judge(task, room, gate, solver, observer, evidence_path=None):
    res = J2C.judge(task, room, gate, solver, observer, evidence_path)
    why_g = []
    check_gate(task, why_g)
    why = list(res.get('why') or []) + why_g
    core = dict(res.get('core') or {})
    core['approach_gate_stats'] = (task or {}).get('approach_gate_stats')
    core['approach_gate_events'] = [e for e in ((task or {}).get('policy_events') or []) if 'approach_gate_pause' in e or 'approach_gate_resume' in e]
    chain = not why
    return dict(res, why=why, judge='coopen_core3', chain_pass=chain, core=core, coopen_gate_registration_ok=not why_g, **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core3.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'offset_hold_ok',
                                             'finger_effort_ok', 'trace_log_ok', 'coopen_registration_ok', 'coopen_gate_registration_ok', 'why')},
                     ensure_ascii=False, default=float))
    print(json.dumps({'core': res.get('core')}, ensure_ascii=False, default=float)[:3000])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
