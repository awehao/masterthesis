#!/usr/bin/env python3
"""核心段判定器 r4（door_coopen_check_core3.py 的版本化修正；Codex 20261027_004139 §3；r1–r3 保留不改）。
r3 全部照用，只改 J4 閘控登錄核對：先檢查型別（int／float、非 bool）與有限性，再逐鍵比較；登錄或紀錄值為非數值、NaN、Inf、缺欄 ⇒
  回不通過理由（coopen_gate_invalid_registered／coopen_gate_invalid_recorded／coopen_gate_missing／coopen_gate_mismatch），不拋例外。
r3 的 J4 以差值比較，perp_pause_m 為 NaN 時 abs(NaN − x) > tol 為假 ⇒ 誤判一致（Codex 重現）。

    python3 evaluation/door_coopen_check_core4.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core4.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core2 as J2C                  # noqa: E402

KEYS = ('perp_pause_m', 'perp_resume_m', 'resume_hold_s', 'along_resume_m', 'rot_resume_rad', 'align_confirm_s')


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def check_gate(task, why):
    try:
        co = (task or {}).get('coopen') or {}
        p = co.get('params_path')
        if not (isinstance(p, str) and os.path.exists(os.path.join(WS, p))):
            why.append('coopen_gate_params_missing')
            return None
        reg = json.load(open(os.path.join(WS, p))).get('approach_gate')
        got = ((task or {}).get('coopen_gate') or {}).get('config')
        if not isinstance(reg, dict) or not isinstance(got, dict):
            why.append('coopen_gate_missing')
            return reg
        if set(reg) != set(KEYS) or not all(_num(reg[k]) for k in KEYS):
            why.append('coopen_gate_invalid_registered')
            return reg
        if set(got) != set(KEYS) or not all(_num(got.get(k)) for k in KEYS):
            why.append('coopen_gate_invalid_recorded')
            return reg
        if any(float(reg[k]) != float(got[k]) for k in KEYS):
            why.append('coopen_gate_mismatch')
        return reg
    except Exception as e:                                  # 判定器不拋例外：轉為不通過理由
        why.append(f'coopen_gate_check_error:{type(e).__name__}')
        return None


def judge(task, room, gate, solver, observer, evidence_path=None):
    res = J2C.judge(task, room, gate, solver, observer, evidence_path)
    why_g = []
    check_gate(task, why_g)
    why = list(res.get('why') or []) + why_g
    core = dict(res.get('core') or {})
    core['approach_gate_stats'] = (task or {}).get('approach_gate_stats')
    core['approach_gate_events'] = [e for e in ((task or {}).get('policy_events') or []) if 'approach_gate_pause' in e or 'approach_gate_resume' in e]
    chain = not why
    return dict(res, why=why, judge='coopen_core4', chain_pass=chain, core=core, coopen_gate_registration_ok=not why_g, **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core4.json')
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
