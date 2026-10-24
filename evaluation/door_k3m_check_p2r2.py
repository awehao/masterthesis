#!/usr/bin/env python3
"""K3-M＋受控暫停判定器 p2 r2（door_k3m_check_p2.py 的版本化擴充；2026-10-24 Codex 無法使用，經 Howard 授權由 Claude 裁定）。

唯一修正：凍結目標核對 `pause_frozen_target_not_last_published` 的旋轉差。
  原判定器以 door_k3m_pause_mode.rot_angle（acos((tr(RaᵀRb) − 1)/2)）計算，門檻 1e-9 rad。
  acos 在單位旋轉附近放大捨入誤差：同一個矩陣與自己比較就得到 3.65e-8 rad（door_k3m_pause_p2c_2 的凍結目標），
  因此 1e-9 在數值上不可能達到 ⇒ 偽拒。
  r2 改用數值穩定的 θ＝2·asin(‖Ra − Rb‖_F ／ (2√2))（對正交旋轉與 acos 定義相同角度，小角度不放大誤差）；
  門檻、序號與「依實際發布順序取第一列 PAUSE 之前最後一筆已發布目標」的規則都不變。
其餘理由全部沿用 door_k3m_check_p2.judge 的輸出；若父判定器給出該理由，以本檔重算結果取代
（stats.frozen_target_r2 記錄重算值、stats.parent_frozen_reason_replaced 記錄是否取代）。

    python3 evaluation/door_k3m_check_p2r2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_p2r2.json；不覆寫既有輸出
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                          # noqa: E402
import door_k3m_check_p2 as C2                         # noqa: E402

REASON = 'pause_frozen_target_not_last_published'
TOL_POS, TOL_ROT = 1e-9, 1e-9


def rot_angle_stable(Ta, Tb):
    Ra = np.asarray(Ta, float).reshape(4, 4)[:3, :3]
    Rb = np.asarray(Tb, float).reshape(4, 4)[:3, :3]
    s = float(np.linalg.norm(Ra - Rb)) / (2.0 * math.sqrt(2.0))
    return 2.0 * math.asin(min(1.0, s))


def target_diff_stable(Ta, Tb):
    a = np.asarray(Ta, float).reshape(4, 4)
    b = np.asarray(Tb, float).reshape(4, 4)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), rot_angle_stable(a, b)


def frozen_target_ok(task):
    """回傳 (ok, detail)。規則同 door_k3m_check_p r3，只換旋轉差的算法。"""
    req = (task.get('k3m_pause') or {}).get('request') or {}
    T_req = CP._T16(req.get('frozen_target_T'))
    trace = task.get('trace') or []
    i_pause = next((i for i, e in enumerate(trace) if e.get('phase') == 'PAUSE'), None)
    if i_pause is None or T_req is None:
        return False, {'why': 'pause_phase_or_request_missing'}
    prior = [e for e in trace[:i_pause] if e.get('target') is not None and isinstance(e.get('pub_seq'), int)]
    if not prior:
        return False, {'why': 'no_prior_published'}
    last = prior[-1]
    T_last = CP._T16(last['target'])
    if T_last is None:
        return False, {'why': 'last_target_invalid'}
    dp, dr = target_diff_stable(T_last, T_req)
    ok = last.get('pub_seq') == req.get('frozen_target_seq') and dp <= TOL_POS and dr <= TOL_ROT
    return ok, {'last_pub_seq': last.get('pub_seq'), 'frozen_target_seq': req.get('frozen_target_seq'), 'dpos_m': dp, 'drot_rad': dr,
                'tol': [TOL_POS, TOL_ROT]}


def judge(task, room, gate, solver):
    res = C2.judge(task, room, gate, solver)
    st = res['stats']
    replaced = REASON in res['why']
    why = [w for w in res['why'] if w != REASON]
    has_req = bool((task.get('k3m_pause') or {}).get('request'))
    ok, det = frozen_target_ok(task) if has_req else (False, {'why': 'no_request'})   # 無請求：父判定器另有 pause_evidence_missing
    st['frozen_target_r2'] = det
    st['parent_frozen_reason_replaced'] = replaced
    if has_req and not ok:
        why.append(REASON)
    return dict(res, why=why, **{'pass': not why})


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    solver = json.load(open(os.path.join(run, 'align_solver.json')))
    res = judge(task, room, gate, solver)
    p = os.path.join(run, 'analysis', 'door_k3m_check_p2r2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'why')},
                     ensure_ascii=False, default=float))
    print(json.dumps(res['stats'].get('frozen_target_r2'), default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
