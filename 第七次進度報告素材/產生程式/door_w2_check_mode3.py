#!/usr/bin/env python3
"""W2 模式轉換逐步證據核對 r3（純函式；Codex 20261025_214144）。door_w2_check_mode.py（凍結 w2a r2）**不改**；
本檔先跑 r2 的 C0–C7（逐字沿用），再加 r3 項目。

輸入：
- steps：執行端逐物理步紀錄（door_w2_exec.W2Exec.rows），每列 {step, t, src, mode_seq, chain_mode, profile, auth_mode_seq,
  confirm_ok, confirm_why, u}；
- auth：ModeAuthority3.stats()（含 events、commit、applied 確認結果、aborted、violations）；
- auth_rows：ModeAuthority3.applied_rows（權威端收到的逐步確認）；
- 其餘同 r2。

r3 核對：
- C8 確認覆蓋：窗口內每一步都有權威端確認列（步號一一對應），且 confirm_ok 為 True；
  權威確認列的來源、mode_seq、鏈模式、九軸值＝執行端紀錄列（兩處紀錄一致，不以其一代替）。
- C9 提交步實際值：commit.applied_confirmed True，且 applied_u 為保持形；提交後有來源的步九軸皆保持形。
- C10 時間：提交筆的選定時刻 − 原始 stamp ≤ cmd_max_age；提交時刻 − 請求登記時刻 ≤ cand_timeout（以 events 的 pending 為準）。
- C11 無中止、無入口違規。
"""
from __future__ import annotations

import math

import door_w2_check_mode as CK2


def _hold(u, hold_v, hold_w, arm_tol, tol=1e-9):
    if not isinstance(u, (list, tuple)) or len(u) != 9 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in u):
        return False
    return math.hypot(u[0], u[1]) <= hold_v + tol and abs(u[2]) <= hold_w + tol and max(abs(x) for x in u[3:]) <= arm_tol + tol


def check(steps, auth, auth_rows, solver_w2, t_req, t_obs_passed=None, stale_injected=(), window=50,
          cmd_max_age_s=0.2, cand_timeout_s=1.0, hold_v=0.005, hold_w=0.02, arm_tol=1e-6):
    r = CK2.check(steps, auth, solver_w2, t_req, t_obs_passed=t_obs_passed, stale_injected=stale_injected, window=window)
    why = list(r['why'])
    c = auth.get('commit') if isinstance(auth, dict) else None
    if not isinstance(c, dict):
        return dict(r, ok=False, why=why + ['C8_no_commit'])
    k = c['step']
    t0 = t_obs_passed if t_obs_passed is not None else t_req
    win = [s for s in steps if isinstance(s.get('t'), (int, float)) and s['t'] >= t0 - 1e-9 and type(s.get('step')) is int
           and s['step'] <= k + window]
    ar = {}
    dup = False
    for a in auth_rows or []:
        if a.get('step') in ar:
            dup = True
        ar[a.get('step')] = a
    if dup or not win or any(s['step'] not in ar for s in win):
        why.append('C8_confirm_not_covering_window')
    if any(s.get('confirm_ok') is not True for s in win):
        why.append('C8_confirm_not_ok')
    for s in win:
        a = ar.get(s['step'])
        if a is None:
            continue
        if (a.get('src'), a.get('mode_seq'), a.get('chain_mode')) != (s.get('src'), s.get('mode_seq'), s.get('chain_mode')) \
                or any(abs(float(x) - float(y)) > 1e-8 for x, y in zip(a.get('u') or [], s.get('u') or [])) \
                or len(a.get('u') or []) != len(s.get('u') or []):
            why.append('C8_records_disagree')
            break
    if c.get('applied_confirmed') is not True or not _hold(c.get('applied_u'), hold_v, hold_w, arm_tol):
        why.append('C9_commit_applied_not_hold')
    if any(s['step'] >= k and s.get('src', -1) >= 0 and not _hold(s.get('u'), hold_v, hold_w, arm_tol) for s in steps
           if type(s.get('step')) is int):
        why.append('C9_post_commit_not_hold')
    pend = [e for e in (auth.get('events') or []) if e.get('kind') == 'pending']
    st_ = c.get('stamp_sim_t')
    if not isinstance(st_, (int, float)) or not c['t'] - st_ <= cmd_max_age_s + 1e-9:
        why.append('C10_commit_source_stale')
    if len(pend) != 1 or not c['t'] - pend[0]['t'] <= cand_timeout_s + 1e-9:
        why.append('C10_commit_after_deadline')
    if auth.get('aborted') is not None or (auth.get('violations') or []):
        why.append('C11_aborted_or_violation')
    return dict(r, ok=not why, why=why, n_confirm=len(ar))
