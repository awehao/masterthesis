#!/usr/bin/env python3
"""W2 模式轉換的逐步證據核對（純函式；Codex 20261025_201543：§2 逐步證據是驗證要求，不單獨等於安全保證）。

輸入：
- steps：執行端（或假世界執行端）逐物理步紀錄，每列 {step, t, src, mode_seq, chain_mode, profile, auth_mode_seq}
  - src：本步套用的 source_seq（−1＝無來源）；mode_seq：該筆命令自身的 mode_seq；
  - chain_mode：0 正常、1 逾時；profile：本步採用的監看 profile；auth_mode_seq：本步權威模式序號。
- auth：ModeAuthority 的 commit、events、n_stale、aborted；
- solver_w2：求解端 stats.w2（committed、first_src_by_mode）；
- stale_injected：假世界注入的延遲舊命令 source_seq 清單。

核對：
- C1 恰一次提交；提交步 k 套用的 src＝commit.first_source_seq，且該步 mode_seq＝1、chain_mode 0。
- C2 無空窗：從「請求」到「k＋window」之間，每步 chain_mode＝0 且 src ≥ 0。
- C3 每步 mode_seq 不減；k 之前全為 0，k 起全為 1；每步命令的 mode_seq＝本步權威 auth_mode_seq（不以最新狀態猜）。
- C4 監看 profile 與該步權威模式一致，k 之前 pause_hold、k 起 release_prep_test（同一步切換）。
- C5 延遲舊命令：被注入的 src 不得出現在 k 之後的套用列；n_stale ≥ 注入數。
- C6 求解端已由權威狀態得知提交（committed＝1），首筆 v3 mode 1 的 source_seq ≤ 首筆套用。
- C7 HELD 持續發布：OBS_PASSED 之後到請求之前每步 chain_mode＝0（不以觀察通過停發）。
"""
from __future__ import annotations

import door_w2_mode as W2M


def check(steps, auth, solver_w2, t_req, t_obs_passed=None, stale_injected=(), window=50):
    why = []
    commits = [e for e in (auth.get('events') or []) if e.get('kind') == 'commit']
    if len(commits) != 1 or not isinstance(auth.get('commit'), dict):
        return {'ok': False, 'why': [f'commit_count_{len(commits)}'], 'k': None}
    c = auth['commit']
    k = c['step']
    by = {r['step']: r for r in steps}
    rk = by.get(k)
    if rk is None or rk.get('src') != c['first_source_seq'] or rk.get('mode_seq') != 1 or rk.get('chain_mode') != 0:
        why.append('C1_first_applied_mismatch')
    win = [r for r in steps if r['t'] >= t_req - 1e-9 and r['step'] <= k + window]
    if not win or any(r.get('chain_mode') != 0 or (r.get('src') is None or r['src'] < 0) for r in win):
        why.append('C2_gap_around_switch')
    ms = [r.get('mode_seq') for r in steps if r.get('src') is not None and r['src'] >= 0]
    if any(b < a for a, b in zip(ms, ms[1:])):
        why.append('C3_mode_seq_backward')
    if any((r['step'] < k and r.get('mode_seq') not in (0, None)) or (r['step'] >= k and r.get('src', -1) >= 0 and r.get('mode_seq') != 1)
           for r in steps):
        why.append('C3_mode_seq_not_switched_at_k')
    if any(r.get('src', -1) >= 0 and r.get('mode_seq') != r.get('auth_mode_seq') for r in steps):
        why.append('C3_cmd_mode_ne_authority')
    if any(r.get('profile') != W2M.PROFILE_OF_SEQ.get(r.get('auth_mode_seq')) for r in steps) \
            or any((r['step'] < k) != (r.get('profile') == 'pause_hold') for r in steps):
        why.append('C4_profile_not_switched_at_k')
    inj = set(stale_injected)
    if inj and any(r['step'] >= k and r.get('src') in inj for r in steps):
        why.append('C5_stale_applied')
    if (auth.get('n_stale') or 0) < len(inj):
        why.append('C5_stale_not_counted')
    sw = solver_w2 or {}
    f1 = (sw.get('first_src_by_mode') or {}).get('1')
    if sw.get('committed') != 1 or f1 is None or f1 > c['first_source_seq']:
        why.append('C6_solver_not_committed')
    if t_obs_passed is not None:
        held = [r for r in steps if t_obs_passed + 0.3 <= r['t'] < t_req]
        if not held or any(r.get('chain_mode') != 0 for r in held):
            why.append('C7_held_not_publishing')
    return {'ok': not why, 'why': why, 'k': k, 'first_source_seq': c['first_source_seq'], 'n_window': len(win),
            'n_stale': auth.get('n_stale')}
