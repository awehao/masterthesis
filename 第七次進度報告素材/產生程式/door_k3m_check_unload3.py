#!/usr/bin/env python3
"""閉爪卸載的獨立判定 r3（door_k3m_check_unload2.py〔r2〕的版本化複本；Codex 20261026_034842 §1、§3）。r3 差異：
  - 時間對帳改為**步號一致＋明確量化容差** TIME_Q＝5.1e-5 s（room 同步列 sim_t 以 1e-4 s 四捨五入，最大誤差 5e-5 s）；
    適用於 set_log 對 room 列、mode 1 提交對 t0（r2 的 1e-6 會誤拒既有 1.5125e-6 s 差）。
  - U8 改為以 door_k3b_transient2 規則逐 room 列**獨立重算卸載瞬態收斂**（位移、速度、資料有效、收斂窗與持續時間），
    並交叉核對宣告：求解端 UNLOADED ≥ max(重算收斂時刻, t_settling＋quiet_s) − DECL_TOL（0.06 s：求解端 20 Hz 週期＋同步 50 Hz 間隔）；
    重算未於 unload_max_s 內收斂 ⇒ U8_unload_transient_timeout。
以下為 r2 說明：
閉爪卸載的獨立判定 r2（door_k3m_check_unload.py〔r1〕的版本化複本；Codex 20261026_025647 必修 C）。
r2 差異：
  - 全部輸入先核型別與有限性；任何缺欄、非法資料或例外 ⇒ 回報理由（U_internal_error:<型別>），不以例外結束；
  - set_log 每筆須為 (int 步, 有限時間, 有限設定, 兩指有限讀回)；**步號唯一**（重複 ⇒ U0_set_log_duplicate，不先轉字典掩蓋）；
    時間須與 room 同步列時間逐步對帳（|Δ| ≤ 1e-6）；
  - solver_outcome **必要**（缺 ⇒ U5_solver_outcome_missing）；求解端 UNLOADED 須不晚於任務端 mode 2 請求；
  - U4 低速窗起點另須 ≥ 執行端 t_settling（斜坡完成、讀回 5 N）；
  - U8 卸載瞬態重算：step0 至執行端 UNLOADED，以 room 逐步 FK（door_k2_check._tcp）核 TCP 位移 ≤ 5 mm／2°、0.1 s 窗速度 ≤ 0.02 m/s。
r1 其餘項目（U0–U7 定義）見 door_k3m_check_unload.py 檔頭。
"""
from __future__ import annotations

import math

import numpy as np

import door_k3b_transient2 as TT
import door_k3b_unload2 as KU
import door_k3m_check_unload as C1

VERSION = 'door_k3m_check_unload/3'
TIME_Q = 5.1e-5
DECL_TOL = 0.06
_fin = KU._fin


def _valid_set_log(sl):
    if not isinstance(sl, list):
        return None, 'not_list'
    out = []
    for x in sl:
        if not (isinstance(x, (list, tuple)) and len(x) == 4 and type(x[0]) is int and _fin(x[1]) and _fin(x[2])
                and isinstance(x[3], (list, tuple)) and len(x[3]) == 2 and _fin(*x[3])):
            return None, 'entry_invalid'
        out.append(x)
    steps = [x[0] for x in out]
    if len(set(steps)) != len(steps):
        return None, 'duplicate'
    return out, None


def recompute_transient(S, I, i0, i1, t_max):
    """以 transient2 規則重算：起點＝第 i0 列（提交前最後一列），依序餵入 (i0, i1] 各列。回傳 (state, why, t_conv, worst)。"""
    import door_k2_check as KC
    tm = TT.TransientMonitor(lambda b, a: KC._tcp(b, a), t_max)
    s0 = S[i0]
    if not tm.start(s0[I['sim_t']], s0[I['base_xyth']], s0[I['q_arm_meas']]):
        return 'FAILED', 'start_invalid', None, tm.worst
    for s in S[i0 + 1:i1 + 1]:
        tm.feed(s[I['sim_t']], s[I['base_xyth']], s[I['q_arm_meas']])
        if tm.state != 'ACTIVE':
            break
    return tm.state, tm.why, tm.t_conv, dict(tm.worst)


def check_unload3(room, task, solver_outcome):
    try:
        return _check(room, task, solver_outcome)
    except Exception as e:                                                       # noqa: BLE001
        return [f'U_internal_error:{type(e).__name__}'], {}


def _check(room, task, solver_outcome):
    why, st = [], {}
    u = ((room or {}).get('w2') or {}).get('unload') if isinstance(room, dict) else None
    if not isinstance(u, dict) or u.get('version') != KU.VERSION:
        return ['U0_unload_record_missing'], st
    r = dict(KU.REG_UNLOAD)
    if not isinstance(u.get('reg'), dict) or any(u['reg'].get(k) != v for k, v in KU.REG_UNLOAD.items()):
        why.append('U0_registration_mismatch')
    I, S = C1._rows(room)
    need = ('base_xyth', 'q_arm_meas', 'door_q_rad', 'leaf_contacts')
    if any(k not in I for k in need) or not S:
        return why + ['U0_room_columns_missing'], st
    ev = ((room.get('w2') or {}).get('auth') or {}).get('events') or []
    c1s = [e for e in ev if isinstance(e, dict) and e.get('kind') == 'commit' and e.get('mode_seq') == 1]
    c2s = [e for e in ev if isinstance(e, dict) and e.get('kind') == 'commit' and e.get('mode_seq') == 2]
    req2s = [e.get('t') for e in ((task or {}).get('events') or []) if isinstance(e, dict)
             and ((e.get('w2_mode_request_published') or {}).get('mode_seq') == 2)]
    t0, k0, ts_ = u.get('t0'), u.get('step0'), u.get('t_settling')
    if len(c1s) != 1 or len(c2s) != 1 or len(req2s) != 1 or not (_fin(t0) and type(k0) is int and _fin(req2s[0])):
        return why + ['U5_commit_or_request_missing_or_dup'], st
    c1, c2, req2 = c1s[0], c2s[0], float(req2s[0])
    if not (type(c1.get('step')) is int and type(c2.get('step')) is int and _fin(c1.get('t')) and _fin(c2.get('t'))):
        return why + ['U5_commit_fields_invalid'], st
    if c1['step'] != k0 or abs(c1['t'] - t0) > TIME_Q:
        why.append('U5_step0_not_mode1_commit')
    k2 = c2['step']
    span = [i for i, s in enumerate(S) if k0 <= s[0] < k2]
    if not span or S[span[0]][0] != k0 or S[span[-1]][0] != k2 - 1 or any(S[a][0] + 1 != S[b][0] for a, b in zip(span, span[1:])):
        return why + ['U0_room_steps_not_contiguous'], st
    sl, w0 = _valid_set_log(u.get('set_log'))
    if sl is None:
        why.append(f'U0_set_log_{w0}')
        sl = []
    by = {x[0]: x for x in sl}
    bad0 = []
    for i in span:
        k, t = S[i][0], S[i][1]
        x = by.get(k)
        if x is None or abs(x[1] - t) > TIME_Q or abs(x[2] - KU.f_cmd(r, t0, x[1])) > KU.TOL or any(abs(y - x[2]) > KU.TOL for y in x[3]):
            bad0.append(k)
    if sl and bad0:
        why.append(f'U0_set_log_{len(bad0)}')
    extra = [k for k in by if not (k0 <= k < k2)]
    if extra:
        why.append(f'U0_set_log_outside_span_{len(extra)}')
    last = by.get(k2 - 1)
    if last is None or any(abs(y - r['f_lo_n']) > KU.TOL for y in last[3]):
        why.append('U6_readback_not_f_lo_at_open')
    b0 = S[span[0]][I['base_xyth']]
    wc, wd, wb, wdeg = math.inf, 0.0, 0.0, 0.0
    n1 = n2 = n3 = 0
    for i in span:
        s = S[i]
        f = C1._fpair(s[I['leaf_contacts']])
        okf = all(_fin(x) for x in f)
        wc = min(wc, min(f)) if okf else -math.inf
        n1 += int(not okf or min(f) < r['contact_min_n'])
        v = C1._speeds(S, I, i, dict(r, arm_win_s=r['win_s']))                    # 門／底盤短窗獨立（不等手臂長窗）
        if v is not None:
            wd = max(wd, v[2])
            n2 += int(v[2] > r['door_rate_protect'])
        b = s[I['base_xyth']]
        dm = math.hypot(b[0] - b0[0], b[1] - b0[1])
        dg = abs(math.degrees(math.atan2(math.sin(b[2] - b0[2]), math.cos(b[2] - b0[2]))))
        wb, wdeg = max(wb, dm), max(wdeg, dg)
        n3 += int(dm > r['base_dev_m'] or dg > r['base_dev_deg'])
    for tag, n in (('U1_contact_lost', n1), ('U2_door_rate', n2), ('U3_base_dev', n3)):
        if n:
            why.append(f'{tag}_{n}')
    st['unload_worst'] = {'contact_min_n': wc, 'door_rate': wd, 'base_dev_m': wb, 'base_dev_deg': wdeg}
    t_win0 = t0 + r['ramp_s'] if not _fin(ts_) else max(t0 + r['ramp_s'], ts_)
    if not _fin(ts_) or ts_ < t0 + r['ramp_s'] - 0.02 - 1e-9:
        why.append('U4_t_settling_missing_or_early')

    def quiet_over(t_end):
        idx = [i for i in span if t_end - r['quiet_s'] - 1e-9 <= S[i][1] <= t_end + 1e-9]
        if not idx or S[idx[0]][1] > t_end - r['quiet_s'] + r['dt'] + 1e-9 or S[idx[0]][1] < t_win0 - 1e-9:
            return False, 'window_not_covered_or_before_settling'
        bad = [S[i][0] for i in idx if not C1._quiet(C1._speeds(S, I, i, r), r)]
        return not bad, (None if not bad else f'not_quiet_at_{bad[0]}')
    ok4, w4 = quiet_over(min(req2, S[span[-1]][1]))
    if not ok4:
        why.append(f'U4_not_quiet_before_open:{w4}')
    tu = (u.get('t') or {}).get('UNLOADED')
    if not _fin(tu):
        why.append('U5_exec_unloaded_missing')
    else:
        if tu - t0 > r['unload_max_s'] + 1e-9:
            why.append('U5_unload_over_max')
        if not (tu <= req2 + 1e-9 and req2 <= c2['t'] + 1e-9):
            why.append('U5_order_unloaded_request_commit')
        ok7, w7 = quiet_over(tu)
        if not ok7:
            why.append(f'U7_exec_unloaded_not_supported:{w7}')
    i_pre = next((i for i in range(span[0] - 1, -1, -1) if S[i][1] <= t0 + TIME_Q), None)
    if i_pre is None:
        why.append('U8_no_row_before_unload')
    else:
        ts8, w8, tc8, wst8 = recompute_transient(S, I, i_pre, span[-1], r['unload_max_s'])
        st['unload_transient'] = {'state': ts8, 'why': w8, 't_conv': tc8, 'worst': wst8}
        if ts8 == 'FAILED':
            why.append(f'U8_unload_transient_{w8}')
        elif ts8 != 'CONVERGED':
            why.append('U8_unload_transient_not_converged')
        elif isinstance(solver_outcome, dict):
            su = (solver_outcome.get('transitions') or {}).get('UNLOADED')
            need = max(tc8, (ts_ if _fin(ts_) else math.inf) + r['quiet_s'])
            if not _fin(su) or su < need - DECL_TOL:
                why.append('U8_solver_unloaded_before_recomputed_convergence')
    if not isinstance(solver_outcome, dict):
        why.append('U5_solver_outcome_missing')
    else:
        tr = solver_outcome.get('transitions') or {}
        a, b = tr.get('UNLOAD'), tr.get('UNLOADED')
        if not (_fin(a) and _fin(b) and t0 - 1e-6 <= a <= b <= req2 + 1e-9):
            why.append('U5_solver_unload_order_or_after_request')
    st['unload_times'] = {'t0': t0, 't_settling': ts_, 'unloaded_exec': tu, 'req2': req2, 'commit2': c2['t'],
                          'solver': None if not isinstance(solver_outcome, dict)
                          else {k: (solver_outcome.get('transitions') or {}).get(k) for k in ('UNLOAD', 'UNLOADED')}}
    return why, st
