#!/usr/bin/env python3
"""閉爪卸載候選的獨立判定（Codex 20261026_023226 §3；純函式，供日後實跑的判定器組合）。
輸入：room（steps_cols／steps＝逐物理步實測；w2.unload＝執行端 ExecUnloadMonitor.report()；w2.auth.events＝權威提交）、
      task（events 中的 w2_mode_request_published）、solver_outcome（求解端監督器 outcome，可為 None）。
**不信任執行端自報的狀態**：除 set_log 外，所有條件由 room 逐步紀錄重算。

  U0 紀錄：w2.unload 存在、版本相符；set_log 自 step0 起逐步連續到 mode 2 提交前一步；每步設定＝登錄斜坡 f_cmd(t)（容差 TOL），讀回＝設定。
  U1 失接觸：step0 至 mode 2 提交前一步，每步兩指（bar＋ambiguous）皆 ≥ contact_min_n。
  U2 門速保護：同區間，0.1 s 門角差分 ≤ door_rate_protect。
  U3 底盤偏離：同區間，相對 step0 實測底盤 ≤ base_dev_m／base_dev_deg。
  U4 張爪前低速：mode 2 請求時刻前連續 quiet_s 內每步（0.1 s 窗底盤、門；0.5 s 窗手臂）皆 ≤ 舊 HELD 門檻，且窗口起點 ≥ t0＋ramp_s。
  U5 時序：mode 1 提交步＝step0；執行端 UNLOADED − t0 ≤ unload_max_s；UNLOADED ≤ mode 2 請求 ≤ mode 2 提交；
     有求解端 outcome 時：UNLOAD ≤ UNLOADED ≤ mode 2 提交。
  U6 交接時上限：mode 2 提交前一步讀回兩指＝f_lo。
  U7 自報一致：執行端 UNLOADED 時刻前 quiet_s 的重算低速成立（不得早於重算支持的時刻宣稱 UNLOADED）。
"""
from __future__ import annotations

import math

import door_k3b_unload as KU


def _fin(*xs):
    return KU._fin(*xs)


def _rows(room):
    I = {c: i for i, c in enumerate(room.get('steps_cols') or [])}
    return I, room.get('steps') or []


def _fpair(lc):
    out = []
    for n in ('uflite_finger1', 'uflite_finger2'):
        v = (lc or {}).get(n) or {}
        b, a = v.get('bar'), v.get('ambiguous', 0.0)
        out.append(float(b) + float(a or 0.0) if _fin(b) and _fin(a or 0.0) else math.nan)
    return out


def _speeds(S, I, i, r):
    """第 i 列的 (底盤線速, 角速, 門速, 手臂)；窗口內步號須連續。"""
    b = S[i]

    def start(win):
        j = i
        while j > 0:
            if S[j][0] - S[j - 1][0] != 1:
                return None
            j -= 1
            if b[1] - S[j][1] >= win - 1e-9:
                return S[j]
        return None
    a, aa = start(r['win_s']), start(r['arm_win_s'])
    if a is None or aa is None:
        return None
    pb, pa = b[I['base_xyth']], a[I['base_xyth']]
    dt, dta = b[1] - a[1], b[1] - aa[1]
    dyaw = math.atan2(math.sin(pb[2] - pa[2]), math.cos(pb[2] - pa[2]))
    return (math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / dt, abs(dyaw) / dt, abs(b[I['door_q_rad']] - a[I['door_q_rad']]) / dt,
            max(abs(u - v) for u, v in zip(b[I['q_arm_meas']], aa[I['q_arm_meas']])) / dta)


def _quiet(v, r):
    return v is not None and v[0] <= r['quiet_base_lin'] and v[1] <= r['quiet_base_ang'] and v[2] <= r['quiet_door'] and v[3] <= r['quiet_arm']


def check_unload(room, task, solver_outcome=None):
    why, st = [], {}
    u = ((room or {}).get('w2') or {}).get('unload')
    if not isinstance(u, dict) or u.get('version') != KU.VERSION:
        return ['U0_unload_record_missing'], st
    r = dict(KU.REG_UNLOAD)                                                       # 判定一律用登錄值，不用紀錄自報值
    if any((u.get('reg') or {}).get(k) != v for k, v in KU.REG_UNLOAD.items()):
        why.append('U0_registration_mismatch')
    I, S = _rows(room)
    ev = ((room.get('w2') or {}).get('auth') or {}).get('events') or []
    c1 = next((e for e in ev if e.get('kind') == 'commit' and e.get('mode_seq') == 1), None)
    c2 = next((e for e in ev if e.get('kind') == 'commit' and e.get('mode_seq') == 2), None)
    req2 = next((e['t'] for e in (task or {}).get('events') or []
                 if ((e.get('w2_mode_request_published') or {}).get('mode_seq') == 2)), None)
    t0, k0 = u.get('t0'), u.get('step0')
    if c1 is None or c2 is None or not _fin(t0) or type(k0) is not int or not _fin(req2):
        return why + ['U5_commit_or_request_missing'], st
    if c1['step'] != k0:
        why.append('U5_step0_not_mode1_commit')
    k2 = c2['step']
    span = [i for i, s in enumerate(S) if k0 <= s[0] < k2]
    if not span or S[span[0]][0] != k0 or S[span[-1]][0] != k2 - 1 or any(S[a][0] + 1 != S[b][0] for a, b in zip(span, span[1:])):
        why.append('U0_room_steps_not_contiguous')
        return why, st
    # U0／U6 set_log
    sl = {x[0]: x for x in u.get('set_log') or []}
    bad0 = []
    for i in span:
        k, t = S[i][0], S[i][1]
        x = sl.get(k)
        if x is None or not _fin(x[2]) or abs(x[2] - KU.f_cmd(r, t0, t)) > KU.TOL or len(x[3]) != 2 or any(abs(y - x[2]) > KU.TOL for y in x[3]):
            bad0.append(k)
    if bad0:
        why.append(f'U0_set_log_{len(bad0)}')
    last = sl.get(k2 - 1)
    if last is None or any(abs(y - r['f_lo_n']) > KU.TOL for y in last[3]):
        why.append('U6_readback_not_f_lo_at_open')
    # U1–U3
    b0 = S[span[0]][I['base_xyth']]
    wc, wd, wb, wdeg = math.inf, 0.0, 0.0, 0.0
    n1 = n2 = n3 = 0
    for i in span:
        s = S[i]
        f = _fpair(s[I['leaf_contacts']])
        wc = min(wc, min(f)) if all(_fin(x) for x in f) else -math.inf
        n1 += int(not all(_fin(x) for x in f) or min(f) < r['contact_min_n'])
        v = _speeds(S, I, i, r)
        if v is not None:
            wd = max(wd, v[2])
            n2 += int(v[2] > r['door_rate_protect'])
        b = s[I['base_xyth']]
        dm = math.hypot(b[0] - b0[0], b[1] - b0[1])
        dg = abs(math.degrees(math.atan2(math.sin(b[2] - b0[2]), math.cos(b[2] - b0[2]))))
        wb, wdeg = max(wb, dm), max(wdeg, dg)
        n3 += int(dm > r['base_dev_m'] or dg > r['base_dev_deg'])
    if n1:
        why.append(f'U1_contact_lost_{n1}')
    if n2:
        why.append(f'U2_door_rate_{n2}')
    if n3:
        why.append(f'U3_base_dev_{n3}')
    st['unload_worst'] = {'contact_min_n': wc, 'door_rate': wd, 'base_dev_m': wb, 'base_dev_deg': wdeg}

    def quiet_over(t_end):
        idx = [i for i in span if t_end - r['quiet_s'] - 1e-9 <= S[i][1] <= t_end + 1e-9]
        if not idx or S[idx[0]][1] > t_end - r['quiet_s'] + r['dt'] + 1e-9 or S[idx[0]][1] < t0 + r['ramp_s'] - 1e-9:
            return False, 'window_not_covered_or_in_ramp'
        bad = [S[i][0] for i in idx if not _quiet(_speeds(S, I, i, r), r)]
        return not bad, (None if not bad else f'not_quiet_at_{bad[0]}')
    ok4, w4 = quiet_over(min(float(req2), S[span[-1]][1]))
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
    if solver_outcome is not None:
        tr = solver_outcome.get('transitions') or {}
        a, b = tr.get('UNLOAD'), tr.get('UNLOADED')
        if not (_fin(a) and _fin(b) and a <= b <= c2['t'] + 1e-9):
            why.append('U5_solver_unload_order')
    st['unload_times'] = {'t0': t0, 'unloaded_exec': tu, 'req2': req2, 'commit2': c2['t'],
                          'solver': None if solver_outcome is None else {k: (solver_outcome.get('transitions') or {}).get(k) for k in ('UNLOAD', 'UNLOADED')}}
    return why, st
