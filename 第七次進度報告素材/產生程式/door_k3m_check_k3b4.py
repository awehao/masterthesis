#!/usr/bin/env python3
"""K3-B 整體判定 r4（閉爪卸載候選；Codex 20261026_025647 必修 D）。door_k3m_check_k3b3 不改；本檔版本化組合並**明列替換項**。

保留（原樣呼叫）：既有開門／停止／保護／紀錄鏈（door_k3m_check_w1f2.judge，只取代兩項舊 DONE 理由 K3B_REPLACED）；
  K3-B：check_multi（三段提交）、K0 room 覆蓋、K1–K3 執行端解除、K5' 逐步幾何與門角帶、K10 退出重算、K11 追蹤落後、K12 一般淨空、
  K13 三維終點保持（k3b3）、K6–K7 底盤與接觸、K9 求解端交接、錨點紀錄。
替換（REPLACED_CHECKS）：
  - K4'（C_CLEAR 後 20→5 N 恢復動作與其後讀回）→ E1–E6 分段證據：
      E1 暫停切換 5→20 N（W1e 紀錄 ok、讀回 20）；
      E2 暫停切換至卸載起點：定期讀回兩指＝20、ok、非手指上限不變、相鄰間隔 ≤ 1.1 s；
      E3 斜坡：執行端 unload_ctrl 每筆 ok、設定介於 5–20 且隨時間不增；斜坡逐步設定＝登錄斜坡由 U0（check_unload2）核；
      E4 卸載起點至 t_settling 的定期讀回＝當時斜坡值；t_settling 至紀錄末：定期讀回兩指＝5、ok、間隔 ≤ 1.1 s、涵蓋到紀錄末前 1.1 s 內；
      E5 不存在 5 N 恢復動作：W1e restore 紀錄為空、5 N 權威無 pending／done（「恢復 5 N」改為讀回確認，不偽造恢復）；
      E6 全部定期讀回 others_same True（只改兩指）。
  - K8（eff_done 讀回）→ K8b：任務端監看序同前；eff_req 為 None；eff_verify 讀回 5、讀回時刻 ≤ 核對時刻、核對時刻 ≥ C_CLEAR；
      unload_gate 為 go；mode 3 請求時求解端子狀態 release_settled；DONE。
  - X1／X3（登錄與時間交叉）→ X1b／X3b：同 k3b2，另核求解端 --w2-hold-r8／--w2-unload、任務端 --w2-unload、卸載登錄；
      X3 的 effort_times（5 N 請求／切換時間）改為：mode 2 請求 ≥ 執行端 UNLOADED 與求解端 UNLOADED、eff_verify ∈ [C_CLEAR, COMPLETE]。
新增：
  - U：door_k3m_check_unload2.check_unload2（U0–U8）。
  - RT 釋放瞬態重算：mode 2 提交起至求解端 RELEASE_SETTLED，room 逐步 FK 的 TCP 位移 ≤ 5 mm／2°、0.1 s 窗速度 ≤ 0.02 m/s；
      RELEASE_SETTLED − t_open ≤ 1.5 s；RELEASE_SETTLED 至 mode 3 提交手臂 0.5 s 窗 ≤ 0.001 rad/s。
  - S8 求解端：監督器 r8；子狀態轉移序 UNLOAD ≤ UNLOADED ≤ RELEASE_OPEN ≤ RELEASE_SETTLED ≤ HANDOVER；兩段瞬態 CONVERGED；t_settling 有限。
所有新項：缺欄或非法 ⇒ 回理由，不以例外結束。

    python3 evaluation/door_k3m_check_k3b4.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_k3b4.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_release4 as KR                         # noqa: E402
import door_k3b_transient as TT                        # noqa: E402
import door_k3b_unload2 as KU                          # noqa: E402
import door_k3m_check_k3b as J1                        # noqa: E402
import door_k3m_check_k3b2 as J2                       # noqa: E402
import door_k3m_check_k3b3 as J3                       # noqa: E402
import door_k3m_check_unload2 as CU2                   # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402
import door_k3m_check_w2c2 as W2C2                     # noqa: E402

VERSION = 'door_k3m_check_k3b4/1'
R = KR.REG
K3B_REPLACED = J1.K3B_REPLACED
REPLACED_CHECKS = {'K4_prime(check_effort2)': 'E1–E6 check_effort_segments', 'K8(check_task)': 'K8b check_task4',
                   'X1/X3(check_reg_id_time)': 'X1b/X3b check_reg_id_time4'}
_fin = KU._fin
TOL = KU.TOL


def _guard(fn, tag, why, *a):
    try:
        return fn(*a)
    except Exception as e:                                                       # noqa: BLE001
        why.append(f'{tag}_internal_error:{type(e).__name__}')
        return None


def _rb_eq(p, v):
    rb = p.get('readback') if isinstance(p, dict) else None
    return isinstance(rb, list) and len(rb) == 2 and all(_fin(x) and abs(x - v) <= TOL for x in rb)


def check_effort_segments(room, why, st):
    fe = (room or {}).get('finger_effort') or {}
    pz = fe.get('pause') or {}
    rw = (room or {}).get('w2') or {}
    u = rw.get('unload') or {}
    t0, ts = u.get('t0'), u.get('t_settling')
    S, I = J1._room(room)
    t_end = S[-1][I['sim_t']] if S else None
    if not (pz.get('ok') is True and _rb_eq(pz, 20.0) and isinstance(pz.get('old'), list) and all(_fin(x) and abs(x - 5.0) <= TOL for x in pz['old'])):
        why.append('E1_pause_switch_not_5_to_20')
    t_sw = pz.get('t')
    per = pz.get('periodic')
    if not isinstance(per, list) or not all(isinstance(p, dict) and _fin(p.get('t')) for p in per) or not (_fin(t_sw) and _fin(t0) and _fin(ts) and _fin(t_end)):
        why.append('E0_effort_record_missing')
        return
    ts_ = [p['t'] for p in per]
    if any(b <= a for a, b in zip(ts_, ts_[1:])):
        why.append('E0_periodic_not_increasing')
    seg2 = [p for p in per if t_sw < p['t'] < t0]
    seg_r = [p for p in per if t0 <= p['t'] < ts]
    seg4 = [p for p in per if p['t'] >= ts]
    if any(p.get('ok') is not True or not _rb_eq(p, 20.0) for p in seg2):
        why.append('E2_readback_not_20_before_unload')

    def gaps(seq, a, b):
        tt = [a] + [p['t'] for p in seq] + [b]
        return max((y - x for x, y in zip(tt, tt[1:])), default=0.0)
    if gaps(seg2, t_sw, t0) > 1.1 + 1e-6:
        why.append('E2_periodic_gap')
    uc = rw.get('unload_ctrl')
    if not isinstance(uc, list) or not uc or any(not isinstance(x, dict) or x.get('ok') is not True or not _fin(x.get('set'))
                                                   or not (5.0 - TOL <= x['set'] <= 20.0 + TOL) for x in uc):
        why.append('E3_unload_ctrl_missing_or_failed')
    else:
        sv = [x['set'] for x in sorted(uc, key=lambda x: x.get('t', 0.0))]
        if any(b > a + TOL for a, b in zip(sv, sv[1:])):
            why.append('E3_ramp_not_nonincreasing')
    if any(p.get('ok') is not True or not _rb_eq(p, KU.f_cmd(KU.REG_UNLOAD, t0, p['t'])) for p in seg_r):
        why.append('E4_readback_not_ramp_during_unload')
    if any(p.get('ok') is not True or not _rb_eq(p, 5.0) for p in seg4):
        why.append('E4_readback_not_5_after_settling')
    if not seg4 or gaps(seg4, ts, t_end) > 1.1 + 1e-6:
        why.append('E4_periodic_coverage_after_settling')
    eff = rw.get('effort') or {}
    evs = eff.get('events') or []
    if pz.get('restore') is not None or eff.get('restore') is not None or any(isinstance(e, dict) and ('pending' in e or 'done' in e) for e in evs):
        why.append('E5_restore_action_present')
    if any(p.get('others_same') is not True for p in per):
        why.append('E6_non_finger_max_effort_changed')
    st['k3b4_effort'] = {'t_switch': t_sw, 't0': t0, 't_settling': ts, 'n_before': len(seg2), 'n_ramp': len(seg_r), 'n_after': len(seg4),
                         'n_ctrl': len(uc) if isinstance(uc, list) else None}


def check_task4(task, why, st):
    k = (task or {}).get('k3b') or {}
    mon, rel = k.get('monitor') or {}, k.get('rel') or {}
    seq = [e.get('to') for e in (mon.get('events') or []) if isinstance(e, dict) and e.get('to') in ('ARMED', 'NEAR', 'C_CLEAR', 'RETREAT_DONE', 'COMPLETE', 'FAILED')]
    if seq != ['ARMED', 'NEAR', 'C_CLEAR', 'RETREAT_DONE', 'COMPLETE'] or mon.get('why') is not None:
        why.append('K8b_monitor_sequence')
    tt = mon.get('t') or {}
    if not (_fin(tt.get('COMPLETE')) and _fin(tt.get('RETREAT_DONE')) and tt['COMPLETE'] - tt['RETREAT_DONE'] >= R['complete_hold_s'] - 1e-6):
        why.append('K8b_complete_hold')
    ev = next((e for e in (mon.get('events') or []) if isinstance(e, dict) and e.get('to') == 'C_CLEAR'), {})
    if not (_fin(ev.get('s_meas')) and ev['s_meas'] >= R['near_s_m'] and _fin(ev.get('pair')) and ev['pair'] >= R['pair_exit_m']):
        why.append('K8b_clear_evidence')
    ve = rel.get('eff_verify') or {}
    tc = tt.get('C_CLEAR')
    if not (ve.get('ok') is True and _rb_eq(ve, 5.0) and _fin(ve.get('t')) and _fin(ve.get('readback_t')) and ve['readback_t'] <= ve['t'] + 1e-9
            and _fin(tc) and ve['t'] >= tc - 1e-9):
        why.append('K8b_effort_verify')
    if rel.get('eff_req') is not None or rel.get('eff_done') is not None:
        why.append('K8b_restore_request_present')
    ug = rel.get('unload_gate') or {}
    if ug.get('gate') != 'go' or not (_fin(ug.get('state_t')) and _fin(ug.get('solver_t'))):
        why.append('K8b_unload_gate')
    if rel.get('solver_sub_at_req3') != 'release_settled':
        why.append('K8b_mode3_before_release_settled')
    if rel.get('state') != 'done' or (task or {}).get('final_phase') != 'DONE':
        why.append('K8b_task_not_done')
    st['k3b4_task'] = {'seq': seq, 't': tt, 'eff_verify': ve, 'unload_gate': ug, 'solver_sub_at_req3': rel.get('solver_sub_at_req3')}


def check_solver4(solver, why, st):
    s = (solver or {}).get('stats') or {}
    o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
    if o.get('version') != 'door_w2_mode8/r8':
        why.append('S8_supervisor_version')
        return o
    tr = o.get('transitions') or {}
    ks = ('UNLOAD', 'UNLOADED', 'RELEASE_OPEN', 'RELEASE_SETTLED', 'HANDOVER')
    if not all(_fin(tr.get(k)) for k in ks) or any(tr[a] > tr[b] + 1e-9 for a, b in zip(ks, ks[1:])):
        why.append('S8_transition_order')
    tm = o.get('transient') or {}
    if any((tm.get(k) or {}).get('state') != 'CONVERGED' for k in ('1', '2')):
        why.append('S8_transient_not_converged')
    if not _fin(o.get('t_settling')):
        why.append('S8_t_settling_missing')
    st['k3b4_solver'] = {'transitions': {k: tr.get(k) for k in ks}, 't_settling': o.get('t_settling'),
                         'transient': {k: (tm.get(k) or {}).get('worst') for k in ('1', '2')}}
    return o


def check_release_transient(room, o, cs, why, st):
    import door_k2_check as KC
    S, I = J1._room(room)
    tr = (o or {}).get('transitions') or {}
    t_set, k2 = tr.get('RELEASE_SETTLED'), cs[2]['step']
    by = {s[I['step']]: s for s in S}
    if not _fin(t_set) or (k2 - 1) not in by:
        why.append('RT_rows_or_settled_missing')
        return
    r = TT.REG_TRANSIENT
    s0 = by[k2 - 1]
    T0 = KC._tcp(s0[I['base_xyth']], s0[I['q_arm_meas']])
    rows = [s for s in S if k2 <= s[I['step']] and s[I['sim_t']] <= t_set + 1e-9]
    worst = {'disp_m': 0.0, 'rot_rad': 0.0, 'v_m_s': 0.0}
    hist = []
    for s in rows:
        T = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        p = T[:3, 3]
        Rm = T[:3, :3] @ T0[:3, :3].T
        worst['disp_m'] = max(worst['disp_m'], float(np.linalg.norm(p - T0[:3, 3])))
        worst['rot_rad'] = max(worst['rot_rad'], math.acos(max(-1.0, min(1.0, (float(np.trace(Rm)) - 1) / 2))))
        hist.append((s[I['sim_t']], p))
        a = next((h for h in reversed(hist[:-1]) if s[I['sim_t']] - h[0] >= r['win_s'] - 1e-9), None)
        if a is not None:
            worst['v_m_s'] = max(worst['v_m_s'], float(np.linalg.norm(p - a[1])) / (s[I['sim_t']] - a[0]))
        while len(hist) > 2 and s[I['sim_t']] - hist[0][0] > r['win_s'] + 0.05:
            hist.pop(0)
    if worst['disp_m'] > r['disp_max_m'] or worst['rot_rad'] > r['rot_max_rad'] or worst['v_m_s'] > r['v_max_m_s']:
        why.append('RT_release_transient_bounds')
    if t_set - cs[2]['t'] > TT.T_MAX_RELEASE + 1e-9:
        why.append('RT_release_settled_late')
    t3 = cs[3]['t']
    post = [s for s in S if t_set <= s[I['sim_t']] <= t3 + 1e-9]
    bad = 0
    for s in post:
        a = next((x for x in S if x[I['sim_t']] <= s[I['sim_t']] - r['arm_win_s'] + 1e-9 and x[I['sim_t']] >= s[I['sim_t']] - r['arm_win_s'] - 0.02), None)
        if a is not None:
            v = max(abs(x - y) for x, y in zip(s[I['q_arm_meas']], a[I['q_arm_meas']])) / (s[I['sim_t']] - a[I['sim_t']])
            bad += int(v > r['quiet_arm'] + 1e-9)
    if bad:
        why.append(f'RT_arm_not_quiet_before_retreat_{bad}')
    st['k3b4_release_transient'] = dict(worst, t_settled_minus_open=t_set - cs[2]['t'], n_post=len(post))


def check_reg_id_time4(task, room, solver, cs, why, st):
    """J2.check_reg_id_time 的版本化複本（X1b／X3b）。"""
    ta = (task or {}).get('args') or {}
    k = (task or {}).get('k3b') or {}
    mon, rel = k.get('monitor') or {}, k.get('rel') or {}
    rw = (room or {}).get('w2') or {}
    rc = rw.get('config') or {}
    sa = (solver or {}).get('args') or {}
    RUN = ta.get('run_id')
    reg_bad = []
    for name, reg in (('task_monitor', mon.get('reg')), ('exec_release', (rw.get('release') or {}).get('reg'))):
        if not isinstance(reg, dict) or any(not (_fin(reg.get(kk)) and abs(reg[kk] - v) < 1e-9) for kk, v in R.items() if isinstance(v, (int, float))):
            reg_bad.append(name)
    ur = (rw.get('unload') or {}).get('reg')
    if not isinstance(ur, dict) or any(ur.get(kk) != v for kk, v in KU.REG_UNLOAD.items()):
        reg_bad.append('exec_unload')
    for kk in ('cand_timeout_s', 'cmd_max_age_s', 'post_commit_timeout_s', 'applied_timeout_s', 'dt'):
        if not (_fin(rc.get(kk)) and abs(rc[kk] - W2C2.REG[kk]) < 1e-9):
            reg_bad.append(f'room_config.{kk}')
    if sa.get('w2_retreat') is not True or sa.get('w2_held') is not True or not (_fin(sa.get('w2_held_max_s')) and abs(sa['w2_held_max_s'] - 60.0) < 1e-9) \
            or sa.get('w2_hold_r8') is not True or sa.get('w2_unload') is not True:
        reg_bad.append('solver_args')
    if ta.get('w2_release') is not True or ta.get('w2_unload') is not True:
        reg_bad.append('task_args')
    if reg_bad:
        why.append('X1b_registration:' + ','.join(reg_bad[:6]))
    id_bad = []
    if not isinstance(RUN, str) or ta.get('w2_mode_run_id') != RUN:
        id_bad.append('task_run_id')
    if not (rc.get('w2_mode_run_id') == rc.get('run_id') == rc.get('authority_run_id') == RUN):
        id_bad.append('room_config_run_id')
    if not (sa.get('run_id') == sa.get('w2_mode_run_id') == RUN):
        id_bad.append('solver_run_id')
    rt = rw.get('runtime') or {}
    reqs, rlay = rt.get('requests') or [], rt.get('relayed') or []
    names = ['PAUSE_HELD', 'RELEASE_PREP_TEST', 'RELEASE_OPEN', 'RETREAT']
    want = [{'run_id': RUN, 'req_seq': i, 'from_mode': names[i - 1], 'to_mode': names[i], 'mode_seq': i} for i in (1, 2, 3)]
    if [r.get('req') for r in reqs] != want or any(r.get('result') != 'pending' for r in reqs):
        id_bad.append('exec_requests')
    if [r.get('req_seq') for r in rlay] != [1, 2, 3]:
        id_bad.append('exec_relay')
    if id_bad:
        why.append('X2_identity:' + ','.join(id_bad[:6]))
    tm_bad = []
    pend = {e.get('mode_seq'): e.get('t') for e in ((rw.get('auth') or {}).get('events') or []) if isinstance(e, dict) and e.get('kind') == 'pending'}
    w2t, rt_ = ((task or {}).get('w2') or {}).get('t') or {}, rel.get('t') or {}
    t_req = {1: w2t.get('requested'), 2: rt_.get('req_open'), 3: rt_.get('req_retreat')}
    t_seen = {1: w2t.get('committed'), 2: rt_.get('open_seen'), 3: rt_.get('retreat')}
    for m in (1, 2, 3):
        trq, tp, tc = t_req[m], pend.get(m), cs[m]['t']
        if not all(_fin(x) for x in (trq, tp, tc)):
            tm_bad.append(f'mode{m}_times_missing')
            continue
        if not (trq - J2.CLOCK_TOL <= tp <= trq + J2.DELIVER_MAX):
            tm_bad.append(f'mode{m}_register_window')
        if not (tp <= tc <= tp + W2C2.REG['cand_timeout_s'] + 1e-9):
            tm_bad.append(f'mode{m}_commit_window')
        ts = t_seen[m]
        if m != 2 and (not _fin(ts) or not (tc - J2.CLOCK_TOL <= ts <= tc + W2C2.REG['state_stale_s'])):
            tm_bad.append(f'mode{m}_task_seen_window')
    if not (_fin(rt_.get('open')) and abs(rt_['open'] - cs[2]['t']) <= J2.CLOCK_TOL):
        tm_bad.append('t_open_task_ne_exec')
    u = rw.get('unload') or {}
    o = (((solver or {}).get('stats') or {}).get('k3m_pause') or {}).get('supervisor_outcome') or {}
    tu_x, tu_s = (u.get('t') or {}).get('UNLOADED'), (o.get('transitions') or {}).get('UNLOADED')
    if not all(_fin(x) for x in (t_req[2], tu_x, tu_s)) or t_req[2] < max(tu_x, tu_s) - 1e-9:
        tm_bad.append('mode2_request_before_unloaded')
    tcl, tco, ve = (mon.get('t') or {}).get('C_CLEAR'), (mon.get('t') or {}).get('COMPLETE'), (rel.get('eff_verify') or {}).get('t')
    if not all(_fin(x) for x in (tcl, tco, ve)) or not (tcl - 1e-9 <= ve <= tco + 1e-9):
        tm_bad.append('effort_verify_times')
    S, I = J1._room(room)
    t_end = S[-1][I['sim_t']] if S else None
    if not (_fin(tco) and _fin(t_end) and tco <= t_end + 1e-9 and _fin(rt_.get('done')) and rt_['done'] >= tco - 1e-9):
        tm_bad.append('complete_times')
    if tm_bad:
        why.append('X3b_time:' + ','.join(tm_bad[:8]))
    st['k3b4_cross'] = {'reg_bad': reg_bad, 'id_bad': id_bad, 'time_bad': tm_bad, 'run_id': RUN, 'pending': pend, 't_req': t_req}


def check_k3b4(task, room, gate, solver, st):
    why = []
    rw = (room or {}).get('w2') or {}
    cs = _guard(J1.check_multi, 'K_multi', why, rw, why)
    if cs is None:
        return why
    _guard(J2.check_room_coverage, 'K0', why, room, cs, why, st)
    _guard(J1.check_exec_release, 'K1_K3', why, room, rw, cs, why, st)
    _guard(check_effort_segments, 'E', why, room, why, st)
    g = _guard(J2.check_geom_steps, 'K5', why, room, cs, why, st)
    _guard(J2.check_exit_recompute, 'K10', why, room, task, g, why, st)
    _guard(J2.check_tracking, 'K11', why, task, g, why, st)
    _guard(J2.check_general_clearance, 'K12', why, room, cs, why, st)
    _guard(J3.check_end_hold_3d, 'K13', why, room, task, g, why, st)
    _guard(J1.check_base_contacts, 'K6_K7', why, room, gate, cs, why, st)
    _guard(check_task4, 'K8b', why, task, why, st)
    _guard(J1.check_solver, 'K9', why, solver, why, st)
    o = _guard(check_solver4, 'S8', why, solver, why, st)
    _guard(check_release_transient, 'RT', why, room, o, cs, why, st)
    _guard(check_reg_id_time4, 'X', why, task, room, solver, cs, why, st)
    wu, su = CU2.check_unload2(room, task, o)
    why += wu
    st['k3b4_unload'] = su
    _guard(J2.anchors, 'anchors', why, room, gate, solver, cs, st)
    return why


def judge(task, room, gate, solver, observer):
    res = F2.judge(task, room, gate, solver, observer)
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if w in K3B_REPLACED]
    why_old = [w for w in why_all if w not in K3B_REPLACED]
    st = dict(res.get('stats') or {})
    why_k = check_k3b4(task, room, gate, solver, st)
    why = why_old + why_k
    return dict(res, why=why, stats=st, judge='k3b4', judge_version=VERSION, replaced_old_reasons=replaced, k3b_replaced_names=list(K3B_REPLACED),
                replaced_checks=REPLACED_CHECKS, old_pass_excluding_replaced=not why_old, k3b_ok=not why_k, **{'pass': not why},
                semantics='DONE＝閉爪卸載至 5 N（讀回確認）、解除、退出近區、5 N 讀回核對、退離終點並三維收斂保持 ≥ 3 s；不是收臂')


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_k3b4.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3b_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
