#!/usr/bin/env python3
"""W2c 整體判定器 r2（Codex 20261025_223632：三個誤收反例必修）。door_k3m_check_w2c.py（凍結 w2c r1）不改；本檔重用其 check_w2，
但判準參數一律取**固定登錄值**，並加趟別一致性與事件時間交叉核對。起跑前凍結。

一、固定登錄值（REG；待判資料不得改寫判準）
  - G 項以 REG 的 test_hold_s 3.0、commit_timeout_s 2.0 呼叫 w2c.check_w2（不讀 task 自報 params）。
  - 另核各端紀錄的設定＝REG（不符即不通過、缺即不通過）：
    task args（w2_req_delay_s 1.0、w2_commit_timeout_s 2.0、w2_test_hold_s 3.0、w2_state_stale_s 1.5）與 task.w2.params；
    執行端 room.w2.config（cand_timeout_s 1.0、cmd_max_age_s 0.2、post_commit_timeout_s 0.25、applied_timeout_s 0.25、dt 0.01）；
    求解端 args（w2_held True、w2_held_max_s 60）。
二、趟別與請求一致（RUN＝task args run_id；缺證據不通過）
  - task args w2_mode_run_id＝RUN；task.w2.req＝{run_id RUN, req_seq 1, PAUSE_HELD→RELEASE_PREP_TEST, mode_seq 1}；
  - room.w2.config：w2_mode_run_id＝run_id＝authority_run_id＝RUN；執行端登記的請求（runtime.requests[0].req）＝task 請求；轉送恰一次 req_seq 1；
  - 求解端 args run_id＝w2_mode_run_id＝RUN；求解端 w2 events 恰一筆 prepare（req_seq 1）與一筆 committed_seen（mode_seq 1）。
三、事件時間交叉核對（同一模擬時鐘；容差 CLOCK_TOL＝0.05 s＝50 Hz 同步狀態 2 筆間隔＋餘量；送達上限 DELIVER_MAX＝0.5 s）
  - 先後：OBS_PASSED − tol ≤ held_wait；requested − held_wait ≥ req_delay − tol；
    requested − tol ≤ 執行端登記（pending t）≤ requested＋DELIVER_MAX；登記 ≤ 執行端提交 ≤ 登記＋cand_timeout；
    執行端提交 − tol ≤ task committed ≤ 執行端提交＋state_stale；task done − task committed ≥ test_hold − 1e-6；done ≤ room 最後時刻。
  - trace 對齊：task trace 的 w2 狀態序只允許 none→held_wait→requested→committed→done（不跳、不回）；各狀態第一列時刻＝task.w2.t 自報值（≤ 1e-6）。
  - 不靠 task 自報證明保持：執行端逐步列自執行端提交步起，每步有效套用（src ≥ 0、chain_mode 0、mode_seq 1）連續涵蓋到
    執行端提交＋test_hold − tol 以後。

    python3 evaluation/door_k3m_check_w2c2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w2c2.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w1f2 as F2                       # noqa: E402
import door_k3m_check_w2c as W2C                       # noqa: E402

REG = {'test_hold_s': 3.0, 'commit_timeout_s': 2.0, 'req_delay_s': 1.0, 'state_stale_s': 1.5,
       'cand_timeout_s': 1.0, 'cmd_max_age_s': 0.2, 'post_commit_timeout_s': 0.25, 'applied_timeout_s': 0.25, 'dt': 0.01,
       'held_max_s': 60.0}
CLOCK_TOL = 0.05
DELIVER_MAX = 0.5
W2_ORDER = ['none', 'held_wait', 'requested', 'committed', 'done']


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _eq(a, b, tol=1e-9):
    return _fin(a) and _fin(b) and abs(a - b) <= tol


def check_reg_id_time(task, room, solver, why, st):
    task = task or {}
    ta = task.get('args') or {}
    tw = task.get('w2') or {}
    rw = (room or {}).get('w2') or {}
    rc = rw.get('config') or {}
    sa = (solver or {}).get('args') or {}
    ss = ((solver or {}).get('stats') or {}).get('w2') or {}
    RUN = ta.get('run_id')
    # 一、登錄值
    reg_bad = []
    for k_arg, k_reg in (('w2_req_delay_s', 'req_delay_s'), ('w2_commit_timeout_s', 'commit_timeout_s'),
                         ('w2_test_hold_s', 'test_hold_s'), ('w2_state_stale_s', 'state_stale_s')):
        if not _eq(ta.get(k_arg), REG[k_reg]):
            reg_bad.append(f'task_args.{k_arg}')
        if not _eq((tw.get('params') or {}).get(k_reg), REG[k_reg]):
            reg_bad.append(f'task_w2_params.{k_reg}')
    for k in ('cand_timeout_s', 'cmd_max_age_s', 'post_commit_timeout_s', 'applied_timeout_s', 'dt'):
        if not _eq(rc.get(k), REG[k]):
            reg_bad.append(f'room_w2_config.{k}')
    if sa.get('w2_held') is not True or not _eq(sa.get('w2_held_max_s'), REG['held_max_s']):
        reg_bad.append('solver_args.w2_held')
    if reg_bad:
        why.append('w2_registration_mismatch:' + ','.join(reg_bad[:6]))
    # 二、趟別與請求
    id_bad = []
    want_req = {'run_id': RUN, 'req_seq': 1, 'from_mode': 'PAUSE_HELD', 'to_mode': 'RELEASE_PREP_TEST', 'mode_seq': 1}
    if not isinstance(RUN, str) or not RUN:
        id_bad.append('task_run_id_missing')
    if ta.get('w2_mode_run_id') != RUN:
        id_bad.append('task_w2_mode_run_id')
    if tw.get('req') != want_req:
        id_bad.append('task_request')
    if not (rc.get('w2_mode_run_id') == rc.get('run_id') == rc.get('authority_run_id') == RUN):
        id_bad.append('room_config_run_id')
    rt = rw.get('runtime') or {}
    reqs = rt.get('requests') or []
    if len(reqs) != 1 or reqs[0].get('req') != want_req or reqs[0].get('result') != 'pending':
        id_bad.append('exec_registered_request')
    rel = rt.get('relayed') or []
    if len(rel) != 1 or rel[0].get('req_seq') != 1:
        id_bad.append('exec_relay')
    if not (sa.get('run_id') == sa.get('w2_mode_run_id') == RUN):
        id_bad.append('solver_run_id')
    sev = ss.get('events') or []
    prep = [e for e in sev if e.get('kind') == 'prepare']
    seen = [e for e in sev if e.get('kind') == 'committed_seen']
    if len(prep) != 1 or prep[0].get('req_seq') != 1 or len(seen) != 1 or seen[0].get('mode_seq') != 1:
        id_bad.append('solver_prepare_commit_events')
    if id_bad:
        why.append('w2_identity_mismatch:' + ','.join(id_bad[:6]))
    # 三、時間
    tm_bad = []
    tt = tw.get('t') or {}
    hw, rq, cm, dn = (tt.get(k) for k in ('held_wait', 'requested', 'committed', 'done'))
    o = (((solver or {}).get('stats') or {}).get('k3m_pause') or {}).get('supervisor_outcome') or {}
    obs = (o.get('transitions') or {}).get('OBS_PASSED')
    auth = rw.get('auth') or {}
    pend = [e for e in (auth.get('events') or []) if e.get('kind') == 'pending']
    c = auth.get('commit') or {}
    p_t, c_t, c_k = (pend[0].get('t') if len(pend) == 1 else None), c.get('t'), c.get('step')
    S_room = (room or {}).get('steps') or []
    I = {cc: i for i, cc in enumerate((room or {}).get('steps_cols') or [])}
    t_room_end = S_room[-1][I['sim_t']] if (S_room and 'sim_t' in I) else None
    if not all(_fin(x) for x in (hw, rq, cm, dn, obs, p_t, c_t, t_room_end)):
        tm_bad.append('times_missing')
    else:
        checks = {'held_wait_before_obs': hw >= obs - CLOCK_TOL,
                  'request_before_delay': rq - hw >= REG['req_delay_s'] - CLOCK_TOL,
                  'exec_register_window': rq - CLOCK_TOL <= p_t <= rq + DELIVER_MAX,
                  'exec_commit_window': p_t <= c_t <= p_t + REG['cand_timeout_s'] + 1e-9,
                  'task_commit_window': c_t - CLOCK_TOL <= cm <= c_t + REG['state_stale_s'],
                  'task_hold_short': dn - cm >= REG['test_hold_s'] - 1e-6,
                  'done_after_room_end': dn <= t_room_end + 1e-9}
        tm_bad += [k for k, v in checks.items() if not v]
    tr = task.get('trace') or []
    seq = [x.get('w2') for x in tr]
    if not seq or any(s not in W2_ORDER for s in seq) or any(W2_ORDER.index(b) - W2_ORDER.index(a) not in (0, 1) for a, b in zip(seq, seq[1:])) \
            or seq[-1] != 'done':
        tm_bad.append('trace_w2_sequence')
    else:
        first = {}
        for x in tr:
            first.setdefault(x['w2'], x.get('t'))
        for k, v in (('held_wait', hw), ('requested', rq), ('committed', cm), ('done', dn)):
            if not _eq(first.get(k), v, 1e-6):
                tm_bad.append(f'trace_first_{k}_ne_reported')
    rows = rw.get('rows') or []
    if isinstance(c_k, int) and _fin(c_t):
        run_end = None
        for r in rows:
            if r.get('step', -1) < c_k:
                continue
            if isinstance(r.get('src'), int) and r['src'] >= 0 and r.get('chain_mode') == 0 and r.get('mode_seq') == 1:
                run_end = r.get('t')
            else:
                break
        if not _fin(run_end) or run_end < c_t + REG['test_hold_s'] - CLOCK_TOL:
            tm_bad.append('exec_hold_evidence_short')
        st['w2_exec_hold_until'] = run_end
    else:
        tm_bad.append('exec_commit_missing')
    if tm_bad:
        why.append('w2_time_crosscheck:' + ','.join(tm_bad[:8]))
    st['w2_crosscheck'] = {'reg_bad': reg_bad, 'id_bad': id_bad, 'time_bad': tm_bad, 'run_id': RUN,
                           'times': {'obs': obs, 'held_wait': hw, 'requested': rq, 'exec_pending': p_t, 'exec_commit': c_t,
                                     'task_committed': cm, 'done': dn, 'room_end': t_room_end},
                           'clock_tol_s': CLOCK_TOL, 'deliver_max_s': DELIVER_MAX, 'reg': REG}
    return not (reg_bad or id_bad or tm_bad)


def judge(task, room, gate, solver, observer):
    res = F2.judge(task, room, gate, solver, observer)
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if w in W2C.W2_REPLACED]
    why_old = [w for w in why_all if w not in W2C.W2_REPLACED]
    st = dict(res.get('stats') or {})
    why_w2, flags = W2C.check_w2(task, room, solver, st, test_hold_s=REG['test_hold_s'], commit_timeout_s=REG['commit_timeout_s'])
    why_x = []
    x_ok = check_reg_id_time(task, room, solver, why_x, st)
    why = why_old + why_w2 + why_x
    flags = dict(flags, w2_test_complete=bool(flags.get('w2_test_complete')) and x_ok)
    return dict(res, why=why, stats=st, judge='w2c2', replaced_old_reasons=replaced, w2_replaced_names=list(W2C.W2_REPLACED),
                old_pass_excluding_replaced=not why_old, w2_crosscheck_ok=x_ok, **flags, **{'pass': not why})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_w2c2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3m_pass', 'settle_p2_ok',
                                             'pause_command_stop', 'pause_physical_stop', 'finger_effort_ok', 'trace_log_ok',
                                             'w2_obs_passed', 'w2_held', 'w2_commit_ok', 'w2_crosscheck_ok', 'w2_test_complete', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
