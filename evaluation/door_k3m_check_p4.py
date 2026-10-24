#!/usr/bin/env python3
"""K3-M 受控暫停 p4 判定器：**主動保持停止**（Howard 2026-10-24 選定；規格 K3M_pause_p4_active_hold_stop.md）。
door_k3m_check_p.py／_p2.py／_p2r2.py 不改。結構＝door_k3m_check_p2r2：K3-M 判準＋SETTLE p2（S1／S2）照用，暫停核對換成 check_pause_p4：
  P1 握手：同 r3，凍結目標旋轉差用 r2 穩定算法（門檻 1e-9 不變）。
  P2 命令停止（room 獨立重算）：零速段第一筆序號 zero_seq0 起，連續 5 個物理步「來源序號 ≥ zero_seq0、手臂套用 ≤ 1e-6、
     底盤命令在主動保持界內（線速度向量 ≤ 5 mm/s、|ω| ≤ 0.02 rad/s）」⇒ 獨立命令停止時刻；之後每步同條件（否則 pause_old_or_nonzero_after_stop_N）。
     求解端 STOPPED 到 DONE 期間**必須持續發布**保持命令（每列 pause_zero、手臂 0、底盤在保持界內；相鄰間隔 ≤ 0.2 s），DONE 後不得再發布。
  P3 實際停止：**不變**（自 ZERO_PUB 起到命令停止＋10 s，10 步窗口：底盤 1 mm/s、0.005 rad/s、門 0.002、手臂 1e-3）。
  P4 一致性：不變。
報告用語：「主動保持下的停止」，不得寫成「命令歸零即停住」。

    python3 evaluation/door_k3m_check_p4.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_p4.json；不覆寫
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                          # noqa: E402
import door_k3m_check_p2 as C2                         # noqa: E402
import door_k3m_check_p2r2 as R2                       # noqa: E402
import door_k3m_pause_mode as PMODE                    # noqa: E402,F401
import door_k3m_pause_sup_p4 as P4                     # noqa: E402

TH, WIN, N_CONFIRM, TOL_APPLIED, OBS_S = CP.TH, CP.WIN, CP.N_CONFIRM, CP.TOL_APPLIED, CP.OBS_S
_num, room_rows, window_speeds, low, _T16 = CP._num, CP.room_rows, CP.window_speeds, CP.low, CP._T16


def check_pause_p4(task, room, solver, why, st):
    """p4 主動保持停止版（door_k3m_check_p.check_pause r3 的複本，差異見檔頭）。回傳 (cmd_ok, phys_ok)。"""
    kp_t = task.get('k3m_pause') or {}
    kp_s = (solver.get('stats') or {}).get('k3m_pause') or {}
    rk = room.get('k3m') or {}
    req, ea, sa = kp_t.get('request'), rk.get('pause_switch'), (kp_s.get('record') or {}).get('ack')
    # ---- P1 握手 ----
    if not (isinstance(req, dict) and isinstance(ea, dict) and isinstance(sa, dict)):
        why.append('pause_evidence_missing')
        return False, False
    rid = ((task.get('args') or {}).get('run_id'))
    rid_room = ((rk.get('config') or {}).get('run_id'))
    if not rid or any(x != rid for x in (req.get('run_id'), ea.get('run_id'), sa.get('run_id'), rid_room)):
        why.append('pause_run_id_mismatch')
    if req.get('seq') != 2 or req.get('mode') != 'pause' or ea.get('seq') != 2 or sa.get('seq') != 2:
        why.append('pause_seq_or_mode')
    if sa.get('published') is not True:
        why.append('pause_solver_ack_not_published')
    if not (ea.get('switch_step') == sa.get('switch_step') and _num(ea.get('switch_t')) and _num(sa.get('switch_t'))
            and abs(ea['switch_t'] - sa['switch_t']) <= 1e-9):
        why.append('pause_ack_mismatch')
    T_req, T_sa = _T16(req.get('frozen_target_T')), _T16(sa.get('frozen_target_T'))
    trace = task.get('trace') or []
    i_pause = next((i for i, e in enumerate(trace) if e.get('phase') == 'PAUSE'), None)   # r3：請求所在的那一輪（轉入 PAUSE 的紀錄邊界）
    if i_pause is None:
        why.append('pause_phase_missing_in_trace')
        prior = []
    else:                                                                  # 依實際發布順序：該輪之前的已發布目標（同輪請求後重發的不算）
        prior = [e for e in trace[:i_pause] if e.get('target') is not None and isinstance(e.get('pub_seq'), int)]
    last = prior[-1] if prior else None
    T_last = None if last is None else _T16(last['target'])
    if (T_req is None or T_last is None or last.get('pub_seq') != req.get('frozen_target_seq')
            or (lambda d: d[0] > 1e-9 or d[1] > 1e-9)(R2.target_diff_stable(T_last, T_req))):          # r2：穩定旋轉差
        why.append('pause_frozen_target_not_last_published')
    if T_req is None or T_sa is None or (lambda d: d[0] > 1e-6 or d[1] > 1e-6)(PMODE.target_diff(T_sa, T_req)):
        why.append('pause_solver_frozen_target_mismatch')
    ok_t = all(_num(x) for x in (req.get('req_t'), ea.get('switch_t'), sa.get('first_pause_sim_t'), sa.get('publish_sim_t')))
    if not ok_t or not (req['req_t'] <= ea['switch_t'] + 1e-9 <= sa['first_pause_sim_t'] + 2e-9 <= sa['publish_sim_t'] + 3e-9):
        why.append('pause_time_order')
    te, ts_, we, ws_, wr = (kp_t.get(k) for k in ('t_exec_recv', 't_solver_recv', 'wall_exec_recv', 'wall_solver_recv', 'wall_req'))
    if not all(_num(x) for x in (te, ts_, we, ws_, wr)) or max(te, ts_) > req['req_t'] + 1.0 + 1e-9 or max(we, ws_) > wr + 2.0 + 1e-9:
        why.append('pause_ack_deadline')
    if kp_t.get('result') != 'confirmed':
        why.append('pause_not_confirmed')
    # ---- P2 命令停止 ----
    n_why_p1 = len(why)
    S, I = room_rows(room)
    if S is None:
        why.append('room_applied_src_missing')
        return False, False
    log = solver.get('log') or []
    zl = [x for x in log if x.get('published') and x.get('pause_zero')]
    if not zl or not isinstance(zl[0].get('pause_source_seq'), int):
        why.append('pause_zero_publication_missing')
        return False, False
    zero_seq0 = zl[0]['pause_source_seq']
    i_cmd, run = None, 0
    prev_step = None
    for i, s in enumerate(S):
        a = s[I['applied_src']]
        ok = (a is not None and isinstance(a[2], int) and a[2] >= zero_seq0
              and P4.hold_cmd_ok(s[I['base_cmd_body']], s[I['applied_arm']], TOL_APPLIED))            # p4：手臂 0＋底盤在保持界內
        consec = prev_step is not None and int(s[I['step']]) == prev_step + 1
        run = (run + 1 if consec else 1) if ok else 0
        prev_step = int(s[I['step']])
        if run >= N_CONFIRM:
            i_cmd = i
            break
    if i_cmd is None:
        why.append('pause_command_stop_not_confirmed')
        return False, False
    t_cmd = S[i_cmd][I['sim_t']]
    bad_after = [s[I['step']] for s in S[i_cmd + 1:]
                 if (s[I['applied_src']] is not None and 0 <= s[I['applied_src']][2] < zero_seq0)
                 or not P4.hold_cmd_ok(s[I['base_cmd_body']], s[I['applied_arm']], TOL_APPLIED)]          # p4
    if bad_after:
        why.append(f'pause_old_or_nonzero_after_stop_{len(bad_after)}')
    o = kp_s.get('supervisor_outcome') or {}
    trn = o.get('transitions') or {}
    t_stop_solver, t_zero = trn.get('STOPPED'), trn.get('ZERO_PUB')
    t_done_solver = trn.get('DONE')                                       # p4：STOPPED→DONE 期間必須持續發布保持命令、DONE 後不得再發布
    hold_rows = [x for x in log if x.get('published') and _num(t_stop_solver) and t_stop_solver + 0.06 < x.get('sim_t', 0)
                 and (not _num(t_done_solver) or x.get('sim_t', 0) < t_done_solver - 0.06)]
    bad_hold = [x for x in hold_rows if not (x.get('pause_zero') and x.get('pause_u_published') is not None
                                             and P4.hold_cmd_ok(x['pause_u_published'][:3], x['pause_u_published'][3:], TOL_APPLIED))]
    gaps = [b['sim_t'] - a['sim_t'] for a, b in zip(hold_rows, hold_rows[1:]) if b['sim_t'] - a['sim_t'] > 0.2]
    pub_after = [x for x in log if x.get('published') and _num(t_done_solver) and x.get('sim_t', 0) > t_done_solver + 0.06]
    if bad_hold:
        why.append(f'pause_hold_publication_invalid_{len(bad_hold)}')
    if _num(t_stop_solver) and _num(t_done_solver) and (not hold_rows or gaps):
        why.append('pause_hold_publication_gap')
    if pub_after:
        why.append(f'pause_published_after_done_{len(pub_after)}')
    if not _num(t_stop_solver):
        why.append('pause_solver_stop_time_missing')
    cmd_ok = (len(why) == n_why_p1 and not bad_after and not pub_after and not bad_hold and _num(t_stop_solver)
              and not any(w.startswith('pause_') for w in why))          # P1 握手亦須無理由
    # ---- P3 實際停止 ----
    n_why_p3 = len(why)
    if not _num(t_zero):
        why.append('pause_zero_pub_time_missing')
        return cmd_ok, False
    t_end_req = t_cmd + OBS_S
    covered = S[-1][I['sim_t']] - t_cmd
    moved, invalid = [], []
    for i, s in enumerate(S):
        t = s[I['sim_t']]
        if t < t_zero - 1e-9 or t > t_end_req + 1e-9:
            continue
        v = window_speeds(S, I, i)
        if v is None:
            invalid.append(s[I['step']])
        elif not low(v):
            moved.append((s[I['step']], round(t, 4), [round(x, 6) for x in v]))
    if covered < OBS_S - 1e-6:
        why.append('pause_observation_incomplete')
    if invalid:
        why.append(f'pause_physical_data_invalid_{len(invalid)}')
    if moved:
        why.append(f'pause_moved_{len(moved)}')
    phys_ok = cmd_ok and len(why) == n_why_p3
    # ---- P4 一致性 ----
    if not (o.get('state') == 'DONE' and o.get('command_stop_confirmed') and o.get('physical_stop_confirmed')):
        why.append('pause_solver_outcome_not_done')
    if task.get('final_phase') != 'DONE' or (kp_t.get('outcome') or {}).get('state') != 'DONE':
        why.append('pause_task_not_done')
    if not _num(t_stop_solver) or abs(t_stop_solver - t_cmd) > 0.1:
        why.append('pause_stop_time_inconsistent')
    if kp_s.get('infeasible_base_hold_cycles') != 0:
        why.append('pause_base_hold_infeasible')
    ce = room.get('cmd_env') or {}
    if ce.get('n_plain_after_env', 0) != 0:
        why.append('plain_cmd_after_envelope')                         # 展開路徑與封裝入口重疊
    st['pause'] = {'zero_seq0': zero_seq0, 't_zero_pub_solver': t_zero, 't_cmd_stop_independent': t_cmd, 't_stopped_solver': t_stop_solver,
                   'observation_covered_s': covered, 'n_moved': len(moved), 'first_moved': moved[:5], 'n_invalid': len(invalid),
                   'n_old_or_nonzero_after_stop': len(bad_after), 'solver_outcome': o,
                   'command_stop_independent': cmd_ok, 'physical_stop_independent': phys_ok,
                   'stop_definition': 'active_hold', 'n_hold_rows': len(hold_rows), 'n_bad_hold_rows': len(bad_hold)}
    return cmd_ok, phys_ok


def judge(task, room, gate, solver):
    t1 = copy.deepcopy(task)
    t1['schema'] = 'door_k3m_task_p/1' if task.get('schema') == 'door_k3m_task_p2/1' else 'mismatch'
    res = CP.KM.judge(CP.to_k3m_task(t1), room, gate, solver)
    st = dict(res.get('stats') or {})
    why_k = list(res['why'])
    if t1.get('schema') != 'door_k3m_task_p/1':
        why_k.append('task_schema_k3m_p2')
    st['parent_settle_reason_replaced'] = C2.PARENT_SETTLE_REASON in why_k
    why_k = [w for w in why_k if w != C2.PARENT_SETTLE_REASON]
    settle_ok = C2.check_settle_p2(task, room, why_k, st)
    why_p = []
    cmd_ok, phys_ok = check_pause_p4(t1, room, solver, why_p, st)
    return {'pass': not why_k and not why_p, 'k3m_pass': not why_k, 'settle_p2_ok': settle_ok,
            'pause_command_stop': cmd_ok, 'pause_physical_stop': phys_ok, 'stop_definition': 'active_hold',
            'why': why_k + why_p, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    solver = json.load(open(os.path.join(run, 'align_solver.json')))
    res = judge(task, room, gate, solver)
    p = os.path.join(run, 'analysis', 'door_k3m_check_p4.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
