#!/usr/bin/env python3
"""K3-M 判定器（door_k3a_check.py 的版本化擴充；K3-A 檔不改）。規格 K3_mobile_coop_small_open_spec.md draft-3 §6。

K3-A 判準照用（door_k3a_check.judge），兩處調整：
  - 等待切換的列（trace k3m_mode＝waiting，phase＝OPEN、無 q_ref、目標＝最後 HOLD 目標）在 K3-A 判準中視為 HOLD 段（不推進參考）；
  - K3-A 的底盤 20 mm／2° 判準改為：**切換步之前** 20 mm／2°（相對停車錨點，room 逐步）；**切換步起**走廊內側帶＋粗略活動上限（room 逐步重算）。
另核（全部以紀錄獨立重算，不採信單一來源）：
  1. 模式證據：task.k3m.request（run_id、seq＝1）；執行端 ack＝room.k3m.switch（run_id、seq、switch_step、switch_t）；
     room 切換步的 sim_t＝switch_t（≤ 1e-4）；求解器 ack＝align_solver stats.k3m_mode.record.ack；兩 ack 的 switch_step／switch_t 一致；
     req_t ≤ switch_t ≤ first_free_sim_t ≤ publish_sim_t；兩 ack 的到達時刻 ≤ req_t＋timeout；result＝confirmed。
  2. 等待列：每列目標逐元素＝最後 HOLD 目標、無 q_ref；第一列有 q_ref 的時刻 ≥ 兩 ack 到達時刻。
  3. 求解器紀錄：首筆 free 解（slot）之前的已發布列 k3m_mode＝hold；之後全部 free 且 park_hold 為 None。
  4. 執行端走廊紀錄：自切換步起逐步連續、到紀錄結束；每步偏差與 room 重算一致（≤ 1e-5）；room.k3m.violation 為 None。
  5. 幾何證據：自執行端 ack 到達起、OPEN～END_HOLD 的每列都有 k3m_geom（決策、有限的計算耗時）與 k3m_corr；決策不得為 stop。

    python3 evaluation/door_k3m_check.py <run_dir>    # 寫 <run_dir>/analysis/door_k3m_check.json；結束碼 0＝通過
"""
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_check as KC3                          # noqa: E402
import door_k3m_corridor as CO                         # noqa: E402

INNER_MM, INNER_DEG = 20.0, 0.8
ENVELOPE = (-0.15, 0.02, -0.05, 0.10, 6.0)
BASE_M, BASE_YAW_DEG = 0.020, 2.0
MODE_TIMEOUT_S = 1.0
OPENING = ('OPEN', 'SETTLE', 'END_HOLD')


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _M(x):
    if x is None:
        return None
    a = np.asarray(x, float)
    return a.reshape(4, 4) if a.size == 16 and np.isfinite(a).all() else None


def base_task(task):
    t2 = copy.deepcopy(task)
    t2['schema'] = 'door_k3a_task/1' if task.get('schema') == 'door_k3m_task/1' else 'mismatch'
    for e in t2.get('trace') or []:
        if e.get('k3m_mode') == 'waiting' and e.get('phase') == 'OPEN' and e.get('q_ref') is None:
            e['phase'] = 'HOLD'
    return t2


def check_mode(task, room, solver, why, st):
    k = task.get('k3m') or {}
    req, ea, sa = k.get('request'), k.get('exec_ack'), k.get('solver_ack')
    sw = (room.get('k3m') or {}).get('switch')
    sk = ((solver or {}).get('stats') or {}).get('k3m_mode') or {}
    sa_log = (sk.get('record') or {}).get('ack')
    if not isinstance(req, dict) or req.get('seq') != 1 or not req.get('run_id') or not _num(req.get('req_t')):
        why.append('mode_request_missing')
        return None
    if not isinstance(ea, dict) or not isinstance(sw, dict) or any(ea.get(x) != sw.get(x) for x in ('run_id', 'seq', 'mode', 'switch_step', 'switch_t')):
        why.append('mode_exec_ack_not_room_switch')
        return None
    if not isinstance(sa, dict) or sa != sa_log:
        why.append('mode_solver_ack_not_solver_record')
        return None
    if ea['run_id'] != req['run_id'] or sa.get('run_id') != req['run_id']:
        why.append('mode_run_id_mismatch')
    if sa.get('switch_step') != ea['switch_step'] or not _num(sa.get('switch_t')) or abs(sa['switch_t'] - ea['switch_t']) > 1e-9:
        why.append('mode_ack_switch_mismatch')
    if not (all(_num(sa.get(x)) for x in ('first_free_sim_t', 'publish_sim_t')) and sa.get('published') is True
            and req['req_t'] - 1e-9 <= ea['switch_t'] <= sa['first_free_sim_t'] + 1e-9 <= sa['publish_sim_t'] + 2e-9):
        why.append('mode_time_order')
    cols = room.get('steps_cols') or []
    I = {c: i for i, c in enumerate(cols)}
    rs = next((s for s in room.get('steps', []) if s[I['step']] == ea['switch_step']), None)
    if rs is None or abs(rs[I['sim_t']] - ea['switch_t']) > 1e-4:
        why.append('mode_switch_step_time_mismatch')
    t_e, t_s = k.get('t_exec_recv'), k.get('t_solver_recv')
    if not (_num(t_e) and _num(t_s)) or max(t_e, t_s) > req['req_t'] + MODE_TIMEOUT_S + 1e-9:
        why.append('mode_ack_late_or_missing')
    if k.get('result') != 'confirmed':
        why.append('mode_not_confirmed')
    st['mode'] = {'req_t': req['req_t'], 'switch_step': ea['switch_step'], 'switch_t': ea['switch_t'], 't_exec_recv': t_e, 't_solver_recv': t_s,
                  'first_free_slot': sa.get('first_free_slot')}
    return {'switch_step': ea['switch_step'], 't_confirm': max(t_e, t_s) if _num(t_e) and _num(t_s) else math.inf,
            't_exec': t_e if _num(t_e) else math.inf, 'first_free_slot': sa.get('first_free_slot')}


def check_waiting(task, M, why, st):
    tr = task.get('trace') or []
    wt = [e for e in tr if e.get('k3m_mode') == 'waiting' and e.get('phase') == 'OPEN']
    last_hold = None
    for e in tr:
        if e in wt:
            break
        if _M(e.get('target')) is not None:
            last_hold = _M(e['target'])
    st['n_waiting_rows'] = len(wt)
    if not wt or last_hold is None:
        why.append('waiting_rows_missing')
        return
    for e in wt:
        if e.get('q_ref') is not None or _M(e.get('target')) is None or float(np.abs(_M(e['target']) - last_hold).max()) != 0.0:
            why.append('waiting_target_not_held')
            break
    fq = next((e for e in tr if e.get('q_ref') is not None), None)
    if fq is None or fq['t'] < M['t_confirm'] - 1e-9:
        why.append('reference_advanced_before_confirm')


def check_solver_rows(solver, M, why, st):
    log = (solver or {}).get('log') or []
    pubs = [x for x in log if x.get('published')]
    slot = M['first_free_slot']
    bad = 0
    seen_free = False
    for x in pubs:
        if x.get('slot') == slot:
            seen_free = True
        mode = x.get('k3m_mode')
        if not seen_free:
            if mode != 'hold':
                bad += 1
        else:
            if mode != 'free' or x.get('park_hold') is not None:
                bad += 1
    st['solver_rows_bad'] = bad
    if not seen_free:
        why.append('solver_first_free_not_published')
    if bad:
        why.append('solver_mode_rows_inconsistent')


def check_base(task, room, gate, M, why, st):
    cols = room.get('steps_cols') or []
    I = {c: i for i, c in enumerate(cols)}
    anchor = ((gate or {}).get('gate') or {}).get('anchor')
    k3m = room.get('k3m') or {}
    cfgk = k3m.get('config') or {}
    if anchor is None or not k3m.get('active') or k3m.get('violation') is not None:
        why.append('k3m_room_evidence_invalid' if k3m.get('violation') is None else 'k3m_executor_violation')
        return
    tbl = cfgk.get('corridor_table_json')
    C = CO.Corridor.from_json(tbl) if tbl is not None else None
    if C is None:
        why.append('corridor_table_missing')
        return
    ax, ay, ath = (float(v) for v in anchor)
    ca, sa = math.cos(ath), math.sin(ath)
    pre_m, pre_y, worst = 0.0, 0.0, [0.0, 0.0]
    logmap = {r[0]: r for r in k3m.get('log') or []}
    ksw = M['switch_step']
    post_steps = []
    dev_log = 0.0
    for s in room.get('steps', []):
        k, b, q = s[I['step']], s[I['base_xyth']], s[I['door_q_rad']]
        if b is None or q is None:
            continue
        if k < ksw:
            pre_m = max(pre_m, math.hypot(b[0] - ax, b[1] - ay))
            pre_y = max(pre_y, abs(math.degrees(math.atan2(math.sin(b[2] - ath), math.cos(b[2] - ath)))))
            continue
        post_steps.append(k)
        e = C.deviation(b, math.degrees(q))
        worst = [max(worst[0], abs(e[0]), abs(e[1])), max(worst[1], abs(e[2]))]
        fw, lt = ca * (b[0] - ax) + sa * (b[1] - ay), -sa * (b[0] - ax) + ca * (b[1] - ay)
        dy = math.degrees(math.atan2(math.sin(b[2] - ath), math.cos(b[2] - ath)))
        if not (ENVELOPE[0] <= fw <= ENVELOPE[1] and ENVELOPE[2] <= lt <= ENVELOPE[3] and abs(dy) <= ENVELOPE[4]):
            why.append('base_envelope')
        L = logmap.get(k)
        if L is None or L[2] is None:
            dev_log = math.inf
        else:
            dev_log = max(dev_log, max(abs(a_ - b_) for a_, b_ in zip(L[2], e)))
    st.update(pre_switch_base_m=pre_m, pre_switch_base_yaw_deg=pre_y, corridor_worst_mm=worst[0] * 1e3,
              corridor_worst_deg=math.degrees(worst[1]), executor_log_dev=dev_log)
    if pre_m > BASE_M or pre_y > BASE_YAW_DEG:
        why.append('base_drift_before_switch')
    if worst[0] > INNER_MM * 1e-3 + 1e-9 or math.degrees(worst[1]) > INNER_DEG + 1e-9:
        why.append('base_corridor')
    if not post_steps or post_steps != list(range(ksw, ksw + len(post_steps))):
        why.append('room_steps_discontinuous_after_switch')
    if dev_log > 1e-5:
        why.append('executor_corridor_log_mismatch')


def check_geom(task, M, why, st):
    tr = [e for e in task.get('trace') or [] if e.get('phase') in OPENING and e['t'] >= M['t_exec'] - 1e-9]
    bad = stop = 0
    for e in tr:
        g = e.get('k3m_geom')
        if not isinstance(g, dict) or g.get('dec') not in ('ok', 'warn', 'stop') or not _num(g.get('compute_s')) or e.get('k3m_corr') is None:
            bad += 1
        elif g['dec'] == 'stop':
            stop += 1
    st['geom_rows'], st['geom_rows_missing'], st['geom_stop_rows'] = len(tr), bad, stop
    if not tr or bad:
        why.append('geometry_evidence_missing')
    if stop:
        why.append('geometry_stop')


def judge(task, room, gate, solver):
    why, st = [], {}
    res = KC3.judge(base_task(task), room, gate)
    why += [w for w in res['why'] if w != 'base_drift']
    st.update(res.get('stats') or {})
    if task.get('schema') != 'door_k3m_task/1':
        why.append('task_schema_k3m')
    M = check_mode(task, room, solver, why, st)
    if M is not None:
        check_waiting(task, M, why, st)
        check_solver_rows(solver, M, why, st)
        check_base(task, room, gate, M, why, st)
        check_geom(task, M, why, st)
    return {'pass': not why, 'why': why, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    solver = json.load(open(os.path.join(run, 'align_solver.json')))
    res = judge(task, room, gate, solver)
    p = os.path.join(run, 'analysis', 'door_k3m_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'why', 'stats')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
