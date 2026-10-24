#!/usr/bin/env python3
"""K3-M＋受控暫停判定器（door_k3m_check.py r5 的版本化擴充；規格 K3M_pause_integration_spec_draft1.md §3）。

一、K3-M 判準照用（door_k3m_check.judge）：task schema door_k3m_task_p/1 轉為 door_k3m_task/1；policy_events 與 trace 的 PAUSE 相位
    **對應為 END_HOLD**（PAUSE 期間同樣要求完整幾何證據、走廊紀錄與監看；END_HOLD 長度判準 ≥ 3 s 由暫停段滿足）。
二、暫停核對（全部以紀錄獨立重算，不採信線上監督器的結論）：
  P1 握手：task.k3m_pause.request（run_id、seq＝2、frozen_target_seq 對到 trace 同序號的已發布目標、frozen_target_xyz＝該目標位置 ≤ 1e-9）；
     執行端 ack＝room.k3m.pause_switch；求解端 ack＝align_solver stats.k3m_pause.record.ack；switch_step／switch_t 一致；
     req_t ≤ switch_t ≤ first_pause_sim_t ≤ publish_sim_t；兩 ack 到達（t_*_recv）≤ req_t＋1.0 s、牆鐘 ≤ wall_req＋2.0 s；
     求解端凍結目標位置＝請求（≤ 1e-6）。
  P2 命令停止（room applied_src＋solver log）：零速第一筆序號 zero_seq0＝求解端第一筆 pause_zero 已發布列的 pause_source_seq；
     room 逐步中第一段「5 個連續步、source_seq ≥ zero_seq0、底盤命令與手臂套用 ≤ 1e-6」的末步＝獨立命令停止時刻 t_cmd_stop；
     之後不得有 source_seq ∈ [0, zero_seq0) 或任何非零套用；求解端 STOPPED 時刻後 0.06 s 起不得再有已發布列。
  P3 實際停止（room 逐步重算）：10 個連續步窗口、實際 Δt，底盤 ≤ 1 mm/s 且 ≤ 0.005 rad/s、門 ≤ 0.002 rad/s、實測手臂 ≤ 1e-3 rad/s；
     自求解端 ZERO_PUB 時刻起至 t_cmd_stop＋10 s 每步都須低速且資料有效；紀錄須覆蓋到 t_cmd_stop＋10 s（否則 pause_observation_incomplete）。
  P4 一致性：求解端 supervisor_outcome 為 DONE 且命令／實際停止皆確認；task 最終相位 DONE、pause_outcome 為 DONE；
     獨立 t_cmd_stop 與求解端 STOPPED 時刻相差 ≤ 0.1 s；暫停期間不可行 base_hold 輪數 0；
     room.cmd_env.n_plain_after_env＝0（第一筆封裝後，展開節點的舊路徑不得再有命令）。
結論分報：k3m_pass（K3-M 部分）、pause_command_stop、pause_physical_stop；整體 pass＝全部通過。
r2（Codex 20261024_095031 第 4 節）：
  - 分項以**自身必要證據**明確判定（check_pause 回傳 cmd_ok／phys_ok 布林）；缺證據 ⇒ False（未驗證），不由理由字首反推。
  - 趟別：request、兩個 ack、room.k3m.config 的 run_id 都須等於 task.args.run_id。
  - 求解端 ack 的 published 須為 True。
  - 凍結目標：請求時刻以前**最後一筆**已發布目標（trace 中 t ≤ req_t、有 target 的最後一列）的 pub_seq 須等於 frozen_target_seq，
    其完整 4×4 須等於請求的 frozen_target_T（位置 ≤ 1e-9 m、旋轉 ≤ 1e-9 rad）；求解端 ack 的 frozen_target_T 與請求比較（≤ 1e-6）。

    python3 evaluation/door_k3m_check_p.py <run_dir>    # 寫 <run_dir>/analysis/door_k3m_check_p.json；結束碼 0＝通過
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check as KM                            # noqa: E402
import door_k3m_pause_mode as PMODE                    # noqa: E402

TH = {'base_lin': 0.001, 'base_ang': 0.005, 'door': 0.002, 'arm': 1e-3}
WIN, N_CONFIRM, TOL_APPLIED, OBS_S = 10, 5, 1e-6, 10.0


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def to_k3m_task(task):
    t2 = copy.deepcopy(task)
    t2['schema'] = 'door_k3m_task/1' if task.get('schema') == 'door_k3m_task_p/1' else 'mismatch'
    for e in t2.get('policy_events') or []:
        for k in ('to', 'from', 'phase'):
            if e.get(k) == 'PAUSE':
                e[k] = 'END_HOLD'
    for e in t2.get('trace') or []:
        if e.get('phase') == 'PAUSE':
            e['phase'] = 'END_HOLD'
    return t2


def room_rows(room):
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    if 'applied_src' not in I:
        return None, I
    return room['steps'], I


def window_speeds(S, I, i):
    if i - WIN < 0:
        return None
    seg = S[i - WIN:i + 1]
    for a, b in zip(seg, seg[1:]):
        if int(b[I['step']]) != int(a[I['step']]) + 1 or not b[I['sim_t']] > a[I['sim_t']]:
            return None
    a, b = seg[0], seg[-1]
    try:
        ba, bb = [float(x) for x in a[I['base_xyth']]], [float(x) for x in b[I['base_xyth']]]
        qa, qb = float(a[I['door_q_rad']]), float(b[I['door_q_rad']])
        ja, jb = [float(x) for x in a[I['q_arm_meas']]], [float(x) for x in b[I['q_arm_meas']]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in ba + bb + ja + jb + [qa, qb]):
        return None
    dt = b[I['sim_t']] - a[I['sim_t']]
    dyaw = math.atan2(math.sin(bb[2] - ba[2]), math.cos(bb[2] - ba[2]))
    return (math.hypot(bb[0] - ba[0], bb[1] - ba[1]) / dt, abs(dyaw) / dt, abs(qb - qa) / dt,
            max(abs(u - v) for u, v in zip(jb, ja)) / dt)


def low(v):
    return v is not None and v[0] <= TH['base_lin'] and v[1] <= TH['base_ang'] and v[2] <= TH['door'] and v[3] <= TH['arm']


def _T16(x):
    try:
        v = [float(a) for a in np.asarray(x, float).reshape(-1)]
    except (TypeError, ValueError):
        return None
    return v if len(v) == 16 and all(math.isfinite(a) for a in v) else None


def check_pause(task, room, solver, why, st):
    """回傳 (cmd_ok, phys_ok)：各分項只在自身必要證據齊全且核對通過時為 True。"""
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
    prior = [e for e in task.get('trace') or [] if e.get('target') is not None and isinstance(e.get('pub_seq'), int)
             and _num(e.get('t')) and _num(req.get('req_t')) and e['t'] <= req['req_t'] + 1e-9]
    last = prior[-1] if prior else None
    T_last = None if last is None else _T16(last['target'])
    if (T_req is None or T_last is None or last.get('pub_seq') != req.get('frozen_target_seq')
            or (lambda d: d[0] > 1e-9 or d[1] > 1e-9)(PMODE.target_diff(T_last, T_req))):
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
              and max(abs(float(v)) for v in list(s[I['base_cmd_body']]) + list(s[I['applied_arm']])) <= TOL_APPLIED)
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
                 or max(abs(float(v)) for v in list(s[I['base_cmd_body']]) + list(s[I['applied_arm']])) > TOL_APPLIED]
    if bad_after:
        why.append(f'pause_old_or_nonzero_after_stop_{len(bad_after)}')
    o = kp_s.get('supervisor_outcome') or {}
    trn = o.get('transitions') or {}
    t_stop_solver, t_zero = trn.get('STOPPED'), trn.get('ZERO_PUB')
    pub_after = [x for x in log if x.get('published') and _num(t_stop_solver) and x.get('sim_t', 0) > t_stop_solver + 0.06]
    if pub_after:
        why.append(f'pause_published_after_stop_{len(pub_after)}')
    if not _num(t_stop_solver):
        why.append('pause_solver_stop_time_missing')
    cmd_ok = (len(why) == n_why_p1 and not bad_after and not pub_after and _num(t_stop_solver)
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
                   'command_stop_independent': cmd_ok, 'physical_stop_independent': phys_ok}
    return cmd_ok, phys_ok


def judge(task, room, gate, solver):
    why_k, st = [], {}
    res = KM.judge(to_k3m_task(task), room, gate, solver)
    why_k += res['why']
    st.update(res.get('stats') or {})
    if task.get('schema') != 'door_k3m_task_p/1':
        why_k.append('task_schema_k3m_p')
    why_p = []
    cmd_ok, phys_ok = check_pause(task, room, solver, why_p, st)
    return {'pass': not why_k and not why_p, 'k3m_pass': not why_k, 'pause_command_stop': cmd_ok, 'pause_physical_stop': phys_ok,
            'why': why_k + why_p, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    solver = json.load(open(os.path.join(run, 'align_solver.json')))
    res = judge(task, room, gate, solver)
    p = os.path.join(run, 'analysis', 'door_k3m_check_p.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'pause_command_stop', 'pause_physical_stop', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
