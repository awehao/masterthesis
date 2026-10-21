#!/usr/bin/env python3
"""K3-A 判定器（缺資料、非有限、時間不連續一律不通過）。只讀：<run>/task.json、room_run.json、park_gate.json。
door_k2_check.py 不改；共用其輔助函式（FK、門幾何、剛體核對、有限性、相位窗口）。

以**執行端逐物理步紀錄**（room_run.json）為準；任務節點 trace 取相位時刻與已發布目標／參考角。
監看窗＝UNFOLD 開始到紀錄結束（DONE 之後夾持保留，視同 DONE）。逐步：
  門角：OPEN 前 |q| ≤ 1°（同 K2 判定）；OPEN 起 −1° ≤ q ≤ θ_goal＋1°；|q̇| ≤ 0.05 rad/s 全程；
  非預期接觸（含 ambiguous）≤ 0.1 N；牆 ≤ 0.1 N；腕力 < 30 N；底盤 ≤ 20 mm／2°；
  手指—橫桿只在 ENGAGE、HOLD、OPEN、SETTLE、END_HOLD、DONE 允許；
  HOLD～紀錄結束（含 DONE 後）兩指橫桿接觸逐步 ≥ 0.5 N、實測手指 ≤ 50% 開度；抓取漂移（G_T_H_grasp）逐步 ≤ 5 mm、≤ 2°。
任務 trace 對帳（Codex 20261021 實作審查 2、3）：HOLD 起每列夾爪命令＝閉合；已發布目標序號 1…N 連續且 N＝n_target_published；
  HOLD～DONE trace 相鄰間隔 ≤ 0.25 s；OPEN／SETTLE 每列都有 q_ref 與目標（唯一例外＝HOLD→OPEN 轉換列，其目標＝HOLD 目標）；
  END_HOLD 每列目標＝SETTLE 最後已發布目標，涵蓋 END_HOLD 起到 DONE。
事件：ENGAGE → HOLD → OPEN → SETTLE → END_HOLD → DONE；無 RELEASE／RETREAT；無中止；HOLD ≥ 1.0 s、END_HOLD ≥ 3.0 s。
參考角（task trace 逐筆）：q_ref ≤ θ_goal；相鄰差 ≤ r(q_ref_prev)·dt；|q_ref − q_meas| ≤ 0.5°；q_ref 到達 θ_goal；
  OPEN 首筆目標＝最後一筆 HOLD 目標（≤ 1e-9）；OPEN／SETTLE 目標＝首筆目標繞鉸鏈轉 (q_ref − q_ref₀)（≤ 1e-5 m／1e-5 rad）。
到達：SETTLE 末 |q − θ_goal| ≤ 0.5°；SETTLE 末 1.0 s 逐步滿足穩定條件（|q − θ_goal| ≤ 0.5°、|q̇| ≤ 0.002、漂移 ≤ 5 mm、兩指 ≥ 0.5 N）。

    python3 evaluation/door_k3a_check.py <run_dir>    # 寫 <run_dir>/analysis/door_k3a_check.json；結束碼 0＝通過
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3a_policy as KP                          # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

CFG = KP.K3AConfig()
TCFG = CFG.target_cfg()
THETA = TCFG.theta_goal_rad
PRE_MAX_DEG, CONTACT_N, GRIP_MIN_N, WRIST_N, DOOR_RATE, ATTACH_S = 1.0, 0.1, 0.5, 30.0, 0.05, 2.0
DRIFT_M, DRIFT_R = CFG.drift_abort_m, CFG.drift_rot_abort_rad
BASE_M, BASE_YAW = 0.020, math.radians(2.0)
SEQ = ['ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE']
ALLOWED = ('ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE')
GRASP = ('HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE')   # 夾持保留到紀錄結束（Codex 20261021 實作審查 1）
OPEN_MAX_FRAC, FINGER_OPEN = 0.5, 0.0089                 # 夾持後實測手指 ≤ 50% 開度視為未張爪
TRACE_GAP_S = 0.25
FINGERS = KC.FINGERS
AX = np.asarray(CFG.hinge_axis, float)
PT = np.asarray(CFG.hinge_point, float)


def _tgt(x):
    if x is None:
        return None
    a = np.asarray(x, float)
    return a.reshape(4, 4) if a.size == 16 and np.isfinite(a).all() else None


def check_reference(task, why, stats):
    tr = [e for e in (task.get('trace') or []) if e.get('phase') in ('HOLD', 'OPEN', 'SETTLE')]
    hold = [e for e in tr if e['phase'] == 'HOLD']
    opn = [e for e in tr if e['phase'] in ('OPEN', 'SETTLE')]
    if not hold or not opn:
        why.append('reference_trace_missing')
        return
    # 相位轉換那一輪的 trace 已標新相位：唯一允許缺 q_ref 的是第一列 OPEN（HOLD→OPEN 轉換列，目標＝HOLD 目標）
    miss = [i for i, e in enumerate(opn) if e.get('q_ref') is None or _tgt(e.get('target')) is None]
    if miss and not (miss == [0] and opn[0]['phase'] == 'OPEN' and opn[0].get('q_ref') is None and _tgt(opn[0].get('target')) is not None):
        why.append('reference_evidence_incomplete')
    opn = [e for e in opn if e.get('q_ref') is not None]
    if not opn:
        why.append('reference_trace_missing')
        return
    T_hold = None
    for e in tr:
        if e is opn[0]:
            break
        if _tgt(e.get('target')) is not None:
            T_hold = _tgt(e['target'])
    T0 = _tgt(opn[0].get('target'))
    if T_hold is None or T0 is None:
        why.append('open_continuity_unverifiable')
        return
    stats['open_first_vs_last_hold_m'] = float(np.abs(T0 - T_hold).max())
    if stats['open_first_vs_last_hold_m'] > 1e-9:
        why.append('open_first_target_discontinuous')
    q0 = opn[0]['q_ref']
    worst = {'rate': 0.0, 'lead': 0.0, 'goal': -math.inf, 'tgt_p': 0.0, 'tgt_r': 0.0}
    prev = None
    for e in opn:
        q, qm, T = e.get('q_ref'), e.get('q_door'), _tgt(e.get('target'))
        if q is None or qm is None or T is None or not (math.isfinite(q) and math.isfinite(qm)):
            why.append('reference_trace_nonfinite')
            return
        worst['goal'] = max(worst['goal'], q - THETA)
        worst['lead'] = max(worst['lead'], abs(q - qm) - TCFG.lead_max_rad)
        if prev is not None:
            dt = e['t'] - prev['t']
            if not (dt >= 0.0):
                why.append('reference_time_backwards')
                return
            worst['rate'] = max(worst['rate'], abs(q - prev['q_ref']) - TCFG.rate(prev['q_ref']) * dt)
        Te = rotate_about_hinge(T0, PT, AX / np.linalg.norm(AX), q - q0)
        worst['tgt_p'] = max(worst['tgt_p'], float(np.linalg.norm(Te[:3, 3] - T[:3, 3])))
        c = (float(np.trace(Te[:3, :3].T @ T[:3, :3])) - 1.0) / 2.0
        worst['tgt_r'] = max(worst['tgt_r'], math.acos(max(-1.0, min(1.0, c))))
        prev = e
    stats['reference'] = worst
    stats['q_ref_final_deg'] = math.degrees(opn[-1]['q_ref'])
    if worst['goal'] > 1e-12:
        why.append('reference_over_goal')
    if worst['rate'] > 1e-9:
        why.append('reference_rate_violated')
    if worst['lead'] > 1e-9:
        why.append('reference_lead_violated')
    if worst['tgt_p'] > 1e-5 or worst['tgt_r'] > 1e-5:
        why.append('target_not_rigid_rotation_of_first')
    if abs(opn[-1]['q_ref'] - THETA) > 1e-12:
        why.append('reference_not_reached')


def check_trace(task, W, why, st):
    tr = [e for e in (task.get('trace') or []) if 'phase' in e]
    t_hold = (W.get('HOLD') or [(None, None)])[0][0]
    t_end = (W.get('END_HOLD') or [(None, None)])[0][0]
    t_done = (W.get('DONE') or [(None, None)])[0][0]
    if t_hold is None or t_end is None or t_done is None or not tr:
        why.append('trace_unverifiable')
        return
    # 發布序號對帳
    seqs = [e.get('pub_seq') for e in tr if e.get('pub_seq') is not None]
    n = task.get('n_target_published')
    if any((e.get('pub_seq') is None) != (_tgt(e.get('target')) is None) for e in tr) or not isinstance(n, int) \
            or seqs != list(range(1, len(seqs) + 1)) or len(seqs) != n:
        why.append('publish_sequence_mismatch')
    st['n_target_published'] = n
    # HOLD～DONE 覆蓋與間隔
    win = [e for e in tr if t_hold - 1e-9 <= e['t'] <= t_done + 1e-9]
    ts = [e['t'] for e in win]
    if not ts or ts[0] > t_hold + TRACE_GAP_S or ts[-1] < t_done - 1e-9 or any(b_ - a_ > TRACE_GAP_S for a_, b_ in zip(ts, ts[1:])):
        why.append('trace_gap')
    st['max_trace_gap_s'] = max([b_ - a_ for a_, b_ in zip(ts, ts[1:])] or [math.inf])
    # 夾爪命令：HOLD 起全部閉合
    g = [e.get('grip') for e in tr if e['t'] >= t_hold - 1e-9]
    if not g or any(x is None or not math.isfinite(x) or x > CFG.finger_closed + 1e-9 for x in g):
        why.append('grip_command_not_closed_after_grasp')
    # END_HOLD：目標固定＝SETTLE 最後已發布目標
    settle = [e for e in tr if e['phase'] == 'SETTLE' and _tgt(e.get('target')) is not None]
    eh = [e for e in tr if e['phase'] == 'END_HOLD']
    if not settle or not eh or any(_tgt(e.get('target')) is None for e in eh) or eh[0]['t'] > t_end + TRACE_GAP_S \
            or eh[-1]['t'] < t_done - TRACE_GAP_S:
        why.append('end_hold_evidence_missing')
    else:
        Ts = _tgt(settle[-1]['target'])
        dev = max(float(np.abs(_tgt(e['target']) - Ts).max()) for e in eh)
        st['end_hold_target_dev'] = dev
        if dev > 1e-9:
            why.append('end_hold_target_not_fixed')


def judge(task, room, gate):
    why, stats = [], {}
    if task.get('schema') != 'door_k3a_task/1':
        why.append('task_schema')
    seq = [e['to'] for e in task.get('policy_events', []) if 'to' in e]
    if task.get('abort'):
        why.append(f"aborted:{task['abort']}")
    if [p for p in seq if p in SEQ] != SEQ or any(p in seq for p in ('RELEASE', 'RETREAT')):
        why.append(f'phase_sequence:{seq}')
    W = KC.phase_windows(task)
    for ph, need in (('HOLD', CFG.hold_s), ('END_HOLD', CFG.end_hold_s)):
        ws = W.get(ph, [])
        if not ws or ws[0][1] is None or ws[0][1] - ws[0][0] < need - 1e-6:
            why.append(f'{ph.lower()}_too_short_or_missing')
    t_unfold = W.get('UNFOLD', [(None, None)])[0][0]
    if t_unfold is None:
        why.append('no_unfold_event')
        return {'pass': False, 'why': why}
    cols = room.get('steps_cols') or []
    need = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'door_qdot_rad_s', 'q_finger', 'finger_contact_n', 'contact_held_s',
            'leaf_contacts', 'wall_contacts', 'wrist_force_n']
    if any(c not in cols for c in need):
        why.append('room_cols_missing')
        return {'pass': False, 'why': why}
    ix = {c: cols.index(c) for c in need}
    steps = [s for s in room.get('steps', []) if s[ix['sim_t']] >= t_unfold - 1e-9]
    if not steps:
        why.append('no_steps_in_window')
        return {'pass': False, 'why': why}
    sid = [s[ix['step']] for s in steps]
    if sid != list(range(sid[0], sid[0] + len(sid))):
        why.append('step_discontinuity')
    if task.get('final_phase') != 'DONE' or task.get('wall_timeout') or task.get('post_abort_wall_cap_hit') or task.get('received_any_sync') is False:
        why.append(f"termination:{task.get('final_phase')}")
    t_done = (W.get('DONE') or [(None, None)])[0][0]
    if t_done is None or steps[0][ix['sim_t']] > t_unfold + 0.02 or steps[-1][ix['sim_t']] < t_done - 0.02:
        why.append('record_does_not_cover_task_window')
    anchor = (gate or {}).get('gate', {}).get('anchor') if gate else None
    if anchor is None:
        why.append('park_anchor_missing')
    GTH = KC._rigid(task.get('G_T_H'))
    HF = KC._rigid(task.get('H_frozen'))
    if GTH is None or HF is None or KC._rigid((task.get('grasp_basis') or {}).get('T_HG')) is None:
        why.append('grasp_matrices_missing_or_invalid')

    def ph_at(t):
        if t_done is not None and t >= t_done:
            return 'DONE'                                      # 夾持保留到紀錄結束
        for ph, ws in W.items():
            for t0, t1 in ws:
                if t1 is not None and t0 <= t < t1:
                    return ph
        return None
    st = {'max_abs_q_pre_deg': 0.0, 'q_open_min_deg': math.inf, 'q_open_max_deg': -math.inf, 'max_abs_qdot': 0.0,
          'max_unexpected_n': 0.0, 'max_wall_n': 0.0, 'max_wrist_n': 0.0, 'max_base_m': 0.0, 'max_base_yaw_deg': 0.0,
          'min_grasp_finger_bar_n': [math.inf, math.inf], 'max_finger_bar_outside_n': 0.0, 'max_drift_m': 0.0, 'max_drift_rad': 0.0,
          'n_grasp_steps': 0}
    bad = set()
    settle_rows = []
    for s in steps:
        t = s[ix['sim_t']]
        q, qd_, lc, wc, wf, b = (s[ix['door_q_rad']], s[ix['door_qdot_rad_s']], s[ix['leaf_contacts']], s[ix['wall_contacts']],
                                 s[ix['wrist_force_n']], s[ix['base_xyth']])
        qf_, qa_ = s[ix['q_finger']], s[ix['q_arm_meas']]
        if (not KC._fin(q) or not KC._fin(qd_) or lc is None or wc is None or wf is None or not KC._fin(wf) or not KC._fin(b)
                or not KC._fin(qa_) or not KC._fin(lc) or not KC._fin(wc) or qf_ is None or not KC._fin(qf_)):
            bad.add('missing_or_nonfinite')
            continue
        ph = ph_at(t)
        st['max_abs_qdot'] = max(st['max_abs_qdot'], abs(qd_))
        if ph in ('OPEN', 'SETTLE', 'END_HOLD', 'DONE'):
            st['q_open_min_deg'] = min(st['q_open_min_deg'], math.degrees(q))
            st['q_open_max_deg'] = max(st['q_open_max_deg'], math.degrees(q))
        else:
            st['max_abs_q_pre_deg'] = max(st['max_abs_q_pre_deg'], abs(math.degrees(q)))
        fb = [lc.get(f, {}).get('bar', 0.0) for f in FINGERS]
        unexp = max([v.get('panel', 0.0) + v.get('ambiguous', 0.0) + (0.0 if k in FINGERS else v.get('bar', 0.0)) for k, v in lc.items()] or [0.0])
        st['max_unexpected_n'] = max(st['max_unexpected_n'], unexp)
        st['max_wall_n'] = max(st['max_wall_n'], max(wc.values()) if wc else 0.0)
        st['max_wrist_n'] = max(st['max_wrist_n'], wf)
        if ph not in ALLOWED:
            st['max_finger_bar_outside_n'] = max(st['max_finger_bar_outside_n'], max(fb))
        if ph in GRASP:
            st['n_grasp_steps'] += 1
            st['max_finger_q_grasp'] = max(st.get('max_finger_q_grasp', 0.0), max(qf_))
            st['min_grasp_finger_bar_n'] = [min(st['min_grasp_finger_bar_n'][k], fb[k]) for k in range(2)]
            if GTH is not None:
                dp, dr = KP.grasp_drift(KC._tcp(b, qa_), GTH, KC._H_of(q))     # **逐物理步**
                st['max_drift_m'], st['max_drift_rad'] = max(st['max_drift_m'], dp), max(st['max_drift_rad'], dr)
                if ph == 'SETTLE':
                    settle_rows.append((t, q, qd_, dp, min(fb)))
        if anchor is not None:
            st['max_base_m'] = max(st['max_base_m'], math.hypot(b[0] - anchor[0], b[1] - anchor[1]))
            st['max_base_yaw_deg'] = max(st['max_base_yaw_deg'],
                                         abs(math.degrees(math.atan2(math.sin(b[2] - anchor[2]), math.cos(b[2] - anchor[2])))))
    why += sorted(bad)
    # 夾持證據（進 HOLD 時）
    t_hold = (W.get('HOLD') or [(None, None)])[0][0]
    pre = [s for s in steps if t_hold is not None and s[ix['sim_t']] <= t_hold]
    if not pre or not KC._fin(pre[-1][ix['contact_held_s']]) or pre[-1][ix['contact_held_s']] < ATTACH_S - 1e-6 \
            or not KC._fin(pre[-1][ix['finger_contact_n']]) or min(pre[-1][ix['finger_contact_n']]) < GRIP_MIN_N:
        why.append('grasp_evidence_missing')
    if st['n_grasp_steps'] == 0 or GTH is None:
        why.append('grasp_drift_unverifiable')
    elif st['max_drift_m'] > DRIFT_M or st['max_drift_rad'] > DRIFT_R:
        why.append('grasp_drift_over_limit')
    if st['max_abs_q_pre_deg'] > PRE_MAX_DEG:
        why.append('door_angle_pre_open_over_1deg')
    if st['q_open_max_deg'] == -math.inf:
        why.append('no_open_steps')
    elif st['q_open_min_deg'] < CFG.q_min_deg or st['q_open_max_deg'] > CFG.theta_goal_deg + CFG.over_goal_deg:
        why.append('door_angle_out_of_bounds')
    if st['max_abs_qdot'] > DOOR_RATE:
        why.append('door_rate_over_0.05')
    if st['max_unexpected_n'] > CONTACT_N:
        why.append('unexpected_contact')
    if st['max_wall_n'] > CONTACT_N:
        why.append('wall_contact')
    if st['max_wrist_n'] >= WRIST_N:
        why.append('wrist_force')
    if not math.isfinite(min(st['min_grasp_finger_bar_n'])) or min(st['min_grasp_finger_bar_n']) < GRIP_MIN_N:
        why.append('grasp_finger_contact_below_min')
    if st['max_finger_bar_outside_n'] > CONTACT_N:
        why.append('finger_bar_outside_allowed_phase')
    if st['max_base_m'] > BASE_M or st['max_base_yaw_deg'] > math.degrees(BASE_YAW):
        why.append('base_drift')
    # 到達與 SETTLE 穩定條件（逐物理步重算）
    t_end = (W.get('END_HOLD') or [(None, None)])[0][0]
    if not settle_rows or t_end is None:
        why.append('settle_unverifiable')
    else:
        last = [r for r in settle_rows if r[0] >= t_end - CFG.settle_hold_s + 1e-9]
        st['settle_final_q_deg'] = math.degrees(settle_rows[-1][1])
        if abs(settle_rows[-1][1] - THETA) > math.radians(CFG.settle_tol_deg):
            why.append('not_reached_goal')
        if not last or last[0][0] - (t_end - CFG.settle_hold_s) > 0.02 + 1e-9 or not all(
                abs(r[1] - THETA) <= math.radians(CFG.settle_tol_deg) and abs(r[2]) <= CFG.settle_rate and r[3] <= DRIFT_M
                and r[4] >= GRIP_MIN_N for r in last):
            why.append('settle_condition_not_held_1s')
    if st.get('max_finger_q_grasp', math.inf) > OPEN_MAX_FRAC * FINGER_OPEN:
        why.append('fingers_opened_after_grasp')
    check_reference(task, why, st)
    check_trace(task, W, why, st)
    return {'pass': not why, 'why': why, 'stats': st, 'phase_sequence': seq, 'n_steps_window': len(steps)}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    res = judge(task, room, gate)
    p = os.path.join(run, 'analysis', 'door_k3a_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'why', 'stats')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
