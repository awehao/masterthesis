#!/usr/bin/env python3
"""K2 判定器（缺資料、非有限、時間不連續一律不通過）。只讀：<run>/task.json、room_run.json、park_gate.json。

以**執行端逐物理步紀錄**（room_run.json）為準核對（K2 規格 draft-2 §G＋draft-3）；任務節點 trace 只用來取相位時刻。
監看窗＝UNFOLD 開始（任務事件）到結束。逐步：
  門角 |q| ≤ 1°；HOLD 期間兩指對橫桿接觸各 ≥ 0.5 N；非預期接觸（任何剛體對門板／立柱；非手指剛體對橫桿）≤ 0.1 N；
  手指對橫桿只在 ENGAGE～RELEASE 允許（其餘相位 > 0.1 N ⇒ 不通過）；牆接觸 ≤ 0.1 N；腕力 < 30 N；
  底盤相對停車錨點（park_gate.json）≤ 20 mm、偏航 ≤ 2°。
事件：相位序列含 ENGAGE → HOLD → RELEASE → RETREAT → DONE、無中止、HOLD 時長 ≥ 3.0 s。
（Codex 20261021 實作審查增補）另獨立核對：逐步紀錄**完整覆蓋** UNFOLD 開始到 DONE；終止狀態 DONE 且無牆鐘逾時／收尾上限；
夾持成立證據（進 HOLD 時執行端連續橫桿接觸 ≥ 2.0 s 且雙指 ≥ 0.5 N）；HOLD 漂移以逐步 FK（底盤＋手臂實測）與門角幾何重算 ≤ 5 mm；
釋放：進 RETREAT 前 0.5 s 雙指張開 ≥ 80% 且橫桿接觸 < 0.1 N；退離到位：DONE 時 TCP 距凍結把手位姿 s = 0.10 目標 < 5 mm；
門角速率全程 ≤ 0.05 rad/s；接觸值逐層有限；任何必要窗口為空 ⇒ 不通過。

    python3 evaluation/door_k2_check.py <run_dir>    # 寫 <run_dir>/analysis/door_k2_check.json；結束碼 0＝通過
"""
import json
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))

DOOR_MAX_DEG, CONTACT_N, GRIP_MIN_N, WRIST_N = 1.0, 0.1, 0.5, 30.0
DOOR_RATE, ATTACH_S, DRIFT_M, OPEN_FRAC, FINGER_OPEN, RELEASE_S, RETREAT_TOL, S_PRE = 0.05, 2.0, 0.005, 0.8, 0.0089, 0.5, 0.005, 0.10
SPEC_REL = 'src/my_omnibot_description/config/door_local_v2.yaml'
URDF_REL = 'evaluation/models/omni_bot_wholebody_expanded.urdf'
_K = None


def _kin():
    global _K
    if _K is None:
        from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
        _K = WholeBodyKinematics.from_urdf_file(os.path.join(WS, URDF_REL))
    return _K


def _tcp(b, qa):
    return _kin().fk(np.array(list(b) + list(qa), float), 'link_tcp')


def _H_of(q):
    import door_geom_v2 as G2
    D = yaml.safe_load(open(os.path.join(WS, SPEC_REL)))
    return G2.handle_pose(D, q, float(D['handle']['height']))
BASE_M, BASE_YAW = 0.020, math.radians(2.0)
SEQ = ['ENGAGE', 'HOLD', 'RELEASE', 'RETREAT', 'DONE']
FINGERS = ('uflite_finger1', 'uflite_finger2')


def _rigid(T):
    """4×4、有限、旋轉正交且行列式 +1、末列 (0,0,0,1)；否則回 None。"""
    try:
        T = np.asarray(T, float)
    except (TypeError, ValueError):
        return None
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return None
    R = T[:3, :3]
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-6 or np.abs(R.T @ R - np.eye(3)).max() > 1e-6 or np.abs(T[3] - [0, 0, 0, 1]).max() > 1e-9:
        return None
    return T


def _fin(x):
    try:
        if isinstance(x, dict):
            return all(_fin(v) for v in x.values())
        return bool(np.isfinite(np.asarray(x, float)).all())
    except (TypeError, ValueError):
        return False


def phase_windows(task):
    """由任務節點相位事件得 {phase: [(t0, t1), ...]}（模擬時間）。"""
    ev = [e for e in task.get('policy_events', []) if 'to' in e]
    out, tr = {}, task.get('trace') or []
    t_end = tr[-1]['t'] if tr else None
    for i, e in enumerate(ev):
        t1 = ev[i + 1]['t'] if i + 1 < len(ev) else t_end
        out.setdefault(e['to'], []).append((e['t'], t1))
    return out


def judge(task, room, gate):
    why = []
    if task.get('schema') != 'door_k2_task/1':
        why.append('task_schema')
    seq = [e['to'] for e in task.get('policy_events', []) if 'to' in e]
    if task.get('abort'):
        why.append(f"aborted:{task['abort']}")
    if not all(p in seq for p in SEQ) or [p for p in seq if p in SEQ] != SEQ:
        why.append(f'phase_sequence:{seq}')
    W = phase_windows(task)
    hold = W.get('HOLD', [])
    if not hold or hold[0][1] is None or hold[0][1] - hold[0][0] < 3.0 - 1e-6:
        why.append('hold_too_short_or_missing')
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
    # 終止狀態與覆蓋
    if task.get('final_phase') != 'DONE' or task.get('wall_timeout') or task.get('post_abort_wall_cap_hit') or task.get('received_any_sync') is False:
        why.append(f"termination:{task.get('final_phase')}")
    t_done = (W.get('DONE') or [(None, None)])[0][0]
    if t_done is None or steps[0][ix['sim_t']] > t_unfold + 0.02 or steps[-1][ix['sim_t']] < t_done - 0.02:
        why.append('record_does_not_cover_task_window')
    anchor = (gate or {}).get('gate', {}).get('anchor') if gate else None
    if anchor is None:
        why.append('park_anchor_missing')

    def ph_at(t):
        for ph, ws in W.items():
            for t0, t1 in ws:
                if t1 is not None and t0 <= t < t1:
                    return ph
        return None
    stats = {'max_abs_q_deg': 0.0, 'max_unexpected_n': 0.0, 'max_wall_n': 0.0, 'max_wrist_n': 0.0, 'max_base_m': 0.0,
             'max_base_yaw_deg': 0.0, 'min_hold_finger_bar_n': [math.inf, math.inf], 'max_finger_bar_outside_n': 0.0}
    bad = set()
    for s in steps:
        t = s[ix['sim_t']]
        q, lc, wc, wf, b = s[ix['door_q_rad']], s[ix['leaf_contacts']], s[ix['wall_contacts']], s[ix['wrist_force_n']], s[ix['base_xyth']]
        qd_, qf_, fc_, ch_, qa_ = (s[ix['door_qdot_rad_s']], s[ix['q_finger']], s[ix['finger_contact_n']], s[ix['contact_held_s']],
                                   s[ix['q_arm_meas']])
        if (not _fin(q) or not _fin(qd_) or lc is None or wc is None or wf is None or not _fin(wf) or not _fin(b) or not _fin(qa_)
                or not _fin(lc) or not _fin(wc) or qf_ is None or not _fin(qf_)):
            bad.add('missing_or_nonfinite')
            continue
        stats['max_abs_qdot'] = max(stats.get('max_abs_qdot', 0.0), abs(qd_))
        _vb = s[cols.index('meas_vb_body')] if 'meas_vb_body' in cols else None
        if _vb is not None and _fin(_vb):
            stats['report_max_base_speed_m_s'] = max(stats.get('report_max_base_speed_m_s', 0.0), math.hypot(_vb[0], _vb[1]))
            stats['report_max_base_yaw_rate'] = max(stats.get('report_max_base_yaw_rate', 0.0), abs(_vb[2]))
        stats['max_abs_q_deg'] = max(stats['max_abs_q_deg'], abs(math.degrees(q)))
        ph = ph_at(t)
        fb = [lc.get(f, {}).get('bar', 0.0) for f in FINGERS]
        unexp = max([v.get('panel', 0.0) + v.get('ambiguous', 0.0) + (0.0 if k in FINGERS else v.get('bar', 0.0)) for k, v in lc.items()] or [0.0])
        stats['max_unexpected_n'] = max(stats['max_unexpected_n'], unexp)
        stats['max_wall_n'] = max(stats['max_wall_n'], max(wc.values()) if wc else 0.0)
        stats['max_wrist_n'] = max(stats['max_wrist_n'], wf)
        if ph == 'HOLD':
            stats['min_hold_finger_bar_n'] = [min(stats['min_hold_finger_bar_n'][k], fb[k]) for k in range(2)]
        elif ph not in ('ENGAGE', 'RELEASE'):
            stats['max_finger_bar_outside_n'] = max(stats['max_finger_bar_outside_n'], max(fb))
        if anchor is not None:
            stats['max_base_m'] = max(stats['max_base_m'], math.hypot(b[0] - anchor[0], b[1] - anchor[1]))
            stats['max_base_yaw_deg'] = max(stats['max_base_yaw_deg'],
                                            abs(math.degrees(math.atan2(math.sin(b[2] - anchor[2]), math.cos(b[2] - anchor[2])))))
    why += sorted(bad)
    if stats.get('max_abs_qdot', 0.0) > DOOR_RATE:
        why.append('door_rate_over_0.05')
    # ---- 夾持證據、HOLD 漂移、釋放、退離（逐步重算）----
    def win(ph):
        ws = W.get(ph) or []
        if not ws or ws[0][1] is None:
            return []
        t0, t1 = ws[0]
        return [s for s in steps if t0 <= s[ix['sim_t']] < t1 and _fin(s[ix['door_q_rad']]) and _fin(s[ix['q_arm_meas']])]
    hold_steps = win('HOLD')
    t_hold = (W.get('HOLD') or [(None, None)])[0][0]
    pre = [s for s in steps if t_hold is not None and s[ix['sim_t']] <= t_hold]
    if not pre or not _fin(pre[-1][ix['contact_held_s']]) or pre[-1][ix['contact_held_s']] < ATTACH_S - 1e-6 \
            or not _fin(pre[-1][ix['finger_contact_n']]) or min(pre[-1][ix['finger_contact_n']]) < GRIP_MIN_N:
        why.append('grasp_evidence_missing')
    GTH, HF = _rigid(task.get('G_T_H')), _rigid(task.get('H_frozen'))
    THG = _rigid((task.get('grasp_basis') or {}).get('T_HG'))
    if GTH is None or HF is None or THG is None:
        why.append('grasp_matrices_missing_or_invalid')
    if not hold_steps or GTH is None or HF is None:
        why.append('hold_drift_unverifiable')
    else:
        dr = [float(np.linalg.norm((_tcp(s[ix['base_xyth']], s[ix['q_arm_meas']]) @ GTH)[:3, 3] - _H_of(s[ix['door_q_rad']])[:3, 3]))
              for s in hold_steps]                                   # **逐物理步**，不抽樣
        if not all(math.isfinite(x) for x in dr):
            why.append('hold_drift_nonfinite')
        else:
            stats['max_hold_drift_m'] = max(dr)
            if max(dr) > DRIFT_M:
                why.append('hold_drift_over_5mm')
    t_ret = (W.get('RETREAT') or [(None, None)])[0][0]
    rel = [s for s in steps if t_ret is not None and t_ret - RELEASE_S <= s[ix['sim_t']] < t_ret]
    if not rel:
        why.append('release_window_empty')
    else:
        ok_open = all(_fin(s[ix['q_finger']]) and min(s[ix['q_finger']]) >= OPEN_FRAC * FINGER_OPEN for s in rel)
        ok_free = all(max(s[ix['leaf_contacts']].get(f, {}).get('bar', 0.0) for f in FINGERS) < CONTACT_N for s in rel)
        if not (ok_open and ok_free):
            why.append('release_not_confirmed')
    if HF is not None and THG is not None and t_done is not None:
        last = [s for s in steps if s[ix['sim_t']] <= t_done + 1e-9]
        if last:
            import door_k2_policy as KP
            Tt = KP.K2Policy(KP.K2Config(), THG).target(HF, S_PRE)
            e = float(np.linalg.norm(_tcp(last[-1][ix['base_xyth']], last[-1][ix['q_arm_meas']])[:3, 3] - Tt[:3, 3]))
            stats['retreat_err_m'] = e
            if not math.isfinite(e) or e >= RETREAT_TOL:
                why.append('retreat_not_reached')
        else:
            why.append('retreat_unverifiable')
    else:
        why.append('retreat_unverifiable')
    if stats['max_abs_q_deg'] > DOOR_MAX_DEG:
        why.append('door_angle_over_1deg')
    if stats['max_unexpected_n'] > CONTACT_N:
        why.append('unexpected_contact')
    if stats['max_wall_n'] > CONTACT_N:
        why.append('wall_contact')
    if stats['max_wrist_n'] >= WRIST_N:
        why.append('wrist_force')
    if not math.isfinite(min(stats['min_hold_finger_bar_n'])) or min(stats['min_hold_finger_bar_n']) < GRIP_MIN_N:
        why.append('hold_finger_contact_below_min')
    if stats['max_finger_bar_outside_n'] > CONTACT_N:
        why.append('finger_bar_outside_allowed_phase')
    if stats['max_base_m'] > BASE_M or stats['max_base_yaw_deg'] > math.degrees(BASE_YAW):
        why.append('base_drift')
    return {'pass': not why, 'why': why, 'stats': stats, 'phase_sequence': seq, 'n_steps_window': len(steps)}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    res = judge(task, room, gate)
    p = os.path.join(run, 'analysis', 'door_k2_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'why', 'stats')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
