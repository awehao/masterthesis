#!/usr/bin/env python3
"""K3-B 受控釋放整體判定器（規格 K3B_release_clear_spec_draft2.md §8 R1–R8；設計 K3B_impl_design_draft1.md D5；起跑前凍結）。

一、既有判準：door_k3m_check_w1f2.judge 全部照跑；只取代下列明列項（K3B_REPLACED），其餘一律保留：
  - pause_solver_outcome_not_done／pause_task_not_done（同 W2c：監督器 HELD→HANDED_OVER，task pause_outcome 不是 DONE）。
  取代清單依實驗決定：
  - 做法：以 W2c2 的合成趟（60_3 實錄＋合成 W2）把 t_open 後改為 RELEASE 列、夾爪命令張開、目標退離，再跑 W1f2 鏈；
  - 結果：只出現上述兩項。
  - 原擬一併取代的 grip_command_not_closed_after_grasp、可能的 end_hold_evidence_missing 均未觸發，故**不預先取代**。
  - 實跑若出現新舊理由即不通過，不事後放寬。
二、W2 多段提交（執行端逐步紀錄，不信任務端）：
  M1 commits 恰為 mode 1、2、3 依序、皆 applied_confirmed；各提交步實際套用＝first_source_seq、mode_seq 相符、鏈正常。
  M2 自 mode 1 提交起逐步連續；有來源列 mode_seq 不減；profile 與 finger_target 等於該步權威模式的對照值；
     finger_target 恰在 mode 2 提交步由 closed 轉 open；權威確認列覆蓋每一步且 confirm_ok；無中止、無入口違規。
三、K3-B（以 room 逐物理步重算為主）：
  K1 t_open＝mode 2 提交步；執行端 release 報告 open 步＝t_open。
  K2 t_conf 重算：t_open 後首個「兩指開度 ≥ 95 %、兩指 bar＋ambiguous ≤ 0.1 N、資料有效」連續 ≥ 0.5 s 之步；
     與執行端報告相差 ≤ 1 步；t_conf − t_open ≤ 1.5 s；mode 3 提交步晚於 t_conf。
  K3 段 A（t_open～t_conf）配對 ≤ 18 N；t_conf 之後至結束每步配對 ≤ 0.1 N、開度 ≥ 95 %。
  K4 5 N：恢復紀錄 ok、20→5；切換步晚於任務端 C_CLEAR 與 mode 3 提交，且 t_switch − t_conf ≥ 0.5 s；之後定期讀回皆 ok。
  K5 退離幾何（FK 重算，每 10 步）：
     - E＝H(q_rel)⁻¹·TCP；s_meas；|x| ≤ 3 mm、|z| ≤ Z(s)、轉角 ≤ 0.5°（相對 E_rel 沿 a_H 的路徑）；
     - 門角 Δ ∈ [−1°, 帶上界(s_meas)]（判定器不含預測；預測由線上監看）；max s_meas ≥ 0.095。
  K6 底盤：t_open 起相對 t_open 實測底盤位姿 ≤ 20 mm／2°（不是開門前 /park/gate 停位錨點：開門時底盤已協同移動）。K7 非設計接觸：非手指部件 bar／panel／ambiguous、手指 panel、牆皆 ≤ 0.1 N。
  K8 任務端：
     - 監看事件序 ARMED→NEAR→C_CLEAR→RETREAT_DONE→COMPLETE、無失敗；
     - COMPLETE − RETREAT_DONE ≥ 3 s；
     - C_CLEAR 證據 s ≥ 0.02、配對 ≥ 5 mm；rel done、eff_done 讀回 True；final_phase DONE。
  K9 求解端：w2_retreat 交接已發生、候選經實際套用才入歷史（≥ 1）；監督器 held_end＝handover。
  DONE 的意義：解除、退出近區、恢復 5 N、退離終點並保持 ≥ 3 s；**不是收臂**（收臂另包）。

    python3 evaluation/door_k3m_check_k3b.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_k3b.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_release2 as KR                         # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402
import door_k3m_check_w2c as W2C                       # noqa: E402
import door_w2_mode5 as M5                             # noqa: E402

K3B_REPLACED = tuple(W2C.W2_REPLACED)
R = KR.REG
FINGERS = ('uflite_finger1', 'uflite_finger2')


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _room(room):
    I = {c: i for i, c in enumerate((room or {}).get('steps_cols') or [])}
    return (room or {}).get('steps') or [], I


def check_multi(rw, why):
    auth = rw.get('auth') or {}
    cs = auth.get('commits') or []
    rows = rw.get('rows') or []
    by = {r.get('step'): r for r in rows}
    if [c.get('mode_seq') for c in cs] != [1, 2, 3] or any(c.get('applied_confirmed') is not True for c in cs):
        why.append('M1_commits_not_1_2_3_confirmed')
        return None
    for c in cs:
        r = by.get(c['step'])
        if r is None or r.get('src') != c['first_source_seq'] or r.get('mode_seq') != c['mode_seq'] or r.get('chain_mode') != 0:
            why.append(f"M1_commit_step_mismatch_{c['mode_seq']}")
    k1 = cs[0]['step']
    seg = [r for r in rows if r.get('step', -1) >= k1]
    st_ = [r['step'] for r in seg]
    if not seg or any(b - a != 1 for a, b in zip(st_, st_[1:])):
        why.append('M2_rows_not_contiguous')
    ms = [r['mode_seq'] for r in seg if isinstance(r.get('src'), int) and r['src'] >= 0]
    if any(b < a for a, b in zip(ms, ms[1:])):
        why.append('M2_mode_seq_backward')
    if any(r.get('profile') != M5.PROFILE_OF_SEQ5.get(r.get('auth_mode_seq')) or r.get('finger_target') != M5.FINGER_OF_SEQ5.get(r.get('auth_mode_seq'))
           for r in seg):
        why.append('M2_profile_or_finger_target_mismatch')
    k2 = cs[1]['step']
    ft = {r['step']: r.get('finger_target') for r in seg}
    if ft.get(k2 - 1) != 'closed' or ft.get(k2) != 'open':
        why.append('M2_finger_switch_not_at_mode2_commit')
    ar = {a.get('step') for a in (rw.get('auth_rows') or [])}
    if any(r['step'] not in ar for r in seg) or any(r.get('confirm_ok') is not True for r in seg):
        why.append('M2_confirm_coverage')
    if auth.get('aborted') is not None or auth.get('violations'):
        why.append('M2_aborted_or_violation')
    return {c['mode_seq']: c for c in cs}


def _pair_force(lc):
    if not isinstance(lc, dict):
        return None
    try:
        return [float((lc.get(n) or {}).get('bar', 0.0)) + float((lc.get(n) or {}).get('ambiguous', 0.0)) for n in FINGERS]
    except (TypeError, ValueError):
        return None


def check_exec_release(room, rw, cs, why, st):
    S, I = _room(room)
    rel = rw.get('release') or {}
    k_open, t_open = cs[2]['step'], cs[2]['t']
    ev = rel.get('events') or []
    e_open = next((e for e in ev if e.get('to') == 'A_OPENING'), None)
    e_conf = next((e for e in ev if e.get('to') == 'B_CONFIRMED'), None)
    if e_open is None or e_open.get('step') != k_open:
        why.append('K1_exec_open_step_ne_mode2_commit')
    after = [s for s in S if s[I['step']] > k_open]
    ok_t0, t_conf_rc, k_conf_rc = None, None, None
    fa_max, post_bad = 0.0, 0
    for s in after:
        fq, pf = s[I['q_finger']], _pair_force(s[I['leaf_contacts']])
        valid = isinstance(fq, list) and len(fq) == 2 and pf is not None
        t = s[I['sim_t']]
        if t_conf_rc is None:
            if valid and max(pf) > R['f_design_max_n']:
                why.append('K3_segment_A_force')
            if valid:
                fa_max = max(fa_max, max(pf))
            if valid and min(fq) >= R['open_frac'] * R['finger_open_m'] and max(pf) <= R['f_small_n']:
                ok_t0 = ok_t0 if ok_t0 is not None else t
                if t - ok_t0 >= R['confirm_hold_s'] - 1e-6:
                    t_conf_rc, k_conf_rc = t, s[I['step']]
            else:
                ok_t0 = None
        else:
            if not valid or max(pf) > R['f_small_n'] or min(fq) < R['open_frac'] * R['finger_open_m']:
                post_bad += 1
    if t_conf_rc is None:
        why.append('K2_t_conf_not_reached')
    else:
        if e_conf is None or abs(int(e_conf.get('step', -10)) - int(k_conf_rc)) > 1:
            why.append('K2_t_conf_mismatch_exec')
        if t_conf_rc - t_open > R['t_conf_max_s'] + 0.011:
            why.append('K2_t_conf_after_deadline')
        if cs[3]['step'] <= k_conf_rc:
            why.append('K2_mode3_before_t_conf')
    if post_bad:
        why.append(f'K3_after_confirm_force_or_closed_{post_bad}')
    st['k3b_exec'] = {'k_open': k_open, 't_open': t_open, 'k_conf_recomputed': k_conf_rc, 't_conf_recomputed': t_conf_rc,
                      'exec_conf_step': None if e_conf is None else e_conf.get('step'), 'segment_A_force_max_n': fa_max,
                      'n_after_confirm_bad': post_bad}
    return t_conf_rc


def check_effort(room, rw, task, cs, t_conf, why, st):
    eff = rw.get('effort') or {}
    rr = eff.get('restore') or {}
    mon = ((task or {}).get('k3b') or {}).get('monitor') or {}
    t_clear = (mon.get('t') or {}).get('C_CLEAR')
    if rr.get('ok') is not True or not (_fin(rr.get('from')) and abs(rr['from'] - 20.0) < 1e-6 and _fin(rr.get('to')) and abs(rr['to'] - 5.0) < 1e-6):
        why.append('K4_restore_record')
        return
    t_sw = rr.get('t')
    if not _fin(t_clear) or not _fin(t_sw) or t_sw + 1e-9 < t_clear or rr.get('step', -1) < cs[3]['step']:
        why.append('K4_restore_before_clear_or_retreat')
    if t_conf is None or t_sw - t_conf < R['restore_quiet_s'] - 1e-6:
        why.append('K4_restore_quiet_window')
    per = (((room or {}).get('finger_effort') or {}).get('pause') or {}).get('periodic') or []
    if any(p.get('ok') is not True for p in per if _fin(p.get('t')) and p['t'] > t_sw):
        why.append('K4_periodic_after_restore')
    st['k3b_effort'] = {'t_switch': t_sw, 't_clear': t_clear, 't_conf': t_conf, 'readback': rr.get('readback')}


def check_retreat_geom(room, cs, why, st, stride=10):
    import door_k2_check as KC
    S, I = _room(room)
    k_open = cs[2]['step']
    s0 = next((s for s in S if s[I['step']] == k_open), None)
    if s0 is None:
        why.append('K5_no_open_row')
        return
    q_rel = float(s0[I['door_q_rad']])
    Hr = KC._H_of(q_rel)
    E_rel = np.linalg.inv(Hr) @ KC._tcp(s0[I['base_xyth']], s0[I['q_arm_meas']])
    worst = {'x_mm': 0.0, 'z_mm': 0.0, 'rot_deg': 0.0, 'band_margin_deg': math.inf, 's_max': 0.0}
    bad = []
    for s in S[S.index(s0)::stride]:
        E = np.linalg.inv(Hr) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        sm = float((E[:3, 3] - E_rel[:3, 3]) @ KR.A_H)
        d = E[:3, 3] - (E_rel[:3, 3] + sm * KR.A_H)
        Rm = E[:3, :3] @ E_rel[:3, :3].T
        rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2))))
        dq = math.degrees(float(s[I['door_q_rad']]) - q_rel)
        up = KR.band_upper_deg(sm)
        worst.update(x_mm=max(worst['x_mm'], abs(d[0]) * 1e3), z_mm=max(worst['z_mm'], abs(d[2]) * 1e3), rot_deg=max(worst['rot_deg'], rot),
                     band_margin_deg=min(worst['band_margin_deg'], up - dq), s_max=max(worst['s_max'], sm))
        if abs(d[0]) > R['x_max_m'] or abs(d[2]) > KR.z_max_m(max(sm, 0.0)) or rot > R['rot_max_deg'] or dq < R['lower_deg'] or dq > up:
            bad.append((s[I['step']], round(sm, 4), round(dq, 4)))
    if bad:
        why.append(f'K5_geometry_or_band_{len(bad)}')
    if worst['s_max'] < R['s_done_m'] - 1e-6:
        why.append('K5_retreat_not_reached')
    st['k3b_geom'] = dict(worst, n_bad=len(bad), first_bad=bad[:5], q_rel_deg=math.degrees(q_rel), stride=stride)


def check_base_contacts(room, gate, cs, why, st):
    S, I = _room(room)
    k_open = cs[2]['step']
    s0 = next((s for s in S if s[I['step']] == k_open), None)                 # 錨點＝t_open 實測底盤（不是開門前 /park/gate 停位）
    anc = None if s0 is None else list(s0[I['base_xyth']])
    if anc is None:
        why.append('K6_anchor_missing')
    bmax, ymax, nbad = 0.0, 0.0, 0
    for s in S:
        if s[I['step']] < k_open:
            continue
        if anc is not None:
            b = s[I['base_xyth']]
            bmax = max(bmax, math.hypot(b[0] - anc[0], b[1] - anc[1]))
            ymax = max(ymax, abs(math.degrees(math.atan2(math.sin(b[2] - anc[2]), math.cos(b[2] - anc[2])))))
        lc = s[I['leaf_contacts']]
        wc = s[I['wall_contacts']]
        if not isinstance(lc, dict):
            nbad += 1
            continue
        for n, v in lc.items():
            x = float(v.get('panel', 0.0)) + float(v.get('ambiguous', 0.0)) + (0.0 if n in FINGERS else float(v.get('bar', 0.0)))
            if n in FINGERS:
                x = float(v.get('panel', 0.0))
            if x > R['unexpected_max_n']:
                nbad += 1
                break
        if isinstance(wc, dict) and any(float(v) > R['unexpected_max_n'] for v in wc.values()):
            nbad += 1
    if bmax > R['base_dev_m'] or ymax > R['base_dev_deg']:
        why.append('K6_base_anchor')
    if nbad:
        why.append(f'K7_non_design_contact_{nbad}')
    st['k3b_base'] = {'dev_mm': bmax * 1e3, 'yaw_deg': ymax, 'n_contact_bad': nbad}


def check_task(task, why, st):
    k = (task or {}).get('k3b') or {}
    mon, rel = k.get('monitor') or {}, k.get('rel') or {}
    seq = [e.get('to') for e in (mon.get('events') or []) if e.get('to') in ('ARMED', 'NEAR', 'C_CLEAR', 'RETREAT_DONE', 'COMPLETE', 'FAILED')]
    if seq != ['ARMED', 'NEAR', 'C_CLEAR', 'RETREAT_DONE', 'COMPLETE'] or mon.get('why') is not None:
        why.append('K8_monitor_sequence')
    tt = mon.get('t') or {}
    if not (_fin(tt.get('COMPLETE')) and _fin(tt.get('RETREAT_DONE')) and tt['COMPLETE'] - tt['RETREAT_DONE'] >= R['complete_hold_s'] - 1e-6):
        why.append('K8_complete_hold')
    ev = next((e for e in (mon.get('events') or []) if e.get('to') == 'C_CLEAR'), {})
    if not (_fin(ev.get('s_meas')) and ev['s_meas'] >= R['near_s_m'] and _fin(ev.get('pair')) and ev['pair'] >= R['pair_exit_m']):
        why.append('K8_clear_evidence')
    if rel.get('state') != 'done' or (rel.get('eff_done') or {}).get('readback_ok') is not True or (task or {}).get('final_phase') != 'DONE':
        why.append('K8_task_not_done')
    st['k3b_task'] = {'seq': seq, 't': tt, 'worst': mon.get('worst'), 'rel_t': rel.get('t')}


def check_solver(solver, why, st):
    s = (solver or {}).get('stats') or {}
    wr = s.get('w2_retreat') or {}
    o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
    if wr.get('handover') is None or (wr.get('n_hist_insert') or 0) < 1:
        why.append('K9_solver_handover_or_hist')
    if o.get('held_end') != 'handover':
        why.append('K9_supervisor_not_handed_over')
    st['k3b_solver'] = {'handover': wr.get('handover'), 'n_cand_pub': wr.get('n_cand_pub'), 'n_hist_insert': wr.get('n_hist_insert'),
                        'held_end': o.get('held_end')}


def check_k3b(task, room, gate, solver, st):
    why = []
    rw = (room or {}).get('w2') or {}
    cs = check_multi(rw, why)
    if cs is None:
        return why
    t_conf = check_exec_release(room, rw, cs, why, st)
    check_effort(room, rw, task, cs, t_conf, why, st)
    check_retreat_geom(room, cs, why, st)
    check_base_contacts(room, gate, cs, why, st)
    check_task(task, why, st)
    check_solver(solver, why, st)
    return why


def judge(task, room, gate, solver, observer):
    res = F2.judge(task, room, gate, solver, observer)
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if w in K3B_REPLACED]
    why_old = [w for w in why_all if w not in K3B_REPLACED]
    st = dict(res.get('stats') or {})
    why_k = check_k3b(task, room, gate, solver, st)
    why = why_old + why_k
    return dict(res, why=why, stats=st, judge='k3b', replaced_old_reasons=replaced, k3b_replaced_names=list(K3B_REPLACED),
                old_pass_excluding_replaced=not why_old, k3b_ok=not why_k, **{'pass': not why},
                semantics='DONE＝解除、退出近區、恢復 5 N、退離終點並保持 ≥ 3 s；不是收臂')


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_k3b.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3b_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
