#!/usr/bin/env python3
"""核心段判定器 r6（隔離驗證版；Howard 2026-10-27／Codex r15 補充裁定）：r4（J1–J4）全部照用，另加**必要判準 J6 指定初態**，並分項報告。
**不含 r5 的 J5**（本版不用接近段成本候選）。判定器不得以虛構的 APPROACH／ENGAGE 相序補成自主接近成功。
J6 不通過理由（任一 ⇒ 不通過；判定器不拋例外）：
  iso_reg_invalid            登錄（task.json coopen.params_path）缺 grasp_init，或 task.json grasp_init.config ≠ 登錄。
  iso_init_config_mismatch   執行端 room_run config.stow_q 與登錄 qa_grasp 差 > 1e-6 rad（CLI 六位小數）。
  iso_init_meas_mismatch     第一筆 room 步 q_arm_meas 與 qa_grasp 差 > PARK_STOW_TOL 0.02 rad（執行端停車閘門同一容差）。
  iso_setpoint_mismatch      第一筆 wholebody 步：證據 arm_sp 與同步 q_arm_meas 差 > 0.02 rad（設定點與實測不一致）。
  iso_phase_sequence         policy_events 的 'to' 序列不含 UNFOLD → INIT_CHECK → ENGAGE 的先後；或出現 ALIGN_PRE／ALIGN_HOLD／APPROACH（不得偽造接近）。
  iso_contact_before_close   ENGAGE 之前（含生成）任一證據接觸對 fn > 0 或 sep < 0（穿透），或 room finger_contact_n > 0、leaf／wall 接觸非空。
  iso_evidence_missing       room_run、evidence.jsonl 或必要欄位缺。
分項報告（不另成判準；整體通過仍由 chain_pass＝J1–J4＋J6 決定）：
  grasp_established：ENGAGE → HOLD 發生且 chain 理由中無 grasp_*／finger_* 類；door_angle：實測門角最大值與 θ_goal；exit_progress：core（core2 既有出口進展欄位）。
  clearance_min_sep_m：ENGAGE 之前證據接觸對 sep 最小值（只涵蓋進入接觸偏移帶的對，非完整淨空）。

    python3 evaluation/door_coopen_check_core6.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core6.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core4 as J4C                  # noqa: E402

PARK_STOW_TOL = 0.02
CLI_TOL = 1e-6
FAKE = ('ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH')


def _reg(task):
    p = ((task or {}).get('coopen') or {}).get('params_path')
    if not (isinstance(p, str) and os.path.exists(os.path.join(WS, p))):
        return None
    return json.load(open(os.path.join(WS, p))).get('grasp_init')


def check_iso(task, room, evidence_path, why):
    rep = {}
    try:
        reg = _reg(task)
        if not isinstance(reg, dict) or ((task or {}).get('grasp_init') or {}).get('config') != reg:
            why.append('iso_reg_invalid')
            return rep
        qa = [float(v) for v in reg['qa_grasp']]
        seq = [e['to'] for e in (task or {}).get('policy_events', []) if 'to' in e]
        rep['phase_sequence'] = seq
        ok_seq = all(p in seq for p in ('UNFOLD', 'INIT_CHECK', 'ENGAGE')) and seq.index('UNFOLD') < seq.index('INIT_CHECK') < seq.index('ENGAGE')
        if not ok_seq or any(p in seq for p in FAKE):
            why.append('iso_phase_sequence')
        t_eng = next((e['t'] for e in (task or {}).get('policy_events', []) if e.get('to') == 'ENGAGE'), None)
        rep['t_engage'] = t_eng
        if not isinstance(room, dict) or not room.get('steps') or not evidence_path or not os.path.exists(evidence_path):
            why.append('iso_evidence_missing')
            return rep
        c = {k: i for i, k in enumerate(room['steps_cols'])}
        sq = (room.get('config') or {}).get('stow_q')
        if not (isinstance(sq, list) and len(sq) == 6 and max(abs(float(a) - b) for a, b in zip(sq, qa)) <= CLI_TOL):
            why.append('iso_init_config_mismatch')
        st = sorted(room['steps'], key=lambda s: s[c['step']])
        d0 = max(abs(float(a) - b) for a, b in zip(st[0][c['q_arm_meas']], qa))
        rep['init_meas_dev_rad'] = d0
        if not d0 <= PARK_STOW_TOL:
            why.append('iso_init_meas_mismatch')
        own = c['owner(0=nav,1=wb,2=glide)']
        wb0 = next((s for s in st if s[own] == 1), None)
        sp = {}
        fn_max, sep_min, n_pairs = 0.0, math.inf, 0
        t_lim = t_eng if t_eng is not None else math.inf
        for ln in open(evidence_path):
            e = json.loads(ln)
            if isinstance(e.get('step'), int) and isinstance(e.get('arm_sp'), list):
                sp[e['step']] = e['arm_sp']
            if isinstance(e.get('t'), (int, float)) and e['t'] < t_lim and isinstance(e.get('contacts'), dict):
                for lst in e['contacts'].values():
                    for k in lst or []:
                        n_pairs += 1
                        fn_max = max(fn_max, float(k.get('fn', 0.0)))
                        sep_min = min(sep_min, float(k.get('sep', math.inf)))
        if wb0 is None or wb0[c['step']] not in sp:
            why.append('iso_evidence_missing')
        else:
            ds = max(abs(float(a) - float(b)) for a, b in zip(sp[wb0[c['step']]], wb0[c['q_arm_meas']]))
            rep['setpoint_vs_meas_first_wb_rad'] = ds
            if not ds <= PARK_STOW_TOL:
                why.append('iso_setpoint_mismatch')
        pre = [s for s in st if s[c['sim_t']] < t_lim]
        fc = max((max(s[c['finger_contact_n']]) for s in pre), default=0.0)
        lw = sum(1 for s in pre if s[c['leaf_contacts']] or s[c['wall_contacts']])
        rep.update(pre_close_evidence_pairs=n_pairs, pre_close_fn_max=fn_max, clearance_min_sep_m=None if sep_min == math.inf else sep_min,
                   pre_close_finger_contact_n_max=fc, pre_close_leaf_wall_steps=lw)
        if fn_max > 0.0 or sep_min < 0.0 or fc > 0.0 or lw:
            why.append('iso_contact_before_close')
        return rep
    except Exception as e:                                  # 判定器不拋例外：轉為不通過理由
        why.append(f'iso_check_error:{type(e).__name__}')
        return rep


def judge(task, room, gate, solver, observer, evidence_path=None):
    res = J4C.judge(task, room, gate, solver, observer, evidence_path)
    why_i = []
    iso = check_iso(task, room, evidence_path, why_i)
    why = list(res.get('why') or []) + why_i
    seq = iso.get('phase_sequence') or []
    chain_why = list(res.get('why') or [])
    grasp = ('ENGAGE' in seq and 'HOLD' in seq and not any(w.startswith(('grasp_', 'finger_', 'fingers_')) for w in chain_why))
    qmax = None
    try:
        c = {k: i for i, k in enumerate(room['steps_cols'])}
        qmax = max(math.degrees(abs(float(s[c['door_q_rad']]))) for s in room['steps'])
    except Exception:
        pass
    reg_theta = None
    try:
        p = (task.get('coopen') or {}).get('params_path')
        reg_theta = float(json.load(open(os.path.join(WS, p)))['theta_goal_deg'])
    except Exception:
        pass
    items = {'grasp_established': grasp, 'door_angle': {'q_door_max_meas_deg': qmax, 'theta_goal_deg': reg_theta},
             'exit_progress': ((res.get('core') or {}).get('exit_progress_open')), 'core_evidence': (res.get('core') or {}).get('core_evidence')}
    chain = not why
    return dict(res, why=why, judge='coopen_core6_iso', chain_pass=chain, iso_ok=not why_i, iso=iso, items=items,
                scope='指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證（跳過自主接近）', **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core6.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'iso_ok', 'k3m_pass', 'coopen_registration_ok', 'coopen_gate_registration_ok', 'why')},
                     ensure_ascii=False, default=float))
    print(json.dumps({'iso': res.get('iso'), 'items': res.get('items')}, ensure_ascii=False, default=float)[:3000])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
