#!/usr/bin/env python3
"""核心段判定器 r8（door_coopen_check_core7.py 的版本化修正；Codex r19 §1；r7 保留不改，iso_1 既有判定不回改）。
r7 缺口（Codex 核對 iso_1 逐步 leaf_contacts）：原始取證 fn 為負值（壓縮），r7 以 max(fn) > 0 判接觸 ⇒ 把 0.09 s 起 7.7–17.1 N 的 finger1—橫桿接觸報成零。
r8 修正：
  - 接觸以**力的大小** |fn| 判定：閉爪前任一接觸點 |fn| > 0 或 sep < 0 ⇒ iso_contact_before_close。報 pre_close_fn_abs_max、首次接觸步／時刻。
  - 與執行端聚合值對帳：閉爪前每步 room leaf_contacts 中手指對門扇各分類（bar／ambiguous／panel）力的最大值（執行端聚合）與同步證據 |fn| 的有無須一致；
    證據 |fn| > 0 而聚合為 0，或聚合 > 0 而證據無 |fn| > 0 ⇒ 記 n_reconcile_mismatch（> 0 ⇒ iso_contact_evidence_invalid）。
    聚合任一分類 > 0 本身亦算接觸（iso_contact_before_close）；分類值為 0 的條目不算接觸（r7 以「非空」計，會誤計）。
以下為 r7 原說明。

核心段判定器 r7（隔離驗證版；door_coopen_check_core6.py 的版本化修正；Codex r16 §1、§2；r6 保留不改）。
r6 缺口（Codex 直接呼叫重現）：前段 finger_contact_n＝None ⇒ max(None) 拋 TypeError；接觸 fn／sep＝NaN 被誤收；arm_sp 只有一個元素被 zip 截短而誤收。
r7＝r4（J1–J4）＋必要判準 J6 r2（不含 J5）。J6 r2 規則（任一不符 ⇒ 不通過；判定器不拋例外）：
  證據形式：
    - 六軸資料（登錄 qa_grasp、執行端 config.stow_q、room q_arm_meas、證據 arm_sp）須恰為 6 個有限數（非 bool）；否則 iso_six_axis_invalid。
    - 「有效查詢、沒有接觸點」＝證據 contacts 為 dict 且 contacts_err 為 None（空 dict 合法）；「沒有量到」＝contacts 缺／非 dict 或 contacts_err 非 None。
    - 生成至閉爪前（room 第一步至 ENGAGE 之前）的**每一個 room 步**都須有同步號的證據列且為有效查詢；每個接觸點 fn、sep 須為有限數；
      否則 iso_contact_evidence_invalid（缺步、未覆蓋、查詢錯誤、非有限一律拒收）。
    - room finger_contact_n：None 只在該步證據為有效查詢時接受（記 n_finger_unmeasured_supported）；非 None 須為 2 個有限數；否則 iso_contact_evidence_invalid。
      room leaf_contacts／wall_contacts 須為 dict（空＝無接觸）。
  判準：
    iso_reg_invalid            登錄缺 grasp_init，或 task.json grasp_init.config ≠ 登錄。
    iso_init_config_mismatch   config.stow_q 與 qa_grasp 差 > 1e-6。
    iso_init_meas_mismatch     第一筆 room 步 q_arm_meas 與 qa_grasp 差 > 0.02 rad。
    iso_setpoint_mismatch      第一個 wholebody 步：證據 arm_sp 與同步 q_arm_meas 差 > 0.02 rad。
    iso_phase_sequence         不含 UNFOLD → INIT_CHECK → ENGAGE 的先後，或出現 ALIGN_PRE／ALIGN_HOLD／APPROACH。
    iso_contact_before_close   閉爪前任一接觸點 fn > 0 或 sep < 0，或 finger_contact_n 任一 > 0，或 leaf／wall 非空。
    iso_confirm_invalid        閉爪前確認窗（任務 trace 的 init 欄位）不成立：取 init_check:pass 事件的 win_t0 與 ENGAGE 時刻 t_E，
                               [win_t0, t_E] 內 trace 列須每列 init 存在、valid 且 good、err_p_m < grip_tol 0.002、err_r_rad ≤ 登錄 approach_gate.rot_resume_rad、
                               win_t0 相同；相鄰列時間差 ≤ 0.2 s（stale_s）；t_E − win_t0 ≥ 0.5 s（grip_settle_s）；列數 ≥ 2。缺欄位或缺列一律不成立。
    iso_evidence_missing       room_run、evidence.jsonl、任務 trace 缺。
分項報告（不另成判準）：grasp_established、door_angle、exit_progress、core_evidence；clearance_min_sep_m（只涵蓋進入接觸偏移帶的對）。

    python3 evaluation/door_coopen_check_core8.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core8.json；不覆寫
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
GRIP_TOL_M = 0.002
GRIP_SETTLE_S = 0.5
STALE_S = 0.2
FAKE = ('ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH')


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def six(x):
    if not isinstance(x, (list, tuple)) or len(x) != 6 or not all(_num(v) for v in x):
        return None
    return [float(v) for v in x]


def _params(task):
    p = ((task or {}).get('coopen') or {}).get('params_path')
    if not (isinstance(p, str) and os.path.exists(os.path.join(WS, p))):
        return None
    return json.load(open(os.path.join(WS, p)))


def load_trace(path):
    if not path or not os.path.exists(path):
        return None
    rows = []
    for ln in open(path):
        d = json.loads(ln)
        r = d.get('row', d)
        if isinstance(r, dict) and _num(r.get('t')):
            rows.append(r)
    return sorted(rows, key=lambda r: float(r['t']))


def check_confirm(task, trace, rot_tol, why, rep):
    ev = [e for e in (task or {}).get('policy_events', []) if e.get('init_check') == 'pass']
    t_e = next((e['t'] for e in (task or {}).get('policy_events', []) if e.get('to') == 'ENGAGE'), None)
    if len(ev) != 1 or not _num(ev[0].get('win_t0')) or not _num(t_e):
        why.append('iso_confirm_invalid')
        rep['confirm'] = {'why': 'pass_event_or_engage_missing'}
        return
    w0 = float(ev[0]['win_t0'])
    rows = [r for r in (trace or []) if w0 - 1e-9 <= float(r['t']) <= float(t_e) + 1e-9]
    bad = []
    for r in rows:
        i = r.get('init')
        if not (isinstance(i, dict) and i.get('valid') is True and i.get('good') is True and _num(i.get('err_p_m')) and _num(i.get('err_r_rad'))
                and float(i['err_p_m']) < GRIP_TOL_M and float(i['err_r_rad']) <= rot_tol and _num(i.get('win_t0')) and abs(float(i['win_t0']) - w0) <= 1e-9):
            bad.append(float(r['t']))
    ts = [float(r['t']) for r in rows]
    gap = max([b - a for a, b in zip(ts, ts[1:])] or [math.inf])
    dur = float(t_e) - w0
    rep['confirm'] = {'win_t0': w0, 't_engage': float(t_e), 'dur_s': dur, 'n_rows': len(rows), 'max_gap_s': gap if rows else None, 'bad_rows_t': bad[:20]}
    if len(rows) < 2 or bad or gap > STALE_S + 1e-9 or dur < GRIP_SETTLE_S - 1e-9:
        why.append('iso_confirm_invalid')


def check_iso(task, room, evidence_path, why, trace=None):
    rep = {}
    try:
        prm = _params(task)
        reg = (prm or {}).get('grasp_init')
        if not isinstance(reg, dict) or ((task or {}).get('grasp_init') or {}).get('config') != reg:
            why.append('iso_reg_invalid')
            return rep
        qa = six(reg.get('qa_grasp'))
        rot_tol = ((prm or {}).get('approach_gate') or {}).get('rot_resume_rad')
        if qa is None or not _num(rot_tol):
            why.append('iso_reg_invalid')
            return rep
        events = (task or {}).get('policy_events', [])
        seq = [e['to'] for e in events if 'to' in e]
        rep['phase_sequence'] = seq
        ok_seq = all(p in seq for p in ('UNFOLD', 'INIT_CHECK', 'ENGAGE')) and seq.index('UNFOLD') < seq.index('INIT_CHECK') < seq.index('ENGAGE')
        if not ok_seq or any(p in seq for p in FAKE):
            why.append('iso_phase_sequence')
        t_eng = next((e['t'] for e in events if e.get('to') == 'ENGAGE'), None)
        rep['t_engage'] = t_eng
        if not isinstance(room, dict) or not room.get('steps') or not evidence_path or not os.path.exists(evidence_path) or trace is None:
            why.append('iso_evidence_missing')
            return rep
        check_confirm(task, trace, float(rot_tol), why, rep)
        c = {k: i for i, k in enumerate(room['steps_cols'])}
        sq = six((room.get('config') or {}).get('stow_q'))
        st = sorted(room['steps'], key=lambda s: s[c['step']])
        q0 = six(st[0][c['q_arm_meas']])
        if sq is None or q0 is None:
            why.append('iso_six_axis_invalid')
            return rep
        if max(abs(a - b) for a, b in zip(sq, qa)) > CLI_TOL:
            why.append('iso_init_config_mismatch')
        d0 = max(abs(a - b) for a, b in zip(q0, qa))
        rep['init_meas_dev_rad'] = d0
        if not d0 <= PARK_STOW_TOL:
            why.append('iso_init_meas_mismatch')
        ev = {}
        for ln in open(evidence_path):
            e = json.loads(ln)
            if isinstance(e.get('step'), int) and not isinstance(e.get('step'), bool):
                ev[e['step']] = e
        own = c['owner(0=nav,1=wb,2=glide)']
        wb0 = next((s for s in st if s[own] == 1), None)
        if wb0 is None or wb0[c['step']] not in ev:
            why.append('iso_evidence_missing')
        else:
            a_sp, q_wb = six(ev[wb0[c['step']]].get('arm_sp')), six(wb0[c['q_arm_meas']])
            if a_sp is None or q_wb is None:
                why.append('iso_six_axis_invalid')
            else:
                ds = max(abs(a - b) for a, b in zip(a_sp, q_wb))
                rep['setpoint_vs_meas_first_wb_rad'] = ds
                if not ds <= PARK_STOW_TOL:
                    why.append('iso_setpoint_mismatch')
        t_lim = float(t_eng) if _num(t_eng) else math.inf
        pre = [s for s in st if _num(s[c['sim_t']]) and float(s[c['sim_t']]) < t_lim]
        invalid, n_unmeas, fn_max, sep_min, fc_max, lw, n_pairs = [], 0, 0.0, math.inf, 0.0, 0, 0
        first_contact, first_agg, agg_max, n_recon = None, None, 0.0, 0
        for s in pre:
            k = s[c['step']]
            e = ev.get(k)
            ev_contact = False
            if e is None or not isinstance(e.get('contacts'), dict) or e.get('contacts_err') is not None:
                invalid.append((k, 'evidence_missing_or_query_error'))
                continue
            for lst in e['contacts'].values():
                if not isinstance(lst, list):
                    invalid.append((k, 'contact_list'))
                    continue
                for p in lst:
                    if not (isinstance(p, dict) and _num(p.get('fn')) and _num(p.get('sep'))):
                        invalid.append((k, 'contact_nonfinite'))
                        continue
                    n_pairs += 1
                    fa = abs(float(p['fn']))
                    fn_max, sep_min = max(fn_max, fa), min(sep_min, float(p['sep']))
                    ev_contact = ev_contact or fa > 0.0
                    if (fa > 0.0 or float(p['sep']) < 0.0) and first_contact is None:
                        first_contact = (k, float(s[c['sim_t']]), fa, float(p['sep']))
            fcn = s[c['finger_contact_n']]
            if fcn is None:
                n_unmeas += 1                                   # 同步證據為有效查詢 ⇒ 接受（已於上方核對）
            elif isinstance(fcn, (list, tuple)) and len(fcn) == 2 and all(_num(v) for v in fcn):
                fc_max = max(fc_max, max(float(v) for v in fcn))
            else:
                invalid.append((k, 'finger_contact_n'))
            if not (isinstance(s[c['leaf_contacts']], dict) and isinstance(s[c['wall_contacts']], dict)):
                invalid.append((k, 'leaf_wall_not_dict'))
                continue
            bars = [x for v in s[c['leaf_contacts']].values() for x in (v.values() if isinstance(v, dict) else [v])]   # bar／ambiguous／panel 全部分類
            if not all(_num(b) for b in bars):
                invalid.append((k, 'leaf_bar_nonfinite'))
                continue
            agg = max([float(b) for b in bars], default=0.0)
            agg_max = max(agg_max, agg)
            if agg > 0.0 and first_agg is None:
                first_agg = (k, float(s[c['sim_t']]), agg)
            if (agg > 0.0) != ev_contact:
                n_recon += 1
            if any(float(b) > 0.0 for b in bars) or s[c['wall_contacts']]:
                lw += 1
        rep.update(pre_close_steps=len(pre), pre_close_evidence_pairs=n_pairs, pre_close_fn_abs_max=fn_max,
                   first_contact_evidence=first_contact, first_contact_aggregate=first_agg, pre_close_aggregate_bar_max=agg_max,
                   n_reconcile_mismatch=n_recon,
                   clearance_min_sep_m=None if sep_min == math.inf else sep_min, pre_close_finger_contact_n_max=fc_max,
                   n_finger_unmeasured_supported=n_unmeas, pre_close_leaf_wall_steps=lw, contact_evidence_invalid=invalid[:20],
                   n_contact_evidence_invalid=len(invalid))
        if invalid or not pre or n_recon:
            why.append('iso_contact_evidence_invalid')
        if fn_max > 0.0 or sep_min < 0.0 or fc_max > 0.0 or lw:
            why.append('iso_contact_before_close')
        return rep
    except Exception as e:                                  # 判定器不拋例外：轉為不通過理由
        why.append(f'iso_check_error:{type(e).__name__}')
        return rep


def judge(task, room, gate, solver, observer, evidence_path=None, trace_path=None):
    res = J4C.judge(task, room, gate, solver, observer, evidence_path)
    why_i = []
    iso = check_iso(task, room, evidence_path, why_i, load_trace(trace_path))
    why = list(res.get('why') or []) + why_i
    seq = iso.get('phase_sequence') or []
    chain_why = list(res.get('why') or [])
    grasp = ('ENGAGE' in seq and 'HOLD' in seq and not any(w.startswith(('grasp_', 'finger_', 'fingers_')) for w in chain_why))
    qmax = reg_theta = None
    try:
        c = {k: i for i, k in enumerate(room['steps_cols'])}
        qmax = max(math.degrees(abs(float(s[c['door_q_rad']]))) for s in room['steps'] if _num(s[c['door_q_rad']]))
    except Exception:
        pass
    try:
        reg_theta = float(_params(task)['theta_goal_deg'])
    except Exception:
        pass
    items = {'grasp_established': grasp, 'door_angle': {'q_door_max_meas_deg': qmax, 'theta_goal_deg': reg_theta},
             'exit_progress': ((res.get('core') or {}).get('exit_progress_open')), 'core_evidence': (res.get('core') or {}).get('core_evidence')}
    chain = not why
    return dict(res, why=why, judge='coopen_core8_iso', chain_pass=chain, iso_ok=not why_i, iso=iso, items=items,
                scope='指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證（跳過自主接近）；chain_pass 不等於邊開邊繞成功', **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'), os.path.join(run, 'task_trace.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core8.json')
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
