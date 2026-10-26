#!/usr/bin/env python3
"""K3-B 整體判定 r5（Codex 20261026_034842 §1、§3）。door_k3m_check_k3b4 不改；本檔版本化組合。

一、舊判準鏈（door_k3m_check_w1f2.judge）改以**張爪前視圖**執行（prerelease_view；分段，不是廣泛刪除理由）：
  - 截斷點＝執行端 mode 2 **實際提交步** k2（room.w2.auth 的 commit 事件）；張爪前的夾持、開門、暫停、停止、保護、紀錄判準核到 k2 之前；
    k2 起由解除／退離判準（K3-B 項）承接。
  - room：steps 與 w2.rows／auth_rows 只留步號 < k2；W1e 定期讀回只留卸載起點 t0（mode 1 提交）之前（卸載起由分段證據 E 承接）。
  - task：trace 以來源步號 state_step < k2 截斷（無步號者以 t < t2 − TIME_Q）；視圖終點 t_end＝其最後一筆的時刻；trace、trace_log（另寫截斷版到暫存目錄）、policy_events 只留 ≤ t_end（其中若有 RELEASE／DONE／ABORTED 照留，由舊鏈照報）；
    n_target_published 與 loop_timing.n 改為截斷後一致的計數。真趟 final_phase＝DONE 時，視圖於 t_end 附加一筆 {'to': 'DONE', 'k3b5_view': 'prerelease_end'} 作為舊鏈的窗口終點
    （真趟 DONE 另由 K8b 要求，視圖不能取代）；真趟非 DONE 時不附加（舊鏈照報 termination）。
  - solver：stats.k3m_pause.supervisor_outcome 改用張爪前的對應視圖（state HELD、轉移只留 ≤ t2 者）；log 只留 sim_t < t2。
  - 只排除下列舊理由（REPLACED_OLD）：K3B_REPLACED 兩項（舊 DONE 理由）；W1e 的 finger_effort_periodic_invalid_*／
    finger_effort_periodic_short_*（20 N 讀回至卸載起點由 E2 核、斜坡與 5 N 由 E3／E4 核，均更嚴）。其餘舊理由一律保留。
二、K3-B 項：同 k3b4，另：
  - U 改 door_k3m_check_unload3（量化容差、U8 收斂重算交叉核對）；
  - RT5 釋放瞬態**獨立重算收斂**（door_k3b_transient2 規則逐 room 列；位移、速度、資料有效、收斂窗、持續時間、期限 1.5 s），
    交叉核對宣告：求解端 RELEASE_SETTLED ≥ 重算收斂時刻 − DECL_TOL；settled 至 mode 3 提交手臂 0.5 s 窗 ≤ 0.001 rad/s；
  - S9 求解端監督器版本 door_w2_mode9/r9（其餘同 S8）；X1c：求解端旗標 --w2-hold-r9（以 X1b 核，r8 欄位以 r9 值代入）。
任何例外 ⇒ 回理由，不以例外結束。

    python3 evaluation/door_k3m_check_k3b5.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_k3b5.json；不覆寫
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_transient2 as TT2                      # noqa: E402
import door_k3m_check_k3b as J1                        # noqa: E402
import door_k3m_check_k3b2 as J2                       # noqa: E402
import door_k3m_check_k3b3 as J3                       # noqa: E402
import door_k3m_check_k3b4 as J4                       # noqa: E402
import door_k3m_check_unload3 as CU3                   # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402

VERSION = 'door_k3m_check_k3b5/1'
K3B_REPLACED = J1.K3B_REPLACED
REPLACED_OLD_PREFIX = ('finger_effort_periodic_invalid_', 'finger_effort_periodic_short_')
REPLACED_CHECKS = dict(J4.REPLACED_CHECKS, **{'W1e finger_effort_periodic_invalid_*／short_*': 'E2／E3／E4（張爪前視圖只核到卸載起點）',
                                               'old chain window': 'prerelease_view（截到執行端 mode 2 實際提交步）'})
DECL_TOL = CU3.DECL_TOL
_fin = J4._fin
_guard = J4._guard


def _replaced(w):
    return w in K3B_REPLACED or w.startswith(REPLACED_OLD_PREFIX)


def prerelease_view(task, room, solver, tmpdir):
    ev = ((room.get('w2') or {}).get('auth') or {}).get('events') or []
    c2 = [e for e in ev if isinstance(e, dict) and e.get('kind') == 'commit' and e.get('mode_seq') == 2]
    c1 = [e for e in ev if isinstance(e, dict) and e.get('kind') == 'commit' and e.get('mode_seq') == 1]
    if len(c2) != 1 or len(c1) != 1 or type(c2[0].get('step')) is not int or not _fin(c2[0].get('t')) or not _fin(c1[0].get('t')):
        return None, 'view_commits_missing'
    k2, t2, t0 = c2[0]['step'], float(c2[0]['t']), float(c1[0]['t'])
    rv = dict(room)
    rv['steps'] = [s for s in room.get('steps') or [] if s[0] < k2]
    w2 = dict(room.get('w2') or {})
    for key in ('rows', 'auth_rows'):
        if isinstance(w2.get(key), list):
            w2[key] = [x for x in w2[key] if isinstance(x, dict) and isinstance(x.get('step'), int) and x['step'] < k2]
    rv['w2'] = w2
    fe = copy.deepcopy(room.get('finger_effort') or {})
    pz = fe.get('pause') or {}
    if isinstance(pz.get('periodic'), list):
        pz['periodic'] = [p for p in pz['periodic'] if isinstance(p, dict) and _fin(p.get('t')) and p['t'] < t0]
    rv['finger_effort'] = fe
    tv = dict(task)
    trv = [x for x in task.get('trace') or [] if isinstance(x, dict) and _fin(x.get('t'))
           and ((x['state_step'] < k2) if type(x.get('state_step')) is int else (x['t'] < t2 - CU3.TIME_Q))]   # 步號一致（不以浮點時刻比）
    t_end = max((x['t'] for x in trv), default=None)                   # 視圖終點＝提交前最後一筆任務 trace（舊鏈取窗末端須有列）
    if t_end is None:
        return None, 'view_trace_empty'
    pe = []
    for e in task.get('policy_events') or []:
        if not isinstance(e, dict) or not _fin(e.get('t')) or e['t'] > t_end + 1e-9:
            continue
        pe.append(e)                                                          # 視圖終點前的 RELEASE／DONE／ABORTED 照留（異常由舊鏈照報）
    if task.get('final_phase') == 'DONE':
        pe.append({'t': t_end, 'from': 'PAUSE', 'to': 'DONE', 'k3b5_view': 'prerelease_end'})
    tv['policy_events'] = pe
    tv['trace'] = trv
    tv['n_target_published'] = sum(1 for x in trv if x.get('pub_seq') is not None)      # 截斷後一致的計數
    tv['loop_timing'] = dict(task.get('loop_timing') or {}, n=len(trv))
    tl = dict(task.get('trace_log') or {})
    p = tl.get('path')
    if isinstance(p, str) and os.path.exists(p):
        q = os.path.join(tmpdir, 'trace_prerelease.jsonl')
        n = 0
        with open(p, encoding='utf-8') as fh, open(q, 'w', encoding='utf-8') as out:
            for line in fh:
                try:
                    row = json.loads(line).get('row') or {}
                except ValueError:
                    continue
                if _fin(row.get('t')) and row['t'] <= t_end + 1e-12:
                    out.write(line)
                    n += 1
        tl.update(path=q, n_enqueued=n, n_written=n)
        tv['trace_log'] = tl
    sv = dict(solver or {})
    st = dict(sv.get('stats') or {})
    kp = dict(st.get('k3m_pause') or {})
    o = dict(kp.get('supervisor_outcome') or {})
    if o:
        tr = {k: (v if (_fin(v) and v <= t_end) else None) for k, v in (o.get('transitions') or {}).items()}
        o.update(state='HELD', transitions=tr, held_end=None)
        kp['supervisor_outcome'] = o
        st['k3m_pause'] = kp
    sv['stats'] = st
    if isinstance(sv.get('log'), list):
        sv['log'] = [x for x in sv['log'] if isinstance(x, dict) and _fin(x.get('sim_t')) and x['sim_t'] <= t_end + 1e-9]
    return (tv, rv, sv), {'k2': k2, 't2': t2, 't0': t0, 't_view_end': t_end, 'n_steps': len(rv['steps']), 'n_trace': len(tv['trace']), 'view_done': task.get('final_phase') == 'DONE'}


def check_solver5(solver, why, st):
    s = (solver or {}).get('stats') or {}
    o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
    if o.get('version') != 'door_w2_mode9/r9':
        why.append('S9_supervisor_version')
        return o
    tr = o.get('transitions') or {}
    ks = ('UNLOAD', 'UNLOADED', 'RELEASE_OPEN', 'RELEASE_SETTLED', 'HANDOVER')
    if not all(_fin(tr.get(k)) for k in ks) or any(tr[a] > tr[b] + 1e-9 for a, b in zip(ks, ks[1:])):
        why.append('S9_transition_order')
    tm = o.get('transient') or {}
    if any((tm.get(k) or {}).get('state') != 'CONVERGED' or (tm.get(k) or {}).get('version') != TT2.VERSION for k in ('1', '2')):
        why.append('S9_transient_not_converged_or_version')
    if not _fin(o.get('t_settling')):
        why.append('S9_t_settling_missing')
    st['k3b5_solver'] = {'transitions': {k: tr.get(k) for k in ks}, 't_settling': o.get('t_settling'),
                         'transient': {k: (tm.get(k) or {}).get('worst') for k in ('1', '2')}}
    return o


def check_release_transient5(room, o, cs, why, st):
    S, I = J1._room(room)
    tr = (o or {}).get('transitions') or {}
    t_set, k2, t2, t3 = tr.get('RELEASE_SETTLED'), cs[2]['step'], cs[2]['t'], cs[3]['t']
    i0 = next((i for i, s in enumerate(S) if s[I['step']] == k2 - 1), None)
    i1 = next((i for i in range(len(S) - 1, -1, -1) if S[i][I['sim_t']] <= t3 + 1e-9), None)
    if not _fin(t_set) or i0 is None or i1 is None or i1 <= i0:
        why.append('RT5_rows_or_settled_missing')
        return
    state, w, tc, worst = CU3.recompute_transient(S, I, i0, i1, TT2.T_MAX_RELEASE)
    st['k3b5_release_transient'] = {'state': state, 'why': w, 't_conv': tc, 'worst': worst, 'declared': t_set, 't_open': t2}
    if state == 'FAILED':
        why.append(f'RT5_release_transient_{w}')
        return
    if state != 'CONVERGED':
        why.append('RT5_release_not_converged_before_retreat')
        return
    if t_set < tc - DECL_TOL:
        why.append('RT5_settled_declared_before_recomputed_convergence')
    if t_set - t2 > TT2.T_MAX_RELEASE + 1e-9:
        why.append('RT5_release_settled_late')
    r = TT2.REG_TRANSIENT
    bad = 0
    for s in S[i0:i1 + 1]:
        if s[I['sim_t']] < t_set:
            continue
        a = next((x for x in S[i0:i1 + 1] if s[I['sim_t']] - r['arm_win_s'] - 0.011 <= x[I['sim_t']] <= s[I['sim_t']] - r['arm_win_s'] + 1e-9), None)
        if a is not None:
            v = max(abs(x - y) for x, y in zip(s[I['q_arm_meas']], a[I['q_arm_meas']])) / (s[I['sim_t']] - a[I['sim_t']])
            bad += int(v > r['quiet_arm'] + 1e-9)
    if bad:
        why.append(f'RT5_arm_not_quiet_before_retreat_{bad}')


def check_k3b5(task, room, gate, solver, st):
    why = []
    rw = (room or {}).get('w2') or {}
    cs = _guard(J1.check_multi, 'K_multi', why, rw, why)
    if cs is None:
        return why
    _guard(J2.check_room_coverage, 'K0', why, room, cs, why, st)
    _guard(J1.check_exec_release, 'K1_K3', why, room, rw, cs, why, st)
    _guard(J4.check_effort_segments, 'E', why, room, why, st)
    g = _guard(J2.check_geom_steps, 'K5', why, room, cs, why, st)
    _guard(J2.check_exit_recompute, 'K10', why, room, task, g, why, st)
    _guard(J2.check_tracking, 'K11', why, task, g, why, st)
    _guard(J2.check_general_clearance, 'K12', why, room, cs, why, st)
    _guard(J3.check_end_hold_3d, 'K13', why, room, task, g, why, st)
    _guard(J1.check_base_contacts, 'K6_K7', why, room, gate, cs, why, st)
    _guard(J4.check_task4, 'K8b', why, task, why, st)
    _guard(J1.check_solver, 'K9', why, solver, why, st)
    o = _guard(check_solver5, 'S9', why, solver, why, st)
    _guard(check_release_transient5, 'RT5', why, room, o, cs, why, st)
    sa = dict((solver or {}).get('args') or {})
    sv = dict(solver or {}, args=dict(sa, w2_hold_r8=(sa.get('w2_hold_r9') is True)))           # X1c：r8 欄位以 r9 值代入
    _guard(J4.check_reg_id_time4, 'X', why, task, room, sv, cs, why, st)
    wu, su = CU3.check_unload3(room, task, o)
    why += wu
    st['k3b5_unload'] = su
    _guard(J2.anchors, 'anchors', why, room, gate, solver, cs, st)
    return why


def judge(task, room, gate, solver, observer):
    with tempfile.TemporaryDirectory(prefix='k3b5_view_') as td:
        view, meta = (None, None)
        try:
            view, meta = prerelease_view(task or {}, room or {}, solver or {}, td)
        except Exception as e:                                                   # noqa: BLE001
            meta = f'view_internal_error:{type(e).__name__}'
        if view is None:
            res = {'why': [f'OLD_view:{meta}'], 'stats': {}}
        else:
            try:
                res = F2.judge(view[0], view[1], gate, view[2], observer)
            except Exception as e:                                               # noqa: BLE001
                res = {'why': [f'OLD_internal_error:{type(e).__name__}'], 'stats': {}}
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if _replaced(w)]
    why_old = [w for w in why_all if not _replaced(w)]
    st = dict(res.get('stats') or {})
    st['k3b5_view'] = meta
    why_k = check_k3b5(task, room, gate, solver, st)
    why = why_old + why_k
    return dict(res, why=why, stats=st, judge='k3b5', judge_version=VERSION, replaced_old_reasons=replaced,
                replaced_old_rules={'names': list(K3B_REPLACED), 'prefixes': list(REPLACED_OLD_PREFIX)}, replaced_checks=REPLACED_CHECKS,
                old_pass_excluding_replaced=not why_old, k3b_ok=not why_k, **{'pass': not why},
                semantics='DONE＝閉爪卸載至 5 N（讀回確認）、解除、退出近區、5 N 讀回核對、退離終點並三維收斂保持 ≥ 3 s；不是收臂')


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_k3b5.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3b_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
