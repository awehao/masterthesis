#!/usr/bin/env python3
"""核心段判定器 r5（door_coopen_check_core4.py 的版本化延伸；Codex 20261027_011742／r13 §3-2）：r4（J1–J4）全部照用，另加**必要判準 J5 接近段成本切換**。
J5 證據：登錄檔（task.json coopen.params_path）approach_cost、task.json approach_cost（設定／統計）、任務 trace（<run>/task_trace.jsonl 每列 t、phase、acost）、
  求解紀錄 align_solver.json（args.w_a、args.coord_max_age_s、args.rate、coord.launch_values、log 每輪 sim_t 與 coord.{src, w_a}）、求解 stats（stop_rx 等）。
J5 不通過理由（任一 ⇒ 不通過；缺證據、錯值、越界沿用一律不通過；判定器不拋例外）：
  acost_reg_invalid      登錄 approach_cost 缺、鍵不符或值非 r7（w_a 0.01、phases ['APPROACH']、latch True、index 12、launch 0.05、deadline 0.55）。
  acost_task_mismatch    task.json approach_cost.config ≠ 登錄。
  acost_launch_mismatch  求解啟動 w_a ≠ launch_w_a，或 coord_max_age_s＋1/rate ≠ restore_deadline_s（1e-9）。
  acost_rows_missing     求解紀錄缺列、缺 coord 或 w_a 非有限數。trace 缺或無 phase。
  acost_not_reached      trace 從未進入 APPROACH（候選未受測）。
  acost_task_conflict    task 統計 n_conflict > 0，或 trace acost 與相位不符（APPROACH 且未離開 ⇒ 候選值；其他 ⇒ None）。
  acost_pre_approach     APPROACH 之前的求解輪 w_a ≠ 啟動值。
  acost_not_effective    APPROACH 期間沒有任何 src＝topic 且 w_a＝候選值的求解輪。
  acost_dropout          [t_A0 ＋ deadline, t_A1) 內有求解輪 w_a ≠ 候選值（候選中途失效）。
  acost_restore_late     t_A1 ＋ deadline 之後仍有求解輪 w_a ≠ 啟動值。
  acost_open_phase       任一求解輪所在任務相位（trace 中 ≤ 該 sim_t 的最新一列）∈ {OPEN, SETTLE, PAUSE} 而 w_a ≠ 啟動值。
  acost_restore_unverified 已離開 APPROACH，但 t_A1 ＋ deadline 之後沒有求解輪，且沒有停止證據（stats.stop_rx 存在且 n_published_after_stop_rx＝0）。
恢復狀態另報 restore_status：restored／stopped_no_restore_rows（中止後求解端已停發，不當正常恢復）／not_exited。

    python3 evaluation/door_coopen_check_core5.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core5.json；不覆寫
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core4 as J4C                  # noqa: E402

R7 = {'w_a': 0.01, 'phases': ['APPROACH'], 'latch_after_exit': True, 'coord_index': 12, 'launch_w_a': 0.05, 'restore_deadline_s': 0.55}
OPEN_PHASES = ('OPEN', 'SETTLE', 'PAUSE')
TOL = 1e-9


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def load_trace(path):
    if not path or not os.path.exists(path):
        return None
    rows = []
    for ln in open(path):
        try:
            d = json.loads(ln)
        except ValueError:
            return None
        r = d.get('row', d)
        if isinstance(r, dict) and _num(r.get('t')) and isinstance(r.get('phase'), str):
            rows.append(r)
    return rows


def check_acost(task, solver, trace, why):
    """回傳報告 dict；不通過理由附加到 why。"""
    rep = {'restore_status': None}
    try:
        co = (task or {}).get('coopen') or {}
        p = co.get('params_path')
        reg = json.load(open(os.path.join(WS, p))).get('approach_cost') if isinstance(p, str) and os.path.exists(os.path.join(WS, p)) else None
        if reg != R7:
            why.append('acost_reg_invalid')
            return rep
        if (((task or {}).get('approach_cost') or {}).get('config')) != reg:
            why.append('acost_task_mismatch')
            return rep
        cand, launch, ddl = float(reg['w_a']), float(reg['launch_w_a']), float(reg['restore_deadline_s'])
        a = (solver or {}).get('args') or {}
        lv = ((solver or {}).get('coord') or {}).get('launch_values') or {}
        if not (_num(a.get('w_a')) and float(a['w_a']) == launch and _num(lv.get('w_a')) and float(lv['w_a']) == launch
                and _num(a.get('coord_max_age_s')) and _num(a.get('rate')) and abs(float(a['coord_max_age_s']) + 1.0 / float(a['rate']) - ddl) <= TOL):
            why.append('acost_launch_mismatch')
            return rep
        log = (solver or {}).get('log')
        if not isinstance(log, list) or not log or trace is None or not trace:
            why.append('acost_rows_missing')
            return rep
        rows = []
        for x in log:
            c = x.get('coord') if isinstance(x, dict) else None
            if not (isinstance(c, dict) and _num(c.get('w_a')) and _num(x.get('sim_t'))):
                why.append('acost_rows_missing')
                return rep
            rows.append((float(x['sim_t']), float(c['w_a']), c.get('src')))
        tr = sorted(trace, key=lambda r: float(r['t']))
        tt = [float(r['t']) for r in tr]
        iA = next((i for i, r in enumerate(tr) if r['phase'] == 'APPROACH'), None)
        if iA is None:
            why.append('acost_not_reached')
            return rep
        tA0 = tt[iA]
        iX = next((i for i in range(iA, len(tr)) if tr[i]['phase'] != 'APPROACH'), None)
        tA1 = tt[iX] if iX is not None else None
        rep.update(t_A0=tA0, t_A1=tA1, exit_phase=None if iX is None else tr[iX]['phase'])
        st = (task or {}).get('approach_cost', {}).get('stats') or {}
        bad_tr = 0
        for i, r in enumerate(tr):
            exp = cand if (r['phase'] == 'APPROACH' and (iX is None or i < iX)) else None
            got = r.get('acost', 'missing')
            if got == 'missing' or (exp is None and got is not None) or (exp is not None and not (_num(got) and float(got) == exp)):
                bad_tr += 1
        rep['trace_acost_mismatch'] = bad_tr
        if bad_tr or not _num(st.get('n_conflict', None)) or int(st['n_conflict']) != 0:
            why.append('acost_task_conflict')

        def phase_at(t):
            j = bisect.bisect_right(tt, t + TOL) - 1
            return tr[j]['phase'] if j >= 0 else None
        hi = tA1 if tA1 is not None else float('inf')
        pre = [r for r in rows if r[0] < tA0 - TOL]
        inA = [r for r in rows if tA0 - TOL <= r[0] < hi - TOL]
        eff = [r for r in inA if r[2] == 'topic' and r[1] == cand]
        drop = [r for r in inA if r[0] >= tA0 + ddl - TOL and r[1] != cand]
        win = [r for r in rows if tA1 is not None and tA1 - TOL <= r[0] < tA1 + ddl - TOL]
        late = [r for r in rows if tA1 is not None and r[0] >= tA1 + ddl - TOL and r[1] != launch]
        after = [r for r in rows if tA1 is not None and r[0] >= tA1 + ddl - TOL]
        openp = [r for r in rows if phase_at(r[0]) in OPEN_PHASES and r[1] != launch]
        rep.update(n_rows=len(rows), n_pre=len(pre), n_pre_bad=sum(1 for r in pre if r[1] != launch), n_in_approach=len(inA), n_effective=len(eff),
                   effective_fraction=(len(eff) / len(inA)) if inA else None, first_effective_t=eff[0][0] if eff else None,
                   n_dropout=len(drop), dropout_t=[r[0] for r in drop][:20], n_exit_window=len(win), exit_window_w_a=[r[1] for r in win],
                   n_after_deadline=len(after), n_restore_late=len(late), n_open_phase_bad=len(openp),
                   first_launch_after_exit_t=next((r[0] for r in rows if tA1 is not None and r[0] >= tA1 - TOL and r[1] == launch), None))
        if rep['n_pre_bad']:
            why.append('acost_pre_approach')
        if not eff:
            why.append('acost_not_effective')
        if drop:
            why.append('acost_dropout')
        if late:
            why.append('acost_restore_late')
        if openp:
            why.append('acost_open_phase')
        if tA1 is None:
            rep['restore_status'] = 'not_exited'
        elif after:
            rep['restore_status'] = 'restored' if not late else 'late'
        else:
            s = (solver or {}).get('stats') or {}
            ok_stop = s.get('stop_rx') is not None and s.get('n_published_after_stop_rx') == 0
            rep['stop_evidence'] = {'stop_rx': s.get('stop_rx'), 'last_publish_sim_t': s.get('last_publish_sim_t'), 'n_published_after_stop_rx': s.get('n_published_after_stop_rx')}
            rep['restore_status'] = 'stopped_no_restore_rows' if ok_stop else 'unverified'
            if not ok_stop:
                why.append('acost_restore_unverified')
        return rep
    except Exception as e:                                  # 判定器不拋例外：轉為不通過理由
        why.append(f'acost_check_error:{type(e).__name__}')
        return rep


def judge(task, room, gate, solver, observer, evidence_path=None, trace_path=None):
    res = J4C.judge(task, room, gate, solver, observer, evidence_path)
    why_a = []
    acost = check_acost(task, solver, load_trace(trace_path), why_a)
    why = list(res.get('why') or []) + why_a
    chain = not why
    return dict(res, why=why, judge='coopen_core5', chain_pass=chain, approach_cost_ok=not why_a, approach_cost=acost, **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'), os.path.join(run, 'task_trace.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core5.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'offset_hold_ok',
                                             'finger_effort_ok', 'trace_log_ok', 'coopen_registration_ok', 'coopen_gate_registration_ok', 'approach_cost_ok', 'why')},
                     ensure_ascii=False, default=float))
    print(json.dumps({'approach_cost': res.get('approach_cost')}, ensure_ascii=False, default=float)[:2000])
    print(json.dumps({'core': res.get('core')}, ensure_ascii=False, default=float)[:3000])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
