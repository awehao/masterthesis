#!/usr/bin/env python3
"""W1d 判定器（door_k3m_check_w1c.py r2 的版本化擴充；Codex 20261025_010414）：w1c 全部判準照用（w1b 全部＋偏移估計保持 H1–H6），
另加**保持段伺服錨點一次性重設**證據核對（登錄 solver_change.pause_hold_reanchor 為 True 時必查；缺證據一律不通過）。

  A1 align_solver stats.pause_reanchor 存在，record.status＝applied、n_calls＝1；old／new 為 3 個有限值；
  A2 old＝k3m_pause.events 唯一一筆 stage＝hold 的 anchor；events 恰一筆 stage＝reanchor，其 anchor＝new、old_anchor＝old；
  A3 new＝重設來源步的實測底盤位姿：room_run 中 sim_t＝source_sim_t（≤ 1e-6 s）那一步的 base_xyth（≤ 1e-5，room_run 為 6 位小數）；
     age_s＝edge_sim_t − source_sim_t ∈ [0, max_age_s]；
  A4 求解紀錄恰一列帶 pause_reanchor_edge，該列 pause_state＝ZERO_PUB 且是第一列 ZERO_PUB、pause_anchor_used＝old（本輪已用舊錨點）；
  A5 effective.slot＞邊緣 slot；自 effective.slot 起每一列 pause_anchor_used 非空者都＝new，之前沒有任何列＝new；
  A6 first_published 精確對上 effective 之後第一筆真實已發布列（slot、pause_source_seq、publish_sim_t_exact）；
  A7 監控參考未移動：暫停期每列 park_anchor 恆定且＝park_gate.json gate.anchor；凍結 TCP 目標存在
     （求解端 k3m_pause.record.frozen_target_T 為 16 個有限值、只此一份）；d̂ 由 w1c H1–H6 核對（保持不受重設影響）。
  A9 求解端 args.pause_hold_reanchor 為 True 且 pause_reanchor_max_age_s＝登錄值（否則 w1d_solver_args_mismatch）。
  A8 承接與限制：暫停期每列 pause_published_feasible.ok；重設前後沒有發布值跳變以外的豁免（不另放寬）。

    python3 evaluation/door_k3m_check_w1d.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1d.json；不覆寫
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_k3m_check_w1b as W1B                       # noqa: E402
import door_k3m_check_w1c as W1C                       # noqa: E402

POSE_TOL = 1e-5
T_TOL = 1e-6


def _fin3(v):
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(W1B._num(x) for x in v)


def _close(a, b, tol):
    return _fin3(a) and _fin3(b) and max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def check_reanchor(room, gate, solver, why, st):
    s = (solver or {}).get('stats') or {}
    pr = s.get('pause_reanchor')
    r = (pr or {}).get('record') if isinstance(pr, dict) else None
    if not isinstance(r, dict) or r.get('status') != 'applied' or pr.get('n_calls') != 1 or \
            not _fin3(r.get('old_anchor')) or not _fin3(r.get('new_anchor')):
        why.append('reanchor_missing_or_not_applied')
        st['reanchor_check'] = {'record': pr}
        return False
    ok = True
    old, new = r['old_anchor'], r['new_anchor']
    ev = (s.get('k3m_pause') or {}).get('events') or []
    eh = [e for e in ev if e.get('stage') == 'hold']
    er = [e for e in ev if e.get('stage') == 'reanchor']
    if len(eh) != 1 or not _close(eh[0].get('anchor'), old, 0.0):
        why.append('reanchor_old_not_hold_anchor')
        ok = False
    if len(er) != 1 or not _close(er[0].get('anchor'), new, 0.0) or not _close(er[0].get('old_anchor'), old, 0.0):
        why.append(f'reanchor_event_count_{len(er)}' if len(er) != 1 else 'reanchor_event_mismatch')
        ok = False
    # A3 實測來源
    tsrc, tedge, mx = r.get('source_sim_t'), r.get('edge_sim_t'), pr.get('max_age_s')
    st_src = None
    try:
        I = {c: i for i, c in enumerate(room['steps_cols'])}
        if W1B._num(tsrc):
            st_src = next((x for x in room['steps'] if abs(float(x[1]) - float(tsrc)) <= T_TOL), None)
    except (TypeError, KeyError):
        I = None
    if st_src is None or not _close(st_src[I['base_xyth']], new, POSE_TOL):
        why.append('reanchor_new_not_measured_pose')
        ok = False
    if not (W1B._num(tedge) and W1B._num(tsrc) and W1B._num(mx) and 0.0 <= tedge - tsrc <= mx + 1e-9):
        why.append('reanchor_source_age_invalid')
        ok = False
    # A4 邊緣列
    log = [x for x in (solver.get('log') or []) if isinstance(x.get('slot'), int)]
    edge = [x for x in log if isinstance(x.get('pause_reanchor_edge'), dict)]
    zp = next((x for x in log if x.get('pause_state') == 'ZERO_PUB'), None)
    if len(edge) != 1 or zp is None or edge[0] is not zp or not _close(edge[0].get('pause_anchor_used'), old, 0.0):
        why.append('reanchor_edge_row_invalid')
        ok = False
    # A5 生效
    eff = r.get('effective') or {}
    es = edge[0]['slot'] if edge else None
    ef = eff.get('slot')
    if not (isinstance(ef, int) and es is not None and ef > es):
        why.append('reanchor_effective_not_after_edge')
        ok = False
        ef = None
    else:
        aft = [x for x in log if x['slot'] >= ef and x.get('pause_anchor_used') is not None]
        bef = [x for x in log if x['slot'] < ef and x.get('pause_anchor_used') is not None]
        if not aft or not all(_close(x['pause_anchor_used'], new, 0.0) for x in aft) or \
                any(_close(x['pause_anchor_used'], new, 0.0) for x in bef):
            why.append('reanchor_anchor_used_inconsistent')
            ok = False
    # A6 首筆已發布
    fp = r.get('first_published') or {}
    fpub = None if ef is None else next((x for x in log if x['slot'] >= ef and x.get('published')
                                         and W1B._num(x.get('publish_sim_t_exact'))), None)
    if fpub is None or fp.get('slot') != fpub['slot'] or fp.get('source_seq') != fpub.get('pause_source_seq') or \
            fp.get('publish_sim_t_exact') != fpub.get('publish_sim_t_exact'):
        why.append('reanchor_first_published_invalid')
        ok = False
    # A7 監控參考
    pa = [x.get('park_anchor') for x in log if 'pause_anchor_used' in x]
    ga = ((gate or {}).get('gate') or {}).get('anchor')
    if not pa or not all(_close(p, ga, 0.0) for p in pa):
        why.append('reanchor_park_anchor_moved')
        ok = False
    ft = ((s.get('k3m_pause') or {}).get('record') or {}).get('frozen_target_T')
    if not (isinstance(ft, list) and len(ft) == 16 and all(W1B._num(v) for v in ft)):                 # 4×4 列優先攤平
        why.append('reanchor_frozen_target_missing')
        ok = False
    # A8 發布可行
    bad = [x['slot'] for x in log if 'pause_published_feasible' in x and not (x['pause_published_feasible'] or {}).get('ok')]
    if bad:
        why.append(f'reanchor_published_infeasible_{len(bad)}')
        ok = False
    st['reanchor_check'] = {'old': old, 'new': new, 'shift': r.get('anchor_shift'), 'edge_slot': es, 'effective_slot': ef,
                            'source_sim_t': tsrc, 'edge_sim_t': tedge, 'first_published': fp, 'n_rows_with_anchor': len(pa), 'ok': ok}
    return ok


def judge_with(task, room, gate, solver, observer, params):
    res = W1C.judge_with(task, room, gate, solver, observer, params)
    why_r, st = [], dict(res['stats'])
    sc = params.get('solver_change') or {}
    if sc.get('pause_hold_reanchor') is True:
        ra_ok = check_reanchor(room, gate, solver, why_r, st)
        ag = (solver or {}).get('args') or {}
        if ag.get('pause_hold_reanchor') is not True or ag.get('pause_reanchor_max_age_s') != sc.get('pause_reanchor_max_age_s'):
            why_r.append('w1d_solver_args_mismatch')                                  # 求解端實際參數＝登錄
            ra_ok = False
    else:
        why_r.append('w1d_solver_change_not_registered')
        ra_ok = False
    why = list(res['why']) + why_r
    return dict(res, why=why, stats=st, **{'pass': not why, 'reanchor_ok': ra_ok})


def judge(task, room, gate, solver, observer):
    w = task.get('w1b') or {}
    p = w.get('params_path')
    if not p:
        return {'pass': False, 'why': ['w1b_params_missing'], 'stats': {}}
    pp = p if os.path.isabs(p) else os.path.join(WS, p)
    if hashlib.sha256(open(pp, 'rb').read()).hexdigest() != w.get('params_sha256'):
        return {'pass': False, 'why': ['w1b_params_sha_mismatch'], 'stats': {}}
    return judge_with(task, room, gate, solver, observer, json.load(open(pp)))


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_w1d.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop',
                                             'w1b_registration_ok', 'w1b_guidance_ok', 'offset_hold_ok', 'reanchor_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
