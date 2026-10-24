#!/usr/bin/env python3
"""W1d 判定器（door_k3m_check_w1c.py r2 的版本化擴充；Codex 20261025_010414）：w1c 全部判準照用（w1b 全部＋偏移估計保持 H1–H6），
另加**保持段伺服錨點一次性重設**證據核對（登錄 solver_change.pause_hold_reanchor 為 True 時必查；缺證據一律不通過）。

  A1 align_solver stats.pause_reanchor 存在，record.status＝applied、n_calls＝1；old／new 為 3 個有限值；
  A2 old＝k3m_pause.events 唯一一筆 stage＝hold 的 anchor；events 恰一筆 stage＝reanchor，其 anchor＝new、old_anchor＝old；
  A3 new＝重設來源步的實測底盤位姿：room_run 中 sim_t＝source_sim_t（≤ 5e-6 s）那一步的 base_xyth（≤ 1e-5，room_run 為 6 位小數）；
     age_s＝edge_sim_t − source_sim_t ∈ [0, max_age_s]；
  A4 求解紀錄恰一列帶 pause_reanchor_edge，該列 pause_state＝ZERO_PUB 且是第一列 ZERO_PUB、pause_anchor_used＝old（本輪已用舊錨點）；
  A5 effective.slot＞邊緣 slot；自 effective.slot 起每一列 pause_anchor_used 非空者都＝new，之前沒有任何列＝new；
  A6 first_published 精確對上 effective 之後第一筆真實已發布列（slot、pause_source_seq、publish_sim_t_exact）；
  A7 監控參考未移動：暫停期每列 park_anchor 恆定且＝park_gate.json gate.anchor；凍結 TCP 目標存在
     （求解端 k3m_pause.record.frozen_target_T 為 16 個有限值、只此一份）；d̂ 由 w1c H1–H6 核對（保持不受重設影響）。
  A9 求解端 args.pause_hold_reanchor 為 True 且 pause_reanchor_max_age_s＝登錄值（否則 w1d_solver_args_mismatch）。
  A8 承接與限制：暫停期每列 pause_published_feasible.ok；重設前後沒有發布值跳變以外的豁免（不另放寬）。

r2（Codex 20261025_012250：判定器誤收缺口，不改控制）：
  - 作用期覆蓋：自 applied_pause 那一輪起每個求解列（帶 solve_in）都必須有 pause_state、pause_anchor_used、park_anchor，缺欄不篩掉；
    hold 事件 slot 前（減速段）錨點必須為空，之後依 slot 判定：< effective＝old、≥ effective＝new（不靠數值是否相同，old＝new 合法）；
    已發布列 pause_published_feasible.ok 必須明確為 True，未發布求解列只容許 pause_no_publish；暫停期已發布列必須是求解列。
  - 來源：source_step.slot／snap_sim_t＝邊緣列 slot／sim_t、source_sim_t＝snap_sim_t（底盤來源以同輪快照時間定位，設定點步號只作註記）；
    邊緣列 pause_reanchor_edge 與 stats 紀錄逐欄相同；reanchor 事件在邊緣列；age_s 重算＝edge_sim_t − source_sim_t 且 ≤ 登錄上限、stats max_age＝登錄；
    edge_sim_t 落在邊緣列 [sim_t, publish_sim_t_exact]。
  - 生效：effective.slot／sim_t＝邊緣之後第一筆求解紀錄列；goal_u＝以新錨點、該列 room_run 實測位姿與求解端伺服參數重算的 park_hold_cmd（≤ 1e-4）。

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
import park_fixed as PF                                 # noqa: E402

POSE_TOL = 1e-5
T_TOL = 5e-6         # room_run sim_t 與求解紀錄 sim_t 實測差 1.0e-6（float32 時間）；物理步 0.01 s，不會配錯步
GOAL_TOL = 1e-4      # 伺服 k_p 2.0 × room_run 6 位小數位姿誤差 ≤ ~2e-5，留餘裕


def _fin3(v):
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(W1B._num(x) for x in v)


def _close(a, b, tol):
    return _fin3(a) and _fin3(b) and max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def _room_pose(room, t):
    """room_run 中 sim_t＝t（≤ T_TOL）那一步的 base_xyth；找不到 ⇒ None。"""
    try:
        I = {c: i for i, c in enumerate(room['steps_cols'])}
        if not W1B._num(t):
            return None
        st_ = next((x for x in room['steps'] if abs(float(x[1]) - float(t)) <= T_TOL), None)
        return None if st_ is None else st_[I['base_xyth']]
    except (TypeError, KeyError, IndexError):
        return None


def check_reanchor(room, gate, solver, why, st, max_age_reg):
    s = (solver or {}).get('stats') or {}
    pr = s.get('pause_reanchor')
    r = (pr or {}).get('record') if isinstance(pr, dict) else None
    if not isinstance(r, dict) or r.get('status') != 'applied' or pr.get('n_calls') != 1 or \
            not _fin3(r.get('old_anchor')) or not _fin3(r.get('new_anchor')):
        why.append('reanchor_missing_or_not_applied')
        st['reanchor_check'] = {'record': pr}
        return False
    ok = True

    def bad(tag):
        nonlocal ok
        why.append(tag)
        ok = False
    old, new = r['old_anchor'], r['new_anchor']
    log = [x for x in (solver.get('log') or []) if isinstance(x.get('slot'), int)]
    ev = (s.get('k3m_pause') or {}).get('events') or []
    e_pause = [e for e in ev if e.get('applied_pause')]
    eh = [e for e in ev if e.get('stage') == 'hold']
    er = [e for e in ev if e.get('stage') == 'reanchor']
    # A2 事件
    if len(eh) != 1 or not _close(eh[0].get('anchor'), old, 0.0) or not isinstance(eh[0].get('slot'), int):
        bad('reanchor_old_not_hold_anchor')
    if len(er) != 1 or not _close(er[0].get('anchor'), new, 0.0) or not _close(er[0].get('old_anchor'), old, 0.0):
        bad(f'reanchor_event_count_{len(er)}' if len(er) != 1 else 'reanchor_event_mismatch')
    # A4 邊緣列（與 stats、事件互相對照）
    edge = [x for x in log if isinstance(x.get('pause_reanchor_edge'), dict)]
    zp = next((x for x in log if x.get('pause_state') == 'ZERO_PUB'), None)
    E = edge[0] if len(edge) == 1 else None
    if E is None or zp is None or E is not zp or not _close(E.get('pause_anchor_used'), old, 0.0):
        bad('reanchor_edge_row_invalid')
    else:
        pe = E['pause_reanchor_edge']
        keys = ('edge_sim_t', 'source_step', 'source_sim_t', 'old_anchor', 'new_anchor', 'status', 'age_s')
        if any(pe.get(k) != r.get(k) for k in keys):
            bad('reanchor_edge_record_mismatch')
        if len(er) == 1 and (er[0].get('slot') != E['slot'] or not W1B._num(er[0].get('sim_t'))
                             or abs(er[0]['sim_t'] - E['sim_t']) > T_TOL):
            bad('reanchor_event_not_at_edge')
    # A3 來源：底盤來源以同輪快照時間（source_sim_t＝snap.t_base）定位；設定點步號只作註記、不當底盤步號
    ss, tsrc, tedge = r.get('source_step'), r.get('source_sim_t'), r.get('edge_sim_t')
    if not (isinstance(ss, dict) and E is not None and ss.get('slot') == E['slot'] and W1B._num(ss.get('snap_sim_t'))
            and abs(ss['snap_sim_t'] - E['sim_t']) <= T_TOL and W1B._num(tsrc) and abs(tsrc - ss['snap_sim_t']) <= T_TOL):
        bad('reanchor_source_not_edge_snapshot')
    if not _close(_room_pose(room, tsrc), new, POSE_TOL):
        bad('reanchor_new_not_measured_pose')
    mx = pr.get('max_age_s')
    age = (tedge - tsrc) if (W1B._num(tedge) and W1B._num(tsrc)) else None
    if age is None or not W1B._num(max_age_reg) or mx != max_age_reg or not (0.0 <= age <= max_age_reg + 1e-9) or \
            not W1B._num(r.get('age_s')) or abs(r['age_s'] - age) > 1e-9:
        bad('reanchor_source_age_invalid')
    if E is not None and W1B._num(tedge) and not (E['sim_t'] - 1e-9 <= tedge and
                                                  (not W1B._num(E.get('publish_sim_t_exact')) or tedge <= E['publish_sim_t_exact'] + 1e-9)):
        bad('reanchor_edge_time_out_of_cycle')
    # A5 生效：effective＝邊緣之後第一筆求解紀錄列（slot、sim_t），目標＝以新錨點對該列實測位姿重算的伺服命令
    eff = r.get('effective') or {}
    es = E['slot'] if E is not None else None
    nxt = None if es is None else next((x for x in log if x['slot'] > es), None)
    ef = eff.get('slot')
    if nxt is None or ef != nxt['slot'] or not W1B._num(eff.get('sim_t')) or abs(eff['sim_t'] - nxt['sim_t']) > T_TOL:
        bad('reanchor_effective_not_next_solve')
        ef = None
    servo = next((x['park_hold'].get('servo') for x in log if isinstance(x.get('park_hold'), dict)
                  and isinstance(x['park_hold'].get('servo'), dict)), None)
    gu = eff.get('goal_u')
    if ef is not None:
        try:
            sv = PF.HoldServo(**{k: float(servo[k]) for k in ('k_p', 'k_yaw', 'v_max', 'w_max')})
            pose = _room_pose(room, nxt['sim_t'])
            ref = PF.park_hold_cmd(new, pose, sv) if _fin3(pose) else None
        except (TypeError, KeyError, ValueError):
            ref = None
        if ref is None or not _fin3(gu) or max(abs(a - b) for a, b in zip(gu, ref)) > GOAL_TOL:
            bad('reanchor_effective_goal_mismatch')
    # A5／A7／A8 作用期覆蓋：自 applied_pause 那一輪起每個求解列都必須帶暫停證據；暫停觀察（無 solve_in）另計
    p0 = e_pause[0].get('slot') if e_pause else None
    hs = eh[0].get('slot') if len(eh) == 1 else None
    ga = ((gate or {}).get('gate') or {}).get('anchor')
    req = [] if p0 is None else [x for x in log if x['slot'] >= p0 and isinstance(x.get('solve_in'), dict)]
    n_miss = n_bad_anchor = n_bad_park = n_bad_feas = 0
    for x in req:
        if 'pause_state' not in x or 'pause_anchor_used' not in x or 'park_anchor' not in x:
            n_miss += 1
            continue
        a = x['pause_anchor_used']
        if hs is None or x['slot'] < hs:
            want_none = True
            okA = a is None                                                   # 減速段：合法空錨點
        else:
            want_none = False
            okA = _close(a, new if (ef is not None and x['slot'] >= ef) else old, 0.0)
        if not okA or (ef is None and not want_none):
            n_bad_anchor += 1
        if not _close(x['park_anchor'], ga, 0.0):
            n_bad_park += 1
        if x.get('published'):
            if (x.get('pause_published_feasible') or {}).get('ok') is not True:
                n_bad_feas += 1
        elif x.get('reason') not in ('pause_no_publish',):
            n_bad_feas += 1                                                   # 求解列未發布只容許監督器不發布
    pubs_out = [x for x in log if p0 is not None and x['slot'] >= p0 and x.get('published') and not isinstance(x.get('solve_in'), dict)]
    if p0 is None or not req or n_miss or pubs_out:
        bad(f'reanchor_pause_rows_missing_{n_miss + len(pubs_out)}' if req else 'reanchor_no_pause_rows')
    if n_bad_anchor:
        bad(f'reanchor_anchor_used_inconsistent_{n_bad_anchor}')
    if n_bad_park or not _fin3(ga):
        bad('reanchor_park_anchor_moved')
    if n_bad_feas:
        bad(f'reanchor_published_infeasible_{n_bad_feas}')
    # A6 首筆已發布
    fp = r.get('first_published') or {}
    fpub = None if ef is None else next((x for x in log if x['slot'] >= ef and x.get('published')
                                         and W1B._num(x.get('publish_sim_t_exact'))), None)
    if fpub is None or fp.get('slot') != fpub['slot'] or fp.get('source_seq') != fpub.get('pause_source_seq') or \
            fp.get('publish_sim_t_exact') != fpub.get('publish_sim_t_exact'):
        bad('reanchor_first_published_invalid')
    # A7 凍結 TCP 目標
    ft = ((s.get('k3m_pause') or {}).get('record') or {}).get('frozen_target_T')
    if not (isinstance(ft, list) and len(ft) == 16 and all(W1B._num(v) for v in ft)):          # 4×4 列優先攤平
        bad('reanchor_frozen_target_missing')
    st['reanchor_check'] = {'old': old, 'new': new, 'shift': r.get('anchor_shift'), 'edge_slot': es, 'effective_slot': ef,
                            'source_sim_t': tsrc, 'edge_sim_t': tedge, 'age_s': age, 'first_published': fp,
                            'n_required_rows': len(req), 'n_missing': n_miss, 'n_bad_anchor': n_bad_anchor,
                            'n_bad_park_anchor': n_bad_park, 'n_bad_feasible': n_bad_feas, 'ok': ok}
    return ok


def judge_with(task, room, gate, solver, observer, params):
    res = W1C.judge_with(task, room, gate, solver, observer, params)
    why_r, st = [], dict(res['stats'])
    sc = params.get('solver_change') or {}
    if sc.get('pause_hold_reanchor') is True:
        ra_ok = check_reanchor(room, gate, solver, why_r, st, sc.get('pause_reanchor_max_age_s'))
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
