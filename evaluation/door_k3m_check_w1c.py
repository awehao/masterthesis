#!/usr/bin/env python3
"""W1c 判定器（door_k3m_check_w1b.py r2 的版本化擴充；Codex 20261025_002129）：w1b 全部判準照用，另加**偏移估計保持證據**核對。

一、w1b：登錄核對（θ、over_goal、OPEN 逾時、走廊表、內帶、包絡、觀察器上限）、K3-M（θ）、SETTLE p2、主動保持停止 P1–P4、
   手臂 0.5 s 窗口、零長度 END_HOLD 修正、引導相位；登錄檔＝task.json w1b.params_path（W1c 登錄另含 solver_change）。
二、偏移估計保持（登錄 solver_change.offset_hold_on_free 為 True 時必查；缺證據一律不通過）：
  H1 align_solver stats.offset_hold 存在，sim_t 有限、d_frozen 為 6 個有限值、n_upd_before 為非負整數；
  H2 凍結時刻＝求解端首次套用 free 的那一輪：stats.k3m_mode.events 中第一筆 applied_free 的 sim_t ≤ 凍結 sim_t ≤ 其＋0.06 s；
  H3 凍結起（sim_t ≥ 凍結 sim_t）每一列 offset_interval：why 以 blocked: 開頭（offset_hold，或停止／失效／閘門等優先理由），
     沒有 used＝updated，d_after 逐元素等於 d_frozen（≤ 1e-12）；凍結前最後一筆 d_after＝d_frozen；
  H4 first_published 存在，publish_sim_t_exact ≥ 凍結 sim_t，d_hat_in＝d_frozen；凍結後已發布列都標 offset_hold；
  H5 報告 offset_free.n_updates＝n_upd_before（凍結後計數不再增加）。
r2（Codex 20261025_003515：保持證據誤收缺口）：
  H3' 凍結起**每個實際求解列**（帶 solve_in）都必須有 offset_interval，缺欄不得被篩掉；不帶 solve_in 的診斷列（例如暫停觀察）另列計數；
      已發布列必須是求解列；
  H6  實際使用值：每個求解列 solve_in.offset_d_hat＝d_frozen、solve_in.arm_bias＝α⊙d_frozen（α 取求解端 args.arm_ident 的辨識檔；讀不到 ⇒ 不通過）；
  H4' first_published 必須精確對上凍結後第一筆真實已發布列：slot、publish_sim_t_exact 相同，d_hat_in＝該列 solve_in.offset_d_hat＝d_frozen。

    python3 evaluation/door_k3m_check_w1c.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1c.json；不覆寫
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

TOL = 1e-12


def _fin6(v):
    return isinstance(v, (list, tuple)) and len(v) == 6 and all(W1B._num(x) for x in v)


def check_offset_hold(solver, why, st):
    s = (solver or {}).get('stats') or {}
    h = s.get('offset_hold')
    if not isinstance(h, dict) or not W1B._num(h.get('sim_t')) or not _fin6(h.get('d_frozen')) or \
            not (isinstance(h.get('n_upd_before'), int) and h['n_upd_before'] >= 0):
        why.append('offset_hold_missing_or_invalid')
        st['offset_hold_check'] = {'record': h}
        return False
    th, dz = float(h['sim_t']), [float(x) for x in h['d_frozen']]
    ok = True
    ev = ((s.get('k3m_mode') or {}).get('events') or [])
    tf = next((e.get('sim_t') for e in ev if e.get('applied_free')), None)
    if not W1B._num(tf) or not (tf - 1e-9 <= th <= tf + 0.06 + 1e-9):
        why.append('offset_hold_boundary_mismatch')
        ok = False
    log = [x for x in (solver.get('log') or []) if W1B._num(x.get('sim_t'))]
    rows = [x for x in log if isinstance(x.get('offset_interval'), dict)]
    pre = [x for x in rows if x['sim_t'] < th - 1e-9 and x['offset_interval'].get('d_after') is not None]
    if not pre or not _fin6(pre[-1]['offset_interval']['d_after']) or \
            max(abs(a - b) for a, b in zip(pre[-1]['offset_interval']['d_after'], dz)) > TOL:
        why.append('offset_hold_frozen_value_not_last_available')
        ok = False
    after = [x for x in log if x['sim_t'] >= th - 1e-9]
    solve_rows = [x for x in after if isinstance(x.get('solve_in'), dict)]
    diag_rows = [x for x in after if not isinstance(x.get('solve_in'), dict)]
    alpha = None
    try:
        alpha = [float(v) for v in json.load(open(((solver.get('args') or {}).get('arm_ident'))))['alpha']]
    except (TypeError, OSError, KeyError, ValueError):
        alpha = None
    if alpha is None or len(alpha) != 6:
        why.append('offset_hold_alpha_unavailable')
        ok = False
    bias_ref = None if alpha is None else [a_ * d_ for a_, d_ in zip(alpha, dz)]
    bad, bad_use = [], []
    for x in solve_rows:
        oi = x.get('offset_interval')
        da = None if not isinstance(oi, dict) else oi.get('d_after')
        if not isinstance(oi, dict) or oi.get('used') == 'updated' or not str(oi.get('why') or '').startswith('blocked:') or \
                not _fin6(da) or max(abs(a - b) for a, b in zip(da, dz)) > TOL:
            bad.append((x['sim_t'], None if not isinstance(oi, dict) else oi.get('used'), None if not isinstance(oi, dict) else oi.get('why')))
        si = x['solve_in']
        dh, bb = si.get('offset_d_hat'), si.get('arm_bias')
        if not _fin6(dh) or max(abs(a - b) for a, b in zip(dh, dz)) > TOL or bias_ref is None or not _fin6(bb) or \
                max(abs(a - b) for a, b in zip(bb, bias_ref)) > 1e-12:
            bad_use.append(x['sim_t'])
    pub_not_solve = [x['sim_t'] for x in after if x.get('published') and not isinstance(x.get('solve_in'), dict)]
    if not solve_rows or bad or pub_not_solve:
        why.append(f'offset_hold_not_held_{len(bad) + len(pub_not_solve)}' if solve_rows else 'offset_hold_no_rows_after_freeze')
        ok = False
    if bad_use:
        why.append(f'offset_hold_solve_input_mismatch_{len(bad_use)}')
        ok = False
    fp = h.get('first_published')
    first_pub = next((x for x in after if x.get('published') and W1B._num(x.get('publish_sim_t_exact'))), None)
    if not isinstance(fp, dict) or first_pub is None or fp.get('slot') != first_pub.get('slot') or \
            fp.get('publish_sim_t_exact') != first_pub.get('publish_sim_t_exact') or not _fin6(fp.get('d_hat_in')) or \
            not isinstance(first_pub.get('solve_in'), dict) or not _fin6(first_pub['solve_in'].get('offset_d_hat')) or \
            max(abs(a - b) for a, b in zip(fp['d_hat_in'], first_pub['solve_in']['offset_d_hat'])) > TOL or \
            max(abs(a - b) for a, b in zip(fp['d_hat_in'], dz)) > TOL:
        why.append('offset_hold_first_published_invalid')
        ok = False
    pubs = [x for x in after if x.get('published')]
    if not pubs or not all(x.get('offset_hold') for x in pubs):
        why.append('offset_hold_published_rows_unmarked')
        ok = False
    n_rep = ((solver.get('offset_free') or {}).get('n_updates'))
    if not (isinstance(n_rep, int) and n_rep == h['n_upd_before']):
        why.append('offset_hold_update_count_mismatch')
        ok = False
    st['offset_hold_check'] = {'freeze_sim_t': th, 'free_applied_sim_t': tf, 'n_solve_rows_after': len(solve_rows),
                               'n_diag_rows_after': len(diag_rows), 'n_bad': len(bad), 'n_bad_solve_input': len(bad_use),
                               'first_bad': bad[:5], 'n_updates_report': n_rep, 'n_upd_before': h['n_upd_before'], 'ok': ok}
    return ok


def judge_with(task, room, gate, solver, observer, params):
    res = W1B.judge_with(task, room, gate, solver, observer, params)
    why_h, st = [], dict(res['stats'])
    sc = params.get('solver_change') or {}
    if sc.get('offset_hold_on_free') is True:
        hold_ok = check_offset_hold(solver, why_h, st)
    else:
        why_h.append('w1c_solver_change_not_registered')
        hold_ok = False
    why = list(res['why']) + why_h
    return dict(res, why=why, stats=st, **{'pass': not why, 'offset_hold_ok': hold_ok})


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
    p = os.path.join(run, 'analysis', 'door_k3m_check_w1c.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop',
                                             'w1b_registration_ok', 'w1b_guidance_ok', 'offset_hold_ok', 'why')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
