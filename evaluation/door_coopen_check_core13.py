#!/usr/bin/env python3
"""核心段判定器 r13（door_coopen_check_core12.py 的版本化擴充；Codex r36 §2 C：C2 接線可獨立核對；r12 保留不改）。
r12 全部判準照用。登錄 control_cost_change.closure_guidance.enabled 為 True 時另核 C2（任一不符 ⇒ 鏈理由 closure_*，鏈不通過）：
  Q1 登錄一致：task.json closure.enabled 為 True、cfg 與登錄逐鍵相同、module_sha256＝現行 door_coopen_closure_guidance2.py。
  Q2 每列 closure 欄位齊全（why ∈ {None, not_active, step_not_advanced, input_invalid}；o、o_prev 兩個有限值；ref_orig／ref_mod 三個有限值）。
  Q3 輸出界限（**以實際輸出 o 判**）：|o| ≤ c_max；|o − o_prev| ≤ rate_norm·dt（dt＝本列與上一列 trace 的模擬時間差）。
  Q4 連續：o_prev＝上一列 o；非 updated 列（why 非 None）o＝o_prev。
  Q5 updated 列：相位 ∈ {OPEN, SETTLE}；s＝smoothstep(實測門角)；步號嚴格大於上一筆 updated 列。
  Q6 參考：ref_orig＝CorridorRef(走廊表).pose(門角)（由判定器自走廊表重算）；ref_mod＝ref_orig＋(o, 0)（偏航不變）。
  Q7 已發布引導：trace guide.v_body＝door_wide_guidance.phase_coord(ShiftedTable(CorridorRef, o), 登錄 guidance cfg, 相位, 模式, 底盤位姿, 門角, q_rate)
     的 v_body（逐元素 1e-12；判定器重算，不信任務端自報）。
  Q8 無效輸入：連續 input_invalid 跨度 ≥ invalid_hold_s ⇒ 必有中止 closure_input_invalid；有該中止則必有對應跨度。
分報：o 最大範數、s 首次 > 0 與到 1 的門角、計數、每輪計算耗時 p50／p99／最大、Q7 最大差。

    python3 evaluation/door_coopen_check_core13.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core13.json；不覆寫
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core12 as J12                 # noqa: E402
import door_coopen_check_core8 as J8                   # noqa: E402
import door_coopen_closure_guidance2 as CG2            # noqa: E402
import door_k3m_corridor as K3MCOR                     # noqa: E402
import door_wide_guidance as WG                        # noqa: E402

TOL = 1e-12
WHY_OK = (None, 'not_active', 'step_not_advanced', 'input_invalid')


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _fin(v, n):
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(x) for x in v)


def check_closure(task, params, why, st):
    out, bad = {}, []

    def fail(tag, n=None):
        if (tag if n is None else f'{tag}_{n}') not in bad:
            bad.append(tag if n is None else f'{tag}_{n}')
    cgp = ((params or {}).get('control_cost_change') or {}).get('closure_guidance') or {}
    tc = (task or {}).get('closure') or {}
    cfg = cgp.get('cfg') or {}
    if tc.get('enabled') is not True or tc.get('cfg') != cfg or \
            tc.get('module_sha256') != hashlib.sha256(open(CG2.__file__, 'rb').read()).hexdigest():
        fail('closure_registration')                                                    # Q1
    try:
        c = CG2.ClosureCfg2(**{k: float(v) for k, v in cfg.items()})
    except TypeError:
        fail('closure_registration')
        why.extend(bad)
        st['closure_check'] = {'ok': False, 'why': bad}
        return False
    try:
        args = task.get('args') or {}
        cp = args['corridor_table'] if os.path.isabs(args['corridor_table']) else os.path.join(WS, args['corridor_table'])
        ref = WG.CorridorRef(K3MCOR.Corridor.from_json(json.load(open(cp))['table']))
        gcfg = WG.GuidanceCfg(**((task.get('w1b') or {}).get('guidance_cfg') or {}))
    except (KeyError, OSError, TypeError, ValueError):
        fail('closure_reference_unavailable')
        ref = gcfg = None
    rows = [r for r in (task.get('trace') or []) if isinstance(r.get('closure'), dict)]
    if not rows:
        fail('closure_records_missing')
    prev, prev_upd_step, inv_t0, inv_spans = None, None, None, []
    omax, dv_max, s_first, s_full = 0.0, 0.0, None, None
    for r in rows:
        cr = r['closure']
        w = cr.get('why')
        if w not in WHY_OK or not _fin(cr.get('o'), 2) or not _fin(cr.get('o_prev'), 2) or not _fin(cr.get('ref_orig'), 3) or not _fin(cr.get('ref_mod'), 3) \
                or not _num(r.get('t')) or not _num(r.get('q_door')):
            fail('closure_record_invalid')                                              # Q2
            prev = None
            continue
        o, op = np.asarray(cr['o'], float), np.asarray(cr['o_prev'], float)
        omax = max(omax, float(np.linalg.norm(o)))
        if float(np.linalg.norm(o)) > c.c_max_m + TOL:
            fail('closure_output_over_max')                                             # Q3
        if prev is not None:
            dt = float(r['t']) - float(prev['t'])
            if float(np.linalg.norm(o - op)) > c.rate_norm_m_s * max(dt, 0.0) + TOL:
                fail('closure_output_rate')
            if float(np.max(np.abs(op - np.asarray(prev['closure']['o'], float)))) > TOL:
                fail('closure_discontinuity')                                           # Q4
        if w is not None and float(np.max(np.abs(o - op))) > TOL:
            fail('closure_held_changed')
        if w is None:                                                                   # Q5
            if r.get('phase') not in CG2.GUIDE_PHASES:
                fail('closure_update_outside_phase')
            s = CG2.smoothstep(math.degrees(float(r['q_door'])), c.q_on_deg, c.q_full_deg)
            if not _num(cr.get('s')) or abs(float(cr['s']) - s) > TOL:
                fail('closure_weight_mismatch')
            if s > 0 and s_first is None:
                s_first = math.degrees(float(r['q_door']))
            if s >= 1.0 and s_full is None:
                s_full = math.degrees(float(r['q_door']))
            if not isinstance(cr.get('step'), int) or (prev_upd_step is not None and cr['step'] <= prev_upd_step):
                fail('closure_step_not_advanced')
            prev_upd_step = cr.get('step')
        if ref is not None:                                                             # Q6
            ro = ref.pose(float(r['q_door']))
            if max(abs(a - b) for a, b in zip(ro, cr['ref_orig'])) > 1e-9:
                fail('closure_ref_orig_mismatch')
            if abs(cr['ref_mod'][0] - cr['ref_orig'][0] - o[0]) > 1e-12 or abs(cr['ref_mod'][1] - cr['ref_orig'][1] - o[1]) > 1e-12 \
                    or cr['ref_mod'][2] != cr['ref_orig'][2]:
                fail('closure_ref_mod_mismatch')
            if gcfg is not None and _fin(cr.get('base_pose'), 3) and _num(cr.get('q_rate')):   # Q7
                coord, _, _ = WG.phase_coord(CG2.ShiftedTable(ref, o), gcfg, r.get('phase'), r.get('k3m_mode'), tuple(cr['base_pose']),
                                             float(r['q_door']), float(cr['q_rate']))
                vb = (r.get('guide') or {}).get('v_body')
                if not isinstance(vb, (list, tuple)) or len(vb) != 3:
                    fail('closure_guide_missing')
                else:
                    a_, b_ = np.asarray(coord[2:5], float), np.asarray(vb, float)
                    both_nan = np.isnan(a_) & np.isnan(b_)
                    d = np.where(both_nan, 0.0, np.abs(a_ - b_))
                    dv_max = max(dv_max, float(np.nanmax(np.where(np.isnan(d), np.inf, d))))
                    if np.any(np.isnan(d)) or float(np.max(d)) > TOL:
                        fail('closure_guide_mismatch')
            else:
                fail('closure_guide_inputs_missing')
        if w == 'input_invalid':                                                        # Q8
            inv_t0 = float(r['t']) if inv_t0 is None else inv_t0
        elif inv_t0 is not None:
            inv_spans.append((inv_t0, float(prev['t']) if prev is not None else inv_t0))
            inv_t0 = None
        prev = r
    if inv_t0 is not None and prev is not None:
        inv_spans.append((inv_t0, float(prev['t'])))
    long_inv = [s_ for s_ in inv_spans if s_[1] - s_[0] >= c.invalid_hold_s - 1e-9]
    ab = (task.get('abort') == 'closure_input_invalid') or any(e.get('abort') == 'closure_input_invalid' for e in (task.get('policy_events') or []))
    if bool(long_inv) != bool(ab):
        fail('closure_invalid_abort_mismatch')
    cs = [float(r['closure']['compute_s']) for r in rows if _num(r['closure'].get('compute_s'))]
    out.update(ok=not bad, why=bad, n_rows=len(rows), o_max_mm=omax * 1e3, s_first_deg=s_first, s_full_deg=s_full, counts=tc.get('counts'),
               guide_recompute_max_diff=dv_max, invalid_spans=inv_spans,
               compute_s=None if not cs else {'p50': float(np.percentile(cs, 50)), 'p99': float(np.percentile(cs, 99)), 'max': max(cs)})
    why.extend(bad)
    st['closure_check'] = out
    return not bad


def judge(task, room, gate, solver, observer, evidence_path=None, trace_path=None):
    res = J12.judge(task, room, gate, solver, observer, evidence_path, trace_path)
    p = J8._params(task) or {}
    enabled = (((p.get('control_cost_change') or {}).get('closure_guidance') or {}).get('enabled') is True)
    why, st = list(res['why']), dict(res.get('stats') or {})
    if enabled:
        check_closure(task, p, why, st)
    chain = not why
    return dict(res, why=why, stats=st, judge='coopen_core13_bench', closure_guidance_registered=enabled, chain_pass=chain, **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'), os.path.join(run, 'task_trace.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core13.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'iso_ok', 'bench_ok', 'closure_guidance_registered', 'why')}, ensure_ascii=False, default=float))
    print(json.dumps((res.get('stats') or {}).get('closure_check'), ensure_ascii=False, default=float)[:1500])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
