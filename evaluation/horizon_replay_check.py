#!/usr/bin/env python3
"""H5 紀錄完整性與重播驗收（H1／H5 比較的前置關卡）。

只用趟次自己的紀錄：原本的 N、原紀錄的暖啟動、當輪的目標／協同／偏差。
缺欄位的輪次列為缺紀錄，**不猜值**。開迴路逐輪重解，不推進物理。

**覆蓋（整趟驗收的前提，任一不合即不通過）**
  C1  每一列都能歸類：求解列（有 sqp_stop／residual／n_sqp）或合法的非求解列
      （reason 屬於節點既定的等待／閘門理由，見 NON_SOLVE_REASONS）；其餘 = 無法歸類
  C2  每個求解列都有 solve_in 與全部重播欄位，形狀正確、數值有限
  C3  列數與節點自己的統計相符：求解列數 = stats.n_solve_calls（節點獨立計的求解器
      呼叫數）、published 列數 = stats.published、求解失敗列數 = stats.no_solution
      （抓整列被刪除，包括「求解接受但因過期／閘門未發布」的列）；統計缺項 = 不通過
  C4  每列 solver_N = args.N
  非求解列另列計數，**不算缺紀錄，也不參與重播**。

**驗收門檻**（事前固定，見 results/horizon_ablation/experiment_spec_C0.yaml）：
  A1  兩邊都接受的輪次：首筆命令差 |Δu0| / vmax（逐軸取最大）p95 ≤ 1e-3、max ≤ 1e-2
  A2  SQP 停止理由一致率 ≥ 95%
  A3  原解接受的輪次：原解與重播解的原單位限制殘差都有限且 ≤ r_tol
  A4  求解器接受判定（ok）一致率 = 100%。比的是求解器的 ok，**不是是否發布**
      （解被接受後仍可能因過期或健康閘門不發布，那屬於節點層，不在重播範圍）
  殘差 NaN 只在 qp_failed／no_accepted_candidate（核心本來就不算殘差）時允許。
另報：完整序列差、停止理由配對、計算時效（記錄值，非重播）。

    python3 evaluation/horizon_replay_check.py runs/<RUN> [--out x.json] [--max-cycles K]

離開碼：0 = 整趟通過；1 = 不通過；2 = 紀錄不足以驗收（無可用求解列）；
       3 = 只做了部分輪次（--max-cycles），**僅供診斷，永遠不算通過**。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)
import horizon_replay as HR                                      # noqa: E402

THRESH = {'A1_p95': 1e-3, 'A1_max': 1e-2, 'A2_stop_agree': 0.95,
          'A4_ok_agree': 1.0}
FIELDS = ('q_pred', 's_pred', 'u_prev', 'T_cyc', 'U_warm', 'U_sol',
          'arm_bias', 'solver_N')
SOLVE_KEYS = ('sqp_stop', 'residual', 'n_sqp')
# wgmpc_wg2_node.py 主迴圈中「沒有呼叫求解器」就記錄的理由（逐一對照原始碼）
NON_SOLVE_REASONS = {
    'chain_fail_latched', 'inflight_unavoidable_breach', 'sp_gate_failed',
    'sp_handshake_init', 'sp_gate_hold', 'snapshot_duplicate',
    'snapshot_not_newer', 'snapshot_not_paired', 'stale_before_solve',
    'no_u_prev'}
# 核心在這些理由下不計算殘差（res.max_residual 維持 NaN）
NAN_RESIDUAL_OK = {'qp_failed', 'no_accepted_candidate'}


def pct(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    return {'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
            'max': float(v.max()), 'n': int(len(v))}


def finite(x):
    try:
        a = np.asarray(x, float)
    except (TypeError, ValueError):
        return False
    return a.size > 0 and bool(np.all(np.isfinite(a)))


def classify(L):
    if 'solve_in' in L or any(k in L for k in SOLVE_KEYS):
        return 'solve'
    if L.get('reason') in NON_SOLVE_REASONS:
        return 'wait'
    return 'unknown'


def record_problems(L, N):
    """回傳這個求解列的問題清單（空 = 可重播）。"""
    si = L.get('solve_in')
    if not isinstance(si, dict):
        return ['missing:solve_in']
    p = [f'missing:{f}' for f in FIELDS if f not in si]
    if p:
        return p
    ok = bool(L.get('ok'))
    shapes = {'q_pred': (9,), 's_pred': (6,), 'u_prev': (9,), 'T_cyc': (16,),
              'arm_bias': (6,)}
    for f, sh in shapes.items():
        if not finite(si[f]) or np.asarray(si[f], float).shape != sh:
            p.append(f'bad:{f}')
    for f in ('U_warm', 'U_sol'):
        v = si[f]
        if v is None:
            if f == 'U_sol' and ok:
                p.append('missing:U_sol(接受輪)')
            continue
        if not finite(v) or np.asarray(v, float).shape != (N, 9):
            p.append(f'bad:{f}')
    if not isinstance(si['solver_N'], int) or si['solver_N'] != N:
        p.append('bad:solver_N')
    res = L.get('residual')
    if ok:
        if res is None or not finite(res):
            p.append('nonfinite:residual(接受輪)')
    elif res is not None and not finite(res) \
            and L.get('reason') not in NAN_RESIDUAL_OK:
        p.append('nonfinite:residual')
    if 'sqp_stop' not in L:
        p.append('missing:sqp_stop')
    return p


def check(run_dir, max_cycles=None):
    sol, ident = HR.load_run(run_dir)
    args = sol['args']
    N = int(args['N'])
    K = WholeBodyKinematics.from_urdf_file(args['urdf'])
    cfg0 = HR.base_cfg(args, ident)
    vmax = cfg0.vmax()
    r_tol = cfg0.r_tol
    need_coord = bool(args.get('coord_topic'))
    log = sol.get('log') or []
    stats = sol.get('stats') or {}
    kinds = collections.Counter(classify(L) for L in log)
    wait_reasons = collections.Counter(L.get('reason') for L in log
                                       if classify(L) == 'wait')
    unknown = [L.get('sim_t') for L in log if classify(L) == 'unknown']
    solve_rows = [L for L in log if classify(L) == 'solve']

    # ---- C3：與節點統計對帳（整趟；抓整列被刪）----
    n_pub = sum(1 for L in log if L.get('published') is True)
    n_fail = sum(1 for L in solve_rows if not L.get('ok'))
    # 外部關閉時，最後一輪可能已呼叫求解器、尚未寫入紀錄；節點以獨立計數報出
    # （n_solve_calls_unlogged）。只在 stop_why = external_shutdown 且 ≤ 1 時扣除，並明列。
    _unlogged = stats.get('n_solve_calls_unlogged') or 0
    _unlogged_ok = (stats.get('stop_why') == 'external_shutdown' and _unlogged in (0, 1))
    _calls = stats.get('n_solve_calls')
    stats_check = {
        'n_solve_calls': {'rows': len(solve_rows),
                          'stats': (None if _calls is None else
                                    _calls - (_unlogged if _unlogged_ok else 0)),
                          'unlogged_at_external_shutdown': (_unlogged if _unlogged_ok else None)},
        'published': {'rows': n_pub, 'stats': stats.get('published')},
        'no_solution': {'rows': n_fail, 'stats': stats.get('no_solution')},
    }
    c3_ok = (all(v['stats'] is not None and v['rows'] == v['stats']
                 for v in stats_check.values())
             and (_unlogged == 0 or _unlogged_ok))

    partial = bool(max_cycles)
    rows = solve_rows[:max_cycles] if partial else solve_rows
    problems = collections.Counter()
    problem_t = collections.defaultdict(list)
    usable = []
    for L in rows:
        p = record_problems(L, N)
        if need_coord and 'coord' not in L:
            p.append('missing:coord')
        for x in p:
            problems[x] += 1
            if len(problem_t[x]) < 5:
                problem_t[x].append(L.get('sim_t'))
        if not p:
            usable.append(L)

    out = {'run': os.path.basename(run_dir.rstrip('/')), 'N_args': N,
           'scope': ('PARTIAL（前 %d 個求解列，僅供診斷）' % max_cycles
                     if partial else 'FULL'),
           'rows': {'total': len(log), 'solve': kinds['solve'],
                    'wait_or_gate': kinds['wait'], 'unclassified': len(unknown)},
           'wait_reasons': dict(wait_reasons),
           'unclassified_sim_t': unknown[:5],
           'n_checked_solve_rows': len(rows), 'n_usable': len(usable),
           'record_problems': dict(problems),
           'record_problems_first_sim_t': dict(problem_t),
           'stats_reconcile': stats_check,
           'thresholds': THRESH, 'r_tol': r_tol}
    if not usable:
        out['verdict'] = 'INCOMPLETE'
        return out, 2

    du0, dU, stop_same, ok_same = [], [], [], []
    res_o, res_r, a3_bad = [], [], []
    ok_mismatch_t = []
    stop_pairs = collections.Counter()
    for L in usable:
        si = L['solve_in']
        rr, info = HR.replay_cycle(K, args, ident, L)
        ok_o = bool(L.get('ok'))
        ok_same.append(rr.ok == ok_o)
        if rr.ok != ok_o and len(ok_mismatch_t) < 5:
            ok_mismatch_t.append(L.get('sim_t'))
        stop_same.append(rr.sqp_stop_reason == L.get('sqp_stop'))
        stop_pairs[(L.get('sqp_stop'), rr.sqp_stop_reason)] += 1
        if ok_o:
            ro = float(L['residual'])
            rp = (float(rr.max_residual) if rr.max_residual is not None
                  else math.nan)
            res_o.append(ro)
            res_r.append(rp)
            if not (math.isfinite(ro) and math.isfinite(rp)
                    and ro <= r_tol and rp <= r_tol):
                a3_bad.append(L.get('sim_t'))
        if ok_o and rr.ok:
            Uo = np.asarray(si['U_sol'], float)
            d = float(np.max(np.abs(rr.u0 - Uo[0]) / vmax))
            du0.append(d)
            if rr.U.shape == Uo.shape:
                dU.append(float(np.max(np.abs(rr.U - Uo) / np.tile(vmax, (len(Uo), 1)))))
            else:
                dU.append(math.inf)
    a1 = pct(du0)
    a1_finite = a1 is not None and all(math.isfinite(x) for x in du0)
    a2 = float(np.mean(stop_same))
    a4 = float(np.mean(ok_same))
    passed = {
        'C1_all_rows_classified': not unknown,
        'C2_solve_rows_complete': not problems,
        'C3_stats_reconcile': c3_ok,
        'A1': (a1_finite and a1['p95'] <= THRESH['A1_p95']
               and a1['max'] <= THRESH['A1_max']),
        'A2': a2 >= THRESH['A2_stop_agree'],
        'A3': (len(res_o) > 0 and not a3_bad),
        'A4': a4 >= THRESH['A4_ok_agree'],
    }
    tm = [(L.get('timing_ms') or {}).get('total') for L in solve_rows]
    cw = [L.get('cycle_wall_ms') for L in solve_rows]
    out.update({
        'A1_du0_over_vmax': a1, 'A2_stop_agree': a2,
        'A3_residual_max_accepted': {
            'original': max(res_o) if res_o else None,
            'replay': max(res_r) if res_r else None,
            'n_bad': len(a3_bad), 'bad_first_sim_t': a3_bad[:5]},
        'A4_ok_agree': a4, 'ok_mismatch_first_sim_t': ok_mismatch_t,
        'full_sequence_diff_over_vmax': pct(dU),
        'stop_reason_pairs': {f'{a}->{b}': n for (a, b), n in stop_pairs.items()},
        'timing_recorded_ms': {
            'core_total': pct([x for x in tm if x is not None]),
            'cycle_wall': pct([x for x in cw if x is not None]),
            'note': ('紀錄值。cycle_wall 含排程等待，只能看控制週期是否退化，'
                     '不能直接當成紀錄的執行成本')},
        'passed': passed,
    })
    if partial:
        out['verdict'] = 'PARTIAL'
        return out, 3
    out['verdict'] = 'PASS' if all(passed.values()) else 'FAIL'
    return out, 0 if out['verdict'] == 'PASS' else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-cycles', type=int, default=None,
                    help='只核前 K 個求解列（僅診斷；結論為 PARTIAL，不會 PASS）')
    a = ap.parse_args()
    out, rc = check(a.run_dir, a.max_cycles)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=str)
    return rc


if __name__ == '__main__':
    sys.exit(main())
