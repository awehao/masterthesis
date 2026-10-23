#!/usr/bin/env python3
"""K3-A-B 判定器（door_k3a_check.py 的版本化擴充；K3-A 檔不改）。規格 K3_finger_balance_compliance_spec.md draft-2 §7＋Codex 20261023。

K3-A 判準照用（door_k3a_check.judge），但參考角／純鉸鏈旋轉／END_HOLD 固定等目標核對改在**基礎目標**（trace base_target）上做；
實際 TCP 漂移、接觸、腕力等安全判定仍用 room 實測與原始 G_T_H_grasp（不以基礎目標代替）。另做：
  1. 發布＝基礎＋柔順位移：每列旋轉相等（≤ 1e-9）、e12_world＝R_base·(0, −1, 0)（≤ 1e-9）、平移差＝delta_m·e12_world（≤ 1e-9）、|Δ| ≤ 1 mm。
  2. 時序（依 ph0＝該輪開始時相位）：首個 ph0＝OPEN 列以前 Δ＝0（精確）；首個 OPEN 列 Δ＝0、**實際發布目標**＝前一筆已發布目標（≤ 1e-9）；
     END_HOLD 列實際發布目標＝進入前最後一筆已發布目標（精確）、Δ 不變、不消費；ph0＝DONE 不發布。
  3. 調整律獨立重算（不匯入策略程式；常數為登錄值，並核 task config 相符）：
     - balance.s0＝balance_start 事件 s0＝首個 OPEN 列 state_step；逐步紀錄步號自 s0＋1 連續；
     - 每筆 f1／f2 與 room 同一步 leaf_contacts 的 finger1／finger2 'bar' 相同（≤ 1e-9）、sim_t 相同（≤ 1e-9）；
     - 依步序重算 ẽ 與 Δ（≤ 1e-12），速率界與上限同時核對；
     - 每列 bal_consumed 範圍連續、不重疊（每步只消費一次）、合起來＝全部紀錄、上界 ≤ 該列 state_step（不提前使用）；
       該列 delta_m＝重算 Δ 於其最後消費步（無消費 ⇒ 沿用前值）。
  4. balance.stopped 或緩衝 fault 存在 ⇒ 不通過。任何欄位缺漏或非有限 ⇒ 不通過。

    python3 evaluation/door_k3ab_check.py <run_dir>    # 寫 <run_dir>/analysis/door_k3ab_check.json；結束碼 0＝通過
"""
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3a_check as KC3                          # noqa: E402

REG = {'tau_f_s': 0.05, 'dt_p_s': 0.01, 'deadband_n': 0.5, 'k_i_m_per_ns': 0.25e-3, 'v_max_mps': 1.0e-3, 'delta_max_m': 1.0e-3,
       'dt_tol_s': 1e-4, 'stale_sim_s': 0.10}
E12 = np.array([0.0, -1.0, 0.0])
FINGERS = KC.FINGERS
TOL = 1e-9


def _M(x):
    if x is None:
        return None
    a = np.asarray(x, float)
    return a.reshape(4, 4) if a.size == 16 and np.isfinite(a).all() else None


def _vec(x, n=3):
    try:
        a = np.asarray(x, float).reshape(n)
    except (TypeError, ValueError):
        return None
    return a if np.isfinite(a).all() else None


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def recompute(records):
    """依登錄常數逐步重算 (ef, delta)。"""
    beta = 1.0 - math.exp(-REG['dt_p_s'] / REG['tau_f_s'])
    ef, d, out = None, 0.0, []
    for r in records:
        e = r['f2'] - r['f1']
        ef = e if ef is None else ef + beta * (e - ef)
        u = math.copysign(max(abs(ef) - REG['deadband_n'], 0.0), ef)
        v = max(-REG['v_max_mps'], min(REG['v_max_mps'], REG['k_i_m_per_ns'] * u))
        d = max(-REG['delta_max_m'], min(REG['delta_max_m'], d + v * REG['dt_p_s']))
        out.append((ef, d))
    return out


def base_task(task, why):
    t2 = copy.deepcopy(task)
    t2['schema'] = 'door_k3a_task/1' if task.get('schema') == 'door_k3ab_task/1' else 'mismatch'
    miss = 0
    for e in t2.get('trace') or []:
        if e.get('target') is not None:
            if _M(e.get('base_target')) is None:
                miss += 1
            e['target'] = e.get('base_target')
    if miss:
        why.append('base_target_missing')
    return t2


def check_composition(task, why, st):
    worst = {'rot': 0.0, 'e12': 0.0, 'p': 0.0, 'abs_delta': 0.0}
    bad = False
    for e in task.get('trace') or []:
        T = _M(e.get('target'))
        if T is None:
            continue
        Tb, e12, d = _M(e.get('base_target')), _vec(e.get('e12_world')), e.get('delta_m')
        if Tb is None or e12 is None or not _num(d):
            bad = True
            continue
        worst['rot'] = max(worst['rot'], float(np.abs(T[:3, :3] - Tb[:3, :3]).max()))
        worst['e12'] = max(worst['e12'], float(np.abs(e12 - Tb[:3, :3] @ E12).max()))
        worst['p'] = max(worst['p'], float(np.abs((T[:3, 3] - Tb[:3, 3]) - d * e12).max()))
        worst['abs_delta'] = max(worst['abs_delta'], abs(d))
    st['composition'] = worst
    if bad:
        why.append('composition_evidence_missing')
    if worst['rot'] > TOL:
        why.append('published_rotation_not_base')
    if worst['e12'] > TOL:
        why.append('e12_not_tool_minus_y')
    if worst['p'] > TOL:
        why.append('published_not_base_plus_delta')
    if worst['abs_delta'] > REG['delta_max_m'] + 1e-12:
        why.append('delta_over_bound')


def check_timing(task, why, st):
    tr = [e for e in (task.get('trace') or []) if 'phase' in e]
    t_hold = next((e['t'] for e in task.get('policy_events', []) if e.get('to') == 'HOLD'), None)
    rows = [e for e in tr if t_hold is not None and e['t'] >= t_hold - 1e-9]
    if not rows or any(e.get('ph0') is None for e in rows):
        why.append('ph0_missing')
        return
    i_open = next((i for i, e in enumerate(rows) if e['ph0'] == 'OPEN'), None)
    if i_open is None:
        why.append('no_open_cycle')
        return
    for e in rows[:i_open + 1]:
        if e.get('target') is not None and e.get('delta_m') != 0.0:
            why.append('delta_nonzero_before_second_open_cycle')
            break
    prev = next((e for e in reversed(rows[:i_open]) if _M(e.get('target')) is not None), None)
    T1 = _M(rows[i_open].get('target'))
    if prev is None or T1 is None:
        why.append('open_first_published_unverifiable')
    else:
        st['open_first_published_dev'] = float(np.abs(T1 - _M(prev['target'])).max())
        if st['open_first_published_dev'] > TOL:
            why.append('open_first_published_discontinuous')
    eh = [e for e in rows if e['ph0'] == 'END_HOLD']
    i_eh = next((i for i, e in enumerate(rows) if e['ph0'] == 'END_HOLD'), None)
    if i_eh is not None:
        last = next((e for e in reversed(rows[:i_eh]) if _M(e.get('target')) is not None), None)
        for e in eh:
            if e.get('target') is None:
                continue
            if last is None or float(np.abs(_M(e['target']) - _M(last['target'])).max()) != 0.0 or e.get('delta_m') != last.get('delta_m'):
                why.append('end_hold_published_not_frozen')
                break
        if any(e.get('bal_consumed') is not None for e in eh):
            why.append('end_hold_consumed_pairs')
    if any(e['ph0'] == 'DONE' and e.get('target') is not None for e in rows):
        why.append('done_published_target')


def check_balance(task, room, why, st):
    bal = task.get('balance')
    cfg = ((task.get('config') or {}).get('balance')) or {}
    if any(not _num(cfg.get(k)) or abs(float(cfg[k]) - v) > 1e-15 for k, v in REG.items()):
        why.append('balance_config_not_registered')
    if not isinstance(bal, dict) or not isinstance(bal.get('records'), list):
        why.append('balance_evidence_missing')
        return
    if bal.get('stopped') is not None or bal.get('pairs_fault') is not None:
        why.append('balance_stopped')
    tr = [e for e in (task.get('trace') or []) if 'phase' in e]
    first_open = next((e for e in tr if e.get('ph0') == 'OPEN'), None)
    ev = next((e['balance_start'] for e in task.get('policy_events', []) if 'balance_start' in e), None)
    s0 = bal.get('s0')
    if first_open is None or ev is None or s0 is None or ev.get('s0') != s0 or first_open.get('state_step') != s0:
        why.append('balance_start_mismatch')
        return
    recs = bal['records']
    try:
        steps = [int(r['step']) for r in recs]
        if any(not all(_num(r[k]) for k in ('sim_t', 'f1', 'f2', 'ef', 'delta')) for r in recs):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        why.append('balance_records_invalid')
        return
    if steps != list(range(s0 + 1, s0 + 1 + len(steps))):
        why.append('balance_records_not_contiguous')
    cols = room.get('steps_cols') or []
    I = {c: i for i, c in enumerate(cols)}
    rs = {s[I['step']]: s for s in room.get('steps', [])}
    src = 0.0
    for r in recs:
        s = rs.get(r['step'])
        if s is None or s[I['leaf_contacts']] is None:
            src = math.inf
            break
        lc = s[I['leaf_contacts']]
        fb = [(lc.get(f) or {}).get('bar', 0.0) for f in FINGERS]
        src = max(src, abs(fb[0] - r['f1']), abs(fb[1] - r['f2']), abs(s[I['sim_t']] - r['sim_t']))
    st['pair_vs_room_dev'] = src
    if src > TOL:
        why.append('pairs_not_same_step_room_source')
    rc = recompute(recs)
    dev = max([max(abs(a[0] - r['ef']), abs(a[1] - r['delta'])) for a, r in zip(rc, recs)] or [0.0])
    st['recompute_dev'] = dev
    if dev > 1e-12:
        why.append('balance_recompute_mismatch')
    by_step = {r['step']: a[1] for r, a in zip(recs, rc)}
    cons = []
    d_prev = 0.0
    future = mism = False
    for e in tr:
        c = e.get('bal_consumed')
        if c is not None:
            if not (isinstance(c, list) and len(c) == 2 and all(isinstance(x, int) for x in c) and c[0] <= c[1]):
                why.append('bal_consumed_invalid')
                return
            cons.append(c)
            if not isinstance(e.get('state_step'), int) or c[1] > e['state_step']:
                future = True
            d_prev = by_step.get(c[1], math.nan)
        if e.get('ph0') in ('OPEN', 'SETTLE') and e.get('target') is not None and e is not first_open:
            if not _num(e.get('delta_m')) or not math.isfinite(d_prev) or abs(e['delta_m'] - d_prev) > 1e-12:
                mism = True
    flat = [s for a, b in cons for s in range(a, b + 1)]
    if flat != steps:
        why.append('bal_consumed_not_each_step_once')
    if future:
        why.append('bal_consumed_future_step')
    if mism:
        why.append('row_delta_not_recomputed')
    st['n_balance_steps'] = len(recs)
    st['delta_final_m'] = rc[-1][1] if rc else 0.0
    st['n_saturated_steps'] = sum(1 for r in recs if r.get('sat'))


def judge(task, room, gate):
    why = []
    t2 = base_task(task, why)
    res = KC3.judge(t2, room, gate)
    why += list(res['why'])
    if task.get('schema') != 'door_k3ab_task/1':
        why.append('task_schema_k3ab')
    st = dict(res.get('stats') or {})
    check_composition(task, why, st)
    check_timing(task, why, st)
    check_balance(task, room, why, st)
    return {'pass': not why, 'why': why, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    res = judge(task, room, gate)
    p = os.path.join(run, 'analysis', 'door_k3ab_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'why', 'stats')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
