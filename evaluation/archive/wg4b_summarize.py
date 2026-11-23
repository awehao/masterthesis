#!/usr/bin/env python3
"""WG4-B 正式批次彙整（分析口徑見 results/wg4b/wg4b_formal_plan.yaml，起跑前鎖定）。只讀實錄。

  全任務主表   S1–S6、完成／未完成、最終相位、終止原因、實際觀察時長、重播判定
  共同區段     ALIGN、ENGAGE_WAIT、OPEN 分相位：段長、追蹤 err_p p50／p95、err_r p95、
               命令增量 du_base／du_arm p95、手臂相鄰翻號率、核心計算 p50／p95（不混入 OPEN_HOLD）
  OPEN_HOLD 另表 帶內最長連續、最大開度、首次超帶時刻、觀察長度、該相位 err_p p50
  完成時間     只對到達 DONE 的趟報（開始移動 → DONE）；未完成標「時限內未完成」，不填數值
  配對         A／B／C 三對，B1 − P 的共同區段差值全列；任一側缺值 ⇒ NA。n = 3，不做顯著性

需要每趟已有 analysis/physical_check.json、analysis/motm_metrics.json、analysis/horizon_metrics.json、
重播結果（B1：analysis/b1_replay.json；P：analysis/replay_check.json）。缺檔 ⇒ 該欄 NA 並列入 missing。

    python3 evaluation/wg4b_summarize.py [排程.tsv] [輸出.json]
預設排程 results/wg4b/wg4b_formal_schedule.tsv、輸出 results/wg4b/wg4b_formal_results.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
COMMON = ('ALIGN', 'ENGAGE_WAIT', 'OPEN')
S_KEYS = ('S1_open_hold', 'S2_close_hold', 'S3_grasp', 'S4_drift', 'S5_home', 'S6_contact')


def load(path, missing):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        missing.append(os.path.relpath(path, HERE))
        return None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def verdict(v):
    return 'PASS' if v is True else 'FAIL' if v is False else '證據不足／NA'


def run_row(rid, method, missing):
    R = os.path.join(RUNS, rid)
    A = os.path.join(R, 'analysis')
    task = load(os.path.join(R, 'task.json'), missing)
    sol = load(os.path.join(R, 'align_solver.json'), missing)
    pc = load(os.path.join(A, 'physical_check.json'), missing)
    mm = load(os.path.join(A, 'motm_metrics.json'), missing)
    hm = load(os.path.join(A, 'horizon_metrics.json'), missing)
    rp = load(os.path.join(A, 'b1_replay.json' if method == 'B1' else 'replay_check.json'), missing)
    final = g(task, 'final_phase')
    done = final == 'DONE'
    ph = {}
    for e in (task or {}).get('events', []):
        if 'phase' in e and e['phase'] not in ph:
            ph[e['phase']] = float(e['sim_t'])
    order = [p for p, _ in sorted(ph.items(), key=lambda x: x[1])]
    st = g(sol, 'stats') or {}
    row = {'rid': rid, 'method': method,
           'solver_kind': (sol or {}).get('solver_kind', 'wgmpc'),
           'kp': g(sol, 'args', 'b1_kp') if method == 'B1' else None,
           'completed': done, 'final_phase': final,
           'termination': st.get('stop_why'),
           # 求解節點自己的時間跨度（首次發布 → 停止），不是整趟觀察時長
           'solver_time_span_s': st.get('task_sim_span_s'),
           'replay_verdict': g(rp, 'verdict'),
           'S': {k: verdict(g(pc, 'results', k)) for k in S_KEYS},
           'S_pass_count': sum(1 for k in S_KEYS if g(pc, 'results', k) is True),
           'completion_time_s': (round(g(mm, 'whole_run', 'done_sim_t')
                                       - g(mm, 'whole_run', 'moving_from_sim_t'), 2)
                                 if done and g(mm, 'whole_run', 'done_sim_t') is not None
                                 else '時限內未完成' if not done else None)}
    seg = {}
    for p in COMMON:
        b = g(hm, 'by_phase', p)
        nxt = order[order.index(p) + 1] if p in order and order.index(p) + 1 < len(order) else None
        seg[p] = {
            'duration_s': (round(ph[nxt] - ph[p], 2) if p in ph and nxt else None),
            'err_p_mm_p50': None if b is None else _mm(g(b, 'tracking', 'err_p_m', 'p50')),
            'err_p_mm_p95': None if b is None else _mm(g(b, 'tracking', 'err_p_m', 'p95')),
            'err_r_rad_p95': g(b, 'tracking', 'err_r_rad', 'p95'),
            'du_base_p95': g(b, 'smoothness', 'du_base', 'p95'),
            'du_arm_p95': g(b, 'smoothness', 'du_arm', 'p95'),
            'arm_reversal_rate_per_s': g(b, 'smoothness', 'arm_reversals', 'adjacent_rate_per_s'),
            'core_ms_p50': g(b, 'compute', 'core_ms', 'p50'),
            'core_ms_p95': g(b, 'compute', 'core_ms', 'p95')}
    row['common_segment'] = seg
    # ---- 逐物理步紀錄（room_run.json）：觀察窗、開始移動、OPEN_HOLD 相位內統計 ----
    room = load(os.path.join(R, 'room_run.json'), missing)
    band = g(pc, 'S1_open_hold', 'band_mm')
    row['record_window'] = None
    row['open_hold'] = {'reached': 'OPEN_HOLD' in ph}
    if room is not None:
        ci = room['steps_cols'].index
        t = np.array([x[1] for x in room['steps']], float)
        xy = np.array([x[ci('base_xyth')][:2] for x in room['steps']], float)
        op = np.array([x[ci('opening_m')] for x in room['steps']], float) * 1e3
        dt = float(np.median(np.diff(t)))
        w = max(1, int(round(0.1 / dt)))            # 同 motm_metrics：0.1 s 視窗底盤速率
        v = np.full(len(t), np.nan)
        v[w:] = np.hypot(xy[w:, 0] - xy[:-w, 0], xy[w:, 1] - xy[:-w, 1]) / (w * dt)
        row['record_window'] = {
            'from_sim_t': round(float(t[0]), 2), 'to_sim_t': round(float(t[-1]), 2),
            'moving_from_sim_t': round(float(t[int(np.argmax(v > 0.02))]), 2),
            'end': ('DONE @ %.2f' % ph['DONE']) if 'DONE' in ph else
                   '模擬時限截止 @ %.2f（未到 DONE）' % float(t[-1])}
        if 'OPEN_HOLD' in ph:
            i_nx = order.index('OPEN_HOLD') + 1
            t_end = ph[order[i_nx]] if i_nx < len(order) else float(t[-1])
            m = (t >= ph['OPEN_HOLD']) & (t <= t_end)
            tin, oin = t[m], op[m]
            inb = (oin >= band[0]) & (oin <= band[1]) if band else np.zeros(len(oin), bool)
            best = cur = 0
            for f in inb:
                cur = cur + 1 if f else 0
                best = max(best, cur)
            over = np.where(oin > band[1])[0] if band else []
            row['open_hold'].update({
                'source': 'room_run.json 逐物理步',
                'phase_observed_s': round(t_end - ph['OPEN_HOLD'], 2),
                'phase_end': order[i_nx] if i_nx < len(order) else '紀錄末端（時限）',
                'in_phase_longest_in_band_s': round(best * dt, 2),
                'in_phase_max_opening_mm': round(float(oin.max()), 2),
                'first_over_band_sim_t': (round(float(tin[over[0]]), 2) if len(over) else None)})
    row['open_hold'].update({
        # S1 判準：全物理窗內帶內最長連續時間（不限 OPEN_HOLD 相位）
        'S1_longest_in_band_s_any_phase': g(pc, 'S1_open_hold', 'longest_s'),
        'S1_max_opening_mm_any_phase': g(pc, 'S1_open_hold', 'max_opening_mm'),
        'err_p_mm_p50': _mm(g(hm, 'by_phase', 'OPEN_HOLD', 'tracking', 'err_p_m', 'p50'))})
    row['later_phases_reached'] = [p for p in order if order.index(p) > (order.index('OPEN_HOLD')
                                                                         if 'OPEN_HOLD' in order else 99)]
    return row


def _mm(x):
    return None if x is None else round(float(x) * 1e3, 3)


def main():
    plan = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'wg4b',
                                                               'wg4b_formal_schedule.tsv')
    rows = list(csv.DictReader(open(plan), delimiter='\t'))
    missing = []
    table = [run_row(r['rid'], r['method'] if r['method'] == 'B1' else 'P', missing) for r in rows]
    by = {t['rid']: t for t in table}
    pairs = {}
    for pr in sorted({r['pair'] for r in rows}):
        rb = next(r['rid'] for r in rows if r['pair'] == pr and r['method'] == 'B1')
        rpp = next(r['rid'] for r in rows if r['pair'] == pr and r['method'] != 'B1')
        d = {}
        for p in COMMON:
            d[p] = {}
            for k, vb in by[rb]['common_segment'][p].items():
                vp = by[rpp]['common_segment'][p][k]
                d[p][k] = (round(vb - vp, 6) if isinstance(vb, (int, float))
                           and isinstance(vp, (int, float)) else 'NA')
        ratio = {p: (round(by[rpp]['common_segment'][p]['core_ms_p50']
                           / by[rb]['common_segment'][p]['core_ms_p50'], 2)
                     if by[rb]['common_segment'][p]['core_ms_p50'] else 'NA') for p in COMMON}
        pairs[pr] = {'B1': rb, 'P': rpp, 'order': next(r['order'] for r in rows if r['pair'] == pr),
                     'diff_B1_minus_P': d, 'core_ms_p50_ratio_P_over_B1': ratio}
    out = {'schema': 'wg4b_formal_results/1', 'plan': os.path.relpath(plan, HERE),
           'runs': table, 'pairs': pairs, 'missing_inputs': missing,
           'n_completed': {m: sum(1 for t in table if t['method'] == m and t['completed'])
                           for m in ('B1', 'P')},
           'rules': ['未完成趟不填完成時間、不算倍數', '後續相位未到達 ⇒ NA', 'n = 3／組，配對差全列，不做顯著性',
                     '共同區段不混入 OPEN_HOLD；OPEN_HOLD 觀察長度不同，另表列出'],
           'limitation': ('本比較評估事前定義、協同權重數值沿用 P、僅在有限預算內調整 kp 的 B1 改編基線；'
                          '不代表單步 QP 的最佳性能，也不能將差異單獨歸因於缺少多步預測。'
                          'kp = 1.0 為依登錄順序選定，非性能最佳；調參趟不併入正式樣本')}
    dst = (sys.argv[2] if len(sys.argv) > 2
           else os.path.join(HERE, 'results', 'wg4b', 'wg4b_formal_results.json'))
    json.dump(out, open(dst, 'w'), ensure_ascii=False, indent=1)
    for t in table:
        print(t['rid'], t['method'], 'done' if t['completed'] else f"未完成({t['final_phase']})",
              t['S'], 'replay', t['replay_verdict'], 'T', t['completion_time_s'])
    if missing:
        print('缺檔：', missing)
    return 0


if __name__ == '__main__':
    sys.exit(main())
