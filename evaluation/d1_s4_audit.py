#!/usr/bin/env python3
"""D1 S4 線上 shadow 一趟的稽核（只讀實錄；Codex reviews/20261005_144036_reply.md 的三部分通過定義）。

    python3 evaluation/d1_s4_audit.py runs/<RUN> [--out runs/<RUN>/analysis/d1_s4_audit.json]

一 通路與對帳（以來源端影格序號 n 逐格對帳，不只比總數）
   來源：wrist_live/capture.jsonl（attempts → published／rejected／no_new_frame）
   節點：d1_shadow/d1_shadow.jsonl（processed／overwritten／shutdown_unprocessed／contract_reject／evicted）
   每個來源已發布影格恰有一個去向；節點完全沒看到者＝transport_drop（ROS 有界接收佇列丟棄或節點未上線）。
   通過條件：至少 1 格 processed 且已發布；逐格去向閉合（無重複、無遺漏）；節點有界收尾摘要存在。
二 有效觀測：ALIGN 之前（接近窗口）至少 1 格 L2 中心觀測。全拒絕 ⇒ 只能說通路連通。有效率與誤差只報告。
三 任務與隔離：S1–S6（motm_physical_check 既有判定）、求解節點時槽統計、熱紀錄、節點只發布 /d1/handle_obs（靜態核對）。
   年齡隨時間的趨勢只作診斷。算圖同步占用模擬主迴圈 ⇒ 只稱命令／資料介面隔離。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R = a.run
    out = {'run': os.path.basename(R.rstrip('/')), 'missing': []}

    def need(p):
        if not os.path.exists(p):
            out['missing'].append(os.path.relpath(p, R))
            return False
        return True

    cap = jl(os.path.join(R, 'wrist_live', 'capture.jsonl'))
    truth = jl(os.path.join(R, 'wrist_live', 'truth.jsonl'))
    wmeta_p = os.path.join(R, 'wrist_live', 'meta.json')
    evs = jl(os.path.join(R, 'd1_shadow', 'd1_shadow.jsonl'))
    summ_p = os.path.join(R, 'd1_shadow', 'd1_shadow_summary.json')
    for p in (os.path.join(R, 'wrist_live', 'capture.jsonl'), wmeta_p, os.path.join(R, 'd1_shadow', 'd1_shadow.jsonl'),
              summ_p, os.path.join(R, 'task.json'), os.path.join(R, 'room_run.json')):
        need(p)
    if cap is None or evs is None:
        out['verdict'] = {'pipeline': 'INSUFFICIENT', 'why': '缺來源或節點紀錄'}
        return dump(out, a, R)
    wmeta = json.load(open(wmeta_p)) if os.path.exists(wmeta_p) else {}
    summ = json.load(open(summ_p)) if os.path.exists(summ_p) else None

    # ---- 一 來源
    src = {'attempts': len(cap)}
    for k in ('published', 'rejected', 'no_new_frame'):
        src[k] = sum(1 for c in cap if c['ev'] == k)
    src['rejected_why'] = {}
    for c in cap:
        if c['ev'] == 'rejected':
            src['rejected_why'][c['why']] = src['rejected_why'].get(c['why'], 0) + 1
    pubs = {c['n']: c for c in cap if c['ev'] == 'published'}
    ts = sorted(c['rendering_time'] for c in pubs.values())
    src['new_frame_rate_sim_hz'] = (None if len(ts) < 2 else round((len(ts) - 1) / (ts[-1] - ts[0]), 3))
    src['render_ms'] = wmeta.get('render_ms')
    rmr = [c['read_minus_render_s'] for c in pubs.values()]
    src['read_minus_render_s'] = None if not rmr else {'min': min(rmr), 'max': max(rmr)}
    out['source'] = src

    # ---- 一 節點去向（逐格）
    by_stamp = {tuple(c['stamp']): n for n, c in pubs.items()}
    fate = {n: [] for n in pubs}
    unknown = []
    proc = {}
    for e in evs:
        k = e['ev']
        if k in ('processed', 'overwritten', 'shutdown_unprocessed'):
            n = e.get('n')
            (fate[n].append(k) if n in fate else unknown.append((k, n)))
            if k == 'processed':
                proc[n] = e
        elif k == 'contract_reject':
            n = e.get('n')
            (fate[n].append('contract_reject:' + e['why']) if n in fate else unknown.append((k, n)))
        elif k == 'evicted':
            n = by_stamp.get(tuple(e['stamp']))
            (fate[n].append('evicted:' + e['why']) if n is not None else unknown.append((k, e['stamp'])))
    for n in fate:
        if not fate[n]:
            fate[n] = ['transport_drop']
    hist = {}
    for v in fate.values():
        for x in v:
            hist[x] = hist.get(x, 0) + 1
    dup = {n: v for n, v in fate.items() if len(v) != 1}
    out['reconciliation'] = {'per_frame_fate_counts': hist, 'frames_with_multiple_fates': dup,
                             'node_events_not_matching_source': unknown[:20], 'n_unknown': len(unknown),
                             'node_summary': None if summ is None else {
                                 'why': summ['why'], 'counters': summ['counters'],
                                 'unprocessed_in_slot': summ['unprocessed_in_slot'],
                                 'unpaired_at_exit': summ['unpaired_at_exit'],
                                 'worker_alive_at_exit': summ['worker_alive_at_exit'],
                                 'detector_sha256': summ['detector_sha256']}}

    # ---- 處理、年齡、輸出率
    P = [proc[n] for n in sorted(proc)]
    if P:
        dm = np.array([p['detect_ms'] for p in P])
        sa = np.array([p['sim_age_s'] for p in P if p['sim_age_s'] is not None])
        wa = np.array([p['wall_age_s'] for p in P if p['wall_age_s'] is not None])
        tc = np.array([pubs[p['n']]['rendering_time'] for p in P])
        wt = np.array([p['wall_t'] for p in P])
        q = lambda x: None if not len(x) else {'p50': float(np.median(x)), 'p95': float(np.percentile(x, 95)),  # noqa: E731
                                                'max': float(x.max())}
        slope = None
        if len(wa) >= 5:
            slope = float(np.polyfit(tc[:len(wa)], wa, 1)[0])
        out['processing'] = {'n_processed': len(P), 'detect_ms': q(dm), 'sim_age_s': q(sa), 'wall_age_s': q(wa),
                             'output_rate_sim_hz': (None if len(tc) < 2 else round((len(tc) - 1) / (tc.max() - tc.min()), 3)),
                             'output_rate_wall_hz': (None if len(wt) < 2 else round((len(wt) - 1) / (wt.max() - wt.min()), 3)),
                             'wall_age_trend_s_per_sim_s（診斷）': slope}

    # ---- 二 有效觀測（離線評估：凍結 evaluate() 對 truth.jsonl）
    import d1_handle_detect as D1
    tr = {t['n']: t for t in (truth or [])}
    K = np.array(wmeta.get('intrinsics_readback_K')) if wmeta.get('intrinsics_readback_K') else None
    task = json.load(open(os.path.join(R, 'task.json'))) if os.path.exists(os.path.join(R, 'task.json')) else {}
    ph = {e['phase']: float(e['sim_t']) for e in task.get('events', []) if 'phase' in e}
    t_align = ph.get('ALIGN')
    rows = []
    for p in P:
        n = p['n']
        if n not in tr or K is None:
            continue
        f = {'handle_center_world_at_capture': tr[n]['handle_center_world_at_capture']}
        det = p['det']
        T = np.array(p['T'])
        ev = D1.evaluate(det, f, T, K, np.array([1.0, 0.0, 0.0]))
        rows.append({'n': n, 'src_t': tr[n]['t_cap'], 'detect_ms': p['detect_ms'], 'reject': det['reject'],
                     'reject_L2': det['reject_L2'], 'L1': det['L1'], 'L2': det['L2'], 'eval': ev})
    from d1_dev_summary import summarize
    out['observation'] = {'t_ALIGN': t_align, 'by_distance': summarize(rows) if rows else None,
                          'n_eval': len(rows)}
    pre = [r for r in rows if t_align is not None and r['src_t'] < t_align]
    out['observation']['approach_window'] = {'n_processed': len(pre), 'n_L2': sum(1 for r in pre if r['L2']),
                                             'n_L2_violation': sum(1 for r in rows if r['eval']['L2_violation'])}
    json.dump({'rows': rows}, open(os.path.join(os.path.dirname(summ_p) if summ else R, 'd1_s4_eval_rows.json'), 'w'),
              ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))

    # ---- 三 任務與隔離
    pc = os.path.join(R, 'analysis', 'physical_check.json')
    out['task'] = {'final_phase': task.get('final_phase'), 'abort': task.get('abort'),
                   'S': (json.load(open(pc)).get('results') if os.path.exists(pc) else 'NA（未跑 motm_physical_check）')}
    sol = os.path.join(R, 'align_solver.json')
    if os.path.exists(sol):
        st = json.load(open(sol)).get('stats') or {}
        out['control'] = {k: st.get(k) for k in ('slot_basis', 'nominal_period_sim_s', 'n_solve_rows_logged',
                                                  'n_missed_slot', 'n_dup_skip', 'n_warm_discard', 'n_sim_stall',
                                                  'task_sim_span_s')}
    th = os.path.join(R, 'thermal.csv')
    if os.path.exists(th):
        vals = []
        for line in open(th):
            for tok in line.strip().split(',')[1:]:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        out['thermal_max_c'] = max(vals) if vals else None
    src_node = open(os.path.join(HERE, 'd1_shadow_node.py')).read()
    pubs_topics = re.findall(r"create_publisher\(\s*\w+\s*,\s*'([^']+)'", src_node)
    out['isolation'] = {'node_publishers（靜態）': pubs_topics,
                        'only_obs_topic': pubs_topics == ['/d1/handle_obs'],
                        '說法': '命令／資料介面隔離；算圖同步占用模擬主迴圈，不稱算力或時序隔離'}

    # ---- 判定
    closed = not dup and not unknown and summ is not None
    has_out = bool(P) and summ is not None and summ['counters']['published'] >= 1
    out['verdict'] = {
        '通路與對帳': 'PASS' if (closed and has_out) else 'FAIL',
        '有效觀測（接近窗口 ≥ 1 格 L2）': ('PASS' if out['observation']['approach_window']['n_L2'] >= 1
                                    else ('只能說通路連通' if has_out else 'FAIL')),
        '任務與隔離': '見 task／control／thermal／isolation（各自報告）'}
    return dump(out, a, R)


def dump(out, a, R):
    dst = a.out or os.path.join(R, 'analysis', 'd1_s4_audit.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(json.dumps(out.get('verdict'), ensure_ascii=False))
    print('寫出', dst)
    return 0


if __name__ == '__main__':
    sys.exit(main())
