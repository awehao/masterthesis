#!/usr/bin/env python3
"""D1 S4 線上 shadow 一趟的稽核（只讀實錄）。r2：依 Codex reviews/20261005_145312_reply.md 四項必修。

    python3 evaluation/d1_s4_audit.py runs/<RUN> [--out runs/<RUN>/analysis/d1_s4_audit.json]

一 通路與對帳（以來源影格序號 n 逐格；不只比總數）
   每個來源已發布影格恰有一個去向：
     processed／overwritten／contract_reject:<原因>／worker_error:<階段>／shutdown_unprocessed／shutdown_in_progress／
     evicted:<原因>（以擷取戳對應）／
     transport_drop（推定：節點沒有收到該戳的任何一段）／
     received_no_fate（節點收到過、但沒有去向 ⇒ **證據不足**，不補成傳輸遺失）
   通過條件（全部成立）：至少 1 格 processed 並發布；無 received_no_fate、無重複去向、無對不上來源的節點事件；
   節點收尾摘要存在、工作執行緒已結束（worker_alive_at_exit = False）、worker_error = 0、無 shutdown_in_progress；
   節點計數器與事件逐項對帳（received 各段、paired、published = processed）。
二 有效觀測（只報告，不設門檻）
   有效率分兩種分母：來源新影格（已發布＋擷取端拒絕；拒絕者距離未知另列）與已處理影格。
   連續有效窗沿來源影格時序計算：任何未處理、拒絕、非 L2 的影格或間隔 > 0.4 s 都中斷。
   接近窗口（ALIGN 之前）至少 1 格 L2 ⇒ 有效觀測成立；全部拒絕 ⇒ 只能說通路連通。必要真值或內參缺失 ⇒ 證據不足。
三 任務與隔離：S1–S6、求解節點時槽、CPU 溫度（thermal.csv 的 cpu_c 欄）、節點只發布 /d1/handle_obs（靜態核對）。
   只稱命令／資料介面隔離（算圖同步占用模擬主迴圈）。年齡趨勢只作診斷。
時間欄位定義
   processed_capture_cadence_hz：已處理影格的**擷取時刻**序列頻率（不是輸出頻率）
   output_rate_sim_hz：以節點**輸出當下的模擬時間**（out_sim_t）計算的輸出頻率
   wall_age_s：模擬器讀取並匹配影格之後 → 節點輸出（**不含**此前的算圖延遲；算圖耗時另見 source.render_ms）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GAP_S = 0.4
BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]
TERMINAL_BY_N = ('processed', 'overwritten', 'contract_reject', 'worker_error', 'shutdown_unprocessed',
                 'shutdown_in_progress')


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else None


def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    return None if not len(x) else {'n': int(len(x)), 'p50': float(np.median(x)),
                                    'p95': float(np.percentile(x, 95)), 'max': float(x.max())}


def rate(ts):
    ts = sorted(t for t in ts if t is not None)
    return None if len(ts) < 2 or ts[-1] <= ts[0] else round((len(ts) - 1) / (ts[-1] - ts[0]), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R = a.run
    out = {'schema': 'd1_s4_audit/2', 'run': os.path.basename(R.rstrip('/')), 'missing': [],
           'evidence_gaps': []}
    paths = {k: os.path.join(R, *v) for k, v in {
        'capture': ('wrist_live', 'capture.jsonl'), 'truth': ('wrist_live', 'truth.jsonl'),
        'wmeta': ('wrist_live', 'meta.json'), 'events': ('d1_shadow', 'd1_shadow.jsonl'),
        'summary': ('d1_shadow', 'd1_shadow_summary.json'), 'task': ('task.json',)}.items()}
    for k, p in paths.items():
        if not os.path.exists(p):
            out['missing'].append(os.path.relpath(p, R))
    cap, evs = jl(paths['capture']), jl(paths['events'])
    if cap is None or evs is None:
        out['verdict'] = {'通路與對帳': 'INSUFFICIENT（缺來源或節點紀錄）'}
        return dump(out, a, R)
    truth = jl(paths['truth']) or []
    wmeta = json.load(open(paths['wmeta'])) if os.path.exists(paths['wmeta']) else {}
    summ = json.load(open(paths['summary'])) if os.path.exists(paths['summary']) else None

    # ================= 一 來源
    pubs = {c['n']: c for c in cap if c['ev'] == 'published'}
    rej = [c for c in cap if c['ev'] == 'rejected']
    src = {'attempts': len(cap), 'published': len(pubs), 'rejected': len(rej),
           'no_new_frame': sum(1 for c in cap if c['ev'] == 'no_new_frame'), 'rejected_why': {}}
    for c in rej:
        src['rejected_why'][c['why']] = src['rejected_why'].get(c['why'], 0) + 1
    src['new_frame_capture_rate_sim_hz'] = rate([c['rendering_time'] for c in pubs.values()])
    src['render_ms'] = wmeta.get('render_ms')
    src['read_minus_render_s'] = q([c.get('read_minus_render_s') for c in pubs.values()])
    out['source'] = src

    # ================= 一 逐格去向
    by_stamp = {tuple(c['stamp']): n for n, c in pubs.items()}
    fate = {n: [] for n in pubs}
    unknown, proc, recv_stamps = [], {}, set()
    recv_by_part = {}
    for e in evs:
        k = e['ev']
        if k == 'received':
            recv_stamps.add(tuple(e['stamp']))
            recv_by_part[e['part']] = recv_by_part.get(e['part'], 0) + 1
            if tuple(e['stamp']) not in by_stamp:
                unknown.append(('received', e['stamp']))
        elif k in TERMINAL_BY_N:
            n = e.get('n')
            tag = k + (':' + e['why'] if k == 'contract_reject' else '') + (':' + e['stage'] if k == 'worker_error' else '')
            (fate[n].append(tag) if n in fate else unknown.append((k, n)))
            if k == 'processed' and n in fate:
                proc[n] = e
        elif k == 'evicted':
            n = by_stamp.get(tuple(e['stamp']))
            (fate[n].append('evicted:' + e['why']) if n is not None else unknown.append((k, e['stamp'])))
    for n, c in pubs.items():
        if not fate[n]:
            fate[n] = ['received_no_fate' if tuple(c['stamp']) in recv_stamps else 'transport_drop']
    hist = {}
    for v in fate.values():
        for x in v:
            hist[x] = hist.get(x, 0) + 1
    dup = {n: v for n, v in fate.items() if len(v) != 1}
    n_nofate = hist.get('received_no_fate', 0)

    # 計數器對帳
    recon = {}
    if summ is not None:
        C = summ['counters']
        paired_fates = sum(v for k, v in hist.items()
                           if k.split(':')[0] in TERMINAL_BY_N)
        recon = {
            'received_parts_counter_vs_events': {p: [C.get('received_' + p), recv_by_part.get(p, 0)]
                                                 for p in ('depth', 'info', 'pose', 'meta')},
            'paired_counter_vs_terminal_fates': [C.get('paired'), paired_fates],
            'published_equals_processed': C.get('published') == C.get('processed') == len(proc),
            'worker_error': C.get('worker_error', 0),
            'worker_alive_at_exit': summ.get('worker_alive_at_exit'),
            'in_progress_at_exit_n': summ.get('in_progress_at_exit_n'),
            'unprocessed_in_slot': summ.get('unprocessed_in_slot'),
            'unpaired_at_exit': summ.get('unpaired_at_exit'),
            'shutdown_why': summ.get('why'), 'detector_sha256': summ.get('detector_sha256')}
        recon['counters_close'] = (
            all(v[0] == v[1] for v in recon['received_parts_counter_vs_events'].values())
            and recon['paired_counter_vs_terminal_fates'][0] == recon['paired_counter_vs_terminal_fates'][1]
            and recon['published_equals_processed'])
    out['reconciliation'] = {'per_frame_fate_counts': hist, 'frames_with_multiple_fates': dup,
                             'node_events_not_matching_source': unknown[:20], 'n_unknown': len(unknown),
                             'counters': recon}

    # ================= 處理與時間
    P = [proc[n] for n in sorted(proc)]
    if P:
        tcap = [pubs[p['n']]['rendering_time'] for p in P]
        wa = [p.get('wall_age_s') for p in P]
        slope = None
        pts = [(t, w) for t, w in zip(tcap, wa) if w is not None]
        if len(pts) >= 5:
            slope = float(np.polyfit(*np.array(pts).T, 1)[0])
        out['processing'] = {
            'n_processed': len(P), 'detect_ms': q([p['detect_ms'] for p in P]),
            'processed_capture_cadence_hz（已處理影格的擷取時序頻率，非輸出頻率）': rate(tcap),
            'output_rate_sim_hz（以輸出當下模擬時間）': rate([p.get('out_sim_t') for p in P]),
            'output_rate_wall_hz': rate([p['wall_t'] for p in P]),
            'sim_age_s（輸出當下 /clock − 擷取戳）': q([p.get('sim_age_s') for p in P]),
            'wall_age_s（模擬器讀取匹配後 → 節點輸出；不含算圖延遲）': q(wa),
            'wall_age_trend_s_per_sim_s（診斷）': slope}

    # ================= 二 有效觀測
    import d1_handle_detect as D1
    tr = {t['n']: t for t in truth}
    K = np.array(wmeta['intrinsics_readback_K']) if wmeta.get('intrinsics_readback_K') else None
    if K is None:
        out['evidence_gaps'].append('缺內參（wrist_live/meta.json intrinsics_readback_K）')
    task = json.load(open(paths['task'])) if os.path.exists(paths['task']) else {}
    ph = {e['phase']: float(e['sim_t']) for e in task.get('events', []) if 'phase' in e}
    t_align = ph.get('ALIGN')
    if t_align is None:
        out['evidence_gaps'].append('缺 ALIGN 時刻（task.json）⇒ 接近窗口無法界定')

    def dist(n):
        t = tr.get(n)
        if t is None or t.get('cam_pos_world') is None:
            return None
        return float(np.linalg.norm(np.array(t['handle_center_world_at_capture']) - np.array(t['cam_pos_world'])))

    rows, miss_truth = [], []
    for p in P:
        n = p['n']
        if n not in tr or K is None:
            miss_truth.append(n)
            continue
        ev = D1.evaluate(p['det'], {'handle_center_world_at_capture': tr[n]['handle_center_world_at_capture']},
                         np.array(p['T']), K, np.array([1.0, 0.0, 0.0]))
        rows.append({'n': n, 'src_t': tr[n]['t_cap'], 'detect_ms': p['detect_ms'], 'reject': p['det']['reject'],
                     'reject_L2': p['det']['reject_L2'], 'L1': p['det']['L1'], 'L2': p['det']['L2'], 'eval': ev})
    if miss_truth:
        out['evidence_gaps'].append(f'{len(miss_truth)} 格已處理影格缺真值或內參，未納入評估：{miss_truth[:10]}')
    from d1_dev_summary import summarize
    by_n = {r['n']: r for r in rows}
    src_bins = {}
    no_dist = [n for n in pubs if dist(n) is None]
    for lo, hi in BINS:
        ns = [n for n in pubs if dist(n) is not None and lo <= dist(n) < hi]
        if not ns:
            continue
        npro = sum(1 for n in ns if n in by_n)
        nl2 = sum(1 for n in ns if n in by_n and by_n[n]['L2'])
        src_bins[f'{lo}-{hi}'] = {'n_source_published': len(ns), 'n_processed': npro, 'n_L2': nl2,
                                  'L2_rate_of_source（含丟格）': round(nl2 / len(ns), 3),
                                  'L2_rate_of_processed': (None if not npro else round(nl2 / npro, 3))}
    # 連續有效窗：沿來源時序（已發布＋擷取端拒絕），任何非有效影格或間隔 > GAP_S 中斷
    seq = sorted([(c['rendering_time'], n) for n, c in pubs.items()]
                 + [(c.get('rendering_time'), None) for c in rej if c.get('rendering_time') is not None])
    wins, st, prev = [], None, None
    for t, n in seq:
        ok = n is not None and n in by_n and bool(by_n[n]['L2'])
        if ok and st is not None and t - prev <= GAP_S + 1e-6:
            prev = t
        elif ok:
            if st is not None:
                wins.append((st, prev))
            st = prev = t
        elif st is not None:
            wins.append((st, prev))
            st = None
    if st is not None:
        wins.append((st, prev))
    pre = [r for r in rows if t_align is not None and r['src_t'] < t_align]
    out['observation'] = {
        't_ALIGN': t_align,
        'by_distance_source_denominator': src_bins,
        'source_frames_distance_unknown': {'published_without_truth': len(no_dist), 'capture_rejected': len(rej)},
        'by_distance_processed_denominator': summarize(rows) if rows else None,
        'L2_continuous_windows_on_source_timeline_s': [[round(x, 2), round(y, 2)] for x, y in wins],
        'approach_window': {'n_processed': len(pre), 'n_L2': sum(1 for r in pre if r['L2'])},
        'n_L2_violation': sum(1 for r in rows if r['eval']['L2_violation'])}
    json.dump({'rows': rows}, open(os.path.join(R, 'd1_shadow', 'd1_s4_eval_rows.json')
                                   if os.path.isdir(os.path.join(R, 'd1_shadow')) else os.devnull, 'w'),
              ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))

    # ================= 三 任務與隔離
    pc = os.path.join(R, 'analysis', 'physical_check.json')
    out['task'] = {'final_phase': task.get('final_phase'), 'abort': task.get('abort'),
                   'S': (json.load(open(pc)).get('results') if os.path.exists(pc) else 'NA（未跑 motm_physical_check）')}
    sol = os.path.join(R, 'align_solver.json')
    if os.path.exists(sol):
        stt = json.load(open(sol)).get('stats') or {}
        out['control'] = {k: stt.get(k) for k in ('slot_basis', 'nominal_period_sim_s', 'n_solve_rows_logged',
                                                  'n_missed_slot', 'n_dup_skip', 'n_warm_discard', 'n_sim_stall',
                                                  'task_sim_span_s')}
    th = os.path.join(R, 'thermal.csv')
    if os.path.exists(th):
        cc = []
        for r in csv.DictReader(open(th)):
            try:
                cc.append(float(r['cpu_c']))
            except (KeyError, TypeError, ValueError):
                pass
        out['thermal_cpu_c_max'] = max(cc) if cc else None
    src_node = open(os.path.join(HERE, 'd1_shadow_node.py')).read()
    pubs_topics = re.findall(r"create_publisher\(\s*\w+\s*,\s*'([^']+)'", src_node)
    out['isolation'] = {'node_publishers（靜態）': pubs_topics, 'only_obs_topic': pubs_topics == ['/d1/handle_obs'],
                        '說法': '命令／資料介面隔離；算圖同步占用模擬主迴圈，不稱算力或時序隔離'}

    # ================= 判定
    rc = recon
    fail = []
    if not P:
        fail.append('沒有任何 processed 影格')
    if n_nofate:
        fail.append(f'{n_nofate} 格節點收到但沒有去向（證據不足）')
    if dup:
        fail.append(f'{len(dup)} 格多重去向')
    if unknown:
        fail.append(f'{len(unknown)} 個節點事件對不上來源')
    if summ is None:
        fail.append('缺節點收尾摘要')
    else:
        if rc['worker_alive_at_exit']:
            fail.append('收尾時工作執行緒仍在執行')
        if rc['worker_error']:
            fail.append(f"worker_error {rc['worker_error']} 次")
        if rc['in_progress_at_exit_n'] is not None:
            fail.append(f"收尾時影格 {rc['in_progress_at_exit_n']} 仍在處理")
        if not rc['counters_close']:
            fail.append('節點計數器與事件不閉合')
    obs_gap = bool(out['evidence_gaps'])
    out['verdict'] = {
        '通路與對帳': 'PASS' if not fail else 'FAIL：' + '；'.join(fail),
        '有效觀測（接近窗口 ≥ 1 格 L2）': ('證據不足：' + '；'.join(out['evidence_gaps']) if obs_gap else
                                    'PASS' if out['observation']['approach_window']['n_L2'] >= 1 else
                                    ('只能說通路連通' if P else 'FAIL')),
        '任務與隔離': '見 task／control／thermal_cpu_c_max／isolation（各自報告）'}
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
