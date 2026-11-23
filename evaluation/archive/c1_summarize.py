#!/usr/bin/env python3
"""C1 批次彙整：依**凍結排程**逐趟跑判定工具，再按案例算配對差（H1 − H5）。

預期行程（--expect-open-m）一律取自排程檔，不取自待驗收趟次自己的紀錄。
趟次狀態取自批次腳本寫的 <rid>.status.json；非 executed 的趟次列為證據／協定問題，不進配對差。
不新增指標定義 —— 指標與 c0_summarize.analyse 相同。

    python3 evaluation/c1_summarize.py --plan results/horizon_ablation/c1a_schedule.tsv \\
        --out results/horizon_ablation/c1a_b1_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c0_summarize as C0                                        # noqa: E402

RUNS = os.path.join(HERE, 'runs')


def run_problems(r, n_sched):
    """單趟不能進配對差的原因（空 = 可用）。資料完整的任務 FAIL **不是**原因。"""
    why = []
    if r is None:
        return ['無此趟']
    if r.get('status') != 'executed':
        why.append(f"批次狀態 {r.get('status')}")
        return why
    if r.get('tool_errors'):
        why.append('工具異常：' + '；'.join(r['tool_errors']))
    if r.get('N') is None:
        why.append('落盤 N 缺失')
    elif int(r['N']) != int(n_sched):
        why.append(f"落盤 N={r['N']} 與排程 N={n_sched} 不符")
    if r.get('replay') != 'PASS':
        why.append(f"重播 {r.get('replay')}")
    if r.get('physical') not in ('PASS', 'FAIL'):
        why.append(f"物理判定 {r.get('physical')}")
    if r.get('physical_insufficient'):
        why.append('物理判定證據不足：' + '、'.join(r['physical_insufficient']))
    if r.get('evidence_issues'):
        why += [e for e in r['evidence_issues'] if not e.startswith('物理判定證據不足')]
    return why


def pair_problems(r5, r1, n5, n1, cfg_rc):
    why = [f'H5：{w}' for w in run_problems(r5, n5)] + [f'H1：{w}' for w in run_problems(r1, n1)]
    if cfg_rc != 0:
        why.append(f'配對設定核對 rc={cfg_rc}（N 以外有差異或無法核對）')
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.plan), delimiter='\t'))
    sched_n = {r['rid']: int(r['N']) for r in rows}
    runs = {}
    for r in rows:
        rid = r['rid']
        st_f = os.path.join(RUNS, rid + '.status.json')
        st = json.load(open(st_f)) if os.path.exists(st_f) else {'class': 'no_status'}
        ent = {'idx': r['idx'], 'case': r['case'], 'method': r['method'],
               'pair': r['pair'], 'order': r['order'],
               'expect_open_m': float(r['open_m']), 'status': st.get('class')}
        if st.get('class') == 'executed':
            ent.update(C0.analyse(rid, expect_open_m=float(r['open_m'])))
        ent['problems'] = run_problems(ent, r['N'])
        runs[rid] = ent
    # 配對內設定核對（只允許 solver.N）
    pairs = {}
    for r in rows:
        pairs.setdefault(r['pair'], {})[r['method']] = r['rid']
    cfg = {}
    for p, m in pairs.items():
        if 'H5' in m and 'H1' in m and all(not runs[m[k]]['problems'] for k in ('H5', 'H1')):
            rc, txt = C0.sh(['horizon_config_check.py', os.path.join(RUNS, m['H5']),
                             os.path.join(RUNS, m['H1']), '--allow', 'solver.N'])
            cfg[p] = {'rc': rc, 'text': txt.strip()}
    # 按案例的配對差
    by_case = {}
    for p, m in pairs.items():
        r5, r1 = runs.get(m.get('H5')), runs.get(m.get('H1'))
        case = (r5 or r1)['case']
        why = pair_problems(r5, r1, sched_n.get(m.get('H5')), sched_n.get(m.get('H1')),
                            cfg.get(p, {}).get('rc'))
        ok = not why
        entry = {'pair': p, 'order': (r5 or r1)['order'], 'usable': ok,
                 'excluded_because': why,
                 'task_outcome': {'H5': (r5 or {}).get('physical'), 'H1': (r1 or {}).get('physical')}}
        if ok:
            entry['diff_H1_minus_H5'] = {k: (None if r5['m'][k] is None or r1['m'][k] is None
                                             else round(r1['m'][k] - r5['m'][k], 6))
                                         for k in r5['m']}
            entry['H5'] = r5['m']
            entry['H1'] = r1['m']
            entry['physical'] = {'H5': r5['physical'], 'H1': r1['physical']}
        by_case.setdefault(case, []).append(entry)
    summary = {}
    for case, ents in by_case.items():
        use = [e for e in ents if e['usable']]
        keys = list(use[0]['diff_H1_minus_H5']) if use else []
        summary[case] = {
            'n_pairs_usable': len(use), 'pairs': ents,
            'direction': {k: ('兩對同向↑' if len(use) == 2 and all((e['diff_H1_minus_H5'][k] or 0) > 0 for e in use)
                              else '兩對同向↓' if len(use) == 2 and all((e['diff_H1_minus_H5'][k] or 0) < 0 for e in use)
                              else '未呈現一致方向') for k in keys}}
    out = {'plan': a.plan, 'runs': runs, 'config_check': cfg, 'by_case': summary,
           'note': ('每案例 2 對，探索性比較；只報各對差與方向，不估成功率、不做顯著性；'
                    '非 executed 趟次列為證據／協定問題，不進配對差')}
    print(json.dumps({rid: {k: v.get(k) for k in ('case', 'method', 'status', 'replay', 'physical',
                                                  'physical_failed', 'physical_insufficient',
                                                  'problems')}
                      for rid, v in runs.items()}, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
