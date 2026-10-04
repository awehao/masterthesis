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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.plan), delimiter='\t'))
    runs = {}
    for r in rows:
        rid = r['rid']
        st_f = os.path.join(RUNS, rid + '.status.json')
        st = json.load(open(st_f)) if os.path.exists(st_f) else {'class': 'no_status'}
        ent = {'idx': r['idx'], 'case': r['case'], 'method': r['method'],
               'pair': r['pair'], 'order': r['order'],
               'expect_open_m': float(r['open_m']), 'status': st.get('class')}
        if st.get('class') == 'executed':
            res = C0.analyse(rid, expect_open_m=float(r['open_m']))
            if res.get('N') is not None and int(res['N']) != int(r['N']):
                ent['protocol_issue'] = f"落盤 N={res['N']} 與排程 N={r['N']} 不符"
            ent.update(res)
        runs[rid] = ent
    # 配對內設定核對（只允許 solver.N）
    pairs = {}
    for r in rows:
        pairs.setdefault(r['pair'], {})[r['method']] = r['rid']
    cfg = {}
    for p, m in pairs.items():
        if 'H5' in m and 'H1' in m and all(runs[m[k]]['status'] == 'executed' for k in ('H5', 'H1')):
            rc, txt = C0.sh(['horizon_config_check.py', os.path.join(RUNS, m['H5']),
                             os.path.join(RUNS, m['H1']), '--allow', 'solver.N'])
            cfg[p] = {'rc': rc, 'text': txt.strip()}
    # 按案例的配對差
    by_case = {}
    for p, m in pairs.items():
        r5, r1 = runs.get(m.get('H5')), runs.get(m.get('H1'))
        case = (r5 or r1)['case']
        ok = (r5 and r1 and 'm' in r5 and 'm' in r1 and cfg.get(p, {}).get('rc') == 0
              and r5.get('replay') == 'PASS' and r1.get('replay') == 'PASS')
        entry = {'pair': p, 'order': (r5 or r1)['order'], 'usable': bool(ok)}
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
                                                  'protocol_issue')}
                      for rid, v in runs.items()}, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
