#!/usr/bin/env python3
"""各模式／階段控制拍計算時間（p50、p95、p99、最大）與超過預算的拍數。用法：tick_budget.py <d2w.npz> [預算 ms，預設 50]"""
import json, sys, collections
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
L = [json.loads(str(x)) for x in z['log']]
by = collections.defaultdict(list)
for e in L:
    if e.get('compute_ms') is not None and e.get('mode'):
        by[(e['mode'], e.get('phase'))].append(e['compute_ms'])
allc = np.concatenate([np.array(v) for v in by.values()])
print('%-26s %6s %6s %6s %6s %6s %s' % ('mode/phase', 'n', 'p50', 'p95', 'p99', 'max', '>%.0fms' % budget))
for k, v in sorted(by.items(), key=lambda kv: -max(kv[1])):
    a = np.array(v)
    print('%-26s %6d %6.1f %6.1f %6.1f %6.1f %d' % (str(k)[:26], len(a), np.median(a), np.percentile(a, 95), np.percentile(a, 99), a.max(), int((a > budget).sum())))
print('ALL n=%d p50 %.1f p99 %.1f max %.1f  >%.0fms: %d (%.2f%%)' % (len(allc), np.median(allc), np.percentile(allc, 99), allc.max(), budget, (allc > budget).sum(), 100 * (allc > budget).mean()))
