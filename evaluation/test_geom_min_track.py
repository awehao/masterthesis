#!/usr/bin/env python3
"""已知序列測試：最小值在中段時，MinTracker 取到中段、舊寫法取到末列（防止再把末列值當最小值）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom_min_track import MinTracker, buggy_min  # noqa: E402

seq = [0.150, 0.080, 0.020, 0.060, 0.140]           # m；真最小 20 mm 在第 2 列
dets = [{'classes': {'finger': {'d_pred': d, 'who': f'w{i}'}}} for i, d in enumerate(seq)]
t = MinTracker()
for i, det in enumerate(dets):
    t.update(det, i)
ok = True
got = t.as_mm()['finger']
ok &= got[0] == 20.0 and got[2] == 2
bug = buggy_min(dets)['finger']
ok &= bug[0] == 140.0 and bug[2] == 4                # 舊寫法回傳末列
# 單一元素、全相等、負值
for s, exp in (([0.03], (30.0, 0)), ([0.05, 0.05, 0.05], (50.0, 0)), ([0.01, -0.002, 0.004], (-2.0, 1))):
    m = MinTracker()
    for i, d in enumerate(s):
        m.update({'classes': {'base': {'d_pred': d, 'who': 'x'}}}, i)
    g = m.as_mm()['base']
    ok &= (g[0], g[2]) == exp
print('MinTracker：真最小', got, '；舊寫法', bug, '；', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
