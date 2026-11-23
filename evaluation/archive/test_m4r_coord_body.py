#!/usr/bin/env python3
"""door_regrasp_m4r_policy2._coord_body 測試（10/31 R3 本體逐軸限幅）：
  1. 上限很大時與 policy1._coord（上限同樣很大）逐元素相同；
  2. 上限有效時，本體 vx、vy ∈ [−v_cap, v_cap]、ω ∈ [−w_cap, w_cap]，且未飽和分量與無限幅值相同；
  3. 位姿非有限 ⇒ None。
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_regrasp_m4r_policy2 as M4R                  # noqa: E402
import door_regrasp_policy1 as P1                       # noqa: E402

cls = M4R.m4r(object)
rng = np.random.default_rng(7)
ok = True
for _ in range(500):
    th = float(rng.uniform(-math.pi, math.pi))
    m = SimpleNamespace(base_pose=[float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), th])
    b_ref = [m.base_pose[0] + float(rng.normal(0, 0.02)), m.base_pose[1] + float(rng.normal(0, 0.02)), th + float(rng.normal(0, 0.1))]
    ff = rng.normal(0, 0.04, 3)
    big = SimpleNamespace(r=dict(P1.REG, v_cap=1e9, w_cap=1e9))
    a = P1.Policy1._coord(big, m, b_ref, ff) if hasattr(P1, 'Policy1') else None
    if a is None:                                                      # policy1 的類名不固定：直接取函式
        fn = next(v for v in vars(P1).values() if isinstance(v, type) and hasattr(v, '_coord'))._coord
        a = fn(big, m, b_ref, ff)
    b = cls._coord_body(big, m, b_ref, ff, 1e9, 1e9)
    ok &= np.allclose(np.nan_to_num(a, nan=-7), np.nan_to_num(b, nan=-7), atol=1e-15, rtol=0)
    v, w = 0.0525, 0.285
    c = cls._coord_body(big, m, b_ref, ff, v, w)
    ok &= abs(c[2]) <= v + 1e-15 and abs(c[3]) <= v + 1e-15 and abs(c[4]) <= w + 1e-15
    for j, lim in ((2, v), (3, v), (4, w)):
        if abs(b[j]) < lim:
            ok &= abs(c[j] - b[j]) < 1e-15
ok &= cls._coord_body(big, SimpleNamespace(base_pose=[0.0, float('nan'), 0.0]), [0, 0, 0], [0, 0, 0], 0.05, 0.2) is None
print('m4r _coord_body：', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
