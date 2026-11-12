#!/usr/bin/env python3
"""scan_safety_shield._limit：新版（wheel_rows 預設 False）與舊版（versions/scan_safety_shield_0d228dc4.py）逐值相同核對。隨機命令×隨機點群。"""
import importlib.util, json, os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import nav_guard_shield_chain_check as C                            # noqa: E402  （帶入 ROS 模組替身）
sp = importlib.util.spec_from_file_location('shield_old', os.path.join(HERE, 'versions', 'scan_safety_shield_0d228dc4.py'))
OLD = importlib.util.module_from_spec(sp); sp.loader.exec_module(OLD)
rng = np.random.default_rng(1)
N, mx, nfb = 5000, 0.0, 0
for i in range(N):
    n = int(rng.integers(0, 200))
    ang, r = rng.uniform(-np.pi, np.pi, n), rng.uniform(0.33, 2.3, n)
    pts = np.c_[r * np.cos(ang), r * np.sin(ang)]
    u = rng.uniform(-1, 1, 3) * [0.4, 0.4, 1.5]
    age = float(rng.uniform(0, 0.4))
    a = OLD.ScanSafetyShield._limit(C.S, u.copy(), pts.copy(), age)
    b = C.SH.ScanSafetyShield._limit(C.S, u.copy(), pts.copy(), age)
    mx = max(mx, float(np.abs(a[0] - b[0]).max()), *(abs(float(x) - float(y)) for x, y in zip(a[1:6], b[1:6]) if np.isfinite(x) and np.isfinite(y)))
    assert a[6] == b[6]
    nfb += bool(a[6])
out = dict(n=N, max_abs_diff=mx, n_fallback=nfb, equal=bool(mx == 0.0))
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(out); sys.exit(0 if out['equal'] else 1)
