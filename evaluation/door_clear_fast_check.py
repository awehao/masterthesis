#!/usr/bin/env python3
"""door_clear_fast.clear_full 與原 cfg_clear_full 的等值核對：隨機全身構型（關節限位內、底盤在門前區）＋把手位姿擾動。逐項比數值與 ok。"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_clear_fast as F                                          # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402

R3 = F.R3
reg = json.load(open(os.path.join(HERE, 'results/vision/m8_door_reaim_viewpoints_r1.json')))
T0 = np.array(reg['source']['T_WH_est'])
rng = np.random.default_rng(20261112)
lo, hi = np.asarray(LITE6_SAFE.lower, float), np.asarray(LITE6_SAFE.upper, float)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
mx = dict(env=0.0, base=0.0, self=0.0, bar_tool=0.0)
n_ok_diff, n_ok, t_a, t_b, who_diff = 0, 0, 0.0, 0.0, 0
vq = [np.array(v['q_arm'], float) for v in reg['viewpoints'] if v['status'] == 'ok']
for i in range(N):
    q = np.zeros(9)
    q[:3] = [rng.uniform(0.2, 1.2), rng.uniform(-1.8, -0.3), rng.uniform(1.0, 2.1)]
    q[3:] = rng.uniform(lo, hi) if i % 2 == 0 else np.clip(vq[i % len(vq)] + rng.normal(0, 0.25, 6), lo, hi)     # 一半全域隨機、一半在已登錄視點附近
    T = T0.copy()
    T[:3, 3] += rng.normal(0, 0.02, 3)
    w = time.perf_counter()
    a = R3.cfg_clear_full(q, T)
    t_a += time.perf_counter() - w
    w = time.perf_counter()
    b = F.clear_full(q, T)
    t_b += time.perf_counter() - w
    for k in mx:
        mx[k] = max(mx[k], abs(a[k] - b[k]))
    n_ok_diff += a['ok'] != b['ok']
    n_ok += a['ok']
    who_diff += a['env_who'] != b['env_who']
out = dict(n=N, max_abs_diff_m=mx, ok_mismatch=int(n_ok_diff), n_ok_true=int(n_ok), env_who_mismatch=int(who_diff), ms_per_cfg=dict(original=round(t_a / N * 1e3, 2), fast=round(t_b / N * 1e3, 2)),
           equal=bool(max(mx.values()) < 1e-9 and n_ok_diff == 0))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(out)
sys.exit(0 if out['equal'] else 1)
