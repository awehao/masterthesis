#!/usr/bin/env python3
"""build_carry 報告欄索引（Codex 20261103_075357）：已重取樣的路徑再重取樣可能少一點，舊寫法索引超界（IndexError）。確認修正後能回報，且承接路徑不受報告欄影響。
smooth_shift_max 是重取樣對應點的位移統計，不是嚴格的最大幾何偏移界。
    python3 evaluation/test_bottle_place_report_index.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_n2_path as NP                             # noqa: E402
import bottle_place_ctrl as PC                          # noqa: E402

R = PC.REG
n_over = 0
for fn, key in (('f1_offline_plan_rev_box_lchan_dx0.6.json', 'carry_far'), ('f1_offline_plan_rev_box_lchan_dx0.6.json', 'carry_near'), ('f1_offline_plan_rev_box.json', 'carry_far')):
    d = json.load(open(os.path.join(HERE, 'results', 'bottle', fn), encoding='utf-8'))[key]
    P = np.asarray(d['P_region'], float)[:, :2]
    raw0 = NP.resample(P)
    n_over += int(len(NP.resample(raw0)) < len(raw0))
    cp, info = PC.build_carry(P, P[0], np.array([1.0, 0.0]), float(d['goal'][0]), -0.994, R, v_max=0.25)
    assert cp is not None and np.isfinite(info['smooth_shift_max']), (fn, key, info)
    # 路徑不受報告欄影響：照同一流程重建，逐點相同
    raw = PC.smooth_path(raw0, R['smooth_iters'])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    k = int(np.argmin(np.linalg.norm(raw - P[0], axis=1)[s <= 1.0]))
    k2 = int(np.searchsorted(s, s[k] + R['conn_len']))
    t3 = raw[k2 + 1] - raw[k2]
    conn = PC.bezier(P[0], np.array([1.0, 0.0]), raw[k2], t3 / max(np.linalg.norm(t3), 1e-9))
    J = PC.join_lane_y(np.vstack([conn, raw[k2 + 1:]]), float(d['goal'][0]), -0.994, R['conn_len'])
    assert np.array_equal(J, cp.P), (fn, key)
    print(fn, key, '通過', 'shift', round(info['smooth_shift_max'], 4), '舊寫法會越界', len(NP.resample(raw0)) < len(raw0))
assert n_over >= 1, '這組路徑沒有一條會讓舊寫法越界，測試沒有覆蓋到原缺陷'
print('全部通過；舊寫法會越界的路徑數', n_over)
