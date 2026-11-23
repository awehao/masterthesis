#!/usr/bin/env python3
"""v2.1 運動中偏移觀測的必要案例（Codex 20261005_125840）：靜止、正／負等速、加減速暫態、缺資料。

合成資料用手臂模型本身：每物理步 q⁺ = q + α(s − q) + α d、s⁺ = s + dt_p·ṡ；每 0.05 s 取樣一次。
"""
import math
import sys

import numpy as np

from offset_moving import MovingCfg, MovingOffsetObserver

ALPHA, DTP, DT = 0.095, 0.01, 0.05
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def simulate(sdot_fn, d_true=0.020, T=3.0):
    """回傳 [(t, s, q)]，每 DT 一筆；sdot_fn(t) 給第 0 軸的設定點速率（其他軸 0）。"""
    q = np.zeros(6) + d_true
    s = np.zeros(6)
    out, t, n = [], 0.0, int(round(DT / DTP))
    for k in range(int(T / DTP)):
        if k % n == 0:
            out.append((t, s.copy(), q.copy()))
        v = np.zeros(6)
        v[0] = sdot_fn(t)
        q = q + ALPHA * (s - q) + ALPHA * d_true
        s = s + DTP * v
        t += DTP
    return out


def run(samples):
    ob = MovingOffsetObserver(MovingCfg(), [ALPHA] * 6, DTP)
    return [ob.observe(t, s, q) for t, s, q in samples]


# 1 靜止：d_obs ≈ d_true（20 mrad）
r = run(simulate(lambda t: 0.0))
ok = [x for x in r if x['ok']]
check('static_obs_equals_d', ok and abs(ok[-1]['d_obs'][0] - 0.020) < 1e-6, ok[-1] if ok else r[-1])
# 2 正等速 0.1 rad/s：模型一致公式 ⇒ 20 mrad（舊提案 0.095 s 會偏 1 mrad）
r = run(simulate(lambda t: 0.1))
ok = [x for x in r if x['ok']]
check('pos_const_vel_obs_equals_d', ok and abs(ok[-1]['d_obs'][0] - 0.020) < 1e-4, ok[-1]['d_obs'][0] if ok else None)
naive = ok[-1]['d_obs'][0] - (DTP / ALPHA - 0.095) * 0.1 if ok else None
check('old_coefficient_would_be_biased', naive is not None and abs(naive - 0.019) < 2e-4, naive)
# 3 負等速 −0.1 rad/s
r = run(simulate(lambda t: -0.1))
ok = [x for x in r if x['ok']]
check('neg_const_vel_obs_equals_d', ok and abs(ok[-1]['d_obs'][0] - 0.020) < 1e-4, ok[-1]['d_obs'][0] if ok else None)
# 4 加速暫態：速度在 1.0 s 由 0 跳到 0.1 rad/s ⇒ 1.0 s 後 0.3 s 窗內不給觀測；之後恢復
smp = simulate(lambda t: 0.0 if t < 1.0 else 0.1)
r = run(smp)
win = [x for (t, _, _), x in zip(smp, r) if 1.04 < t < 1.25]   # 1.0 s 那個樣本取在跳變之前（仍靜止），不算暫態
after = [x for (t, _, _), x in zip(smp, r) if t > 1.6 and x['ok']]
check('accel_transient_no_obs', win and not any(x['ok'] for x in win), [x['why'] for x in win])
check('after_transient_obs_recovers', after and abs(after[-1]['d_obs'][0] - 0.020) < 1e-4, after[-1] if after else None)
# 5 減速（線性降速）⇒ 窗內 ṡ 變化 > 門檻時不給觀測
smp = simulate(lambda t: max(0.0, 0.2 - 0.2 * t))
r = run(smp)
check('decel_ramp_no_obs', not any(x['ok'] for (t, _, _), x in zip(smp, r) if 0.4 < t < 0.9), [x['why'] for x in r[8:18]])
# 6 缺資料：非有限、時間不前進、形狀錯 ⇒ 不給觀測
ob = MovingOffsetObserver(MovingCfg(), [ALPHA] * 6, DTP)
for t, s, q in simulate(lambda t: 0.0)[:10]:
    ob.observe(t, s, q)
x = ob.observe(0.5, [float('nan')] + [0.0] * 5, np.zeros(6))
check('nonfinite_no_obs', not x['ok'] and x['why'] == 'nonfinite', x)
x = ob.observe(0.45, np.zeros(6), np.zeros(6))
check('time_not_increasing_no_obs', not x['ok'] and x['why'] == 'time_not_increasing', x)
x = ob.observe(0.6, np.zeros(5), np.zeros(6))
check('bad_shape_no_obs', not x['ok'] and x['why'] == 'bad_shape', x)
# 7 窗口未滿 ⇒ 不給觀測
ob = MovingOffsetObserver(MovingCfg(), [ALPHA] * 6, DTP)
xs = [ob.observe(t, s, q) for t, s, q in simulate(lambda t: 0.1)[:4]]
check('window_not_full_no_obs', not any(x['ok'] for x in xs), [x['why'] for x in xs])

print(f'{12 - len(fails)} 通過、{len(fails)} 失敗')
sys.exit(1 if fails else 0)
